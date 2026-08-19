"""
run_wpe_online_mird.py
======================
Ejecucion de prueba para el WPE ONLINE (frame-by-frame, RLS) educacional
(``dereverberation/nara_study/wpe_online.py``).

Misma escena y forma de evaluar que ``run_wpe_batch_mird.py``: una fuente de voz
reverberada con RIRs REALES del dataset MIRD (Bar-Ilan) en el entorno de MAYOR
reverberacion (T60 = 610 ms). Se envuelve ``frame_online_WPE`` (que opera en el
dominio STFT) con STFT/iSTFT para procesar directo desde el tiempo, y se guardan
los WAVs para inspeccion auditiva.

USO:
    conda activate tesis_beam
    python tests/run_wpe_online_mird.py
"""

import os
import sys

import numpy as np
import scipy.signal as signal
from scipy.io import wavfile

# --- Rutas de import ---------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from propagation.simulate_acoustics_v1 import SimAcoustic
from propagation.mird_loader import MirdDatasetProvider, generate_mird_linear_array
from dereverberation.nara_study.wpe_online import frame_online_WPE

# --- Configuracion -----------------------------------------------------------
FS = 16000                 # frecuencia de trabajo (las RIRs de 48 kHz se remuestrean)
DURATION = 8               # s
ISIR_DB = 0                # irrelevante: no hay interferencia

MIRD_ROOT = os.path.join(REPO_ROOT, "tools", "data", "rirs", "mird")
SOURCE_WAV = os.path.join(REPO_ROOT, "tools", "data", "signals",
                          "p002_emo_adoration_sentences.wav")

TARGET_T60 = 0.610         # ENTORNO DE MAYOR RT DISPONIBLE (160 / 360 / 610 ms)
SPACING_CFG = "4-4-4-8-4-4-4"   # geometria que emula generate_mird_linear_array()
ARRAY_CENTER = np.array([3.0, 3.0, 1.2])

# Parametros del WPE online (mismos STFT que el batch para poder comparar)
TAPS = 15          # longitud del filtro de prediccion (historia de reverb)
DELAY = 3          # retardo de prediccion (protege el early)
DELTA = 2          # semiventana para estimar la varianza (potencia)
ALPHA = 0.99999    # factor de olvido del RLS
NPERSEG = 512
NOVERLAP = 384

REF_MIC = 0
OUT_DIR = os.path.join(REPO_ROOT, "tests", "wpe_online_mird_out")


def build_reverberant_scene():
    """Escena de una fuente con RIRs reales MIRD a T60=610 ms."""
    mics = generate_mird_linear_array() + ARRAY_CENTER
    scene = SimAcoustic(mics, array_mismatch=0.0, duration=DURATION, fs=FS, seed=0)

    # Fuente frente al array (broadside, eje +X), a 1 m -> snap a grid MIRD.
    source_pos = ARRAY_CENTER + np.array([1.0, 0.0, 0.0])
    scene.set_source(SOURCE_WAV, gain=1.0, position=source_pos.reshape(1, 3))

    scene.import_rirs(MirdDatasetProvider(MIRD_ROOT), target_t60=TARGET_T60,
                      array_center=ARRAY_CENTER, spacing_cfg=SPACING_CFG)
    scene.convolve_signals(t_early=0.050)
    data = scene.mix_and_normalize(iSIR_dB=ISIR_DB, inter_normalization=False)
    return data


def process_wpe_online_time_domain(audio_time, fs=FS, taps=TAPS, delay=DELAY,
                                   delta=DELTA, alpha=ALPHA,
                                   nperseg=NPERSEG, noverlap=NOVERLAP):
    """Envuelve frame_online_WPE (dominio STFT) para procesar desde el tiempo."""
    if audio_time.ndim == 1:
        audio_time = audio_time[np.newaxis, :]

    # STFT: SciPy devuelve (D, F, T); el WPE espera (F, D, T)
    _, _, Zxx = signal.stft(audio_time, fs=fs, nperseg=nperseg, noverlap=noverlap)
    Y_in = Zxx.transpose(1, 0, 2)

    X_hat = frame_online_WPE(Y_in, taps, delay, delta, alpha)

    Zxx_out = X_hat.transpose(1, 0, 2)
    _, audio_dereverb = signal.istft(Zxx_out, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return audio_dereverb


def save_wav_shared_scale(path, fs, sig, scale):
    """Guarda un WAV mono aplicando una escala compartida (comparacion justa)."""
    x = np.real(sig) * scale
    x = np.clip(x, -1.0, 1.0)
    wavfile.write(path, fs, (x * 32767).astype(np.int16))
    print(f"  -> {os.path.basename(path)}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[1/3] Generando escena reverberante MIRD (T60={TARGET_T60*1000:.0f} ms)...")
    data = build_reverberant_scene()

    mic_signals = data["mic_signals"]        # (M, N) mezcla reverberante
    target_early = data["target_early"]      # (M, N) referencia "ideal" (early)
    print(f"      mezcla: {mic_signals.shape}  (M canales x N muestras)")

    print(f"[2/3] Corriendo WPE ONLINE (RLS, alpha={ALPHA}, taps={TAPS}, delay={DELAY})...")
    wpe_out = process_wpe_online_time_domain(mic_signals)   # (M, N_out)

    # Alinea longitudes (la iSTFT puede devolver un largo distinto).
    n = min(mic_signals.shape[1], wpe_out.shape[1], target_early.shape[1])
    x_in = mic_signals[REF_MIC, :n]
    y_out = wpe_out[REF_MIC, :n]
    e_ref = target_early[REF_MIC, :n]

    # Escala compartida -> se puede comparar el nivel a oido (reverb suprimida).
    peak = max(np.max(np.abs(x_in)), np.max(np.abs(y_out)), np.max(np.abs(e_ref)))
    scale = 0.9 / (peak + 1e-12)

    print(f"[3/3] Guardando WAVs en {os.path.relpath(OUT_DIR, REPO_ROOT)}/ (mic {REF_MIC})...")
    save_wav_shared_scale(os.path.join(OUT_DIR, "1_input_reverberant.wav"), FS, x_in, scale)
    save_wav_shared_scale(os.path.join(OUT_DIR, "2_wpe_online_output.wav"), FS, y_out, scale)
    save_wav_shared_scale(os.path.join(OUT_DIR, "3_target_early_reference.wav"), FS, e_ref, scale)

    print("\nListo. Compara auditivamente 1_input vs 2_wpe_online_output:")
    print("la cola reverberante deberia reducirse progresivamente (el RLS converge en el tiempo).")


if __name__ == "__main__":
    main()
