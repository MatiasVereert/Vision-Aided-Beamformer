"""
run_wpe_batch_mird.py
=====================
Ejecucion de prueba para el WPE batch educacional
(``dereverberation/nara_study.py/wpe_batch.py``).

Genera una escena reverberante multicanal con RIRs REALES medidas del dataset
MIRD (Bar-Ilan) usando ``propagation.simulate_acoustics_v1.SimAcoustic``, corre
el WPE batch iterativo y guarda los WAVs de entrada / salida para inspeccion
auditiva. Se usa el entorno de MAYOR reverberacion disponible (T60 = 610 ms).

Escena: una unica fuente de voz (sin interferencia) -> estudio puro de
dereverberacion. La referencia "ideal" es la parte early (directo + primeras
reflexiones) que el WPE deberia recuperar al suprimir la cola tardia.

USO:
    conda activate tesis_beam
    python tests/run_wpe_batch_mird.py
"""

import os
import sys

import numpy as np
from scipy.io import wavfile

# --- Rutas de import ---------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# El WPE batch vive en una carpeta cuyo nombre lleva un punto ("nara_study.py"),
# asi que no es importable como paquete: la agregamos directo al path.
WPE_DIR = os.path.join(SRC_DIR, "dereverberation", "nara_study.py")
if WPE_DIR not in sys.path:
    sys.path.insert(0, WPE_DIR)

from propagation.simulate_acoustics_v1 import SimAcoustic
from propagation.mird_loader import MirdDatasetProvider, generate_mird_linear_array
from wpe_batch import process_wpe_time_domain

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

# Parametros del WPE batch
WPE_PARAMS = dict(
    fs=FS,
    taps=15,        # longitud del filtro de prediccion (historia de reverb)
    delay=3,        # retardo de prediccion (protege el early)
    delta=2,        # semiventana para estimar la varianza (potencia)
    iterations=3,
    nperseg=512,
    noverlap=384,
)

REF_MIC = 0
OUT_DIR = os.path.join(REPO_ROOT, "tests", "wpe_batch_mird_out")


def build_reverberant_scene():
    """Escena de una fuente con RIRs reales MIRD a T60=610 ms."""
    mics = generate_mird_linear_array() + ARRAY_CENTER
    scene = SimAcoustic(mics, array_mismatch=0.0, duration=DURATION, fs=FS, seed=0)

    # Fuente frente al array (broadside, eje +X), a 1 m -> snap a grid MIRD.
    source_pos = ARRAY_CENTER + np.array([1.0, 0.0, 0.0])
    scene.set_source(SOURCE_WAV, gain=1.0, position=source_pos.reshape(1, 3))

    # Carga RIRs medidas, convoluciona y mezcla (solo fuente, sin interferencia).
    scene.import_rirs(MirdDatasetProvider(MIRD_ROOT), target_t60=TARGET_T60,
                      array_center=ARRAY_CENTER, spacing_cfg=SPACING_CFG)
    scene.convolve_signals(t_early=0.050)
    data = scene.mix_and_normalize(iSIR_dB=ISIR_DB, inter_normalization=False)
    return data


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

    print(f"[2/3] Corriendo WPE batch ({WPE_PARAMS['iterations']} iteraciones, "
          f"taps={WPE_PARAMS['taps']}, delay={WPE_PARAMS['delay']})...")
    wpe_out = process_wpe_time_domain(mic_signals, **WPE_PARAMS)   # (M, N_out)

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
    save_wav_shared_scale(os.path.join(OUT_DIR, "2_wpe_output.wav"), FS, y_out, scale)
    save_wav_shared_scale(os.path.join(OUT_DIR, "3_target_early_reference.wav"), FS, e_ref, scale)

    print("\nListo. Compara auditivamente 1_input vs 2_wpe_output:")
    print("la cola reverberante deberia reducirse acercandose a 3_target_early.")


if __name__ == "__main__":
    main()
