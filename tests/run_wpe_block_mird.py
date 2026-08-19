"""
run_wpe_block_mird.py
=====================
Prueba de escucha del WPE BLOCK-ONLINE (Opcion B: re-solve por bloque via
Cholesky sobre ventana trailing) contra el WPE RLS-ONLINE ya existente, en la
MISMA escena reverberante con RIRs REALES del dataset MIRD (Bar-Ilan) a
T60 = 610 ms (entorno de mayor reverberacion disponible).

Exporta WAVs para inspeccion auditiva (mismo mic de referencia y escala
compartida, comparacion justa):
    1_input_reverberant.wav      -- mezcla reverberante
    2_wpe_online_output.wav      -- RLS online (nara_wrappers.process_wpe_online)
    3_wpe_block_output.wav       -- block online (nara_wrappers.process_wpe_block_online)
    4_target_early_reference.wav -- referencia "ideal" (early: directo + primeras refl.)

USO:
    conda activate tesis_beam
    python tests/run_wpe_block_mird.py
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

from propagation.simulate_acoustics_v1 import SimAcoustic
from propagation.mird_loader import MirdDatasetProvider, generate_mird_linear_array
from dereverberation.nara_wrappers import (
    process_wpe_online,
    process_wpe_block_online,
    block_wpe_warmup,
)

# --- Configuracion escena (identica a run_wpe_online_mird.py) ----------------
FS = 16000
DURATION = 8               # s
ISIR_DB = 0                # irrelevante: no hay interferencia

MIRD_ROOT = os.path.join(REPO_ROOT, "tools", "data", "rirs", "mird")
SOURCE_WAV = os.path.join(REPO_ROOT, "tools", "data", "signals",
                          "p002_emo_adoration_sentences.wav")

TARGET_T60 = 0.610
SPACING_CFG = "4-4-4-8-4-4-4"
ARRAY_CENTER = np.array([3.0, 3.0, 1.2])

# --- Parametros WPE (comunes online/block para comparar) ---------------------
TAPS = 15
DELAY = 3
STFT_SIZE = 512
STFT_SHIFT = 128           # frame = 8 ms @ 16 kHz -> F = 257 bins

# RLS online
ALPHA = 0.99999

# Block online (Opcion B)
BLOCK_L = 512              # ventana trailing de estadistica [frames] ~ 4.1 s
BLOCK_SHIFT = 64           # refresco de G cada [frames] ~ 512 ms
BLOCK_ITERS = 3           # iteraciones tipo-offline por re-solve
BLOCK_REG = 1e-6          # carga diagonal relativa (Cholesky)

REF_MIC = 0
OUT_DIR = os.path.join(REPO_ROOT, "tests", "wpe_block_mird_out")


def build_reverberant_scene():
    """Escena de una fuente con RIRs reales MIRD a T60=610 ms."""
    mics = generate_mird_linear_array() + ARRAY_CENTER
    scene = SimAcoustic(mics, array_mismatch=0.0, duration=DURATION, fs=FS, seed=0)

    source_pos = ARRAY_CENTER + np.array([1.0, 0.0, 0.0])
    scene.set_source(SOURCE_WAV, gain=1.0, position=source_pos.reshape(1, 3))

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
    print(f"[1/4] Generando escena reverberante MIRD (T60={TARGET_T60*1000:.0f} ms)...")
    data = build_reverberant_scene()

    mic_signals = data["mic_signals"]        # (M, N) mezcla reverberante
    target_early = data["target_early"]      # (M, N) referencia early
    print(f"      mezcla: {mic_signals.shape}  (M canales x N muestras)")

    print(f"[2/4] WPE RLS-ONLINE (alpha={ALPHA}, taps={TAPS}, delay={DELAY})...")
    online_out = process_wpe_online(
        mic_signals, taps=TAPS, delay=DELAY, alpha=ALPHA,
        stft_size=STFT_SIZE, stft_shift=STFT_SHIFT,
    )

    print(f"[3/4] WPE BLOCK-ONLINE (L={BLOCK_L}, block_shift={BLOCK_SHIFT}, "
          f"iters={BLOCK_ITERS}, taps={TAPS}, delay={DELAY})...")
    block_out = process_wpe_block_online(
        mic_signals, taps=TAPS, delay=DELAY, L=BLOCK_L, block_shift=BLOCK_SHIFT,
        iterations=BLOCK_ITERS, reg=BLOCK_REG, solver="cholesky",
        stft_size=STFT_SIZE, stft_shift=STFT_SHIFT,
    )

    # Alinea longitudes.
    n = min(mic_signals.shape[1], online_out.shape[1],
            block_out.shape[1], target_early.shape[1])
    x_in = mic_signals[REF_MIC, :n]
    y_on = online_out[REF_MIC, :n]
    y_bl = block_out[REF_MIC, :n]
    e_ref = target_early[REF_MIC, :n]

    peak = max(np.max(np.abs(x_in)), np.max(np.abs(y_on)),
               np.max(np.abs(y_bl)), np.max(np.abs(e_ref)))
    scale = 0.9 / (peak + 1e-12)

    print(f"[4/4] Guardando WAVs en {os.path.relpath(OUT_DIR, REPO_ROOT)}/ (mic {REF_MIC})...")
    save_wav_shared_scale(os.path.join(OUT_DIR, "1_input_reverberant.wav"), FS, x_in, scale)
    save_wav_shared_scale(os.path.join(OUT_DIR, "2_wpe_online_output.wav"), FS, y_on, scale)
    save_wav_shared_scale(os.path.join(OUT_DIR, "3_wpe_block_output.wav"), FS, y_bl, scale)
    save_wav_shared_scale(os.path.join(OUT_DIR, "4_target_early_reference.wav"), FS, e_ref, scale)

    # --- Metricas: SIEMPRE descartando el warmup del block ---------------------
    # El block-online arranca en frio (bypass) y con ventana parcial hasta juntar
    # L frames; esas muestras NO son del regimen de la Opcion B. Se recorta por la
    # frontera de warmup (ventana llena) el MISMO tramo para online y block, para
    # que la comparacion sea justa (el RLS tambien converge en ese arranque).
    _, warm = block_wpe_warmup(TAPS, DELAY, BLOCK_L, BLOCK_SHIFT, STFT_SHIFT)
    warm = min(warm, n - 1)
    print(f"\nWarmup descartado para metricas: {warm} muestras "
          f"({warm/FS:.2f} s, ventana llena de L={BLOCK_L} frames).")

    x_w, on_w, bl_w = x_in[warm:], y_on[warm:], y_bl[warm:]

    # Metrica cruda de reduccion de energia de cola (proxy rapido, NO es PESQ):
    # energia residual respecto de la entrada (mas bajo = mas supresion).
    def rel_energy(y, ref):
        return 10 * np.log10((np.sum(y**2) + 1e-12) / (np.sum(ref**2) + 1e-12))
    print("Energia relativa a la entrada (proxy, no perceptual, post-warmup):")
    print(f"  online: {rel_energy(on_w, x_w):+.2f} dB   block: {rel_energy(bl_w, x_w):+.2f} dB")
    print("\nListo. Compara auditivamente 1_input vs 2_online vs 3_block "
          "(y 4_target como techo).")


if __name__ == "__main__":
    main()
