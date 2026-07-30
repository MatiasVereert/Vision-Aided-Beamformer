"""
test_frontend_decomposition.py
==============================
Valida los bloques matematicos sobre los que se apoya el fix de las referencias
del ORACLE (que sus covarianzas se computen en el MISMO dominio -- HW mismatch +
WPE -- que la senal que se filtra).

No necesita MIRD ni DTLN: usa una senal multicanal sintetica reverberante-ish.

Chequea dos invariantes EXACTAS:

  1. HW (Microphone._apply_mismatch) es LINEAL:
        _apply_mismatch(target) + _apply_mismatch(interf) == _apply_mismatch(target+interf)
     => hw_target = _apply_mismatch(target_clean)
        hw_noise  = mic_signals_degraded - hw_target
        cumplen  hw_target + hw_noise == mic_signals_degraded   (el ruido termico,
        aditivo, cae entero en hw_noise, que es donde corresponde).

  2. WPE float (process_wpe_online_with_components) da la descomposicion EXACTA:
        - z_u (mezcla dereverberada) == process_wpe_online(mezcla)   (bit-identico:
          el filtrado de componentes NO toca el estado del filtro de la mezcla).
        - wpe_target + wpe_noise == z_u  (== mic_signals_ready).

USO:
    conda activate tesis_beam
    python tests/test_frontend_decomposition.py
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np

from beamforming.array.microphone import Microphone
from dereverberation.nara_wrappers import (
    process_wpe_online,
    process_wpe_online_with_components,
)

FS = 16000
M = 8
DUR = 4.0
SEED = 1234567
WPE = dict(taps=7, delay=3, alpha=0.9999, stft_size=512, stft_shift=128)


def _synthetic_component(rng, n, M, n_taps=1200, tail_scale=0.3):
    """Fuente 1/f coloreada convolucionada con RIRs cortas por canal (direct+cola)."""
    src = rng.standard_normal(n)
    src = np.cumsum(src) - np.cumsum(np.concatenate([[0.0], src[:-1]]))
    src = src / (np.std(src) + 1e-9)
    x = np.zeros((M, n))
    for m in range(M):
        h = np.zeros(n_taps)
        h[10 + m * 3] = 1.0
        h += rng.standard_normal(n_taps) * np.exp(-np.arange(n_taps) / 300.0) * tail_scale
        x[m] = np.convolve(src, h)[:n]
    return x


def main():
    rng = np.random.default_rng(SEED)
    n = int(FS * DUR)

    # Dos componentes limpias multicanal independientes (target e interferencia).
    target_clean = _synthetic_component(rng, n, M, tail_scale=0.25)
    interf_clean = _synthetic_component(rng, n, M, tail_scale=0.40)
    # Escala global comun (como mix_and_normalize): la mezcla es SUMA exacta.
    mix_clean = target_clean + interf_clean
    gscale = 0.99 / (np.max(np.abs(mix_clean)) + 1e-10)
    target_clean *= gscale
    interf_clean *= gscale
    mix_clean = target_clean + interf_clean

    ok = True

    # -----------------------------------------------------------------
    # 1. HW: linealidad del mismatch + descomposicion por diferencia
    # -----------------------------------------------------------------
    for gain_db, phase_deg, snr_db in [(0.0, 0.0, 60.0), (1.5, 4.0, 45.0)]:
        mic = Microphone(fs=FS)
        mic.set_seed(SEED)
        mic.set_custom_errors(std_gain_dB=gain_db, std_phase_deg=phase_deg, snr_dB=snr_db)

        # emulate = mismatch(mezcla) + ruido termico (fija el patron por (seed, M)).
        degraded = mic.emulate(mix_clean)

        # Linealidad de _apply_mismatch (mismo patron ya cacheado por emulate()).
        mm_t = mic._apply_mismatch(target_clean)
        mm_i = mic._apply_mismatch(interf_clean)
        mm_mix = mic._apply_mismatch(mix_clean)
        lin_err = np.max(np.abs((mm_t + mm_i) - mm_mix))

        # Descomposicion del fix: hw_target = mismatch(target); hw_noise = degraded - hw_target.
        hw_target = mm_t
        hw_noise = degraded - hw_target
        sum_err = np.max(np.abs((hw_target + hw_noise) - degraded))
        # el ruido termico debe estar entero en hw_noise: hw_noise - mm_i == termico
        thermal = hw_noise - mm_i
        # termico ~ 0 a snr alto, y NO nulo a snr bajo (sanity: es ruido, no target)
        thermal_energy = np.sqrt(np.mean(thermal ** 2))

        print(f"[HW gain={gain_db}dB phase={phase_deg}deg snr={snr_db}] "
              f"lin_err={lin_err:.2e}  sum_err={sum_err:.2e}  "
              f"thermal_rms={thermal_energy:.2e}")
        # Linealidad exacta (a redondeo fp de la FFT de fase) y suma exacta.
        ok &= lin_err < 1e-9
        ok &= sum_err < 1e-12

    # -----------------------------------------------------------------
    # 2. WPE float: z_u bit-identico + descomposicion exacta target/ruido
    # -----------------------------------------------------------------
    mic = Microphone(fs=FS)
    mic.set_seed(SEED)
    mic.set_custom_errors(std_gain_dB=1.5, std_phase_deg=4.0, snr_dB=45.0)
    degraded = mic.emulate(mix_clean)
    hw_target = mic._apply_mismatch(target_clean)
    hw_noise = degraded - hw_target

    # Mezcla por el camino ORIGINAL (lo que hace hoy Node 4).
    z_ref = process_wpe_online(degraded, **WPE)
    # Mezcla + componentes en un solo pase (lo que hara el fix).
    z_u, (wpe_target, wpe_noise) = process_wpe_online_with_components(
        u=degraded, components=[hw_target, hw_noise], **WPE
    )

    # a) z_u debe ser identico a process_wpe_online(mezcla): la mezcla que filtran
    #    los beamformers (mic_signals_ready) NO cambia con el fix.
    zu_err = np.max(np.abs(z_u - z_ref))
    # b) descomposicion exacta: wpe_target + wpe_noise == z_u == mic_signals_ready.
    decomp_err = np.max(np.abs((wpe_target + wpe_noise) - z_u))

    print(f"[WPE float] z_u vs process_wpe_online = {zu_err:.2e}  "
          f"(target+noise) vs z_u = {decomp_err:.2e}")
    ok &= zu_err < 1e-9
    ok &= decomp_err < 1e-9

    print("\n" + ("[PASS] invariantes del front-end OK" if ok
                  else "[FAIL] alguna invariante no se cumple"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
