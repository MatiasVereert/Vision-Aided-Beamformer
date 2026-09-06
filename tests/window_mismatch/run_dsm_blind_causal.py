"""
NM_MVDR_DSM_BLIND: cuanto cuesta la CADENA CAUSAL end-to-end (y el warp encima).

La loss del banco (dB de SINR contra el oracle) sigue bien al SDR pero NO a PESQ
ni a SIR -- lo dice el propio aviso de tests/dsm_blind_an_sweep.py -- asi que la
decision sobre causalidad hay que tomarla con las metricas del benchmark.

Filas:
    blind_prod        : el sistema de hoy (hamming, pico global + stretch, **4).
    blind_rect_p8     : rect + sintesis con taper + sharpen 8, mascara NO causal.
    blind_rect_p8_cau : lo mismo con causal=True (escala fija, sin stretch).
    blind_rect_p8_warp: causal + warp calibrado (dsm_blind_warp_calib.py).

Uso:
    python tests/window_mismatch/run_dsm_blind_causal.py [--full]
"""

import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = "/home/matias/Documents/Tesis/Vision-Aided-Beamformer"
sys.path.insert(0, os.path.join(ROOT, "src"))

from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search   # noqa: E402
from evaluation.bf_wrappers import NM_MVDR_DSM_BLIND                        # noqa: E402
from propagation.mird_loader import MirdDatasetProvider                     # noqa: E402

MODEL_1 = f"{ROOT}/src/dnn_denoise/models/model_quant_1.tflite"
MODEL_2 = f"{ROOT}/src/dnn_denoise/models/model_quant_2.tflite"
OUT_DIR = os.environ.get("SWEEP_OUT", "tests/dataset_out/dsm_blind_causal")


def build_processors():
    P = NM_MVDR_DSM_BLIND
    procs = {
        "blind_prod": P(win_type='hamming', sharpen_exp=4.0),
        "blind_rect_p8": P(win_type='rect', synth='hann', sharpen_exp=8.0),
        "blind_rect_p8_cau": P(win_type='rect', synth='hann', sharpen_exp=8.0,
                               causal=True),
    }
    npz = os.path.join(ROOT, "tests/dataset_out/dsm_blind_warp_calib/warp_params.npz")
    if os.path.exists(npz):
        z = np.load(npz)
        th = (z['a_s'], z['b_s'], z['a_n'], z['b_n'])
        procs["blind_rect_p8_warp"] = P(win_type='rect', synth='hann',
                                        sharpen_exp=8.0, causal=True, mask_warp=th)
    else:
        print(f"[!] falta {npz}")
    return procs


def main():
    try:
        itp1 = tf.lite.Interpreter(model_path=MODEL_1); itp1.allocate_tensors()
        itp2 = tf.lite.Interpreter(model_path=MODEL_2); itp2.allocate_tensors()
    except Exception as e:
        print(f"[!] DTLN no cargado: {e}")
        itp1 = itp2 = None

    provider = MirdDatasetProvider(root_dir=os.path.abspath(f"{ROOT}/tools/data/rirs/mird"))
    base_config = {
        'fs': 16000, 'duration': 15, 't_early': 0.050,
        'array_center': [3.0, 3.0, 1.2], 'mird_spacing': "3-3-3-8-3-3-3",
        'snr_db': 60.0,
        'source_path': f"{ROOT}/tools/data/signals/p002_emo_adoration_sentences.wav",
        'interf_paths': [f"{ROOT}/tools/data/signals/techno_gated commune.wav"],
        'wpe_taps': 7, 'wpe_delay': 3, 'wpe_alpha': 0.9999,
        'wpe_stft_size': 512, 'wpe_stft_shift': 128,
        'wpe_fixed_bits': None, 'wpe_fixed_round': 'nearest', 'wpe_backend': 'cov',
        'wpe_block_L': 512, 'wpe_block_shift': 2, 'wpe_block_iters': 2,
        'wpe_block_reg': 1e-6, 'wpe_block_solver': 'cholesky', 'wpe_block_mode': 'resolve',
        'stft_window': 512, 'stft_overlap': 384,
        'eval_references': ['anechoic', 'early', 'reverberant'],
        'dtln_model_path': MODEL_1, 'dtln_model2_path': MODEL_2,
    }
    full = "--full" in sys.argv
    param_grid = {
        'rt60': [0.360, 0.610], 'target_angle': [0], 'target_dist': [1.0],
        'interf_configs': ([[(45, 1.0)], [(90, 2.0)]] if full else [[(45, 1.0)]]),
        'isir_db': [-5, 0], 'mismatch_gain': [0], 'mismatch_phase': [0],
        'use_wpe': [True], 'wpe_method': ['online'], 'wpe_taps': [7], 'wpe_delay': [2],
        'error_angle_deg': [0.0], 'error_distance_m': [0.0],
    }
    df = run_mird_grid_search(
        grid_params=param_grid, dataset_provider=provider,
        processors=build_processors(), scene_base_config=base_config,
        output_dir=OUT_DIR, interpreter_1=itp1, interpreter_2=itp2,
        save_catalog=False, apply_dtln_post=False)
    summarize(df)
    return df


def summarize(df, ref='early'):
    metrics = ['PESQ', 'STOI', 'SI-SDR', 'SDR', 'SIR', 'SAR']
    cols = [f"proc_{m}_{ref}" for m in metrics if f"proc_{m}_{ref}" in df.columns]
    pd.set_option('display.width', 220)
    print(f"\n=== DSM_BLIND causal (ref '{ref}') ===")
    print(df.groupby('processor')[cols].mean().round(3).to_string())
    for a, b in (("blind_rect_p8_cau", "blind_rect_p8"),
                 ("blind_rect_p8_cau", "blind_prod"),
                 ("blind_rect_p8_warp", "blind_rect_p8_cau"),
                 ("blind_rect_p8_warp", "blind_prod")):
        A = df[df.processor == a].reset_index(drop=True)
        B = df[df.processor == b].reset_index(drop=True)
        if A.empty or B.empty:
            continue
        n = min(len(A), len(B))
        print(f"\n--- {a} - {b} ({n} escenas) ---")
        for c in cols:
            d = A[c].to_numpy()[:n] - B[c].to_numpy()[:n]
            print(f"   {c:20s} media {np.nanmean(d):+7.3f}  mediana {np.nanmedian(d):+7.3f}"
                  f"  gana {int(np.sum(d > 0))}/{n}")


if __name__ == "__main__":
    main()
