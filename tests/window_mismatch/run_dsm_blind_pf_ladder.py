"""
ESCALERA DE UNA VARIABLE POR VEZ, CON EL POST-FILTRO ENCENDIDO (smooth=0.5).

Motivo: todas las mediciones anteriores de esta linea de trabajo se hicieron con
`smooth=None`. Con el post-filtro la mascara deja de ser solo un PESO de
covarianza y pasa a ser una GANANCIA por bin sobre el espectro de salida, que es
un uso cualitativamente distinto. Una configuracion reportada por el usuario
sugiere que la conclusion ("la cadena causal + rect + taper no pierde") podria no
transferirse a ese camino.

La escalera cambia UN parametro por fila, partiendo del sistema en produccion:

    L1_base        hamming, exp 4, causal=False, sintesis = analisis
    L2_causal      + causal=True            -> aisla la CAUSALIDAD
    L3_exp8        + sharpen_exp=8          -> aisla el EXPONENTE
    L4_rect        + win_type='rect'        -> aisla el ANALISIS
    L5_synhann     + synth='hann'           -> aisla la SINTESIS

Todo lo demas fijo: alpha=0.99, min_loading=1e-9, smooth=0.5, pf_mask_src='fix'
(los defaults del wrapper, que son los que uso el usuario).

Uso:
    python tests/window_mismatch/run_dsm_blind_pf_ladder.py [--full]
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
OUT_DIR = os.environ.get("SWEEP_OUT", "tests/dataset_out/dsm_blind_pf_ladder")

SMOOTH = float(os.environ.get("SMOOTH", "0.5"))
COMMON = dict(min_loading=1e-9, alpha=0.99, smooth=SMOOTH)

LADDER = [
    ("L1_base",     dict(win_type=None,   sharpen_exp=4.0, causal=False, synth=None)),
    ("L2_causal",   dict(win_type=None,   sharpen_exp=4.0, causal=True,  synth=None)),
    ("L3_exp8",     dict(win_type=None,   sharpen_exp=8.0, causal=True,  synth=None)),
    ("L4_rect",     dict(win_type='rect', sharpen_exp=8.0, causal=True,  synth=None)),
    ("L5_synhann",  dict(win_type='rect', sharpen_exp=8.0, causal=True,  synth='hann')),
]


def build_processors():
    return {name: NM_MVDR_DSM_BLIND(**COMMON, **kw) for name, kw in LADDER}


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
    for ref in ('early', 'anechoic', 'reverberant'):
        summarize(df, ref)
    return df


def summarize(df, ref='early'):
    metrics = ['PESQ', 'STOI', 'SI-SDR', 'SDR', 'SIR', 'SAR']
    cols = [f"proc_{m}_{ref}" for m in metrics if f"proc_{m}_{ref}" in df.columns]
    if not cols:
        return
    pd.set_option('display.width', 220)
    order = [n for n, _ in LADDER]
    g = df.groupby('processor')[cols].mean().reindex(order).round(3)
    print(f"\n=== ESCALERA con smooth={SMOOTH} -- referencia '{ref}' ===")
    print(g.to_string())
    print(f"\n--- escalon a escalon (referencia '{ref}') ---")
    for (a, _), (b, _) in zip(LADDER[1:], LADDER[:-1]):
        A = df[df.processor == a].reset_index(drop=True)
        B = df[df.processor == b].reset_index(drop=True)
        if A.empty or B.empty:
            continue
        n = min(len(A), len(B))
        print(f"  {a} - {b}:")
        for c in cols:
            d = A[c].to_numpy()[:n] - B[c].to_numpy()[:n]
            print(f"     {c:22s} media {np.nanmean(d):+7.3f}  mediana {np.nanmedian(d):+7.3f}"
                  f"  gana {int(np.sum(d > 0))}/{n}")
    A = df[df.processor == LADDER[-1][0]].reset_index(drop=True)
    B = df[df.processor == LADDER[0][0]].reset_index(drop=True)
    if not A.empty and not B.empty:
        n = min(len(A), len(B))
        print(f"\n  TOTAL {LADDER[-1][0]} - {LADDER[0][0]}:")
        for c in cols:
            d = A[c].to_numpy()[:n] - B[c].to_numpy()[:n]
            print(f"     {c:22s} media {np.nanmean(d):+7.3f}  mediana {np.nanmedian(d):+7.3f}"
                  f"  gana {int(np.sum(d > 0))}/{n}")


if __name__ == "__main__":
    main()
