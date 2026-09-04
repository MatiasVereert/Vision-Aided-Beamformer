"""
Confirmacion en METRICAS REALES del warp de mascara sobre NM_MVDR_DSM_BLIND.

El barrido del proxy (tests/dsm_blind_an_sweep.py) dice que, sobre la mascara del
front-end ciego, el warp calibrado con a_n ~ 2.5-3 le gana al `stretch_sharpen`
actual en los DOS terminos de la loss (-0.68 dB de L_sinr, -0.21 de L_dist en
rt60 no visto). Pero ese proxy sigue bien al SDR y NO al SIR ni a PESQ, asi que
el valor hay que confirmarlo aca.

Ademas del rendimiento, el cambio saca del camino la UNICA etapa no causal que
le queda al DSM_BLIND: el stretch min-max necesita el min/max de todo el archivo.
Con el warp, la cadena entera es implementable online.

PROCESADORES
------------
    DSM_BLIND        el wrapper como esta hoy (stretch global + **4)
    DSM_BLIND_W25    idem con warp calibrado, a_n = 2.5      <- causal
    DSM_BLIND_W30    idem con warp calibrado, a_n = 3.0      <- causal
    DSM_BLIND_PF     el de hoy + post-filtro espectral (smooth=0.5)
    DSM_BLIND_W25_PF warp + post-filtro
    ORACLE_SCM       cota superior

USO
---
    python tests/dsm_blind_warp_benchmark.py
    python tests/dsm_blind_warp_benchmark.py --quick
    python tests/dsm_blind_warp_benchmark.py --smooth 0.33

Salida: tests/dataset_out/dsm_blind_warp/
"""

import os
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

from propagation.mird_loader import MirdDatasetProvider
from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search
from evaluation.bf_wrappers import NM_MVDR_DSM_BLIND, SOUDEN_ORACLE_SCM

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT_DIR = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "dsm_blind_warp")

# rt60=0.360 con iSIR in {0,10} son las celdas que el ajuste de mascara uso para
# entrenar; el resto es condicion no vista.
FIT_RT60, FIT_ISIR = {0.360}, {0.0, 10.0}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smooth", type=float, default=0.5)
    ap.add_argument("--b-n", type=float, default=-8.0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    i1 = tf.lite.Interpreter(model_path=f"{PROJECT_ROOT}/src/dnn_denoise/models/model_quant_1.tflite")
    i1.allocate_tensors()
    i2 = tf.lite.Interpreter(model_path=f"{PROJECT_ROOT}/src/dnn_denoise/models/model_quant_2.tflite")
    i2.allocate_tensors()
    provider = MirdDatasetProvider(root_dir=f"{PROJECT_ROOT}/tools/data/rirs/mird")

    base_config = {
        'fs': 16000, 'duration': 15, 't_early': 0.050,
        'array_center': [3.0, 3.0, 1.2], 'mird_spacing': "3-3-3-8-3-3-3",
        'snr_db': 30.0,
        'source_path': f"{PROJECT_ROOT}/tools/data/signals/p002_emo_adoration_sentences.wav",
        'interf_paths': [f"{PROJECT_ROOT}/tools/data/signals/techno_gated commune.wav"],
        'wpe_taps': 7, 'wpe_delay': 2, 'wpe_alpha': 0.9999,
        'wpe_stft_size': 512, 'wpe_stft_shift': 128,
        'wpe_fixed_bits': None, 'wpe_fixed_round': 'nearest', 'wpe_backend': 'cov',
        'wpe_block_L': 512, 'wpe_block_shift': 2, 'wpe_block_iters': 2,
        'wpe_block_reg': 1e-6, 'wpe_block_solver': 'cholesky', 'wpe_block_mode': 'resolve',
        'stft_window': 512, 'stft_overlap': 384,
        'eval_references': ['anechoic', 'early', 'reverberant'],
        'dtln_model_path': f"{PROJECT_ROOT}/src/dnn_denoise/models/model_quant_1.tflite",
    }
    param_grid = {
        'rt60': [0.360] if args.quick else [0.360, 0.610],
        'target_angle': [0], 'target_dist': [1.0],
        'interf_configs': [[(45, 1.0)]] if args.quick else [[(45, 1.0)], [(90, 1.0)]],
        'isir_db': [0, 5] if args.quick else [-5, 0, 5, 10],
        'mismatch_gain': [0], 'mismatch_phase': [0],
        'use_wpe': [False], 'wpe_method': ['online'], 'wpe_taps': [7], 'wpe_delay': [2],
        'error_angle_deg': [0.0], 'error_distance_m': [0.0],
    }

    kw = dict(min_loading=1e-9, alpha=0.99)
    procs = {
        "DSM_BLIND":        NM_MVDR_DSM_BLIND(**kw),
        "DSM_BLIND_W25":    NM_MVDR_DSM_BLIND(**kw, mask_warp=(1.0, 0.0, 2.5, args.b_n)),
        "DSM_BLIND_W30":    NM_MVDR_DSM_BLIND(**kw, mask_warp=(1.0, 0.0, 3.0, args.b_n)),
        "DSM_BLIND_PF":     NM_MVDR_DSM_BLIND(**kw, smooth=args.smooth),
        "DSM_BLIND_W25_PF": NM_MVDR_DSM_BLIND(**kw, smooth=args.smooth,
                                              mask_warp=(1.0, 0.0, 2.5, args.b_n)),
        "ORACLE_SCM":       SOUDEN_ORACLE_SCM(min_loading=1e-9, alpha=0.99),
    }

    df = run_mird_grid_search(
        grid_params=param_grid, dataset_provider=provider, processors=procs,
        scene_base_config=base_config, output_dir=args.out_dir,
        interpreter_1=i1, interpreter_2=i2, apply_dtln_post=False,
        save_catalog=False)

    mets = ["PESQ", "STOI", "SDR", "SIR", "SAR"]
    order = ["DSM_BLIND", "DSM_BLIND_W25", "DSM_BLIND_W30", "DSM_BLIND_PF",
             "DSM_BLIND_W25_PF", "ORACLE_SCM"]

    def _show(title, sub):
        if not len(sub):
            return
        cols = [f"Delta_tot_{m}_early" for m in mets]
        cols = [c for c in cols if c in sub.columns]
        t = sub.groupby("processor")[cols].median().round(3)
        t.columns = mets[:len(cols)]
        t = t.reindex([p for p in order if p in t.index])
        print(f"\n{'='*70}\n{title}  ({len(sub)//len(procs)} celdas)\n{'='*70}")
        print(t.to_string())
        for a, b in (("DSM_BLIND_W25", "DSM_BLIND"), ("DSM_BLIND_W30", "DSM_BLIND"),
                     ("DSM_BLIND_W25_PF", "DSM_BLIND_PF")):
            if a in t.index and b in t.index:
                print(f"   {a} - {b}: {(t.loc[a]-t.loc[b]).round(3).to_dict()}")

    _show("TODAS LAS CELDAS", df)
    if "rt60" in df.columns:
        seen = df["rt60"].isin(FIT_RT60) & df["isir_db"].isin(FIT_ISIR)
        _show("CELDAS NO VISTAS POR EL AJUSTE DE MASCARA", df[~seen])

    p = df.pivot_table(index=["rt60", "interf_configs", "isir_db"],
                       columns="processor", values=[f"Delta_tot_{m}_early" for m in mets])
    n = len(df) // len(procs)
    print(f"\n=== victorias celda a celda, W25 vs DSM_BLIND ({n} celdas) ===")
    for m in mets:
        c = f"Delta_tot_{m}_early"
        print(f"  {m:5s}: {(p[c]['DSM_BLIND_W25'] > p[c]['DSM_BLIND']).sum()}/{n}")
    print(f"\n[ok] {args.out_dir}")


if __name__ == "__main__":
    main()
