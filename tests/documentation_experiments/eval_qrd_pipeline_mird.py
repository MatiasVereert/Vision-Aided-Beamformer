"""
eval_qrd_pipeline_mird.py
=========================
End-to-end pipeline evaluation of the inverse-QRD fixed-point Online-WPE against
the covariance-form baseline, on the REAL MIRD scene (Bar-Ilan RIRs, T60=0.61,
M=8), post-beamformer.

For each processor it reports the WPE CONTRIBUTION
    aporte = Delta(with WPE) - Delta(without WPE)
in PESQ (early ref) and SIR (early ref), comparing:
    float  vs  covariance-24b (baseline)  vs  QRD-16b  vs  QRD-14b
(n_iter=1, the definitive setting -- variance refinement is discarded).

This reuses the exact benchmark machinery (run_mird_grid_search) so the numbers
are directly comparable to resultados_fixedpoint_wpe.txt.  DTLN post is OFF
(interpreters=None) to isolate the WPE->beamformer effect.

USO:
    conda activate tesis_beam
    python tests/documentation_experiments/eval_qrd_pipeline_mird.py
"""
import os
import sys
import copy
import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(REPO, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from propagation.mird_loader import MirdDatasetProvider
from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search
from evaluation.bf_wrappers import NM_MVDR, ORACLE_MB_MVDR_SOUDEN

PROC_PESQ = "Delta_tot_PESQ_early"
PROC_SIR = "Delta_tot_SIR_early"

TAPS_LIST = [5, 7]

# (label, use_wpe, wpe_fixed_bits, wpe_backend)
CONDITIONS = [
    ("noWPE", False, None, "cov"),
    ("float", True,  None, "cov"),
    ("cov24", True,  24,   "cov"),
    ("qrd16", True,  16,   "qrd"),
    ("qrd14", True,  14,   "qrd"),
]


def base_config():
    return {
        'fs': 16000,
        'duration': 15,
        't_early': 0.050,
        'array_center': [3.0, 3.0, 1.2],
        'mird_spacing': "3-3-3-8-3-3-3",
        'snr_db': 60.0,
        'source_path': os.path.join(REPO, "tools/data/signals/p002_emo_adoration_sentences.wav"),
        'interf_paths': [os.path.join(REPO, "tools/data/signals/techno_gated commune.wav")],
        'wpe_taps': 7,
        'wpe_delay': 3,
        'wpe_alpha': 0.9999,
        'wpe_stft_size': 512,
        'wpe_stft_shift': 128,
        'wpe_fixed_bits': None,
        'wpe_fixed_round': 'nearest',
        'wpe_backend': 'cov',
        'stft_window': 512,
        'stft_overlap': 384,
        'eval_references': ['anechoic', 'early', 'reverberant'],
        # Absolute path so NM-MVDR's internal DTLN mask estimator finds the model
        # (its default is a Windows-style relative path that fails on linux).
        'dtln_model_path': os.path.join(REPO, "src/dnn_denoise/models/model_quant_1.tflite"),
    }


def run_condition(provider, processors, taps, label, use_wpe, bits, backend, outdir):
    cfg = base_config()
    cfg['wpe_fixed_bits'] = bits
    cfg['wpe_backend'] = backend
    grid = {
        'rt60': [0.610],
        'target_angle': [0],
        'target_dist': [1.0],
        'interf_configs': [[(45, 1.0)]],
        'isir_db': [3],
        'mismatch_gain': [0],
        'mismatch_phase': [0],
        'use_wpe': [use_wpe],
        'wpe_taps': [taps],
        'wpe_delay': [3],
        'error_angle_deg': [0.0],
        'error_distance_m': [0.0],
    }
    df = run_mird_grid_search(
        grid_params=grid, dataset_provider=provider, processors=processors,
        scene_base_config=cfg, output_dir=outdir,
        interpreter_1=None, interpreter_2=None,   # DTLN off
        save_catalog=False, apply_dtln_post=False,
    )
    # Return {processor: (dPESQ, dSIR)}
    out = {}
    for _, r in df.iterrows():
        out[r['processor']] = (float(r[PROC_PESQ]), float(r[PROC_SIR]))
    return out


def main():
    provider = MirdDatasetProvider(root_dir=os.path.join(REPO, "tools/data/rirs/mird"))
    processors = {
        "NM-MVDR_a0.99": NM_MVDR(min_loading=1e-6, alpha=0.99),
        "Oracle-MVDR_a0.99": ORACLE_MB_MVDR_SOUDEN(min_loading=1e-6, alpha=0.99, sharpen_exp=1.0),
    }
    outdir = os.path.join(REPO, "tests/documentation_experiments/results/qrd_pipeline")
    os.makedirs(outdir, exist_ok=True)

    rows = []
    for taps in TAPS_LIST:
        results = {}
        for (label, use_wpe, bits, backend) in CONDITIONS:
            print(f"\n{'='*70}\n[RUN] taps={taps}  condition={label}  "
                  f"(use_wpe={use_wpe}, bits={bits}, backend={backend})\n{'='*70}")
            results[label] = run_condition(provider, processors, taps, label,
                                           use_wpe, bits, backend, outdir)
        # aporte = Delta(cond) - Delta(noWPE)
        base = results["noWPE"]
        for proc in processors.keys():
            row = {'taps': taps, 'processor': proc,
                   'noWPE_PESQ': base[proc][0], 'noWPE_SIR': base[proc][1]}
            for label in ("float", "cov24", "qrd16", "qrd14"):
                dP = results[label][proc][0] - base[proc][0]
                dS = results[label][proc][1] - base[proc][1]
                row[f'{label}_dPESQ'] = dP
                row[f'{label}_dSIR'] = dS
            rows.append(row)

    df = pd.DataFrame(rows)
    csv = os.path.join(outdir, "qrd_pipeline_summary.csv")
    df.to_csv(csv, index=False)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 50)
    print("\n\n" + "=" * 90)
    print("WPE CONTRIBUTION (aporte = with_WPE - without_WPE), MIRD T60=0.61 M=8, n_iter=1")
    print("=" * 90)
    print(df.to_string(index=False))
    print(f"\nSaved: {csv}")


if __name__ == "__main__":
    main()
