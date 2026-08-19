"""
eval_qrd_pipeline_mird_avg.py
=============================
Multi-scene AVERAGED pipeline evaluation of the inverse-QRD fixed-point WPE.

Single-scene BSS-Eval SIR is high-variance (median aggregation swings hard), so
here we average the WPE contribution over several interferer angles to get a
trustworthy dSIR / dPESQ.  Uses NM-MVDR only: unlike the oracle beamformer, its
references (neural masks from the WPE output) are IDENTICAL across the float /
fixed conditions, so the float-vs-fixed comparison is fair (the oracle path
gives float an artificial edge via the exact-decomposition Opcion-A refs).

MIRD T60=0.61, M=8, taps=5 (sweet spot), n_iter=1.

USO:
    conda activate tesis_beam
    python tests/documentation_experiments/eval_qrd_pipeline_mird_avg.py
"""
import os
import sys
import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(REPO, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
sys.path.insert(0, os.path.dirname(__file__))

from propagation.mird_loader import MirdDatasetProvider
from evaluation.bf_wrappers import NM_MVDR
from eval_qrd_pipeline_mird import base_config
from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search

PROC_PESQ = "Delta_tot_PESQ_early"
PROC_SIR = "Delta_tot_SIR_early"

TAPS = 5
INTERF_ANGLES = [30, 45, 60, 75]      # average over these interferer positions (valid MIRD angles)
CONDITIONS = [
    ("noWPE", False, None, "cov"),
    ("float", True,  None, "cov"),
    ("cov24", True,  24,   "cov"),
    ("qrd16", True,  16,   "qrd"),
    ("qrd14", True,  14,   "qrd"),
]


def run_one(provider, procs, taps, use_wpe, bits, backend, interf_angle, outdir):
    cfg = base_config()
    cfg['wpe_fixed_bits'] = bits
    cfg['wpe_backend'] = backend
    grid = {
        'rt60': [0.610], 'target_angle': [0], 'target_dist': [1.0],
        'interf_configs': [[(interf_angle, 1.0)]], 'isir_db': [3],
        'mismatch_gain': [0], 'mismatch_phase': [0], 'use_wpe': [use_wpe],
        'wpe_taps': [taps], 'wpe_delay': [3],
        'error_angle_deg': [0.0], 'error_distance_m': [0.0],
    }
    df = run_mird_grid_search(
        grid_params=grid, dataset_provider=provider, processors=procs,
        scene_base_config=cfg, output_dir=outdir,
        interpreter_1=None, interpreter_2=None, save_catalog=False, apply_dtln_post=False,
    )
    r = df.iloc[0]
    return float(r[PROC_PESQ]), float(r[PROC_SIR])


def main():
    provider = MirdDatasetProvider(root_dir=os.path.join(REPO, "tools/data/rirs/mird"))
    procs = {"NM-MVDR_a0.99": NM_MVDR(min_loading=1e-6, alpha=0.99)}
    outdir = os.path.join(REPO, "tests/documentation_experiments/results/qrd_pipeline_avg")
    os.makedirs(outdir, exist_ok=True)

    # collect per-angle contribution for each condition
    per = {label: {'dPESQ': [], 'dSIR': []} for (label, *_ ) in CONDITIONS if label != "noWPE"}
    for ang in INTERF_ANGLES:
        base_p, base_s = None, None
        vals = {}
        for (label, use_wpe, bits, backend) in CONDITIONS:
            print(f"\n[RUN] taps={TAPS} angle={ang} cond={label}")
            p, s = run_one(provider, procs, TAPS, use_wpe, bits, backend, ang, outdir)
            vals[label] = (p, s)
        base_p, base_s = vals["noWPE"]
        for label in per.keys():
            per[label]['dPESQ'].append(vals[label][0] - base_p)
            per[label]['dSIR'].append(vals[label][1] - base_s)

    rows = []
    for label in ("float", "cov24", "qrd16", "qrd14"):
        dP = np.array(per[label]['dPESQ']); dS = np.array(per[label]['dSIR'])
        rows.append({'condition': label,
                     'dPESQ_mean': dP.mean(), 'dPESQ_std': dP.std(),
                     'dSIR_mean': dS.mean(), 'dSIR_std': dS.std()})
    df = pd.DataFrame(rows)
    csv = os.path.join(outdir, "qrd_pipeline_avg_summary.csv")
    df.to_csv(csv, index=False)
    print("\n\n" + "=" * 78)
    print(f"WPE CONTRIBUTION averaged over interferer angles {INTERF_ANGLES}")
    print(f"NM-MVDR, MIRD T60=0.61, M=8, taps={TAPS}, n_iter=1")
    print("=" * 78)
    print(df.to_string(index=False))
    print(f"\nSaved: {csv}")


if __name__ == "__main__":
    main()
