"""
joint_sweep_block_mird.py
=========================
Barrido CONJUNTO (factorial 2 niveles) para ver INTERACCIONES que los sweeps
aislados no muestran: taps x L x block_shift x rt60. Config block resolve it3.

Ejes: taps {5,10} x L {128,256} x block_shift {8,32} x rt60 {0.16,0.61}, angulo 0.
+ online(RLS) al mismo taps/rt60 como referencia. Beamformers SOUDEN_ORACLE + DS.

16 block + 4 online = 20 WPE. Resultados -> tests/dataset_out/joint_sweep_block/
USO:  conda activate tesis_beam && python tests/joint_sweep_block_mird.py
"""
import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import numpy as np, pandas as pd
from propagation.mird_loader import MirdDatasetProvider
from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search
from evaluation.bf_wrappers import SOUDEN_ORACLE_SCM, DS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "tests", "dataset_out", "joint_sweep_block")
os.makedirs(OUT, exist_ok=True)
provider = MirdDatasetProvider(root_dir=os.path.join(REPO, "tools/data/rirs/mird"))

base_config = {
    'fs': 16000, 'duration': 15, 't_early': 0.008, 'array_center': [3.0, 3.0, 1.2],
    'mird_spacing': "3-3-3-8-3-3-3", 'snr_db': 60.0,
    'source_path': os.path.join(REPO, "tools/data/signals/p002_emo_adoration_sentences.wav"),
    'interf_paths': [os.path.join(REPO, "tools/data/signals/techno_gated commune.wav")],
    'wpe_taps': 10, 'wpe_delay': 2, 'wpe_alpha': 0.9999,
    'wpe_stft_size': 512, 'wpe_stft_shift': 128,
    'wpe_fixed_bits': None, 'wpe_fixed_round': 'nearest', 'wpe_backend': 'cov',
    'wpe_block_L': 256, 'wpe_block_mode': 'resolve', 'wpe_block_iters': 3,
    'wpe_block_shift': 32, 'wpe_block_reg': 1e-6, 'wpe_block_solver': 'cholesky',
    'wpe_block_warm_start': False,
    'stft_window': 512, 'stft_overlap': 384, 'eval_references': ['early'],
}
param_grid = {
    'rt60': [0.16, 0.61],
    'target_angle': [0], 'target_dist': [1.0],
    'interf_configs': [[(90, 1.0)]], 'isir_db': [3],
    'mismatch_gain': [0], 'mismatch_phase': [0], 'use_wpe': [True],
    'wpe_method': ['online', 'block'],
    'wpe_taps': [5, 10],
    'wpe_block_L': [128, 256],
    'wpe_block_shift': [8, 32],
    'wpe_delay': [2], 'error_angle_deg': [0.0], 'error_distance_m': [0.0],
}
procs = {"SOUDEN_ORACLE": SOUDEN_ORACLE_SCM(min_loading=1e-6, alpha=0.99), "DS": DS()}

print(f"[*] Joint sweep taps x L x block_shift x rt60 -> {OUT}")
df = run_mird_grid_search(param_grid, provider, procs, base_config,
                          output_dir=OUT, interpreter_1=None, interpreter_2=None, save_catalog=False)

# ---- tabla: block PESQ por (proc, rt60, taps, L, block_shift) + margen vs online(mismo taps/rt60) ----
PESQ = "Delta_tot_PESQ_early"
onl = (df[df.wpe_method == 'online']
       .groupby(['processor', 'rt60', 'wpe_taps'])[PESQ].first().rename('online_PESQ'))
blk = df[df.wpe_method == 'block'].copy()
blk = blk.merge(onl, on=['processor', 'rt60', 'wpe_taps'], how='left')
blk['margin'] = blk[PESQ] - blk['online_PESQ']
cols = ['processor', 'rt60', 'wpe_taps', 'wpe_block_L', 'wpe_block_shift',
        PESQ, 'online_PESQ', 'margin']
view = blk[cols].sort_values(['processor', 'rt60', 'wpe_taps', 'wpe_block_L', 'wpe_block_shift'])
pd.set_option('display.width', 180)
print("\n===== JOINT: block PESQ y margen vs online (por taps/L/block_shift/rt60) =====")
print(view.rename(columns={PESQ: 'block_PESQ'}).to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

# efectos principales (media del block_PESQ variando cada eje, SOUDEN)
s = blk[blk.processor == 'SOUDEN_ORACLE']
print("\n===== efectos principales sobre block_PESQ (SOUDEN) =====")
for ax in ['wpe_taps', 'wpe_block_L', 'wpe_block_shift', 'rt60']:
    print(f"  por {ax}:", {k: round(v, 3) for k, v in s.groupby(ax)[PESQ].mean().items()})

view.to_csv(os.path.join(OUT, "joint_summary.csv"), index=False)
print(f"\nCSV -> {os.path.join(OUT, 'joint_summary.csv')}")
