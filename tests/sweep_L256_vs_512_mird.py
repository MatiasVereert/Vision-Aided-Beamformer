"""
sweep_L256_vs_512_mird.py
=========================
L=256 vs L=512 con las mejores taps, para ver si mas ventana ayuda (sobre todo a
rt60 alto). Ejes: L {256,512} x taps {5,10} x rt60 {0.16,0.61}, angulo 0,
block_shift=32, resolve it3. + online. SOUDEN_ORACLE + DS.

Nota memoria: L=512 window @16b = 33.7 Mbit -> NO on-chip (config DDR). L=256 @16b
= 16.9 Mbit (on-chip). Este test dice si vale la pena el DDR de L=512.

Resultados -> tests/dataset_out/sweep_L256_vs_512/
"""
import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import pandas as pd
from propagation.mird_loader import MirdDatasetProvider
from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search
from evaluation.bf_wrappers import SOUDEN_ORACLE_SCM, DS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "tests", "dataset_out", "sweep_L256_vs_512")
os.makedirs(OUT, exist_ok=True)
provider = MirdDatasetProvider(root_dir=os.path.join(REPO, "tools/data/rirs/mird"))

base_config = {
    'fs': 16000, 'duration': 16, 't_early': 0.008, 'array_center': [3.0, 3.0, 1.2],
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
    'rt60': [0.16, 0.61], 'target_angle': [0], 'target_dist': [1.0],
    'interf_configs': [[(90, 1.0)]], 'isir_db': [3],
    'mismatch_gain': [0], 'mismatch_phase': [0], 'use_wpe': [True],
    'wpe_method': ['online', 'block'],
    'wpe_taps': [5, 10], 'wpe_block_L': [256, 512],
    'wpe_delay': [2], 'error_angle_deg': [0.0], 'error_distance_m': [0.0],
}
procs = {"SOUDEN_ORACLE": SOUDEN_ORACLE_SCM(min_loading=1e-6, alpha=0.99), "DS": DS()}

print(f"[*] L256 vs L512 -> {OUT}")
df = run_mird_grid_search(param_grid, provider, procs, base_config,
                          output_dir=OUT, interpreter_1=None, interpreter_2=None, save_catalog=False)

PESQ = "Delta_tot_PESQ_early"
onl = df[df.wpe_method == 'online'].groupby(['processor', 'rt60', 'wpe_taps'])[PESQ].first().rename('online_PESQ')
blk = df[df.wpe_method == 'block'].merge(onl, on=['processor', 'rt60', 'wpe_taps'], how='left')
blk['margin'] = blk[PESQ] - blk['online_PESQ']
v = blk[['processor', 'rt60', 'wpe_taps', 'wpe_block_L', PESQ, 'online_PESQ', 'margin']].sort_values(
    ['processor', 'rt60', 'wpe_taps', 'wpe_block_L']).rename(columns={PESQ: 'block_PESQ'})
pd.set_option('display.width', 170)
print("\n===== L=256 vs L=512 (block_PESQ y margen vs online) =====")
print(v.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
print("\n===== ganancia L512 - L256 (block_PESQ) por rt60/taps, SOUDEN =====")
s = blk[blk.processor == 'SOUDEN_ORACLE']
p = s.pivot_table(index=['rt60', 'wpe_taps'], columns='wpe_block_L', values=PESQ)
p['gain_512_minus_256'] = p[512] - p[256]
print(p.to_string(float_format=lambda x: f"{x:+.3f}"))
v.to_csv(os.path.join(OUT, "L_summary.csv"), index=False)
print(f"\nCSV -> {os.path.join(OUT, 'L_summary.csv')}")
