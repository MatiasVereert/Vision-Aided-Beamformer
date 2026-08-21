"""
multiscene_block_vs_online_mird.py
==================================
Comparacion MULTI-ESCENA block-online vs online(RLS) en la config DULCE del block,
para medir el margen de forma robusta (no en una sola escena).

Config dulce (de los sweeps): taps=10, delay=2, L=256, iters=3, block_shift=32,
mode=resolve. online al MISMO taps/delay (alpha=0.9999).

Escenas: rt60 {0.160, 0.360, 0.610} x target_angle {0, 45, 315(-45)} = 9 escenas.
Interferente fijo a 90deg, 1m, iSIR=3dB. Beamformers MVDR + DS (sin DTLN).

Resultados -> tests/dataset_out/multiscene_block_vs_online/
USO:  conda activate tesis_beam && python tests/multiscene_block_vs_online_mird.py
"""
import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import numpy as np
import pandas as pd
from propagation.mird_loader import MirdDatasetProvider
from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search
from evaluation.bf_wrappers import ORACLE_MB_MVDR_SOUDEN, SOUDEN_ORACLE_SCM, DS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "tests", "dataset_out", "multiscene_block_vs_online_oracle_taps14")
os.makedirs(OUT, exist_ok=True)
provider = MirdDatasetProvider(root_dir=os.path.join(REPO, "tools/data/rirs/mird"))

base_config = {
    'fs': 16000, 'duration': 15, 't_early': 0.008, 'array_center': [3.0, 3.0, 1.2],
    'mird_spacing': "3-3-3-8-3-3-3", 'snr_db': 60.0,
    'source_path': os.path.join(REPO, "tools/data/signals/p002_emo_adoration_sentences.wav"),
    'interf_paths': [os.path.join(REPO, "tools/data/signals/techno_gated commune.wav")],
    'wpe_taps': 14, 'wpe_delay': 2, 'wpe_alpha': 0.9999,
    'wpe_stft_size': 512, 'wpe_stft_shift': 128,
    'wpe_fixed_bits': None, 'wpe_fixed_round': 'nearest', 'wpe_backend': 'cov',
    # config DULCE del block
    'wpe_block_L': 256, 'wpe_block_mode': 'resolve', 'wpe_block_iters': 3,
    'wpe_block_shift': 32, 'wpe_block_reg': 1e-6, 'wpe_block_solver': 'cholesky',
    'wpe_block_warm_start': False,
    'stft_window': 512, 'stft_overlap': 384, 'eval_references': ['early'],
}
param_grid = {
    'rt60': [0.160, 0.360, 0.610],
    'target_angle': [0, 45, 315],
    'target_dist': [1.0],
    'interf_configs': [[(90, 1.0)]],
    'isir_db': [3],
    'mismatch_gain': [0], 'mismatch_phase': [0], 'use_wpe': [True],
    'wpe_method': ['online', 'block'],
    'wpe_taps': [14], 'wpe_delay': [2],
    'error_angle_deg': [0.0], 'error_distance_m': [0.0],
}
procs = {"ORACLE_MVDR": ORACLE_MB_MVDR_SOUDEN(min_loading=1e-6, alpha=0.99),
         "SOUDEN_ORACLE": SOUDEN_ORACLE_SCM(min_loading=1e-6, alpha=0.99),
         "DS": DS()}

print(f"[*] Multi-escena block vs online (config dulce) -> {OUT}")
df = run_mird_grid_search(param_grid, provider, procs, base_config,
                          output_dir=OUT, interpreter_1=None, interpreter_2=None, save_catalog=False)

# ---- Analisis del margen block - online por escena ----
METR = ["Delta_tot_PESQ_early", "Delta_tot_STOI_early", "Delta_tot_SIR_early"]
METR = [m for m in METR if m in df.columns]
key = ['processor', 'rt60', 'target_angle']
piv = df.pivot_table(index=key, columns='wpe_method', values=METR, aggfunc='first')

rows = []
for k, r in piv.iterrows():
    proc, rt60, ang = k
    row = {'processor': proc, 'rt60': rt60, 'angle': ang}
    for m in METR:
        b = r.get((m, 'block')); o = r.get((m, 'online'))
        short = m.replace('Delta_tot_', '').replace('_early', '')
        row[f'block_{short}'] = b
        row[f'online_{short}'] = o
        row[f'margin_{short}'] = (b - o) if (pd.notna(b) and pd.notna(o)) else np.nan
    rows.append(row)
res = pd.DataFrame(rows).sort_values(['processor', 'rt60', 'angle'])
pd.set_option('display.width', 200)

print("\n========= MARGEN block - online por ESCENA (config dulce, taps10/L256/it3) =========")
show = ['processor', 'rt60', 'angle', 'block_PESQ', 'online_PESQ', 'margin_PESQ',
        'margin_STOI', 'margin_SIR']
show = [c for c in show if c in res.columns]
print(res[show].to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

print("\n========= RESUMEN del margen (por procesador) =========")
for proc in res['processor'].unique():
    sub = res[res['processor'] == proc]
    mp = sub['margin_PESQ']
    print(f"  {proc}: PESQ margin mean={mp.mean():+.3f}  std={mp.std():.3f}  "
          f"min={mp.min():+.3f}  max={mp.max():+.3f}  block_gana={int((mp>0).sum())}/{mp.notna().sum()}")
    if 'margin_STOI' in sub:
        print(f"        STOI margin mean={sub['margin_STOI'].mean():+.3f}  "
              f"block_gana={int((sub['margin_STOI']>0).sum())}/{sub['margin_STOI'].notna().sum()}")

print("\n========= margen PESQ medio por rt60 =========")
print(res.groupby(['processor', 'rt60'])['margin_PESQ'].mean().to_string(float_format=lambda x: f"{x:+.3f}"))

res.to_csv(os.path.join(OUT, "margin_summary.csv"), index=False)
print(f"\nCSV -> {os.path.join(OUT, 'margin_summary.csv')}")
