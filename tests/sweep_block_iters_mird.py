"""
sweep_block_iters_mird.py
=========================
Efecto de wpe_block_iters (y de mode='sliding' == iters=1) en la CALIDAD de
beamforming, escena MIRD T60=610ms. Responde: ¿iters=1/sliding rinde tan bien
como iters=2-3? (clave: sliding es rapido pero es iters=1).

Config: taps=10, delay=2, L=256, block_shift=20. Beamformers sin DTLN (MVDR, DS).

USO:  conda activate tesis_beam && python tests/sweep_block_iters_mird.py
"""
import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import numpy as np, pandas as pd
from propagation.mird_loader import MirdDatasetProvider
from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search
from evaluation.bf_wrappers import MVDR_Recursive, DS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
provider = MirdDatasetProvider(root_dir=os.path.join(REPO, "tools/data/rirs/mird"))

def base_cfg(iters, mode):
    return {
        'fs': 16000, 'duration': 15, 't_early': 0.008, 'array_center': [3.0, 3.0, 1.2],
        'mird_spacing': "3-3-3-8-3-3-3", 'snr_db': 60.0,
        'source_path': os.path.join(REPO, "tools/data/signals/p002_emo_adoration_sentences.wav"),
        'interf_paths': [os.path.join(REPO, "tools/data/signals/techno_gated commune.wav")],
        'wpe_taps': 10, 'wpe_delay': 2, 'wpe_alpha': 0.9999,
        'wpe_stft_size': 512, 'wpe_stft_shift': 128,
        'wpe_fixed_bits': None, 'wpe_fixed_round': 'nearest', 'wpe_backend': 'cov',
        'wpe_block_L': 256, 'wpe_block_shift': 20, 'wpe_block_iters': iters,
        'wpe_block_reg': 1e-6, 'wpe_block_solver': 'cholesky', 'wpe_block_mode': mode,
        'stft_window': 512, 'stft_overlap': 384, 'eval_references': ['early'],
    }

procs = {"MVDR": MVDR_Recursive(), "DS": DS()}
OUT = os.path.join(REPO, "tests/dataset_out/sweep_block_iters")

runs = [
    ("online",  dict(wpe_method=['online', 'block']), base_cfg(3, 'resolve'), "block_it3 + online"),
    ("it2",     dict(wpe_method=['block']),           base_cfg(2, 'resolve'), "block_it2"),
    ("it1",     dict(wpe_method=['block']),           base_cfg(1, 'resolve'), "block_it1"),
    ("sliding", dict(wpe_method=['block']),           base_cfg(1, 'sliding'), "block_sliding(=it1 rapido)"),
]

frames = []
for tag, extra_grid, cfg, desc in runs:
    print(f"\n########## RUN: {desc} ##########")
    grid = {
        'rt60': [0.610], 'target_angle': [0], 'target_dist': [1.0],
        'interf_configs': [[(45, 1.0)]], 'isir_db': [3],
        'mismatch_gain': [0], 'mismatch_phase': [0], 'use_wpe': [True],
        'wpe_taps': [10], 'wpe_delay': [2],
        'error_angle_deg': [0.0], 'error_distance_m': [0.0],
        **extra_grid,
    }
    df = run_mird_grid_search(grid, provider, procs, cfg,
                              output_dir=os.path.join(OUT, tag),
                              interpreter_1=None, interpreter_2=None, save_catalog=False)
    df['run_tag'] = tag
    df['iters_cfg'] = cfg['wpe_block_iters']
    df['mode_cfg'] = cfg['wpe_block_mode']
    frames.append(df)

full = pd.concat(frames, ignore_index=True)
# etiqueta legible del metodo
def label(r):
    if r['wpe_method'] == 'online':
        return 'online(RLS)'
    return f"block-{r['mode_cfg']}" + ("" if r['mode_cfg'] == 'sliding' else f"-it{int(r['iters_cfg'])}")
full['config'] = full.apply(label, axis=1)
full = full.drop_duplicates(subset=['processor', 'config'])

pd.set_option('display.width', 160)
cols = ["processor", "config", "Delta_tot_PESQ_early", "Delta_tot_STOI_early", "Delta_tot_SIR_early"]
cols = [c for c in cols if c in full.columns]
order = {'online(RLS)': 0, 'block-resolve-it1': 1, 'block-sliding': 2, 'block-resolve-it2': 3, 'block-resolve-it3': 4}
full['_o'] = full['config'].map(order).fillna(9)
view = full.sort_values(["processor", "_o"])[cols]
print("\n============ EFECTO DE ITERS / SLIDING (taps=10, L=256, block_shift=20) ============")
print(view.to_string(index=False))
os.makedirs(OUT, exist_ok=True)
full.to_csv(os.path.join(OUT, "sweep_block_iters_summary.csv"), index=False)
print(f"\nCSV -> {os.path.join(OUT, 'sweep_block_iters_summary.csv')}")
