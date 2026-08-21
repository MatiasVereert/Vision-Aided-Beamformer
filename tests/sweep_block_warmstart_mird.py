"""
sweep_block_warmstart_mird.py
=============================
Prueba del WARM-START del block-online: ¿arrancar la potencia de cada bloque con
el filtro del bloque anterior acerca iters=1 a iters altos? Escena MIRD 610ms,
taps=10, L=256, block_shift=20, MVDR/DS.

USO: conda activate tesis_beam && python tests/sweep_block_warmstart_mird.py
"""
import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import pandas as pd
from propagation.mird_loader import MirdDatasetProvider
from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search
from evaluation.bf_wrappers import MVDR_Recursive, DS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
provider = MirdDatasetProvider(root_dir=os.path.join(REPO, "tools/data/rirs/mird"))

def cfg(iters, warm):
    return {
        'fs': 16000, 'duration': 15, 't_early': 0.008, 'array_center': [3.0, 3.0, 1.2],
        'mird_spacing': "3-3-3-8-3-3-3", 'snr_db': 60.0,
        'source_path': os.path.join(REPO, "tools/data/signals/p002_emo_adoration_sentences.wav"),
        'interf_paths': [os.path.join(REPO, "tools/data/signals/techno_gated commune.wav")],
        'wpe_taps': 10, 'wpe_delay': 2, 'wpe_alpha': 0.9999,
        'wpe_stft_size': 512, 'wpe_stft_shift': 128,
        'wpe_fixed_bits': None, 'wpe_fixed_round': 'nearest', 'wpe_backend': 'cov',
        'wpe_block_L': 256, 'wpe_block_shift': 20, 'wpe_block_iters': iters,
        'wpe_block_reg': 1e-6, 'wpe_block_solver': 'cholesky', 'wpe_block_mode': 'resolve',
        'wpe_block_warm_start': warm,
        'stft_window': 512, 'stft_overlap': 384, 'eval_references': ['early'],
    }

procs = {"MVDR": MVDR_Recursive(), "DS": DS()}
OUT = os.path.join(REPO, "tests/dataset_out/sweep_block_warmstart")
runs = [
    ("it1_cold", cfg(1, False), "it1 frio (actual)"),
    ("it1_warm", cfg(1, True),  "it1 WARM-START"),
    ("it2_warm", cfg(2, True),  "it2 warm"),
    ("it3_cold", cfg(3, False), "it3 frio (mejor previo)"),
]
grid = {'rt60': [0.610], 'target_angle': [0], 'target_dist': [1.0],
        'interf_configs': [[(45, 1.0)]], 'isir_db': [3], 'mismatch_gain': [0],
        'mismatch_phase': [0], 'use_wpe': [True], 'wpe_method': ['block'],
        'wpe_taps': [10], 'wpe_delay': [2], 'error_angle_deg': [0.0], 'error_distance_m': [0.0]}

frames = []
for tag, c, desc in runs:
    print(f"\n######### {desc} #########")
    df = run_mird_grid_search(grid, provider, procs, c, output_dir=os.path.join(OUT, tag),
                              interpreter_1=None, interpreter_2=None, save_catalog=False)
    df['cfg'] = tag
    frames.append(df)

full = pd.concat(frames, ignore_index=True)
pd.set_option('display.width', 160)
cols = ["processor", "cfg", "Delta_tot_PESQ_early", "Delta_tot_STOI_early", "Delta_tot_SIR_early"]
cols = [c for c in cols if c in full.columns]
order = {'it1_cold': 0, 'it1_warm': 1, 'it2_warm': 2, 'it3_cold': 3}
full['_o'] = full['cfg'].map(order)
print("\n===== WARM-START vs iters (taps=10, L=256, block_shift=20) =====")
print(full.sort_values(["processor", "_o"])[cols].to_string(index=False))
full.to_csv(os.path.join(OUT, "warmstart_summary.csv"), index=False)
print(f"\nCSV -> {os.path.join(OUT, 'warmstart_summary.csv')}")
