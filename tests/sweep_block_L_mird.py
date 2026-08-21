"""
sweep_block_L_mird.py
=====================
Barrido del LARGO DE VENTANA (wpe_block_L) del block-online WPE contra el online
(RLS), en la escena MIRD T60=610ms. Objetivo: ver cuanto se puede achicar L
(para caber en memoria interna del FPGA) sin perder la ventaja en beamforming.

Config fija en la mejor zona del sweep previo: taps=10, delay=2, block_shift=20.
Beamformers sin dependencia DTLN (MVDR recursivo + DS) -> robusto y rapido.

USO:
    conda activate tesis_beam
    python tests/sweep_block_L_mird.py
"""
import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import numpy as np
import pandas as pd
from propagation.mird_loader import MirdDatasetProvider
from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search
from evaluation.bf_wrappers import MVDR_Recursive, DS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
provider = MirdDatasetProvider(root_dir=os.path.join(REPO, "tools/data/rirs/mird"))

base_config = {
    'fs': 16000, 'duration': 15, 't_early': 0.008, 'array_center': [3.0, 3.0, 1.2],
    'mird_spacing': "3-3-3-8-3-3-3", 'snr_db': 60.0,
    'source_path': os.path.join(REPO, "tools/data/signals/p002_emo_adoration_sentences.wav"),
    'interf_paths': [os.path.join(REPO, "tools/data/signals/techno_gated commune.wav")],
    'wpe_taps': 10, 'wpe_delay': 2, 'wpe_alpha': 0.9999,
    'wpe_stft_size': 512, 'wpe_stft_shift': 128,
    'wpe_fixed_bits': None, 'wpe_fixed_round': 'nearest', 'wpe_backend': 'cov',
    'wpe_block_shift': 20, 'wpe_block_iters': 3, 'wpe_block_reg': 1e-6, 'wpe_block_solver': 'cholesky',
    'stft_window': 512, 'stft_overlap': 384, 'eval_references': ['early'],
}

# Barrido de L (descendente). L>KM=taps*M=80 esta bien condicionado; por debajo
# depende de la carga diagonal (reg) -> se espera degradacion.
L_VALUES = [512, 384, 256, 192, 128, 96, 64]

param_grid = {
    'rt60': [0.610], 'target_angle': [0], 'target_dist': [1.0],
    'interf_configs': [[(45, 1.0)]], 'isir_db': [3],
    'mismatch_gain': [0], 'mismatch_phase': [0], 'use_wpe': [True],
    'wpe_method': ['online', 'block'],
    'wpe_block_L': L_VALUES,          # eje barrido (se ignora en las celdas online, dedup)
    'wpe_taps': [10], 'wpe_delay': [2],
    'error_angle_deg': [0.0], 'error_distance_m': [0.0],
}

procs = {"MVDR": MVDR_Recursive(), "DS": DS()}

OUT = os.path.join(REPO, "tests/dataset_out/sweep_block_L")
df = run_mird_grid_search(param_grid, provider, procs, base_config,
                          output_dir=OUT, interpreter_1=None, interpreter_2=None,
                          save_catalog=False)

# --- Resumen ordenado: por procesador, online vs block(L) ---
pd.set_option('display.width', 140)
cols = ["processor", "wpe_method", "wpe_block_L", "wpe_block_shift",
        "Delta_tot_PESQ_early", "Delta_tot_STOI_early", "Delta_tot_SIR_early"]
cols = [c for c in cols if c in df.columns]
view = df[cols].copy()
view = view.sort_values(["processor", "wpe_method", "wpe_block_L"], na_position="first")
print("\n================= L-SWEEP (taps=10, delay=2, block_shift=20) =================")
print(view.to_string(index=False))
df.to_csv(os.path.join(OUT, "sweep_block_L_summary.csv"), index=False)
print(f"\nCSV -> {os.path.join(OUT,'sweep_block_L_summary.csv')}")
