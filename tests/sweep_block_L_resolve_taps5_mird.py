"""
sweep_block_L_resolve_mird.py
=============================
Barrido del LARGO DE VENTANA L del block-online a CALIDAD REAL (resolve, iters=3),
escena MIRD T60=610ms. L define la memoria on-chip del buffer de ventana
(L*F*M complejos) -> cuanto se puede achicar sin perder calidad = si entra todo
on-chip sin DDR.

Como vimos que la calidad la fija iters (no block_shift), se usa iters=3 y un
block_shift representativo y barato (32). taps=10 (KM=80): L<~80 queda
sub-determinado (depende de reg). Beamformers sin DTLN (MVDR + DS).

Resultados -> tests/dataset_out/sweep_block_L_resolve_taps5/
USO:  conda activate tesis_beam && python tests/sweep_block_L_resolve_mird.py
"""
import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import pandas as pd
from propagation.mird_loader import MirdDatasetProvider
from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search
from evaluation.bf_wrappers import MVDR_Recursive, DS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "tests", "dataset_out", "sweep_block_L_resolve_taps5")
os.makedirs(OUT, exist_ok=True)
provider = MirdDatasetProvider(root_dir=os.path.join(REPO, "tools/data/rirs/mird"))

L_VALUES = [48, 64, 96, 128, 192, 256]
BLOCK_SHIFT = 32   # representativo y barato (block_shift ~ no afecta calidad)
WORDBITS = 24      # para anexar memoria on-chip del buffer de ventana
F_BINS, M_CH = 257, 8

def win_mbit(L):
    # buffer de ventana on-chip: L*F*M complejos * (2*WORDBITS) bits
    return L * F_BINS * M_CH * (2 * WORDBITS) / 1e6

base_config = {
    'fs': 16000, 'duration': 15, 't_early': 0.008, 'array_center': [3.0, 3.0, 1.2],
    'mird_spacing': "3-3-3-8-3-3-3", 'snr_db': 60.0,
    'source_path': os.path.join(REPO, "tools/data/signals/p002_emo_adoration_sentences.wav"),
    'interf_paths': [os.path.join(REPO, "tools/data/signals/techno_gated commune.wav")],
    'wpe_taps': 5, 'wpe_delay': 2, 'wpe_alpha': 0.9999,
    'wpe_stft_size': 512, 'wpe_stft_shift': 128,
    'wpe_fixed_bits': None, 'wpe_fixed_round': 'nearest', 'wpe_backend': 'cov',
    'wpe_block_shift': BLOCK_SHIFT, 'wpe_block_mode': 'resolve', 'wpe_block_iters': 3,
    'wpe_block_reg': 1e-6, 'wpe_block_solver': 'cholesky', 'wpe_block_warm_start': False,
    'stft_window': 512, 'stft_overlap': 384, 'eval_references': ['early'],
}
param_grid = {
    'rt60': [0.610], 'target_angle': [0], 'target_dist': [1.0],
    'interf_configs': [[(45, 1.0)]], 'isir_db': [3],
    'mismatch_gain': [0], 'mismatch_phase': [0], 'use_wpe': [True],
    'wpe_method': ['online', 'block'], 'wpe_block_L': L_VALUES,
    'wpe_taps': [5], 'wpe_delay': [2], 'error_angle_deg': [0.0], 'error_distance_m': [0.0],
}
procs = {"MVDR": MVDR_Recursive(), "DS": DS()}

print(f"[*] Barrido L (resolve it3, block_shift={BLOCK_SHIFT}) -> {OUT}  L={L_VALUES}")
df = run_mird_grid_search(param_grid, provider, procs, base_config,
                          output_dir=OUT, interpreter_1=None, interpreter_2=None, save_catalog=False)

def cfg_label(r):
    if r['wpe_method'] == 'online':
        return 'online(RLS)'
    L = r.get('wpe_block_L')
    return f"block_L{int(L)}" if pd.notna(L) else 'block'

df['config'] = df.apply(cfg_label, axis=1)
mcols = [c for c in ["Delta_tot_PESQ_early", "Delta_tot_STOI_early", "Delta_tot_SIR_early"] if c in df.columns]
summary = df.drop_duplicates(subset=['processor', 'config'])[['processor', 'config', 'wpe_block_L'] + mcols].copy()
summary['win_Mbit_24b'] = summary['wpe_block_L'].apply(lambda L: round(win_mbit(L), 1) if pd.notna(L) else None)
summary['onchip_fit'] = summary['win_Mbit_24b'].apply(lambda m: ('' if m is None else ('OK' if m < 23 else 'DDR')))
summary = summary.assign(_L=summary['wpe_block_L'].fillna(-1)).sort_values(['processor', '_L']).drop(columns='_L')
pd.set_option('display.width', 170)
print("\n===== CALIDAD vs L (resolve it3, taps=10, block_shift=32) + memoria on-chip =====")
print(summary.to_string(index=False))
summary.to_csv(os.path.join(OUT, "sweep_summary.csv"), index=False)
print(f"\nCSV -> {os.path.join(OUT, 'sweep_summary.csv')}")
