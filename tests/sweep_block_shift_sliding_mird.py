"""
sweep_block_shift_sliding_mird.py
=================================
Barrido de block_shift del block-online WPE (Opcion B), escena MIRD T60=610ms,
para ver la curva CALIDAD de beamforming <-> block_shift.

IMPORTANTE: usa mode='sliding' en TODAS las celdas. sliding == resolve con UNA
sola iteracion (equivalente exacto, rel ~1e-9), y es el camino rapido. Asi todo
el barrido queda en el mismo regimen (iters=1) y es barato. Para el numero final
de un block_shift elegido, despues se corre resolve con iters=2-3 (mas lento).

Config fija: taps=10, delay=2, L=256. Beamformers sin dependencia DTLN
(MVDR recursivo + DS), para correr desatendido de forma robusta.

Resultados -> tests/dataset_out/sweep_block_shift_sliding/
    mird_benchmark_metrics.parquet / .csv   (todas las filas)
    sweep_summary.csv                        (tabla resumida block_shift x proc)

USO:  conda activate tesis_beam && python tests/sweep_block_shift_sliding_mird.py
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
OUT = os.path.join(REPO, "tests", "dataset_out", "sweep_block_shift_sliding")
os.makedirs(OUT, exist_ok=True)

provider = MirdDatasetProvider(root_dir=os.path.join(REPO, "tools/data/rirs/mird"))

BLOCK_SHIFTS = [2, 4, 8, 16, 32, 64, 128]

base_config = {
    'fs': 16000, 'duration': 15, 't_early': 0.008, 'array_center': [3.0, 3.0, 1.2],
    'mird_spacing': "3-3-3-8-3-3-3", 'snr_db': 60.0,
    'source_path': os.path.join(REPO, "tools/data/signals/p002_emo_adoration_sentences.wav"),
    'interf_paths': [os.path.join(REPO, "tools/data/signals/techno_gated commune.wav")],
    'wpe_taps': 10, 'wpe_delay': 2, 'wpe_alpha': 0.9999,
    'wpe_stft_size': 512, 'wpe_stft_shift': 128,
    'wpe_fixed_bits': None, 'wpe_fixed_round': 'nearest', 'wpe_backend': 'cov',
    # --- Block: SLIDING (== resolve iters=1) ---
    'wpe_block_L': 256,
    'wpe_block_mode': 'sliding',   # <-- clave: sliding = iters=1 equivalente, rapido
    'wpe_block_iters': 1,          # (ignorado por sliding, se deja explicito)
    'wpe_block_reg': 1e-6, 'wpe_block_solver': 'cholesky', 'wpe_block_warm_start': False,
    'stft_window': 512, 'stft_overlap': 384, 'eval_references': ['early'],
}

param_grid = {
    'rt60': [0.610], 'target_angle': [0], 'target_dist': [1.0],
    'interf_configs': [[(45, 1.0)]], 'isir_db': [3],
    'mismatch_gain': [0], 'mismatch_phase': [0], 'use_wpe': [True],
    'wpe_method': ['online', 'block'],     # online como referencia
    'wpe_block_shift': BLOCK_SHIFTS,       # <-- EJE del barrido (dedup colapsa online)
    'wpe_taps': [10], 'wpe_delay': [2],
    'error_angle_deg': [0.0], 'error_distance_m': [0.0],
}

procs = {"MVDR": MVDR_Recursive(), "DS": DS()}

print(f"[*] Barrido block_shift (sliding) -> {OUT}")
print(f"[*] block_shift = {BLOCK_SHIFTS}  (+ online baseline)")
df = run_mird_grid_search(param_grid, provider, procs, base_config,
                          output_dir=OUT, interpreter_1=None, interpreter_2=None,
                          save_catalog=False)

# --- Resumen legible: por procesador, online + block(block_shift) ---
def cfg_label(r):
    if r['wpe_method'] == 'online':
        return 'online(RLS)'
    bs = r.get('wpe_block_shift')
    return f"block_bs{int(bs)}" if pd.notna(bs) else 'block'

df['config'] = df.apply(cfg_label, axis=1)
metric_cols = [c for c in ["Delta_tot_PESQ_early", "Delta_tot_STOI_early", "Delta_tot_SIR_early"]
               if c in df.columns]
summary = df.drop_duplicates(subset=['processor', 'config'])[['processor', 'config', 'wpe_block_shift'] + metric_cols]
# ordenar: online primero, luego block por block_shift ascendente
summary = summary.assign(_bs=summary['wpe_block_shift'].fillna(-1)).sort_values(['processor', '_bs']).drop(columns='_bs')

pd.set_option('display.width', 160)
print("\n===== CURVA CALIDAD vs block_shift (block=SLIDING/it1, taps=10, L=256) =====")
print(summary.to_string(index=False))
summary.to_csv(os.path.join(OUT, "sweep_summary.csv"), index=False)
print(f"\nCSV resumen -> {os.path.join(OUT, 'sweep_summary.csv')}")
print(f"Parquet completo -> {os.path.join(OUT, 'mird_benchmark_metrics.parquet')}")
