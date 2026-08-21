"""
sweep_block_qr_vs_chol_mird.py
==============================
QR (Householder sobre la matriz de datos A) vs Cholesky (sobre R=A^H A) en punto
fijo, en el pipeline MIRD/PESQ. Pregunta: ¿QR recupera el PESQ del reg=1e-9
(+0.32 vs online) que el Cholesky fijo NO puede (colapsa aun a 32b)?

reg=1e-9, L=512, buffer block-float mant12 + G 24b (aisla el SOLVE).
block_shift=128 para bajar el # de re-solves (el QR es caro en la emulacion).
Resultado -> tests/dataset_out/block_qr_vs_chol/
"""
import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import numpy as np, pandas as pd
from propagation.mird_loader import MirdDatasetProvider
from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search
from evaluation.bf_wrappers import SOUDEN_ORACLE_SCM

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "tests", "dataset_out", "block_qr_vs_chol")
os.makedirs(OUT, exist_ok=True)
provider = MirdDatasetProvider(root_dir=os.path.join(REPO, "tools/data/rirs/mird"))

REG = 1e-9  # el punto de operacion bueno (+0.32) donde el Cholesky fijo falla


def cfg(fixed_bits, solve_method, solve_bits, solve_int):
    return {
        'fs': 16000, 'duration': 16, 't_early': 0.008, 'array_center': [3.0, 3.0, 1.2],
        'mird_spacing': "3-3-3-8-3-3-3", 'snr_db': 60.0,
        'source_path': os.path.join(REPO, "tools/data/signals/p002_emo_adoration_sentences.wav"),
        'interf_paths': [os.path.join(REPO, "tools/data/signals/techno_gated commune.wav")],
        'wpe_taps': 10, 'wpe_delay': 2, 'wpe_alpha': 0.9999,
        'wpe_stft_size': 512, 'wpe_stft_shift': 128,
        'wpe_fixed_bits': fixed_bits, 'wpe_fixed_round': 'nearest', 'wpe_backend': 'cov',
        'wpe_block_L': 512, 'wpe_block_mode': 'resolve', 'wpe_block_iters': 3,
        'wpe_block_shift': 128, 'wpe_block_reg': REG, 'wpe_block_solver': 'cholesky',
        'wpe_block_warm_start': False, 'wpe_block_window_bits': None,
        'wpe_block_window_mant': 12, 'wpe_block_window_exp': 5, 'wpe_block_g_mant': None,
        'wpe_block_solve_method': solve_method,
        'wpe_block_solve_bits': solve_bits, 'wpe_block_solve_int': solve_int,
        'wpe_block_int_bits': {'in': 1, 'pred': 1, 'g': 6, 'p': 6},
        'stft_window': 512, 'stft_overlap': 384, 'eval_references': ['early'],
    }

# label, fixed_bits, solve_method, solve_bits, solve_int, methods
RUNS = [
    ('float_ref',   None, 'cholesky', None, None, ['online', 'block']),  # target +0.32
    ('qr_s28',      24,   'qr',       28,   10,   ['block']),
    ('qr_s24',      24,   'qr',       24,   10,   ['block']),
    ('qr_s20',      24,   'qr',       20,   10,   ['block']),
    ('chol_s28',    24,   'cholesky', 28,   10,   ['block']),            # roto (contraste)
]
grid_base = dict(rt60=[0.61], target_angle=[0], target_dist=[1.0],
                 interf_configs=[[(90, 1.0)]], isir_db=[3], mismatch_gain=[0],
                 mismatch_phase=[0], use_wpe=[True], wpe_taps=[10], wpe_delay=[2],
                 error_angle_deg=[0.0], error_distance_m=[0.0])
procs = {"SOUDEN_ORACLE": SOUDEN_ORACLE_SCM(min_loading=1e-6, alpha=0.99)}

meta, frames = {}, []
for label, fb, sm, sb, si, methods in RUNS:
    meta[label] = (sm, sb)
    print(f"\n########## RUN: {label} (reg={REG:.0e} solve={sm}/{sb}b int={si}) ##########")
    try:
        df = run_mird_grid_search({**grid_base, 'wpe_method': methods},
                                  provider, procs, cfg(fb, sm, sb, si),
                                  output_dir=os.path.join(OUT, label),
                                  interpreter_1=None, interpreter_2=None, save_catalog=False)
        df['label'] = label
        frames.append(df)
    except Exception as e:
        print(f"  [ERR] {label}: {type(e).__name__}: {e}")

full = pd.concat(frames, ignore_index=True)
PESQ = "Delta_tot_PESQ_early"
onl = (full[full.wpe_method == 'online']
       .groupby(['processor', 'rt60'])[PESQ].first().rename('online_PESQ'))
flt = (full[(full.wpe_method == 'block') & (full.label == 'float_ref')]
       .groupby(['processor', 'rt60'])[PESQ].first().rename('floatblk_PESQ'))
blk = full[full.wpe_method == 'block'].merge(onl, on=['processor', 'rt60'], how='left')
blk = blk.merge(flt, on=['processor', 'rt60'], how='left')
blk['margin_vs_online'] = blk[PESQ] - blk['online_PESQ']
blk['delta_vs_float']   = blk[PESQ] - blk['floatblk_PESQ']
blk['solver'] = blk['label'].map(lambda x: meta[x][0])
blk['bits']   = blk['label'].map(lambda x: meta[x][1])
order = {l: i for i, (l, *_) in enumerate(RUNS)}
blk['_o'] = blk['label'].map(order)
v = blk.sort_values(['processor', '_o'])[
    ['processor', 'solver', 'bits', 'label', PESQ, 'delta_vs_float', 'margin_vs_online']]
pd.set_option('display.width', 170)
print("\n===== QR vs Cholesky en punto fijo (reg=1e-9, L=512). delta_vs_float = costo del solver fijo =====")
print(v.rename(columns={PESQ: 'block_PESQ'}).to_string(
    index=False, float_format=lambda x: f"{x:+.3f}", na_rep=' float'))
v.to_csv(os.path.join(OUT, "summary_qr_vs_chol.csv"), index=False)
print(f"\nCSV -> {os.path.join(OUT, 'summary_qr_vs_chol.csv')}")
