"""
sweep_block_solveprec_mird.py
=============================
Estudio de precision del DATAPATH del SOLVE (Cholesky interno) del block-online WPE.
Aisla el ancho de palabra de la factorizacion+sustitucion (buffer block-float mant12
~transparente, G 24b), y lo barre contra el solve float, en DOS reg:

  * reg=1e-9 (mejor PESQ, +0.32 vs online) -> cond(R)~2e9: se espera que Cholesky
    fijo NO cierre ni con 32b (motiva QR).
  * reg=1e-2 (PESQ menor, +0.12)          -> cond(R)~2e3: Cholesky fijo deberia
    cerrar a ~24b.

Resultado -> tests/dataset_out/block_solveprec/
"""
import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import numpy as np, pandas as pd
from propagation.mird_loader import MirdDatasetProvider
from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search
from evaluation.bf_wrappers import SOUDEN_ORACLE_SCM

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "tests", "dataset_out", "block_solveprec")
os.makedirs(OUT, exist_ok=True)
provider = MirdDatasetProvider(root_dir=os.path.join(REPO, "tools/data/rirs/mird"))


def cfg(reg, fixed_bits, solve_bits, solve_int):
    return {
        'fs': 16000, 'duration': 16, 't_early': 0.008, 'array_center': [3.0, 3.0, 1.2],
        'mird_spacing': "3-3-3-8-3-3-3", 'snr_db': 60.0,
        'source_path': os.path.join(REPO, "tools/data/signals/p002_emo_adoration_sentences.wav"),
        'interf_paths': [os.path.join(REPO, "tools/data/signals/techno_gated commune.wav")],
        'wpe_taps': 10, 'wpe_delay': 2, 'wpe_alpha': 0.9999,
        'wpe_stft_size': 512, 'wpe_stft_shift': 128,
        'wpe_fixed_bits': fixed_bits, 'wpe_fixed_round': 'nearest', 'wpe_backend': 'cov',
        'wpe_block_L': 512, 'wpe_block_mode': 'resolve', 'wpe_block_iters': 3,
        'wpe_block_shift': 32, 'wpe_block_reg': reg, 'wpe_block_solver': 'cholesky',
        'wpe_block_warm_start': False, 'wpe_block_window_bits': None,
        # buffer block-float mant12 (~transparente) y G 24b -> aisla el SOLVE.
        'wpe_block_window_mant': 12, 'wpe_block_window_exp': 5, 'wpe_block_g_mant': None,
        'wpe_block_solve_bits': solve_bits, 'wpe_block_solve_int': solve_int,
        'wpe_block_int_bits': {'in': 1, 'pred': 1, 'g': 6, 'p': 6},
        'stft_window': 512, 'stft_overlap': 384, 'eval_references': ['early'],
    }

# label, reg, fixed_bits, solve_bits, solve_int, methods
# (int_bits del solve = headroom para picos de |G|: reg1e-2 |G|~2 -> int4; reg1e-9 |G|~808 -> int12)
RUNS = [
    ('float_r1e-9', 1e-9, None, None, None, ['online', 'block']),   # target PESQ (+0.32)
    ('float_r1e-2', 1e-2, None, None, None, ['block']),             # penalidad del reg alto
    # reg=1e-2 (cond ~2e3): Cholesky fijo deberia cerrar
    ('r1e-2_s28', 1e-2, 24, 28, 4, ['block']),
    ('r1e-2_s24', 1e-2, 24, 24, 4, ['block']),
    ('r1e-2_s20', 1e-2, 24, 20, 4, ['block']),
    # reg=1e-9 (cond ~2e9): se espera roto aun a 32b
    ('r1e-9_s32', 1e-9, 24, 32, 12, ['block']),
    ('r1e-9_s24', 1e-9, 24, 24, 12, ['block']),
]
grid_base = dict(rt60=[0.61], target_angle=[0], target_dist=[1.0],
                 interf_configs=[[(90, 1.0)]], isir_db=[3], mismatch_gain=[0],
                 mismatch_phase=[0], use_wpe=[True], wpe_taps=[10], wpe_delay=[2],
                 error_angle_deg=[0.0], error_distance_m=[0.0])
procs = {"SOUDEN_ORACLE": SOUDEN_ORACLE_SCM(min_loading=1e-6, alpha=0.99)}

meta, frames = {}, []
for label, reg, fb, sb, si, methods in RUNS:
    meta[label] = (reg, sb)
    print(f"\n########## RUN: {label} (reg={reg:.0e} solve_bits={sb} int={si}) ##########")
    try:
        df = run_mird_grid_search({**grid_base, 'wpe_method': methods},
                                  provider, procs, cfg(reg, fb, sb, si),
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
blk = full[full.wpe_method == 'block'].merge(onl, on=['processor', 'rt60'], how='left')
blk['margin_vs_online'] = blk[PESQ] - blk['online_PESQ']
blk['reg']        = blk['label'].map(lambda x: meta[x][0])
blk['solve_bits'] = blk['label'].map(lambda x: meta[x][1])
order = {l: i for i, (l, *_) in enumerate(RUNS)}
blk['_o'] = blk['label'].map(order)
v = blk.sort_values(['processor', '_o'])[
    ['processor', 'reg', 'solve_bits', 'label', PESQ, 'margin_vs_online']]
pd.set_option('display.width', 170)
print("\n===== Precision del SOLVE (Cholesky interno) vs float (L=512, buffer mant12, G 24b) =====")
print("  solve_bits=NaN -> solve FLOAT (LAPACK). Cae la calidad = piso por cond(R) en fixed.")
print(v.rename(columns={PESQ: 'block_PESQ'}).to_string(
    index=False, float_format=lambda x: f"{x:+.3f}", na_rep=' float'))
v.to_csv(os.path.join(OUT, "summary_solveprec.csv"), index=False)
print(f"\nCSV -> {os.path.join(OUT, 'summary_solveprec.csv')}")
