"""
sweep_block_fixedbuffer_mird.py
===============================
(1) Verifica que reg=1e-2 (necesario para punto fijo) no baje el margen float, y
(2) barre la precision del BUFFER (window_bits) a L=512 para ver si buffer 8-10b
mete L=512 on-chip conservando el margen block>online.

L=512, taps=10, block_shift=32, iters=3, rt60 {0.16,0.61}, angulo 0.
Beamformers SOUDEN_ORACLE + DS. Resultados -> tests/dataset_out/block_fixedbuffer/

Corridas:
  online+float(reg1e-6)  -- baseline y referencia online
  float(reg1e-2)         -- (1) margen con reg alto
  fixed16 buffer=16/12/10/8 b (reg1e-2) -- (2) barrido de bits del buffer
"""
import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import numpy as np, pandas as pd
from propagation.mird_loader import MirdDatasetProvider
from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search
from evaluation.bf_wrappers import SOUDEN_ORACLE_SCM, DS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "tests", "dataset_out", "block_fixedbuffer")
os.makedirs(OUT, exist_ok=True)
provider = MirdDatasetProvider(root_dir=os.path.join(REPO, "tools/data/rirs/mird"))

M_MICS, EXP_BITS, F_BINS = 8, 5, 257   # array MIRD 8 mics, exp block-float, bins rfft(512)

def buffer_MB(L, mant):
    """Memoria on-chip del buffer block-float: F * L * M * (2*mant + exp/M) bits."""
    if mant is None:
        return float('nan')
    bpc = 2 * mant + EXP_BITS / M_MICS
    return F_BINS * L * M_MICS * bpc / 8 / 1e6

def cfg(reg, fixed_bits, window_mant, g_mant, L):
    return {
        'fs': 16000, 'duration': 16, 't_early': 0.008, 'array_center': [3.0, 3.0, 1.2],
        'mird_spacing': "3-3-3-8-3-3-3", 'snr_db': 60.0,
        'source_path': os.path.join(REPO, "tools/data/signals/p002_emo_adoration_sentences.wav"),
        'interf_paths': [os.path.join(REPO, "tools/data/signals/techno_gated commune.wav")],
        'wpe_taps': 10, 'wpe_delay': 2, 'wpe_alpha': 0.9999,
        'wpe_stft_size': 512, 'wpe_stft_shift': 128,
        'wpe_fixed_bits': fixed_bits, 'wpe_fixed_round': 'nearest', 'wpe_backend': 'cov',
        'wpe_block_L': L, 'wpe_block_mode': 'resolve', 'wpe_block_iters': 3,
        'wpe_block_shift': 32, 'wpe_block_reg': reg, 'wpe_block_solver': 'cholesky',
        'wpe_block_warm_start': False, 'wpe_block_window_bits': None,
        'wpe_block_window_mant': window_mant, 'wpe_block_window_exp': EXP_BITS,
        'wpe_block_g_mant': g_mant, 'wpe_block_g_exp': None,   # G 24b uniforme (g_mant=None) -> aisla el BUFFER
        'wpe_block_int_bits': {'in': 1, 'pred': 1, 'g': 6, 'p': 6},
        'stft_window': 512, 'stft_overlap': 384, 'eval_references': ['early'],
    }

# TRADE-OFF L x mant a MEMORIA fija: ¿mant8 con L largo, o acortar L con mant alta?
# reg=1e-9, taps=10, G en 24b (aisla el buffer). float_L* = buffer sin cuantizar
# (efecto puro de L). label, reg, fixed_bits, window_mant, g_mant, L, methods
REG = 1e-9
L_LIST = [256, 384, 512]
MANT_LIST = [8, 9, 10, 12]
RUNS = [('float_L512', REG, None, None, None, 512, ['online', 'block'])]
RUNS += [(f'float_L{L}', REG, None, None, None, L, ['block']) for L in (256, 384)]
RUNS += [(f'L{L}_m{m}', REG, 24, m, None, L, ['block']) for L in L_LIST for m in MANT_LIST]
grid_base = dict(rt60=[0.61], target_angle=[0], target_dist=[1.0],
                 interf_configs=[[(90, 1.0)]], isir_db=[3], mismatch_gain=[0],
                 mismatch_phase=[0], use_wpe=[True], wpe_taps=[10], wpe_delay=[2],
                 error_angle_deg=[0.0], error_distance_m=[0.0])
procs = {"SOUDEN_ORACLE": SOUDEN_ORACLE_SCM(min_loading=1e-6, alpha=0.99)}

meta = {}   # label -> (L, mant)
frames = []
for label, reg, fb, mant, g_mant, L, methods in RUNS:
    meta[label] = (L, mant)
    print(f"\n########## RUN: {label} (reg={reg:.0e} L={L} buf_mant={mant}) ##########")
    try:
        df = run_mird_grid_search({**grid_base, 'wpe_method': methods},
                                  provider, procs, cfg(reg, fb, mant, g_mant, L),
                                  output_dir=os.path.join(OUT, label),
                                  interpreter_1=None, interpreter_2=None, save_catalog=False)
        df['label'] = label
        frames.append(df)
    except Exception as e:
        print(f"  [ERR] {label}: {type(e).__name__}: {e}")

full = pd.concat(frames, ignore_index=True)
PESQ = "Delta_tot_PESQ_early"
onl = (full[(full.wpe_method == 'online')]
       .groupby(['processor', 'rt60'])[PESQ].first().rename('online_PESQ'))
blk = full[full.wpe_method == 'block'].merge(onl, on=['processor', 'rt60'], how='left')
blk['margin_vs_online'] = blk[PESQ] - blk['online_PESQ']
blk['L']    = blk['label'].map(lambda x: meta[x][0])
blk['mant'] = blk['label'].map(lambda x: meta[x][1])
blk['buf_MB'] = blk.apply(lambda r: buffer_MB(r['L'], r['mant']), axis=1)
pd.set_option('display.width', 170)

# (1) Ordenado por MEMORIA: la comparacion iso-memoria (mant8 vs acortar L).
v = blk.sort_values(['processor', 'buf_MB'])[
    ['processor', 'L', 'mant', 'buf_MB', PESQ, 'margin_vs_online']]
print("\n===== Trade-off L x mant, ordenado por MEMORIA del buffer (reg=1e-9, G 24b) =====")
print("  buf_MB=nan -> buffer FLOAT (efecto puro de L, sin storage on-chip real)")
print(v.rename(columns={PESQ: 'block_PESQ'}).to_string(
    index=False, float_format=lambda x: f"{x:+.3f}", na_rep='  float'))

# (2) Pivote L (filas) x mant (columnas) -> PESQ.
piv = blk.pivot_table(index='L', columns='mant', values=PESQ)
print("\n===== PESQ por L (filas) x mant (columnas); 'float' = buffer sin cuantizar =====")
fl = blk[blk['mant'].isna()].set_index('L')[PESQ].rename('float')
piv = piv.join(fl)
print(piv.to_string(float_format=lambda x: f"{x:+.3f}"))

blk.to_csv(os.path.join(OUT, "summary_L_mant.csv"), index=False)
print(f"\nCSV -> {os.path.join(OUT, 'summary_L_mant.csv')}")
