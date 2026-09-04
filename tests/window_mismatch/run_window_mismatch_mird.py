"""
AUDITORIA: la mascara se calcula con framing RECTANGULAR y se aplica sobre una
STFT con ventana HAMMING. Cuanto cuesta esa inconsistencia?

Aisla UNA sola variable a la vez sobre la MISMA cadena (core de Souden,
min_loading=1e-6, alpha=0.99, ref_mic del benchmark):

  * ORA_maskRect  : mascara IRM oracle con el framing rect de get_oracle_masks
                    (lo que hace hoy ORACLE_MB_MVDR_SOUDEN) + BF hamming.
  * ORA_maskBF    : MISMA IRM oracle pero calculada sobre los frames de la STFT
                    del beamformer (hamming) -> mascara y espectro CONSISTENTES.
                    Con oracle no hay efecto "fuera de distribucion" de la red:
                    mide el costo PURO de la inconsistencia.
  * NM_maskRect / NM_maskBF : lo mismo con la mascara del DTLN (la de verdad).
                    NM_maskBF alimenta la red con |STFT_hamming| reescalada por
                    win.sum() (misma convencion que coupled_wrappers).
  * *_PF_*        : idem con el post-filtro de sustraccion espectral encima, que
                    es donde la mascara actua como GANANCIA directa sobre el
                    espectro (no como peso de covarianza) y por lo tanto donde la
                    inconsistencia deberia doler mas.
  * *_bfRect      : control con TODO rectangular (consistente por el otro lado).

Alineacion de frames: el camino rect necesita el shift de 1 frame
(align_mask_frames); el camino "maskBF" sale de la misma STFT, asi que va con
shift 0 por construccion.

Uso:
    python tests/window_mismatch/run_window_mismatch_mird.py [--full]
"""

import os
import sys

import numpy as np
import pandas as pd
import scipy.signal as sig
import tensorflow as tf

ROOT = "/home/matias/Documents/Tesis/Vision-Aided-Beamformer"
sys.path.insert(0, os.path.join(ROOT, "src"))

from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search   # noqa: E402
from evaluation.bf_wrappers import resolve_stft_window                      # noqa: E402
from beamforming.mask.dtln_masks import (get_dtln_masks_sharpen,            # noqa: E402
                                         align_mask_frames)
from beamforming.mask.oracle_masks import get_oracle_masks                  # noqa: E402
from beamforming.mask.souden_mvdr import MVDR_Souden_recursive_mask         # noqa: E402
from propagation.mird_loader import MirdDatasetProvider                     # noqa: E402

MODEL_1 = f"{ROOT}/src/dnn_denoise/models/model_quant_1.tflite"
MODEL_2 = f"{ROOT}/src/dnn_denoise/models/model_quant_2.tflite"
OUT_DIR = "tests/dataset_out/window_mismatch"


class MaskFramingProc:
    """NM_MVDR / ORACLE_MB_MVDR_SOUDEN con el FRAMING DE LA MASCARA como variable."""

    _itp_cache = {}

    def __init__(self, mask_source='dtln', mask_framing='rect', win_type='hamming',
                 nperseg=512, noverlap=384, min_loading=1e-6, alpha=0.99,
                 sharpen_exp=None, smooth=None):
        assert mask_source in ('dtln', 'oracle')
        assert mask_framing in ('rect', 'bf')
        self.mask_source = mask_source
        self.mask_framing = mask_framing
        self.win_type = win_type
        self.nperseg, self.noverlap = nperseg, noverlap
        self.min_loading, self.alpha = min_loading, alpha
        # defaults historicos de cada camino: DTLN **4, oracle suave
        self.sharpen_exp = sharpen_exp if sharpen_exp is not None else (
            4.0 if mask_source == 'dtln' else 1.0)
        self.smooth = smooth

    # ------------------------------------------------------------------
    def _masks_bf_framing(self, mic_signals, scene_config, ref, win, model_path):
        """Mascara calculada SOBRE LOS FRAMES DE LA STFT DEL BEAMFORMER."""
        fs = scene_config['fs']
        L, nov = self.nperseg, self.noverlap
        w = win if isinstance(win, np.ndarray) else sig.get_window(win, L, fftbins=True)

        def mag(x):
            _, _, X = sig.stft(x, fs=fs, window=w, nperseg=L, noverlap=nov, nfft=L)
            return np.abs(X) * w.sum()      # deshace el escalado de scipy

        if self.mask_source == 'oracle':
            S = mag(np.asarray(scene_config['oracle_target'])[ref])
            N = mag(np.asarray(scene_config['oracle_noise'])[ref])
            m = (S ** 2) / (S ** 2 + N ** 2 + 1e-10)
            mask_s = m ** self.sharpen_exp
            mask_n = (1.0 - m) ** self.sharpen_exp
            return mask_s.astype(np.float32), mask_n.astype(np.float32)

        # DTLN alimentado con los frames de la STFT (misma normalizacion de pico)
        from ai_edge_litert.interpreter import Interpreter
        itp = MaskFramingProc._itp_cache.get(model_path)
        if itp is None:
            itp = Interpreter(model_path=str(model_path))
            itp.allocate_tensors()
            MaskFramingProc._itp_cache[model_path] = itp
        i1, o1 = itp.get_input_details(), itp.get_output_details()
        st = np.zeros(i1[1]['shape'], dtype=np.float32)

        x = np.asarray(mic_signals)[ref]
        peak = np.max(np.abs(x))
        X = mag(x / (peak if peak > 0 else 1.0))
        m = np.zeros(X.shape, dtype=np.float32)
        for t in range(X.shape[1]):
            buf = np.ascontiguousarray(X[:, t].reshape(1, 1, -1), dtype=np.float32)
            itp.set_tensor(i1[1]['index'], st)
            itp.set_tensor(i1[0]['index'], buf)
            itp.invoke()
            m[:, t] = np.squeeze(itp.get_tensor(o1[0]['index']).copy())
            st = itp.get_tensor(o1[1]['index']).copy()

        m = (m - m.min()) / (m.max() - m.min() + 1e-12)      # mismo stretch global
        return m ** self.sharpen_exp, (1.0 - m) ** self.sharpen_exp

    # ------------------------------------------------------------------
    def process(self, mic_signals, scene_config):
        fs = scene_config['fs']
        L = scene_config.get('stft_window', self.nperseg)
        nov = scene_config.get('stft_overlap', self.noverlap)
        H = L - nov
        self.nperseg, self.noverlap = L, nov
        ref = int(scene_config.get('ref_mic_idx', mic_signals.shape[0] // 2))
        model_path = scene_config.get('dtln_model_path', MODEL_1)
        win = resolve_stft_window(scene_config, self.win_type, L)

        if self.mask_framing == 'rect':
            if self.mask_source == 'dtln':
                mask_s, mask_n = get_dtln_masks_sharpen(
                    mic_signals, ref, model_path, block_len=L, block_shift=H,
                    sharpen_exp=self.sharpen_exp)
            else:
                mask_s, mask_n = get_oracle_masks(
                    scene_config['oracle_target'], scene_config['oracle_noise'],
                    ref_mic=ref, block_len=L, block_shift=H,
                    sharpen_exp=self.sharpen_exp)
            # el bloque i del buffer == frame i-1 de scipy
            mask_s, mask_n = align_mask_frames((mask_s, mask_n), 1)
        else:
            mask_s, mask_n = self._masks_bf_framing(
                mic_signals, scene_config, ref, win, model_path)

        _, _, Zxx = sig.stft(mic_signals, fs=fs, window=win, nperseg=L,
                             noverlap=nov, nfft=L)
        X_stft = np.transpose(Zxx, (1, 2, 0))

        T = min(X_stft.shape[1], mask_s.shape[1])
        X_stft, mask_s, mask_n = X_stft[:, :T, :], mask_s[:, :T], mask_n[:, :T]

        Y_stft, weights = MVDR_Souden_recursive_mask(
            X_stft, mask_s, mask_n, min_loading=self.min_loading,
            save_weights=True, alpha=self.alpha, ref_mic_idx=ref)

        if self.smooth is not None:
            soft = np.clip(mask_s ** (1.0 / self.sharpen_exp), 0.0, 1.0)
            G = self.smooth + (1.0 - self.smooth) * soft
            Y_stft = Y_stft * G

        _, y = sig.istft(Y_stft, fs=fs, window=win, nperseg=L, noverlap=nov, nfft=L)
        return y[:mic_signals.shape[1]], weights


def build_processors():
    P = MaskFramingProc
    return {
        # --- costo PURO de la inconsistencia (sin la red en el medio)
        "ORA_maskRect":      P('oracle', 'rect', 'hamming'),
        "ORA_maskBF":        P('oracle', 'bf',   'hamming'),
        "ORA_todoRect":      P('oracle', 'rect', 'rect'),
        # --- lo mismo con la mascara real del DTLN
        "NM_maskRect":       P('dtln', 'rect', 'hamming'),
        "NM_maskBF":         P('dtln', 'bf',   'hamming'),
        "NM_todoRect":       P('dtln', 'rect', 'rect'),
        # --- con post-filtro: la mascara como GANANCIA sobre el espectro
        "NM_PF_maskRect":    P('dtln', 'rect', 'hamming', smooth=0.33),
        "NM_PF_maskBF":      P('dtln', 'bf',   'hamming', smooth=0.33),
    }


def main():
    try:
        itp1 = tf.lite.Interpreter(model_path=MODEL_1); itp1.allocate_tensors()
        itp2 = tf.lite.Interpreter(model_path=MODEL_2); itp2.allocate_tensors()
    except Exception as e:
        print(f"[!] DTLN no cargado: {e}")
        itp1 = itp2 = None

    provider = MirdDatasetProvider(root_dir=os.path.abspath(f"{ROOT}/tools/data/rirs/mird"))

    base_config = {
        'fs': 16000, 'duration': 15, 't_early': 0.050,
        'array_center': [3.0, 3.0, 1.2], 'mird_spacing': "3-3-3-8-3-3-3",
        'snr_db': 60.0,
        'source_path': f"{ROOT}/tools/data/signals/p002_emo_adoration_sentences.wav",
        'interf_paths': [f"{ROOT}/tools/data/signals/techno_gated commune.wav"],
        'wpe_taps': 7, 'wpe_delay': 3, 'wpe_alpha': 0.9999,
        'wpe_stft_size': 512, 'wpe_stft_shift': 128,
        'wpe_fixed_bits': None, 'wpe_fixed_round': 'nearest', 'wpe_backend': 'cov',
        'wpe_block_L': 512, 'wpe_block_shift': 2, 'wpe_block_iters': 2,
        'wpe_block_reg': 1e-6, 'wpe_block_solver': 'cholesky', 'wpe_block_mode': 'resolve',
        'stft_window': 512, 'stft_overlap': 384,
        'eval_references': ['anechoic', 'early', 'reverberant'],
        'dtln_model_path': MODEL_1, 'dtln_model2_path': MODEL_2,
    }

    full = "--full" in sys.argv
    param_grid = {
        'rt60': [0.360, 0.610],
        'target_angle': [0], 'target_dist': [1.0],
        'interf_configs': ([[(45, 1.0)], [(90, 2.0)]] if full else [[(45, 1.0)]]),
        'isir_db': [-5, 0],
        'mismatch_gain': [0], 'mismatch_phase': [0],
        'use_wpe': [True], 'wpe_method': ['online'],
        'wpe_taps': [7], 'wpe_delay': [2],
        'error_angle_deg': [0.0], 'error_distance_m': [0.0],
    }

    df = run_mird_grid_search(
        grid_params=param_grid, dataset_provider=provider,
        processors=build_processors(), scene_base_config=base_config,
        output_dir=OUT_DIR, interpreter_1=itp1, interpreter_2=itp2,
        save_catalog=False, apply_dtln_post=False,
    )
    summarize(df)
    return df


PAIRS = [
    ("costo de la inconsistencia -- ORACLE (sin red)", "ORA_maskBF", "ORA_maskRect"),
    ("costo de la inconsistencia -- DTLN",             "NM_maskBF", "NM_maskRect"),
    ("costo de la inconsistencia -- DTLN + postfiltro", "NM_PF_maskBF", "NM_PF_maskRect"),
    ("todo rectangular vs actual -- ORACLE",           "ORA_todoRect", "ORA_maskRect"),
    ("todo rectangular vs actual -- DTLN",             "NM_todoRect", "NM_maskRect"),
]


def summarize(df, ref='early'):
    metrics = ['PESQ', 'STOI', 'SI-SDR', 'SDR', 'SIR', 'SAR']
    cols = [f"proc_{m}_{ref}" for m in metrics if f"proc_{m}_{ref}" in df.columns]
    pd.set_option('display.width', 220)
    print(f"\n=== MEDIA POR PROCESADOR (referencia '{ref}', {df['processor'].value_counts().iloc[0]} escenas) ===")
    print(df.groupby('processor')[cols].mean().round(3).to_string())

    for title, a, b in PAIRS:
        da = df[df.processor == a].reset_index(drop=True)
        db = df[df.processor == b].reset_index(drop=True)
        if da.empty or db.empty:
            continue
        n = min(len(da), len(db))
        print(f"\n--- {title}:  {a} - {b} ---")
        for c in cols:
            d = da[c].to_numpy()[:n] - db[c].to_numpy()[:n]
            print(f"   {c:24s} media {np.nanmean(d):+7.3f}   mediana {np.nanmedian(d):+7.3f}"
                  f"   por escena {np.round(d, 3)}")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()


def selftest():
    """
    CONTROL de que el camino 'maskBF' es EXACTAMENTE la misma cuenta que el
    camino actual y lo unico que cambia es la ventana: con ventana rectangular
    tiene que reproducir get_oracle_masks + align_mask_frames(1).
    """
    from beamforming.mask.oracle_masks import get_oracle_masks
    rng = np.random.default_rng(1)
    n = 16000 * 4
    s, v = rng.standard_normal((2, n)) * 0.1, rng.standard_normal((2, n)) * 0.1
    cfg = {'fs': 16000, 'oracle_target': s, 'oracle_noise': v, 'ref_mic_idx': 0}
    p = MaskFramingProc('oracle', 'bf', 'rect')
    mb, _ = p._masks_bf_framing(None, cfg, 0,
                                sig.get_window('boxcar', 512, fftbins=True), None)
    mr, _ = get_oracle_masks(s, v, ref_mic=0, block_len=512, block_shift=128,
                             sharpen_exp=1.0)
    mr, _ = align_mask_frames((mr, 1 - mr), 1)
    T = min(mb.shape[1], mr.shape[1]) - 2
    err = np.max(np.abs(mb[:, :T] - mr[:, :T]))
    print(f"[selftest] max|maskBF(rect) - maskRect| = {err:.3e}  "
          f"({'OK' if err < 1e-5 else 'FALLA'})")
