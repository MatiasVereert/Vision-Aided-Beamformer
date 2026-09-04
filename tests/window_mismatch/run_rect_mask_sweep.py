"""
TODO RECTANGULAR + RECALIBRACION DE LA MASCARA.

Hipotesis a probar: el `** 4` de `get_dtln_masks_sharpen` se eligio con el
sistema DESACOPLADO (mascara rect -> STFT hamming). Parte de lo que hace ese
exponente es compensar el mismatch de ventanas al acumular las SCM. Si se pasa
TODO a rectangular (una sola FFT en el sistema, que es lo que se quiere en HW),
el exponente optimo tendria que ser otro.

Se barre el post-proceso de la mascara con la ventana como segunda variable, con
el core BASE de Souden (NM_MVDR) y nada mas:

  sharpen_exp in {1, 2, 3, 4, 6}   x   win in {rect, hamming}

y ademas el WARP LOGIT-AFIN calibrado (`scm_calibration.masks_from_raw`), que es
la familia que contiene al sharpening y ademas desacopla las dos ramas.

La mascara CRUDA del DTLN se calcula UNA VEZ por escena y se cachea: el
post-proceso (stretch + potencia, o warp) se aplica encima. Eso hace que el
barrido cueste casi lo mismo que una sola corrida.

Uso:
    python tests/window_mismatch/run_rect_mask_sweep.py [--warp] [--full]
"""

import os
import sys
import hashlib

import numpy as np
import pandas as pd
import scipy.signal as sig
import tensorflow as tf

ROOT = "/home/matias/Documents/Tesis/Vision-Aided-Beamformer"
sys.path.insert(0, os.path.join(ROOT, "src"))

from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search   # noqa: E402
from evaluation.bf_wrappers import resolve_stft_window                      # noqa: E402
from beamforming.mask.dtln_masks import get_dtln_masks_soft, align_mask_frames  # noqa: E402
from beamforming.mask.scm_calibration import masks_from_raw                 # noqa: E402
from beamforming.mask.souden_mvdr import MVDR_Souden_recursive_mask         # noqa: E402
from propagation.mird_loader import MirdDatasetProvider                     # noqa: E402

MODEL_1 = f"{ROOT}/src/dnn_denoise/models/model_quant_1.tflite"
MODEL_2 = f"{ROOT}/src/dnn_denoise/models/model_quant_2.tflite"
OUT_DIR = os.environ.get("SWEEP_OUT", "tests/dataset_out/rect_mask_sweep")

_RAW_CACHE = {}


def dtln_raw_mask(mic_signals, ref, model_path, L, H):
    """Mascara CRUDA del DTLN (sin stretch ni sharpen), cacheada por escena."""
    key = (hashlib.md5(np.ascontiguousarray(mic_signals[ref]).tobytes()).hexdigest(),
           ref, L, H)
    if key not in _RAW_CACHE:
        m_raw, _ = get_dtln_masks_soft(mic_signals, ref, model_path,
                                       block_len=L, block_shift=H)
        _RAW_CACHE.clear()          # una escena a la vez: no acumular memoria
        _RAW_CACHE[key] = m_raw
    return _RAW_CACHE[key]


class RectMaskProc:
    """
    NM_MVDR (core base de Souden) con el POST-PROCESO DE LA MASCARA y la VENTANA
    como parametros. Con win_type='hamming', post='pow', sharpen_exp=4.0
    reproduce NM_MVDR exactamente (mismo stretch global, mismo align_mask_frames).
    """

    def __init__(self, win_type='rect', post='pow', sharpen_exp=4.0, warp=None,
                 nperseg=512, noverlap=384, min_loading=1e-6, alpha=0.99,
                 synth=None):
        # synth=None -> iSTFT de scipy con la MISMA ventana (comportamiento de
        # todos los wrappers). synth='hann' -> ANALISIS RECTANGULAR + SINTESIS
        # CON TAPER: OLA manual con hann/2, que cumple COLA a hop=L/4 y da
        # reconstruccion perfecta (verificado, error 1e-15 en el interior).
        # Sirve para separar las dos cosas que hace la ventana: el analisis
        # (leakage -> SCM y mascara) y la sintesis (discontinuidades de borde de
        # frame cuando el filtro cambia frame a frame). El analisis sigue siendo
        # rectangular, o sea UNA sola FFT y mascara consistente; el taper cuesta
        # 512 multiplicaciones por frame de salida y ninguna transformada extra.
        self.synth = synth
        self.win_type = win_type
        self.post = post                # 'pow' | 'warp'
        self.sharpen_exp = sharpen_exp
        self.warp = warp                # (a_s, b_s, a_n, b_n)
        self.nperseg, self.noverlap = nperseg, noverlap
        self.min_loading, self.alpha = min_loading, alpha

    def process(self, mic_signals, scene_config):
        fs = scene_config['fs']
        L = scene_config.get('stft_window', self.nperseg)
        nov = scene_config.get('stft_overlap', self.noverlap)
        H = L - nov
        ref = int(scene_config.get('ref_mic_idx', mic_signals.shape[0] // 2))
        model_path = scene_config.get('dtln_model_path', MODEL_1)
        win = resolve_stft_window(scene_config, self.win_type, L)

        m_raw = dtln_raw_mask(mic_signals, ref, model_path, L, H)

        if self.post == 'pow':
            # IDENTICO a get_dtln_masks_sharpen: stretch min-max global + potencia
            m = (m_raw - m_raw.min()) / (m_raw.max() - m_raw.min() + 1e-12)
            mask_s, mask_n = m ** self.sharpen_exp, (1.0 - m) ** self.sharpen_exp
        elif self.post == 'warp':
            mask_s, mask_n = masks_from_raw(m_raw, *self.warp)
        else:
            raise ValueError(self.post)

        mask_s, mask_n = align_mask_frames((mask_s, mask_n), 1)

        _, _, Zxx = sig.stft(mic_signals, fs=fs, window=win, nperseg=L,
                             noverlap=nov, nfft=L)
        X_stft = np.transpose(Zxx, (1, 2, 0))
        T = min(X_stft.shape[1], mask_s.shape[1])
        X_stft, mask_s, mask_n = X_stft[:, :T, :], mask_s[:, :T], mask_n[:, :T]

        Y_stft, weights = MVDR_Souden_recursive_mask(
            X_stft, mask_s, mask_n, min_loading=self.min_loading,
            save_weights=True, alpha=self.alpha, ref_mic_idx=ref)

        n_out = mic_signals.shape[1]
        if self.synth is None:
            _, y = sig.istft(Y_stft, fs=fs, window=win, nperseg=L, noverlap=nov,
                             nfft=L)
            y = y[:n_out]
        else:
            y = _ola_taper(Y_stft, L, H, self.synth, n_out)
        return y, weights


def _ola_taper(Y, L, H, synth, n_out):
    """
    OLA manual con ventana de SINTESIS distinta de la de analisis.

    Y viene de scipy.stft con ventana RECTANGULAR, o sea escalado por 1/L: se
    deshace ese factor para recuperar el frame crudo. La ventana de sintesis se
    normaliza por su suma OLA (hann a hop=L/4 suma 2), asi que sin modificar Y la
    reconstruccion es exacta. El recorte de L//2 al principio compensa el
    boundary='zeros' de scipy.
    """
    w = sig.get_window(synth, L, fftbins=True)
    acc = np.zeros(L)
    for m in range(0, L, H):
        acc += np.roll(w, m)
    w = w / acc.mean()                      # COLA: sum_m w(n - mH) = 1
    frames = np.fft.irfft(Y * L, n=L, axis=0)
    T = Y.shape[1]
    y = np.zeros((T - 1) * H + L)
    for t in range(T):
        y[t * H:t * H + L] += frames[:, t] * w
    return y[L // 2:L // 2 + n_out]


EXPS = [float(e) for e in os.environ.get('SWEEP_EXPS', '1,2,3,4,6').split(',')]
# con --warp alcanza con los dos exponentes de referencia
if '--warp' in sys.argv:
    EXPS = [4.0, 8.0]


def build_processors_synth(warp_sets):
    """
    Modo --synth: separa el efecto del ANALISIS del de la SINTESIS.

    rect_*_synhann tiene ANALISIS RECTANGULAR (una sola FFT, mascara consistente,
    SCM sin la ventana de por medio) y SINTESIS CON TAPER. Si la perdida de PESQ
    del all-rect viene de las discontinuidades de borde de frame y no del
    leakage del analisis, esta fila la recupera sin tocar el DTLN ni agregar
    transformadas.
    """
    P = RectMaskProc
    procs = {
        "hamming_pow8": P(win_type='hamming', post='pow', sharpen_exp=8.0),
        "rect_pow8": P(win_type='rect', post='pow', sharpen_exp=8.0),
        "rect_pow8_synhann": P(win_type='rect', post='pow', sharpen_exp=8.0,
                               synth='hann'),
        "rect_pow4_synhann": P(win_type='rect', post='pow', sharpen_exp=4.0,
                               synth='hann'),
    }
    if 'rectfit' in warp_sets:
        procs["rect_warp"] = P(win_type='rect', post='warp',
                               warp=warp_sets['rectfit'])
        procs["rect_warp_synhann"] = P(win_type='rect', post='warp',
                                       warp=warp_sets['rectfit'], synth='hann')
    if 'hammfit' in warp_sets:
        procs["hamming_warp"] = P(win_type='hamming', post='warp',
                                  warp=warp_sets['hammfit'])
    return procs


def build_processors(warp_sets=None):
    procs = {}
    for win in ('hamming', 'rect'):
        for e in EXPS:
            procs[f"{win}_pow{e:g}"] = RectMaskProc(win_type=win, post='pow',
                                                    sharpen_exp=e)
    for name, th in (warp_sets or {}).items():
        for win in ('hamming', 'rect'):
            procs[f"{win}_warp_{name}"] = RectMaskProc(win_type=win, post='warp',
                                                       warp=th)
    return procs


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

    # Warps CALIBRADOS (tests/scm_mask_calibration_run.py --win {rect,hamming}
    # --nu 0). Se cruzan a proposito: aplicar el ajuste de una ventana en la otra
    # mide cuanto del warp es especifico del dominio y cuanto es de la mascara.
    warp_sets = {}
    if "--warp" in sys.argv or "--synth" in sys.argv:
        for name, sub in (('rectfit', 'scm_mask_calib_rect'),
                          ('hammfit', 'scm_mask_calib_hamm')):
            npz = os.path.join(ROOT, "tests/dataset_out", sub, "mask_calib_params.npz")
            if os.path.exists(npz):
                z = np.load(npz)
                warp_sets[name] = (z['a_s'], z['b_s'], z['a_n'], z['b_n'])
            else:
                print(f"[!] falta {npz}")

    df = run_mird_grid_search(
        grid_params=param_grid, dataset_provider=provider,
        processors=(build_processors_synth(warp_sets) if "--synth" in sys.argv
                    else build_processors(warp_sets)),
        scene_base_config=base_config,
        output_dir=OUT_DIR, interpreter_1=itp1, interpreter_2=itp2,
        save_catalog=False, apply_dtln_post=False,
    )
    summarize(df)
    return df


def summarize(df, ref='early'):
    metrics = ['PESQ', 'STOI', 'SI-SDR', 'SDR', 'SIR', 'SAR']
    cols = [f"proc_{m}_{ref}" for m in metrics if f"proc_{m}_{ref}" in df.columns]
    pd.set_option('display.width', 220)
    g = df.groupby('processor')[cols].mean().round(3)
    print(f"\n=== BARRIDO sharpen x ventana (referencia '{ref}') ===")
    print(g.to_string())
    if "--synth" in sys.argv:
        print("\n=== ordenado por PESQ ===")
        print(g.sort_values(f'proc_PESQ_{ref}', ascending=False).to_string())
        return
    print("\n=== por ventana, ordenado por PESQ ===")
    for win in ('hamming', 'rect'):
        sub = g[[i.startswith(win + "_") for i in g.index]]
        print(f"\n-- {win} --")
        print(sub.sort_values(f'proc_PESQ_{ref}', ascending=False).to_string())


if __name__ == "__main__":
    main()
