"""
DIAGNOSTICO (sin arreglo, sin MVDR): cuanto se parece la mascara calculada sobre
frames RECTANGULARES (los del DTLN / get_oracle_masks) a la mascara "correcta"
para los frames de la STFT del beamformer (HAMMING)?

Motivacion
----------
Toda la familia mask-based del repo hace esto:

    mascara  <- framing RECTANGULAR (buffer deslizante + np.fft.rfft)   [DTLN]
    espectro <- scipy.signal.stft(window='hamming')                     [BF]

y despues aparea mask[k,t] con X[k,t]. Los dos framings cubren LAS MISMAS
muestras (frame t de scipy == bloque t+1 del buffer, ya corregido por
align_mask_frames), pero NO el mismo espectro: la rectangular tiene lobulos
laterales a -13 dB contra -43 dB de la hamming.

Este script mide ese desacople sin meter el beamformer en el medio:
  1) IRM oracle calculada con framing rect  vs  con framing hamming.
  2) Lo mismo para la mascara del DTLN (rect nativo vs hamming, alimentando la
     red con |STFT_hamming| reescalada) -- solo si esta ai_edge_litert.

Uso:
    python tests/window_mismatch/mask_frame_mismatch_diag.py [--dtln]
"""

import os
import sys

import numpy as np
import scipy.signal as sig
import soundfile as sf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from beamforming.mask.oracle_masks import _stft_mag_blocks   # noqa: E402

FS = 16000
L, H = 512, 128
NOV = L - H

SPEECH = f"{ROOT}/tools/data/signals/p002_emo_adoration_sentences.wav"
NOISE = f"{ROOT}/tools/data/signals/techno_gated commune.wav"


def _load(path, n):
    x, fs = sf.read(path, always_2d=False)
    if x.ndim > 1:
        x = x[:, 0]
    if fs != FS:
        x = sig.resample_poly(x, FS, fs)
    if len(x) < n:
        x = np.tile(x, int(np.ceil(n / len(x))))
    return x[:n].astype(np.float64)


def scipy_mag(x, win):
    """|STFT| de scipy con el escalado DESHECHO -> misma convencion que rfft."""
    w = sig.get_window(win, L, fftbins=True) if isinstance(win, str) else win
    _, _, X = sig.stft(x, fs=FS, window=w, nperseg=L, noverlap=NOV, nfft=L)
    return np.abs(X) * w.sum()


def irm(Smag, Nmag, eps=1e-10):
    S, N = Smag ** 2, Nmag ** 2
    return S / (S + N + eps)


def _fit(m, T):
    if m.shape[1] < T:
        m = np.concatenate([m, np.repeat(m[:, -1:], T - m.shape[1], axis=1)], axis=1)
    return m[:, :T]


def align(mask_rect, T):
    """Bloque t+1 del buffer == frame t de scipy (align_mask_frames, shift=1)."""
    return _fit(mask_rect[:, 1:], T)


def band_report(diff, name, freqs, bands=((0, 300), (300, 1000), (1000, 4000), (4000, 8000))):
    print(f"  {name}: MAE global {np.mean(np.abs(diff)):.4f}   RMS {np.sqrt(np.mean(diff**2)):.4f}")
    for lo, hi in bands:
        sel = (freqs >= lo) & (freqs < hi)
        d = diff[sel]
        print(f"      {lo:5d}-{hi:5d} Hz : MAE {np.mean(np.abs(d)):.4f}  "
              f"bias {np.mean(d):+.4f}  p95|d| {np.percentile(np.abs(d), 95):.4f}")


def main():
    n = 15 * FS
    s = _load(SPEECH, n)
    v = _load(NOISE, n)
    # iSIR ~ 0 dB
    v *= np.sqrt(np.sum(s ** 2) / (np.sum(v ** 2) + 1e-20))

    freqs = np.arange(L // 2 + 1) * FS / L

    # ---- IRM con los dos framings -------------------------------------
    S_rect = _stft_mag_blocks(s.astype(np.float32), L, H)
    N_rect = _stft_mag_blocks(v.astype(np.float32), L, H)
    m_rect = irm(S_rect, N_rect)

    for win in ("hamming", "hann"):
        S_w, N_w = scipy_mag(s, win), scipy_mag(v, win)
        m_w = irm(S_w, N_w)
        T = m_w.shape[1]
        d = align(m_rect, T) - m_w
        print(f"\n=== IRM oracle: framing RECT (actual) - framing {win.upper()} (el del BF) ===")
        print(f"  corr = {np.corrcoef(align(m_rect, T).ravel(), m_w.ravel())[0,1]:.4f}")
        band_report(d, "IRM", freqs)

        # cuanto de la diferencia es SOLO el desalineado de 1 frame (control)
        d0 = _fit(m_rect, T) - m_w
        print(f"  [control] sin la correccion de 1 frame: MAE {np.mean(np.abs(d0)):.4f} "
              f"(corr {np.corrcoef(_fit(m_rect, T).ravel(), m_w.ravel())[0,1]:.4f})")

    # ---- efecto sobre el promedio de covarianza (peso del SCM) --------
    # El uso REAL de la mascara en Souden es como PESO de x x^H. El error que
    # importa es el de la ENERGIA que entra a Phi_NN, no el de la mascara sola.
    X_h = scipy_mag(s + v, "hamming")
    T = X_h.shape[1]
    m_h = irm(scipy_mag(s, "hamming"), scipy_mag(v, "hamming"))
    m_r = align(m_rect, T)
    P = X_h ** 2
    w_r = np.sum((1 - m_r) * P, axis=1)
    w_h = np.sum((1 - m_h) * P, axis=1)
    ratio = 10 * np.log10((w_r + 1e-20) / (w_h + 1e-20))
    print("\n=== Energia acumulada en Phi_NN (peso mask_n sobre |X_hamming|^2) ===")
    for lo, hi in ((0, 300), (300, 1000), (1000, 4000), (4000, 8000)):
        sel = (freqs >= lo) & (freqs < hi)
        print(f"  {lo:5d}-{hi:5d} Hz : rect vs hamming = {np.mean(ratio[sel]):+.2f} dB")

    if "--dtln" in sys.argv:
        dtln_part(s + v, freqs)


def dtln_part(mix, freqs):
    from ai_edge_litert.interpreter import Interpreter
    from beamforming.mask.dtln_masks import get_dtln_masks_sharpen
    model = f"{ROOT}/src/dnn_denoise/models/model_quant_1.tflite"

    m_rect, _ = get_dtln_masks_sharpen(mix[None, :], 0, model, L, H, sharpen_exp=1.0)

    # DTLN alimentado con los frames de la STFT hamming (misma normalizacion de
    # pico y mismo desescalado que coupled_wrappers._masks_from_stft)
    itp = Interpreter(model_path=model)
    itp.allocate_tensors()
    i1, o1 = itp.get_input_details(), itp.get_output_details()
    st = np.zeros(i1[1]['shape'], dtype=np.float32)
    peak = np.max(np.abs(mix))
    Xh = scipy_mag(mix / peak, "hamming")
    m_h = np.zeros_like(Xh, dtype=np.float32)
    for t in range(Xh.shape[1]):
        mag = np.ascontiguousarray(Xh[:, t].reshape(1, 1, -1), dtype=np.float32)
        itp.set_tensor(i1[1]['index'], st)
        itp.set_tensor(i1[0]['index'], mag)
        itp.invoke()
        m_h[:, t] = np.squeeze(itp.get_tensor(o1[0]['index']).copy())
        st = itp.get_tensor(o1[1]['index']).copy()

    def stretch(m):
        return (m - m.min()) / (m.max() - m.min() + 1e-12)

    T = m_h.shape[1]
    a, b = stretch(align(m_rect, T)), stretch(m_h)
    print("\n=== Mascara DTLN: entrada RECT (actual) vs entrada HAMMING ===")
    print(f"  corr = {np.corrcoef(a.ravel(), b.ravel())[0,1]:.4f}")
    band_report(a - b, "DTLN", freqs)


if __name__ == "__main__":
    main()
