"""
Verificacion numerica del acople BF+DTLN en una sola STFT.

Corre 4 chequeos (sin dataset MIRD, con senal sintetica):

  1. ALINEACION DE FRAMES: el frame t de scipy.signal.stft(boxcar, 512/128)
     es EXACTAMENTE el bloque t+1 del buffer deslizante del DTLN.
  2. RECONSTRUCCION: iSTFT scipy con ventana rectangular (NOLA) reconstruye
     la senal (no hace falta hamming para tener reconstruccion perfecta).
  3. EQUIVALENCIA DEL ACOPLE: con ventana RECTANGULAR, correr el DTLN sobre los
     frames de la STFT da el MISMO resultado que el lazo nativo
     apply_dtln_post_tflite_realtime() sobre la senal de tiempo.
  4. NO EQUIVALENCIA CON HAMMING: con ventana hamming, el mismo acople NO
     reproduce el camino nativo (cuanto se despega es justamente lo que hay
     que medir con metricas).

Uso:
    python tests/window_coupling/test_coupling_equivalence.py
"""
import os
import sys

import numpy as np
import scipy.signal as sig

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from evaluation.coupled_wrappers import BF_DTLN_COUPLED           # noqa: E402
from dnn_denoise.dtln_lite import apply_dtln_post_tflite_realtime  # noqa: E402

MODEL_1 = "/home/matias/Documents/Tesis/Vision-Aided-Beamformer/src/dnn_denoise/models/model_quant_1.tflite"
MODEL_2 = "/home/matias/Documents/Tesis/Vision-Aided-Beamformer/src/dnn_denoise/models/model_quant_2.tflite"

L, H = 512, 128
FS = 16000


def dtln_frames(x, L=L, H=H):
    """Frames tal como los arma el lazo nativo del DTLN (buffer deslizante)."""
    nb = (len(x) - (L - H)) // H
    out = np.zeros((nb, L))
    buf = np.zeros(L)
    for i in range(nb):
        buf[:-H] = buf[H:]
        buf[-H:] = x[i * H:i * H + H]
        out[i] = buf
    return out


def check_1_frame_alignment(x):
    D = dtln_frames(x)
    _, _, Z = sig.stft(x, fs=FS, window='boxcar', nperseg=L, noverlap=L - H, nfft=L)
    S = np.fft.irfft(Z * L, n=L, axis=0).T      # se deshace el 1/win.sum() de scipy
    n = min(len(D) - 1, len(S))
    err = np.max(np.abs(D[1:1 + n] - S[:n]))
    print(f"[1] frame t de scipy == bloque t+1 del DTLN : max|dif| = {err:.3e}")
    return err < 1e-10


def check_2_reconstruction(x):
    res = {}
    for w in ('boxcar', 'hamming', 'hann'):
        _, _, Z = sig.stft(x, fs=FS, window=w, nperseg=L, noverlap=L - H, nfft=L)
        _, y = sig.istft(Z, fs=FS, window=w, nperseg=L, noverlap=L - H, nfft=L)
        y = y[:len(x)]
        res[w] = np.max(np.abs(y - x))
        print(f"[2] reconstruccion STFT/iSTFT ventana {w:8s}: max|dif| = {res[w]:.3e}")
    return all(v < 1e-10 for v in res.values())


def _coupled_dtln_on_signal(x, win_name):
    """Corre SOLO la etapa DTLN acoplada sobre la STFT de x (sin beamformer)."""
    bf = BF_DTLN_COUPLED(model1_path=MODEL_1, model2_path=MODEL_2, win_type=win_name)
    bf._ensure_interpreters(MODEL_1, MODEL_2)
    win = sig.get_window(win_name if win_name != 'rect' else 'boxcar', L, fftbins=True)

    _, _, Z = sig.stft(x, fs=FS, window=win, nperseg=L, noverlap=L - H, nfft=L)
    peak = float(np.max(np.abs(x)))
    scale = peak if peak > 1e-6 else 1.0
    Y = (Z * float(np.sum(win))) / scale
    return bf._dtln_on_stft(Y, win=win, hop=H, n_out=len(x), scale_factor=scale)


def native_dtln_skipping_block0(itp1, itp2, x):
    """
    Lazo NATIVO del DTLN (copia fiel de apply_dtln_post_tflite_realtime) pero
    SALTEANDO el bloque 0.

    Por que: el frame 0 de scipy.signal.stft equivale al bloque 1 del buffer del
    DTLN, asi que el camino acoplado nunca ve el bloque 0 (queda fuera de la
    grilla de frames). Esta referencia consume exactamente la MISMA secuencia de
    bloques que el acoplado, con los mismos estados LSTM en cero -> permite
    comparar bit a bit el enmarcado + OLA + normalizacion, sin el transitorio de
    arranque de un frame.
    """
    x = np.ascontiguousarray(np.squeeze(x), dtype=np.float32)
    peak = np.max(np.abs(x))
    scale = peak if peak > 1e-6 else 1.0
    xn = x / scale

    out_audio = np.zeros_like(xn)
    in_buffer = np.zeros(L, dtype=np.float32)
    out_buffer = np.zeros(L, dtype=np.float32)

    in1, out1 = itp1.get_input_details(), itp1.get_output_details()
    in2, out2 = itp2.get_input_details(), itp2.get_output_details()
    states_1 = np.zeros(in1[1]['shape'], dtype=np.float32)
    states_2 = np.zeros(in2[1]['shape'], dtype=np.float32)

    num_blocks = (len(xn) - (L - H)) // H
    for idx in range(num_blocks):
        in_buffer[:-H] = in_buffer[H:]
        in_buffer[-H:] = xn[idx * H: idx * H + H]
        if idx == 0:
            continue                      # <-- unico cambio vs el lazo nativo

        in_block_fft = np.fft.rfft(in_buffer)
        in_mag = np.ascontiguousarray(
            np.reshape(np.abs(in_block_fft), (1, 1, -1)), dtype=np.float32)
        in_phase = np.angle(in_block_fft)

        itp1.set_tensor(in1[1]['index'], states_1)
        itp1.set_tensor(in1[0]['index'], in_mag)
        itp1.invoke()
        out_mask = itp1.get_tensor(out1[0]['index']).copy()
        states_1 = itp1.get_tensor(out1[1]['index']).copy()

        est = in_mag * out_mask * np.exp(1j * in_phase)
        est_block = np.ascontiguousarray(
            np.reshape(np.fft.irfft(est, n=L), (1, 1, -1)), dtype=np.float32)

        itp2.set_tensor(in2[1]['index'], states_2)
        itp2.set_tensor(in2[0]['index'], est_block)
        itp2.invoke()
        out_block = itp2.get_tensor(out2[0]['index']).copy()
        states_2 = itp2.get_tensor(out2[1]['index']).copy()

        out_buffer[:-H] = out_buffer[H:]
        out_buffer[-H:] = 0.0
        out_buffer += np.squeeze(out_block)
        out_audio[idx * H: idx * H + H] = out_buffer[:H]

    return out_audio * scale


def _snr_db(a, b):
    """SNR entre dos senales (cuanto se parece b a a). Inf = identicas."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    d = a - b
    den = np.sum(d ** 2)
    if den <= 0:
        return np.inf
    return 10 * np.log10(np.sum(a ** 2) / den)


def check_3_and_4(x):
    """
    Criterio: NO se puede exigir igualdad bit a bit. El DTLN cuantizado (int8) con
    estados LSTM amplifica muchisimo cualquier perturbacion numerica: redondear los
    frames de float64 a float32 ya degrada la salida a ~46 dB de SNR. Entonces la
    equivalencia se mide en SNR entre salidas.
    """
    itp1, itp2 = _tf_interpreters()
    y_ref = native_dtln_skipping_block0(itp1, itp2, x)          # misma secuencia de bloques
    y_native = apply_dtln_post_tflite_realtime(itp1, itp2, x)   # lazo nativo completo

    results = {}
    for win_name in ('boxcar', 'hamming', 'hann'):
        y_coupled = _coupled_dtln_on_signal(x, win_name)
        results[win_name] = _snr_db(y_ref, y_coupled)
        print(f"[3/4] acoplado(win={win_name:8s}) vs DTLN nativo (misma secuencia de "
              f"bloques): SNR = {results[win_name]:6.1f} dB")

    print(f"[info] acoplado(rect) vs lazo nativo COMPLETO (el nativo procesa ademas "
          f"el bloque 0, que la STFT no genera): "
          f"SNR = {_snr_db(y_native, _coupled_dtln_on_signal(x, 'boxcar')):.1f} dB")

    # rect debe ser MUCHO mejor que cualquier ventana con taper
    ok = (results['boxcar'] > 20.0
          and results['boxcar'] - max(results['hamming'], results['hann']) > 15.0)
    return ok


def _tf_interpreters():
    import tensorflow as tf
    i1 = tf.lite.Interpreter(model_path=MODEL_1)
    i1.allocate_tensors()
    i2 = tf.lite.Interpreter(model_path=MODEL_2)
    i2.allocate_tensors()
    return i1, i2


def main():
    rng = np.random.default_rng(0)
    n = FS * 3
    t = np.arange(n) / FS
    # senal de prueba: tonos + ruido (no importa el contenido, si la mecanica)
    x = (0.5 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
         + 0.1 * rng.standard_normal(n))
    x = x.astype(np.float64)

    ok = True
    ok &= check_1_frame_alignment(x)
    ok &= check_2_reconstruction(x)
    ok &= check_3_and_4(x)

    print("\nRESULTADO:", "OK" if ok else "FALLO")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
