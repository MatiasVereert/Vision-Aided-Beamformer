"""Fase 2: referencia del NUCLEO con el lazo ABIERTO.
Vuelca X, m_s, m_n y los pesos resueltos de cada frame, para que el C++ replique
SoudenCore con exactamente las mismas entradas y la red salga de la ecuacion.
"""
import os, sys, numpy as np, scipy.io.wavfile as wav, scipy.signal as sig
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(R, "src"))
from beamforming.mask.ofb import OutputFeedbackMVDR            # noqa: E402

K, M, REF, BL, HOP, FS, T = 257, 12, 6, 512, 128, 16000, 200
OUT = os.path.join(R, "tests", "cpp_parity", "data")

def load16k(p):
    fs, s = wav.read(p); s = s.astype(np.float64)
    if s.ndim > 1: s = s.mean(1)
    if fs != FS: s = sig.resample_poly(s, FS, fs)
    return s / (np.abs(s).max() + 1e-12)

voz   = load16k(os.path.join(R, "tools/data/signals/FA01_09.wav"))
ruido = load16k(os.path.join(R, "tools/data/signals/ruido_rosa_16k_1.wav"))
L = min(len(voz), len(ruido), (T + 5) * HOP + BL)
mics = np.empty((M, L), dtype=np.float32)
for m in range(M):
    mics[m] = (np.roll(voz[:L], m) + 0.3 * np.roll(ruido[:L], 37 * m)).astype(np.float32)

freqs = np.arange(K) * FS / BL
proc = OutputFeedbackMVDR(K, M, REF, freqs,
                          os.path.join(R, "src/dnn_denoise/models/model_quant_1.tflite"),
                          nperseg=1, alpha=0.99, sharpen_exp=8.0, min_loading=1e-5,
                          block_update=1, smooth=0.2, solve_mode="chol")

bufs = np.zeros((M, BL), dtype=np.float32)
Xs, Ms, Ns, Ws = [], [], [], []
for t in range(T):
    bufs[:, :-HOP] = bufs[:, HOP:]
    bufs[:, -HOP:] = mics[:, t*HOP : t*HOP + HOP]
    X = np.fft.rfft(bufs, axis=1).T                 # (K, M)
    proc.step(X)
    m_scm = 0.5 * (proc.m_out + proc.m_ref)
    Xs.append(X.astype(np.complex64))
    Ms.append((m_scm ** 8.0).astype(np.float32))
    Ns.append(((1.0 - m_scm) ** 8.0).astype(np.float32))
    Ws.append(proc.weights.astype(np.complex64))    # el solve de ESTE frame

np.stack(Xs).tofile(f"{OUT}/core_X.bin")      # (T,K,M) complex64
np.stack(Ms).tofile(f"{OUT}/core_ms.bin")     # (T,K)   float32
np.stack(Ns).tofile(f"{OUT}/core_mn.bin")     # (T,K)   float32
np.stack(Ws).tofile(f"{OUT}/core_W.bin")      # (T,K,M) complex64
W = np.stack(Ws)
print(f"T={T}  X (T,{K},{M})  W (T,{K},{M})")
print(f"  |W|: min={np.abs(W).min():.3e} max={np.abs(W).max():.3e}")
