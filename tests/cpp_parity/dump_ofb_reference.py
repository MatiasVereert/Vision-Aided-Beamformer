"""Genera la referencia de OfbMvdr: bloques de entrada (M,512) -> salida (512,),
y la salida en el tiempo de `ola_taper` para comparar contra process_hop.

Se configura para coincidir con cpp/ofb/config.hpp:
  nperseg=1      -> el `_mask` de ofb.py NO reescala, igual que el C++, que le
                    pasa el espectro crudo de kissfft (sin normalizar).
  min_loading=1e-5, solve_mode='chol'  -> la forma que porta el C++.
"""
import os, sys, numpy as np, scipy.io.wavfile as wav, scipy.signal as sig

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(R, "src"))
from beamforming.mask.ofb import OutputFeedbackMVDR            # noqa: E402
from evaluation.bf_wrappers import ola_taper                   # noqa: E402

K, M, REF, BL, HOP, FS, T = 257, 12, 6, 512, 128, 16000, 200
OUT = os.path.join(R, "tests", "cpp_parity", "data")
os.makedirs(OUT, exist_ok=True)

def load16k(p):
    fs, s = wav.read(p); s = s.astype(np.float64)
    if s.ndim > 1: s = s.mean(1)
    if fs != FS: s = sig.resample_poly(s, FS, fs)
    return s / (np.abs(s).max() + 1e-12)

rng = np.random.default_rng(7)
voz   = load16k(os.path.join(R, "tools/data/signals/FA01_09.wav"))
ruido = load16k(os.path.join(R, "tools/data/signals/ruido_rosa_16k_1.wav"))
L = min(len(voz), len(ruido), (T + 5) * HOP + BL)
# escena simple: el target llega con 1 muestra de retardo por mic (una DOA
# cualquiera) y cada mic ve ruido independiente.
mics = np.empty((M, L), dtype=np.float32)
for m in range(M):
    mics[m] = (np.roll(voz[:L], m) + 0.3 * np.roll(ruido[:L], 37 * m)).astype(np.float32)

freqs = np.arange(K) * FS / BL
proc = OutputFeedbackMVDR(K, M, REF, freqs,
                          os.path.join(R, "src/dnn_denoise/models/model_quant_1.tflite"),
                          nperseg=1, alpha=0.99, sharpen_exp=8.0, min_loading=1e-5,
                          block_update=1, smooth=0.2, solve_mode="chol")

bufs = np.zeros((M, BL), dtype=np.float32)
blocks, outs, isir, Ys = [], [], [], []
for t in range(T):
    bufs[:, :-HOP] = bufs[:, HOP:]
    bufs[:, -HOP:] = mics[:, t*HOP : t*HOP + HOP]
    X = np.fft.rfft(bufs, axis=1).T                      # (K, M), SIN normalizar
    Y = proc.step(X)
    blocks.append(bufs.copy())
    outs.append(np.fft.irfft(Y, n=BL).astype(np.float32))
    Ys.append(Y)
    isir.append(proc.isir_db)

np.stack(blocks).astype(np.float32).tofile(f"{OUT}/ofb_blocks.bin")   # (T, M, 512)
np.stack(outs).astype(np.float32).tofile(f"{OUT}/ofb_out.bin")        # (T, 512)
np.array(isir, dtype=np.float32).tofile(f"{OUT}/ofb_isir.bin")        # (T,)

# Streaming = ola_taper SIN el recorte de borde. ola_taper recorta BL//2 para
# deshacer el boundary='zeros' de scipy.signal.stft; aca no hay borde, asi que
# se le anteponen BL//2/HOP frames nulos y el recorte se come justo eso. El
# hop t de process_hop es entonces y[t*HOP : (t+1)*HOP]. Y / BL: ola_taper
# espera la escala de scipy (rect -> 1/nperseg) y la deshace adentro.
Y_stft = np.stack(Ys, axis=1) / BL                                   # (K, T)
pad = np.zeros((K, BL // 2 // HOP), dtype=Y_stft.dtype)
ola = ola_taper(np.concatenate([pad, Y_stft], axis=1), BL, HOP, "hann", T * HOP)
ola.astype(np.float32).tofile(f"{OUT}/ofb_ola.bin")                   # (T*HOP,)
print(f"T={T}  bloques (T,{M},{BL})  salida (T,{BL})")
print(f"  salida: rms={np.sqrt(np.mean(np.stack(outs)**2)):.5f}  pico={np.abs(np.stack(outs)).max():.5f}")
print(f"  ola: rms={np.sqrt(np.mean(ola**2)):.5f}  (T*HOP={T*HOP} muestras)")
print(f"  isir_db: {min(isir):.2f} .. {max(isir):.2f} dB")
