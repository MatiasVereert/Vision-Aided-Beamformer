"""
test_souden_fixed.py -- compara la Souden ORIGINAL vs la variante FIXED a UN iSNR.
Mismo mask y misma referencia (ch6). Fixes evaluados por separado:
  - fixed        : loading relativo + Hermitiana + solve (misma STFT hamming)
  - fixed_rank1  : + Phi_XX rank-1 (autovector principal)
  - fixed_boxcar : fixed pero STFT boxcar (matchea la ventana de la mascara DTLN)
USO: python src/evaluation/test_souden_fixed.py senal.wav ruido.wav [--snr 0]
"""
import os, sys, argparse, contextlib
import numpy as np
import soundfile as sf
import scipy.signal as sig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from beamforming.mask.dtln_masks import get_dtln_masks_sharpen
from beamforming.mask.souden_mvdr import MVDR_Souden_recursive_mask
from beamforming.mask.souden_mvdr import MVDR_Souden_recursive_mask_fixed
from evaluation.full_benchmark_real import DTLN_MODEL_1
from evaluation.metrics import evaluate_full_pipeline
from evaluation.nonintrusive import compute_nonintrusive


@contextlib.contextmanager
def quiet():
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        yield


def stft_ch(x_cf, fs, win):
    _, _, Z = sig.stft(x_cf, fs=fs, window=win, nperseg=512, noverlap=384, nfft=512)
    return np.transpose(Z, (1, 2, 0))  # (K,T,M)


def istft_ch(Y, fs, win, n):
    _, y = sig.istft(Y, fs=fs, window=win, nperseg=512, noverlap=384, nfft=512)
    return np.asarray(y[:n], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("senal"); ap.add_argument("ruido")
    ap.add_argument("--snr", type=float, default=0.0)
    ap.add_argument("--ref-mic", type=int, default=None)
    ap.add_argument("--eval-start", type=float, default=5.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    s, fs = sf.read(args.senal, dtype="float64", always_2d=True)
    n, _ = sf.read(args.ruido, dtype="float64", always_2d=True)
    M = s.shape[1]
    ref = args.ref_mic if args.ref_mic is not None else M // 2
    N = min(len(s), len(n)); s = s[:N]; n = n[:N]
    ps = float(np.mean(s[:, ref] ** 2)); pn = float(np.mean(n[:, ref] ** 2))
    g = float(np.sqrt(ps / (pn * 10 ** (args.snr / 10))))
    mix_cf = (s + g * n).T                       # (M,N)
    ref_clean = s[:, ref].astype(np.float64)
    print(f"[*] M={M} fs={fs} {N/fs:.1f}s ref=ch{ref}  iSNR={args.snr:+.0f} dB (g={g:.4f})", flush=True)

    print("[*] mascara DTLN (sharpen=4)...", flush=True)
    with quiet():
        ms, mn = get_dtln_masks_sharpen(mix_cf, ref, DTLN_MODEL_1, block_len=512, block_shift=128, sharpen_exp=4.0)

    Xh = stft_ch(mix_cf, fs, "hamming")
    Xb = stft_ch(mix_cf, fs, "boxcar")
    mf = min(Xh.shape[1], Xb.shape[1], ms.shape[1])
    Xh, Xb, ms, mn = Xh[:, :mf], Xb[:, :mf], ms[:, :mf], mn[:, :mf]

    outs = {"ref_mic_raw": mix_cf[ref]}
    print("[*] souden ORIGINAL...", flush=True)
    with quiet():
        Y = MVDR_Souden_recursive_mask(Xh, ms, mn, min_loading=1e-6, alpha=0.99)
    outs["souden_orig"] = istft_ch(Y, fs, "hamming", N)
    print("[*] souden FIXED (loading rel + herm + solve)...", flush=True)
    with quiet():
        Y = MVDR_Souden_recursive_mask_fixed(Xh, ms, mn, min_loading=1e-2, alpha=0.99)
    outs["souden_fixed"] = istft_ch(Y, fs, "hamming", N)
    print("[*] souden FIXED + rank1...", flush=True)
    with quiet():
        Y = MVDR_Souden_recursive_mask_fixed(Xh, ms, mn, min_loading=1e-2, alpha=0.99, rank1=True)
    outs["souden_fixed_rank1"] = istft_ch(Y, fs, "hamming", N)
    print("[*] souden FIXED + boxcar (match ventana mascara)...", flush=True)
    with quiet():
        Y = MVDR_Souden_recursive_mask_fixed(Xb, ms, mn, min_loading=1e-2, alpha=0.99)
    outs["souden_fixed_boxcar"] = istft_ch(Y, fs, "boxcar", N)

    print(f"\n=== Souden ORIGINAL vs FIXED @ iSNR={args.snr:+.0f} dB (vs referencia limpia ch{ref}) ===")
    cols = ["PESQ", "STOI", "SI-SDR", "SDR", "SIG", "BAK", "OVRL"]
    print(f"{'proc':<22} " + " ".join(f"{c:>7}" for c in cols))
    print("-" * (22 + 8 * len(cols)))
    for name, y in outs.items():
        m = evaluate_full_pipeline(ref_clean, y, fs, eval_start_s=args.eval_start)
        ni = compute_nonintrusive(y, fs)
        vals = [m.get("PESQ"), m.get("STOI"), m.get("SI-SDR"), m.get("SDR"),
                ni.get("DNSMOS_SIG"), ni.get("DNSMOS_BAK"), ni.get("DNSMOS_OVRL")]
        print(f"{name:<22} " + " ".join(f"{(v if v is not None else float('nan')):>7.3f}" for v in vals))

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        for name, y in outs.items():
            yn = y / (np.max(np.abs(y)) + 1e-12)
            sf.write(os.path.join(args.out, f"{name}.wav"), yn.astype(np.float32), fs, subtype="FLOAT")
        print(f"[*] WAVs en {args.out}/")


if __name__ == "__main__":
    main()
