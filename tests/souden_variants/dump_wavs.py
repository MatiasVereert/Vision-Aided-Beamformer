"""
dump_wavs.py -- genera WAVs normalizados de las variantes Souden a UN iSNR, para
escuchar. Incluye input crudo (ch6), referencia limpia (ch6), fixed, mwf_mu4 y
specsub (smooth 0.33 y 0.5). Mascara original = ms**(1/sharpen).
USO: python src/evaluation/dump_wavs.py senal.wav ruido.wav out_dir [--snr 0]
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
from beamforming.mask.souden_mvdr import (MVDR_Souden_recursive_mask_fixed,
                                            MVDR_Souden_recursive_mask_MWF,
                                            MVDR_Souden_recursive_mask_specsub)
from evaluation.full_benchmark_real import DTLN_MODEL_1


@contextlib.contextmanager
def quiet():
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        yield


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("senal"); ap.add_argument("ruido"); ap.add_argument("out_dir")
    ap.add_argument("--snr", type=float, default=0.0)
    ap.add_argument("--sharpen", type=float, default=4.0)
    ap.add_argument("--ref-mic", type=int, default=None)
    args = ap.parse_args()

    s, fs = sf.read(args.senal, dtype="float64", always_2d=True)
    n, _ = sf.read(args.ruido, dtype="float64", always_2d=True)
    M = s.shape[1]; ref = args.ref_mic if args.ref_mic is not None else M // 2
    N = min(len(s), len(n)); s = s[:N]; n = n[:N]
    g = float(np.sqrt(np.mean(s[:, ref] ** 2) / (np.mean(n[:, ref] ** 2) * 10 ** (args.snr / 10))))
    mix_cf = (s + g * n).T
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[*] iSNR={args.snr:+.0f} dB (g={g:.4f}), ref=ch{ref}", flush=True)

    def istft(Y):
        _, y = sig.istft(Y, fs=fs, window="hamming", nperseg=512, noverlap=384, nfft=512)
        return np.asarray(y[:N], dtype=np.float64)

    with quiet():
        ms, mn = get_dtln_masks_sharpen(mix_cf, ref, DTLN_MODEL_1, block_len=512,
                                        block_shift=128, sharpen_exp=args.sharpen)
        _, _, Z = sig.stft(mix_cf, fs=fs, window="hamming", nperseg=512, noverlap=384, nfft=512)
        Xh = np.transpose(Z, (1, 2, 0)); mf = min(Xh.shape[1], ms.shape[1])
        Xh, ms, mn = Xh[:, :mf], ms[:, :mf], mn[:, :mf]
        ms_soft = np.clip(ms ** (1.0 / args.sharpen), 0.0, 1.0)
        outs = {
            "00_input_raw_ch6": mix_cf[ref],
            "01_ref_clean_ch6": s[:, ref],
            "02_souden_fixed": istft(MVDR_Souden_recursive_mask_fixed(Xh, ms, mn, min_loading=1e-2, alpha=0.99)),
            "03_souden_mwf_mu4": istft(MVDR_Souden_recursive_mask_MWF(Xh, ms, mn, min_loading=1e-2, alpha=0.99, mu=4.0)),
            "04_souden_specsub_s033": istft(MVDR_Souden_recursive_mask_specsub(Xh, ms, mn, ms_soft, min_loading=1e-2, alpha=0.99, smooth=0.33)),
            "05_souden_specsub_s05": istft(MVDR_Souden_recursive_mask_specsub(Xh, ms, mn, ms_soft, min_loading=1e-2, alpha=0.99, smooth=0.5)),
        }

    for name, y in outs.items():
        y = np.asarray(y, dtype=np.float64)
        yn = y / (np.max(np.abs(y)) + 1e-12)
        path = os.path.join(args.out_dir, f"{name}.wav")
        sf.write(path, yn.astype(np.float32), fs, subtype="FLOAT")
        print(f"    -> {path}")
    print(f"[*] {len(outs)} WAVs (normalizados) en {args.out_dir}/")


if __name__ == "__main__":
    main()
