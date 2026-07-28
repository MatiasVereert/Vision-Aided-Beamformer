"""
specsub_sweep_smooth.py
=======================
Barrido del FACTOR DE SUAVIZADO `smooth` del post-filtro de SUSTRACCION ESPECTRAL
(mask-based, mascara ORIGINAL sin realce) sobre el MVDR de Souden FIXED, en funcion
del iSNR. Para cada iSNR corre el beamformer UNA sola vez (Y_bf) y luego solo re-
aplica la ganancia G = smooth + (1-smooth)*mask_orig por cada smooth (es lo unico
que cambia). Mucho mas liviano que re-correr el beamformer.

smooth: 1.0 = sin filtro (== fixed) ; 0.0 = mascara suave pura (sustraccion dura) ;
0.33 = default. Mascara original recuperada exacta como ms**(1/sharpen_exp).

Grafica metrica-vs-iSNR (PESQ/STOI/SI-SDR/SDR + DNSMOS SIG/BAK/OVRL), una linea por
smooth, para ver el trade-off: bajar smooth sube PESQ/BAK pero cuesta DNSMOS_SIG
(distorsion de voz) y SI-SDR.

USO:
    conda activate tesis_beam
    python src/evaluation/specsub_sweep_smooth.py senal.wav ruido.wav [out_dir] \
           [--lo -5] [--hi 10] [--step 1] [--smooths 0 0.15 0.33 0.5 0.66 1] \
           [--sharpen 4] [--ref-mic 6]
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
from beamforming.mask.souden_mvdr import MVDR_Souden_recursive_mask_fixed
from evaluation.full_benchmark_real import DTLN_MODEL_1
from evaluation.metrics import evaluate_full_pipeline
from evaluation.nonintrusive import compute_nonintrusive

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@contextlib.contextmanager
def quiet():
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        yield


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("senal"); ap.add_argument("ruido")
    ap.add_argument("out_dir", nargs="?", default=os.path.join(REPO_ROOT, "tests", "souden_variants_out", "specsub_smooth_sweep"))
    ap.add_argument("--lo", type=float, default=-5.0)
    ap.add_argument("--hi", type=float, default=10.0)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--smooths", type=float, nargs="+", default=[0.0, 0.15, 0.33, 0.5, 0.66, 1.0])
    ap.add_argument("--sharpen", type=float, default=4.0)
    ap.add_argument("--ref-mic", type=int, default=None)
    ap.add_argument("--eval-start", type=float, default=5.0)
    args = ap.parse_args()

    s, fs = sf.read(args.senal, dtype="float64", always_2d=True)
    n, fn = sf.read(args.ruido, dtype="float64", always_2d=True)
    assert fs == fn and s.shape[1] == n.shape[1]
    M = s.shape[1]
    ref = args.ref_mic if args.ref_mic is not None else M // 2
    N = min(len(s), len(n)); s = s[:N]; n = n[:N]
    ref_clean = s[:, ref].astype(np.float64)
    p_s = float(np.mean(s[:, ref] ** 2)) + 1e-20
    p_n = float(np.mean(n[:, ref] ** 2)) + 1e-20
    os.makedirs(args.out_dir, exist_ok=True)

    snrs = list(np.arange(args.lo, args.hi + 1e-9, args.step))
    intr_keys = ["PESQ", "STOI", "SI-SDR", "SDR"]
    ni_keys = ["DNSMOS_SIG", "DNSMOS_BAK", "DNSMOS_OVRL"]
    all_keys = intr_keys + ni_keys
    procs = [f"smooth={sm:g}" for sm in args.smooths]
    val = {}
    print(f"[*] M={M}, fs={fs}, {N/fs:.1f}s, ref=ch{ref}. Barriendo iSNR "
          f"{snrs[0]:.0f}..{snrs[-1]:.0f} dB ({len(snrs)} pts) x {len(args.smooths)} smooth "
          f"(sharpen={args.sharpen}).", flush=True)

    for si, snr in enumerate(snrs):
        g = float(np.sqrt(p_s / (p_n * 10.0 ** (snr / 10.0))))
        mix_cf = (s + g * n).T  # (M,N)
        with quiet():
            ms, mn = get_dtln_masks_sharpen(mix_cf, ref, DTLN_MODEL_1,
                                            block_len=512, block_shift=128, sharpen_exp=args.sharpen)
            _, _, Z = sig.stft(mix_cf, fs=fs, window="hamming", nperseg=512, noverlap=384, nfft=512)
            Xh = np.transpose(Z, (1, 2, 0))
            mfr = min(Xh.shape[1], ms.shape[1]); Xh, msc, mnc = Xh[:, :mfr], ms[:, :mfr], mn[:, :mfr]
            ms_soft = np.clip(msc ** (1.0 / args.sharpen), 0.0, 1.0)  # mascara ORIGINAL sin realce
            Y_bf = MVDR_Souden_recursive_mask_fixed(Xh, msc, mnc, min_loading=1e-2, alpha=0.99)
            for sm, proc in zip(args.smooths, procs):
                G = sm + (1.0 - sm) * ms_soft
                Y = Y_bf.copy(); Y *= G
                _, y = sig.istft(Y, fs=fs, window="hamming", nperseg=512, noverlap=384, nfft=512)
                y = np.asarray(y[:N], dtype=np.float64)
                m = evaluate_full_pipeline(ref_clean, y, fs, eval_start_s=args.eval_start)
                ni = compute_nonintrusive(y, fs)
                for k in intr_keys:
                    val[(snr, proc, k)] = m.get(k, np.nan)
                for k in ni_keys:
                    val[(snr, proc, k)] = ni.get(k, np.nan)
        print(f"    [{si+1}/{len(snrs)}] iSNR={snr:+.0f} dB  ok", flush=True)

    csv_path = os.path.join(args.out_dir, "specsub_smooth_metrics.csv")
    with open(csv_path, "w") as f:
        f.write("iSNR_dB,smooth," + ",".join(all_keys) + "\n")
        for snr in snrs:
            for sm, proc in zip(args.smooths, procs):
                f.write(f"{snr:.0f},{sm:g}," +
                        ",".join(f"{val.get((snr,proc,k),float('nan')):.4f}" for k in all_keys) + "\n")
    print(f"[*] CSV: {csv_path}")

    panels = all_keys
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    axes = axes.ravel()
    color = {p: c for p, c in zip(procs, plt.cm.viridis(np.linspace(0, 0.9, len(procs))))}
    for ax, key in zip(axes, panels):
        for p in procs:
            ys = [val.get((snr, p, key), np.nan) for snr in snrs]
            ax.plot(snrs, ys, marker="o", ms=3, lw=1.6, label=p, color=color[p])
        ax.set_title(key); ax.set_xlabel("iSNR entrada (dB)"); ax.grid(alpha=0.3)
    for ax in axes[len(panels):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[len(panels)].legend(handles, labels, loc="center", fontsize=10,
                             title="sustraccion espectral\nsmooth (1=sin filtro)")
    fig.suptitle("Barrido factor de suavizado (sustraccion espectral) vs iSNR — 12 mics", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png = os.path.join(args.out_dir, "specsub_smooth_vs_isnr.png")
    fig.savefig(png, dpi=130)
    print(f"[*] Grafico: {png}")


if __name__ == "__main__":
    main()
