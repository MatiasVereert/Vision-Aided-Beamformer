"""
mwf_sweep_isnr.py
=================
Barrido de iSNR aislando el efecto del POST-FILTRO DE WIENER (MWF paramétrico
PMWF-beta de Souden 2010) sobre el MVDR de Souden FIXED. Para cada iSNR calcula
UNA sola vez la máscara DTLN + STFT de la mezcla y corre el beamformer para
mu = 0 (== fixed / MVDR puro) y varios mu>0. Mucho más liviano que sweep_isnr.py
(no re-corre DTLN mono / MB-MVDR / BAN / cascadas).

Grafica métrica-vs-iSNR (PESQ/STOI/SI-SDR/SDR + DNSMOS SIG/BAK/OVRL), una línea
por mu, para ver el crossover: a iSNR bajo el post-filtro es mejora multi-métrica;
a iSNR alto es puro trade-off (sube PESQ, baja SI-SDR/SDR/DNSMOS).

USO:
    conda activate tesis_beam
    python src/evaluation/mwf_sweep_isnr.py senal.wav ruido.wav [out_dir] \
           [--lo -5] [--hi 10] [--step 1] [--mus 0 0.5 1 2 4] [--ref-mic 6]
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
from beamforming.mask.souden_mvdr import MVDR_Souden_recursive_mask_MWF
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
    ap.add_argument("out_dir", nargs="?", default=os.path.join(REPO_ROOT, "tests", "souden_variants_out", "mwf_mu_sweep"))
    ap.add_argument("--lo", type=float, default=-5.0)
    ap.add_argument("--hi", type=float, default=10.0)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--mus", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 4.0])
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
    procs = [f"mu={mu:g}" for mu in args.mus]
    val = {}  # (snr, proc, key) -> value
    print(f"[*] M={M}, fs={fs}, {N/fs:.1f}s, ref=ch{ref}. Barriendo iSNR "
          f"{snrs[0]:.0f}..{snrs[-1]:.0f} dB ({len(snrs)} pts) x {len(args.mus)} mu.", flush=True)

    for si, snr in enumerate(snrs):
        g = float(np.sqrt(p_s / (p_n * 10.0 ** (snr / 10.0))))
        mix_cf = (s + g * n).T  # (M,N)
        with quiet():
            ms, mn = get_dtln_masks_sharpen(mix_cf, ref, DTLN_MODEL_1,
                                            block_len=512, block_shift=128, sharpen_exp=4.0)
            _, _, Z = sig.stft(mix_cf, fs=fs, window="hamming", nperseg=512, noverlap=384, nfft=512)
            Xh = np.transpose(Z, (1, 2, 0))
            mfr = min(Xh.shape[1], ms.shape[1]); Xh, msc, mnc = Xh[:, :mfr], ms[:, :mfr], mn[:, :mfr]
            for mu, proc in zip(args.mus, procs):
                Y = MVDR_Souden_recursive_mask_MWF(Xh, msc, mnc, min_loading=1e-2, alpha=0.99, mu=mu)
                _, y = sig.istft(Y, fs=fs, window="hamming", nperseg=512, noverlap=384, nfft=512)
                y = np.asarray(y[:N], dtype=np.float64)
                m = evaluate_full_pipeline(ref_clean, y, fs, eval_start_s=args.eval_start)
                ni = compute_nonintrusive(y, fs)
                for k in intr_keys:
                    val[(snr, proc, k)] = m.get(k, np.nan)
                for k in ni_keys:
                    val[(snr, proc, k)] = ni.get(k, np.nan)
        print(f"    [{si+1}/{len(snrs)}] iSNR={snr:+.0f} dB  ok", flush=True)

    csv_path = os.path.join(args.out_dir, "mwf_sweep_metrics.csv")
    with open(csv_path, "w") as f:
        f.write("iSNR_dB,mu," + ",".join(all_keys) + "\n")
        for snr in snrs:
            for mu, proc in zip(args.mus, procs):
                f.write(f"{snr:.0f},{mu:g}," +
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
                             title="post-filtro Wiener\nmu (0=MVDR puro)")
    fig.suptitle("MWF paramétrico (post-filtro Wiener) vs iSNR — 12 mics", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png = os.path.join(args.out_dir, "mwf_metrics_vs_isnr.png")
    fig.savefig(png, dpi=130)
    print(f"[*] Grafico: {png}")


if __name__ == "__main__":
    main()
