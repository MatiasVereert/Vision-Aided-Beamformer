"""
sweep_isnr.py
=============
Barrido de iSNR (12 mics): arma la mezcla senal+ruido a distintos SNR de entrada
(-5..10 dB, paso 1), corre todos los procesadores y calcula metricas INTRUSIVAS
(PESQ/STOI/SI-SDR/SDR vs referencia limpia) y NO-INTRUSIVAS (DNSMOS SIG/BAK/OVRL).
Guarda un CSV y un grafico metrica-vs-iSNR (una linea por procesador).

USO:
    conda activate tesis_beam
    python src/evaluation/sweep_isnr.py senal.wav ruido.wav [out_dir] \
           [--lo -5] [--hi 10] [--step 1] [--ref-mic 6] [--eval-start 5]
"""
import os, sys, argparse, contextlib
import numpy as np
import soundfile as sf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import scipy.signal as sig
import tensorflow as tf
from dnn_denoise.dtln_lite import apply_dtln_post_tflite_realtime
from evaluation.full_benchmark_real import (
    DTLN_Souden_MVDR_Processor, DTLN_Souden_BAN_MVDR_Processor,
    build_placeholder_geometry, energy_vad, DTLN_MODEL_1, DTLN_MODEL_2,
)
from beamforming.mask.dtln_masks import get_dtln_masks_sharpen
from beamforming.mask.souden_mvdr import (MVDR_Souden_recursive_mask_fixed,
                                            MVDR_Souden_recursive_mask_MWF,
                                            MVDR_Souden_recursive_mask_BAN_MWF,
                                            MVDR_Souden_recursive_mask_specsub)
from evaluation.metrics import evaluate_full_pipeline
from evaluation.nonintrusive import compute_nonintrusive

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@contextlib.contextmanager
def quiet():
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        yield


def run_all(mix_cf, cfg, i1, i2):
    """mix_cf: (M,N) channels-first. Devuelve dict nombre->senal 1D."""
    ref = mix_cf[cfg["ref_mic_idx"]]
    out = {"ref_mic_raw": ref}
    with quiet():
        out["dtln_mono"] = apply_dtln_post_tflite_realtime(i1, i2, ref)
        y_s, _ = DTLN_Souden_MVDR_Processor(sharpen_exp=4.0, alpha=0.99).process(mix_cf, cfg)
        out["dtln_souden_mvdr"] = y_s
        # variante FIXED (loading relativo + Hermitiana + solve) sobre la misma mascara/STFT
        ms, mn = get_dtln_masks_sharpen(mix_cf, cfg["ref_mic_idx"], cfg["dtln_model_path"],
                                        block_len=512, block_shift=128, sharpen_exp=4.0)
        _, _, Z = sig.stft(mix_cf, fs=cfg["fs"], window="hamming", nperseg=512, noverlap=384, nfft=512)
        Xh = np.transpose(Z, (1, 2, 0))
        mfr = min(Xh.shape[1], ms.shape[1]); Xh, ms, mn = Xh[:, :mfr], ms[:, :mfr], mn[:, :mfr]
        Yf = MVDR_Souden_recursive_mask_fixed(Xh, ms, mn, min_loading=1e-2, alpha=0.99)
        _, y_fx = sig.istft(Yf, fs=cfg["fs"], window="hamming", nperseg=512, noverlap=384, nfft=512)
        out["dtln_souden_mvdr_fixed"] = y_fx[:mix_cf.shape[1]]
        # MWF paramétrico (post-filtro Wiener) sobre el fixed, mu=4 (mas supresion)
        Ymwf = MVDR_Souden_recursive_mask_MWF(Xh, ms, mn, min_loading=1e-2, alpha=0.99, mu=4.0)
        _, y_mwf = sig.istft(Ymwf, fs=cfg["fs"], window="hamming", nperseg=512, noverlap=384, nfft=512)
        out["dtln_souden_mwf_mu4"] = y_mwf[:mix_cf.shape[1]]
        # Souden fixed + BAN + MWF(mu=4). OJO: BAN es invariante a escala -> anula mu
        # (== fixed+BAN); se incluye igual como punto nuevo (BAN con ref=M//2 + loading rel).
        Ybmwf = MVDR_Souden_recursive_mask_BAN_MWF(Xh, ms, mn, min_loading=1e-2, alpha=0.99, mu=4.0)
        _, y_bmwf = sig.istft(Ybmwf, fs=cfg["fs"], window="hamming", nperseg=512, noverlap=384, nfft=512)
        out["dtln_souden_ban_mwf_mu4"] = y_bmwf[:mix_cf.shape[1]]
        # Souden fixed + sustraccion espectral post-BF con la mascara ORIGINAL (sin
        # realce): recuperada como ms**(1/sharpen_exp). Suavizado smooth=0.33.
        ms_soft = ms ** (1.0 / cfg["souden_sharpen_exp"])
        Yss = MVDR_Souden_recursive_mask_specsub(Xh, ms, mn, ms_soft, min_loading=1e-2,
                                                 alpha=0.99, smooth=0.33)
        _, y_ss = sig.istft(Yss, fs=cfg["fs"], window="hamming", nperseg=512, noverlap=384, nfft=512)
        out["dtln_souden_specsub_s033"] = y_ss[:mix_cf.shape[1]]
        Yss5 = MVDR_Souden_recursive_mask_specsub(Xh, ms, mn, ms_soft, min_loading=1e-2,
                                                  alpha=0.99, smooth=0.5)
        _, y_ss5 = sig.istft(Yss5, fs=cfg["fs"], window="hamming", nperseg=512, noverlap=384, nfft=512)
        out["dtln_souden_specsub_s05"] = y_ss5[:mix_cf.shape[1]]
        y_ban, _ = DTLN_Souden_BAN_MVDR_Processor(sharpen_exp=4.0, alpha=0.99).process(mix_cf, cfg)
        out["dtln_souden_ban_mvdr"] = y_ban
        out["dtln_souden_ban_then_dtln"] = apply_dtln_post_tflite_realtime(i1, i2, y_ban)
        out["dtln_mvdr_then_dtln"] = apply_dtln_post_tflite_realtime(i1, i2, y_mvdr)
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("senal"); ap.add_argument("ruido")
    ap.add_argument("out_dir", nargs="?", default=os.path.join(REPO_ROOT, "tests", "souden_variants_out", "sweep_final"))
    ap.add_argument("--lo", type=float, default=-5.0)
    ap.add_argument("--hi", type=float, default=10.0)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--ref-mic", type=int, default=None)
    ap.add_argument("--eval-start", type=float, default=5.0)
    args = ap.parse_args()

    sig, fs = sf.read(args.senal, dtype="float64", always_2d=True)
    noi, fn = sf.read(args.ruido, dtype="float64", always_2d=True)
    assert fs == fn and sig.shape[1] == noi.shape[1]
    M = sig.shape[1]
    ref_mic = args.ref_mic if args.ref_mic is not None else M // 2
    N = min(len(sig), len(noi)); sig = sig[:N]; noi = noi[:N]
    ref_clean = sig[:, ref_mic].astype(np.float64)

    p_s = float(np.mean(sig[:, ref_mic] ** 2)) + 1e-20
    p_n = float(np.mean(noi[:, ref_mic] ** 2)) + 1e-20
    os.makedirs(args.out_dir, exist_ok=True)

    i1 = tf.lite.Interpreter(model_path=DTLN_MODEL_1); i1.allocate_tensors()
    i2 = tf.lite.Interpreter(model_path=DTLN_MODEL_2); i2.allocate_tensors()
    mic_coords, source_pos = build_placeholder_geometry(M=M, fs=fs)

    snrs = list(np.arange(args.lo, args.hi + 1e-9, args.step))
    intr_keys = ["PESQ", "STOI", "SI-SDR", "SDR"]
    ni_keys = ["DNSMOS_SIG", "DNSMOS_BAK", "DNSMOS_OVRL"]
    rows = []  # (snr, proc, metric, value)
    print(f"[*] M={M}, fs={fs}, {N/fs:.1f}s, ref=ch{ref_mic}. Barriendo iSNR {snrs[0]:.0f}..{snrs[-1]:.0f} dB "
          f"({len(snrs)} puntos). Puede tardar ~15-25 min.", flush=True)

    for si, snr in enumerate(snrs):
        g = float(np.sqrt(p_s / (p_n * 10.0 ** (snr / 10.0))))
        mix_cf = (sig + g * noi).T  # (M,N)
        cfg = {"fs": fs, "stft_window": 512, "stft_overlap": 384,
               "dtln_model_path": DTLN_MODEL_1, "per_channel_norm": False,
               "souden_sharpen_exp": 4.0, "souden_alpha": 0.99,
               "mic_coords": mic_coords, "source_pos": source_pos,
               "ref_mic_idx": ref_mic, "VAD": energy_vad(mix_cf[ref_mic], fs)}
        outs = run_all(mix_cf, cfg, i1, i2)
        for proc, y in outs.items():
            m = evaluate_full_pipeline(ref_clean, y, fs, eval_start_s=args.eval_start)
            ni = compute_nonintrusive(y, fs)
            for k in intr_keys:
                rows.append((snr, proc, k, m.get(k, np.nan)))
            for k in ni_keys:
                rows.append((snr, proc, k, ni.get(k, np.nan)))
        print(f"    [{si+1}/{len(snrs)}] iSNR={snr:+.0f} dB  ok", flush=True)

    # CSV (formato ancho: snr,proc,PESQ,STOI,SI-SDR,SDR,SIG,BAK,OVRL)
    procs = list(outs.keys())
    all_keys = intr_keys + ni_keys
    csv_path = os.path.join(args.out_dir, "sweep_metrics.csv")
    val = {(s, p, k): v for (s, p, k, v) in rows}
    with open(csv_path, "w") as f:
        f.write("iSNR_dB,procesador," + ",".join(all_keys) + "\n")
        for s in snrs:
            for p in procs:
                f.write(f"{s:.0f},{p}," + ",".join(f"{val.get((s,p,k),float('nan')):.4f}" for k in all_keys) + "\n")
    print(f"[*] CSV: {csv_path}")

    # grafico: 7 paneles (PESQ, STOI, SI-SDR, SDR, SIG, BAK, OVRL) vs iSNR
    panels = intr_keys + ni_keys
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    axes = axes.ravel()
    color = {p: c for p, c in zip(procs, plt.cm.tab10(np.linspace(0, 1, len(procs))))}
    for ax, key in zip(axes, panels):
        for p in procs:
            ys = [val.get((s, p, key), np.nan) for s in snrs]
            ax.plot(snrs, ys, marker="o", ms=3, lw=1.6, label=p, color=color[p])
        ax.set_title(key); ax.set_xlabel("iSNR entrada (dB)"); ax.grid(alpha=0.3)
    # leyenda unica en un panel libre
    for ax in axes[len(panels):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[len(panels)].legend(handles, labels, loc="center", fontsize=9, title="procesador")
    fig.suptitle("Metricas vs iSNR — 12 mics (intrusivas + DNSMOS)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png = os.path.join(args.out_dir, "metrics_vs_isnr.png")
    fig.savefig(png, dpi=130)
    print(f"[*] Grafico: {png}")


if __name__ == "__main__":
    main()
