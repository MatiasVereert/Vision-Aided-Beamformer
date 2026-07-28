"""
boxplot_improvement.py
======================
Boxplots de MEJORA (delta = metrica_out - metrica_in) por procesador, sobre todos
los puntos de iSNR de un CSV de sweep_isnr.py. El "input" es ref_mic_raw a cada
iSNR. Muestra PESQ y STOI. Una caja por procesador (distribucion sobre los iSNR).

USO:
    python src/evaluation/boxplot_improvement.py <sweep_metrics.csv> [out_png] \
           [--metrics PESQ STOI] [--input-proc ref_mic_raw]
"""
import os, sys, argparse, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("out_png", nargs="?", default=None)
    ap.add_argument("--metrics", nargs="+", default=["PESQ", "STOI"])
    ap.add_argument("--input-proc", default="ref_mic_raw")
    args = ap.parse_args()

    # leer: val[(snr, proc, metric)] = value
    val = {}; snrs = []; procs = []
    with open(args.csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            snr = float(row["iSNR_dB"]); proc = row["procesador"]
            if snr not in snrs: snrs.append(snr)
            if proc not in procs: procs.append(proc)
            for k, v in row.items():
                if k in ("iSNR_dB", "procesador"): continue
                try: val[(snr, proc, k)] = float(v)
                except (ValueError, TypeError): val[(snr, proc, k)] = np.nan

    procs_eval = [p for p in procs if p != args.input_proc]

    # delta[proc][metric] = lista sobre iSNR de (proc - input)
    delta = {p: {m: [] for m in args.metrics} for p in procs_eval}
    for m in args.metrics:
        for snr in snrs:
            base = val.get((snr, args.input_proc, m), np.nan)
            for p in procs_eval:
                delta[p][m].append(val.get((snr, p, m), np.nan) - base)

    # ordenar por mediana de mejora de la 1ra metrica (mejor arriba)
    order = sorted(procs_eval, key=lambda p: np.nanmedian(delta[p][args.metrics[0]]))

    fig, axes = plt.subplots(1, len(args.metrics), figsize=(7.5 * len(args.metrics), 8), sharey=True)
    if len(args.metrics) == 1: axes = [axes]
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(order)))
    for ax, m in zip(axes, args.metrics):
        data = [np.array(delta[p][m], dtype=float) for p in order]
        data = [d[~np.isnan(d)] for d in data]
        bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.6,
                        medianprops=dict(color="black", lw=1.6),
                        flierprops=dict(marker="o", ms=3, alpha=0.5))
        for patch, c in zip(bp["boxes"], cmap):
            patch.set_facecolor(c); patch.set_alpha(0.85)
        ax.axvline(0, color="crimson", ls="--", lw=1.2, alpha=0.8)
        ax.set_yticks(range(1, len(order) + 1)); ax.set_yticklabels(order, fontsize=9)
        ax.set_xlabel(f"Mejora  Δ{m}  ({m}_out - {m}_in)")
        ax.set_title(f"{m}: mejora vs {args.input_proc}")
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle(f"Mejora por procesador sobre iSNR {int(min(snrs))}..{int(max(snrs))} dB "
                 f"(n={len(snrs)} puntos)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = args.out_png or os.path.join(os.path.dirname(args.csv_path), "boxplot_improvement.png")
    fig.savefig(out, dpi=130)
    print(f"[*] Boxplot: {out}")

    # tabla resumen (mediana de mejora)
    print(f"\n{'procesador':<26} " + " ".join(f"med d{m:>6}" for m in args.metrics))
    for p in reversed(order):
        print(f"{p:<26} " + " ".join(f"{np.nanmedian(delta[p][m]):>11.3f}" for m in args.metrics))


if __name__ == "__main__":
    main()
