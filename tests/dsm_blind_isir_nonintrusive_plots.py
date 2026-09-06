"""
Version NO INTRUSIVA del barrido de iSIR (tests/dsm_blind_isir_sweep.py).

No hace falta correr nada de nuevo: cada celda isir<N>/ que genero ese barrido
ya tiene diagnostics_real.csv, que es la salida del benchmark NO intrusivo del
repo (src/evaluation/full_benchmark_real.py -> run_real_benchmark), corrido
puertas adentro de run_intrusive_benchmark sobre la MISMA mezcla real. Esto solo
junta esos CSV y arma las mismas barras/curvas que la version intrusiva, pero
con las metricas SIN referencia (DNSMOS + segSNR estimado; SQUIM da NaN en este
entorno, se ignora).

Salida (en el mismo out-dir del barrido):
    nonintrusive_sweep_metrics.csv
    global_bars_nonintrusive.png
    isir_curves_nonintrusive.png

Uso
---
    python tests/dsm_blind_isir_nonintrusive_plots.py
"""
import os
import re
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dsm_blind_isir_sweep import PROC_ORDER, PROC_LABELS, DEFAULT_OUT, _order

METRIC_COLS = ["DNSMOS_SIG", "DNSMOS_BAK", "DNSMOS_OVRL", "DNSMOS_P808", "segSNR_est_db"]


def load_sweep(sweep_dir):
    rows = []
    for name in sorted(os.listdir(sweep_dir)):
        m = re.fullmatch(r"isir(-?\d+(?:\.\d+)?)", name)
        if not m:
            continue
        csv_path = os.path.join(sweep_dir, name, "diagnostics_real.csv")
        if not os.path.isfile(csv_path):
            continue
        isir = float(m.group(1))
        cell = pd.read_csv(csv_path)
        cell["isir_db"] = isir
        rows.append(cell)
    if not rows:
        raise SystemExit(f"[!] No encontre ninguna celda isir<N>/diagnostics_real.csv en {sweep_dir}")
    return pd.concat(rows, ignore_index=True).rename(columns={"senal": "processor"})


def plot_global_bars(df, out_path):
    order = _order(df["processor"].unique())
    means = df.groupby("processor")[METRIC_COLS].mean().reindex(order)
    labels = [PROC_LABELS.get(p, p) for p in order]
    colors = plt.cm.tab10(np.linspace(0, 1, len(order)))

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, metric in zip(axes.flat, METRIC_COLS):
        vals = means[metric].values
        bars = ax.bar(labels, vals, color=colors)
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                            textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    for ax in axes.flat[len(METRIC_COLS):]:
        ax.axis("off")
    fig.suptitle(f"Promedio global SIN referencia sobre iSIR = {sorted(df['isir_db'].unique())} dB "
                f"(senal12/ruido12, {len(df['isir_db'].unique())} puntos)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[*] {out_path}")


def plot_isir_curves(df, out_path):
    order = _order(df["processor"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(order)))
    isirs = sorted(df["isir_db"].unique())

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, metric in zip(axes.flat, METRIC_COLS):
        for proc, color in zip(order, colors):
            sub = df[df["processor"] == proc].set_index("isir_db").reindex(isirs)
            ax.plot(isirs, sub[metric].values, marker="o", label=PROC_LABELS.get(proc, proc),
                    color=color)
        ax.set_title(metric)
        ax.set_xlabel("iSIR [dB]")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(METRIC_COLS):]:
        ax.axis("off")
    axes.flat[0].legend(loc="best", fontsize=8)
    fig.suptitle("Metricas SIN referencia (DNSMOS/segSNR) vs iSIR (senal12/ruido12)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[*] {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", default=DEFAULT_OUT,
                    help="carpeta del barrido ya corrido (con subcarpetas isir<N>/)")
    args = ap.parse_args()

    df = load_sweep(args.sweep_dir)
    csv_path = os.path.join(args.sweep_dir, "nonintrusive_sweep_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"[*] CSV: {csv_path}")

    plot_global_bars(df, os.path.join(args.sweep_dir, "global_bars_nonintrusive.png"))
    plot_isir_curves(df, os.path.join(args.sweep_dir, "isir_curves_nonintrusive.png"))
    print(f"\n[ok] {args.sweep_dir}")


if __name__ == "__main__":
    main()
