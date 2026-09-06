"""
Barrido de iSIR (SNR objetivo target/ruido en el mic de referencia) sobre
SENALES REALES (senal12.wav / ruido12.wav), comparando NM_MVDR_DSM_BLIND (con
y sin post-filtro) contra los baselines que ya trae run_real_benchmark
(dtln_mono, dtln_souden_mvdr, dtln_souden_ban_mvdr, dtln_souden_ban_then_dtln)
y el mic crudo.

Para cada iSIR de la grilla se corre tests/dsm_blind_real_run.py por dentro
(reusa run_intrusive_benchmark: mezcla senal12+ruido12 a ese iSIR, corre todos
los procesadores, mide PESQ/STOI/SI-SDR/SDR/SAR contra el target limpio). Junta
todo en un CSV y arma:
    isir_sweep_metrics.csv    todas las celdas (isir_db, processor, metricas)
    global_bars.png           barras: promedio de cada metrica sobre TODOS los
                              iSIR, una barra por procesador
    isir_curves.png           curvas: cada metrica en funcion del iSIR, una
                              linea por procesador

Uso
---
    conda activate tesis_beam
    python tests/dsm_blind_isir_sweep.py
    python tests/dsm_blind_isir_sweep.py --isir -15 -10 -5 0 5 10
"""
import os
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation.intrusive_benchmark_real import (
    run_intrusive_benchmark, default_base_config, DTLN_MODEL_1, DTLN_MODEL_2,
)
from evaluation.bf_wrappers import NM_MVDR_DSM_BLIND, NM_MVDR_DSM_FB

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CAPTURE = "/home/matias/pdm_mic_interface/kria_app/capture/wavs_paso5"
DEFAULT_SENAL = os.path.join(CAPTURE, "senal12.wav")
DEFAULT_RUIDO = os.path.join(CAPTURE, "ruido12.wav")
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "tests", "real_benchmark_out", "dsm_blind_isir_sweep")

# Orden fijo para que barras/curvas salgan siempre en el mismo orden.
PROC_ORDER = ["ref_mic_raw", "dtln_mono", "dtln_souden_mvdr", "dtln_souden_ban_mvdr",
             "dtln_souden_ban_then_dtln", "NM_MVDR_DSM_BLIND", "NM_MVDR_DSM_BLIND_PF",
             "NM_MVDR_DSM_FB_8"]
PROC_LABELS = {"ref_mic_raw": "mic crudo", "dtln_mono": "DTLN mono",
              "dtln_souden_mvdr": "Souden (actual)", "dtln_souden_ban_mvdr": "Souden+BAN",
              "dtln_souden_ban_then_dtln": "BAN->DTLN",
              "NM_MVDR_DSM_BLIND": "DSM_BLIND", "NM_MVDR_DSM_BLIND_PF": "DSM_BLIND+PF",
              "NM_MVDR_DSM_FB_8": "DSM_FB (sharpen=8, PF=0.5)"}
METRIC_COLS = ["PESQ", "STOI", "SI-SDR", "SDR"]


def run_sweep(args):
    os.makedirs(args.out_dir, exist_ok=True)

    try:
        interp1 = tf.lite.Interpreter(model_path=DTLN_MODEL_1); interp1.allocate_tensors()
        interp2 = tf.lite.Interpreter(model_path=DTLN_MODEL_2); interp2.allocate_tensors()
        print("[*] Interpretes DTLN TF-Lite cargados.")
    except Exception as e:
        print(f"[!] Sin modelos DTLN (sigo sin las cascadas): {e}")
        interp1, interp2 = None, None

    extra_processors = {
        "NM_MVDR_DSM_BLIND": NM_MVDR_DSM_BLIND(min_loading=args.min_loading, alpha=args.alpha),
        "NM_MVDR_DSM_BLIND_PF": NM_MVDR_DSM_BLIND(min_loading=args.min_loading, alpha=args.alpha,
                                                   smooth=args.smooth),
        "NM_MVDR_DSM_FB_8": NM_MVDR_DSM_FB(mode="fb", win_type="rect", synth="hann",
                                           sharpen_exp=8.0, smooth=0.5, alpha=0.99),
    }

    rows_all = []
    for isir in args.isir:
        tag = f"isir{isir:g}"
        cell_out = os.path.join(args.out_dir, tag)
        print(f"\n[*] === iSIR {isir:+g} dB -> {cell_out} ===")
        base_config = default_base_config(fs=16000)
        res = run_intrusive_benchmark(
            senal_path=args.senal,
            ruido_path=args.ruido,
            output_dir=cell_out,
            base_config=base_config,
            interpreter_1=interp1,
            interpreter_2=interp2,
            snr=isir,
            ref_mic=args.ref_mic,
            eval_start_s=args.eval_start,
            use_wpe=False,
            extra_processors=extra_processors,
        )
        for name, metrics in res["rows"]:
            row = {"isir_db": isir, "processor": name}
            for c in METRIC_COLS + ["SAR"]:
                row[c] = metrics.get(c, np.nan)
            rows_all.append(row)

    df = pd.DataFrame(rows_all)
    csv_path = os.path.join(args.out_dir, "isir_sweep_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[*] CSV del barrido: {csv_path}")
    return df


def _order(names):
    return [p for p in PROC_ORDER if p in names] + [p for p in names if p not in PROC_ORDER]


def plot_global_bars(df, out_path):
    order = _order(df["processor"].unique())
    means = df.groupby("processor")[METRIC_COLS].mean().reindex(order)
    labels = [PROC_LABELS.get(p, p) for p in order]
    colors = plt.cm.tab10(np.linspace(0, 1, len(order)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
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
    fig.suptitle(f"Promedio global sobre iSIR = {sorted(df['isir_db'].unique())} dB "
                f"(senal12/ruido12, {len(df['isir_db'].unique())} puntos)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[*] {out_path}")


def plot_isir_curves(df, out_path):
    order = _order(df["processor"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(order)))
    isirs = sorted(df["isir_db"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric in zip(axes.flat, METRIC_COLS):
        for proc, color in zip(order, colors):
            sub = df[df["processor"] == proc].set_index("isir_db").reindex(isirs)
            ax.plot(isirs, sub[metric].values, marker="o", label=PROC_LABELS.get(proc, proc),
                    color=color)
        ax.set_title(metric)
        ax.set_xlabel("iSIR [dB]")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
    axes.flat[0].legend(loc="best", fontsize=8)
    fig.suptitle("Metricas intrusivas vs iSIR (senal12/ruido12, mic crudo real)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[*] {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--senal", default=DEFAULT_SENAL)
    ap.add_argument("--ruido", default=DEFAULT_RUIDO)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--isir", type=float, nargs="+", default=[-10, -5, 0, 5, 10])
    ap.add_argument("--ref-mic", type=int, default=None)
    ap.add_argument("--eval-start", type=float, default=5.0)
    ap.add_argument("--min-loading", type=float, default=1e-9)
    ap.add_argument("--alpha", type=float, default=0.99)
    ap.add_argument("--smooth", type=float, default=0.5)
    args = ap.parse_args()

    for path in (args.senal, args.ruido):
        if not os.path.isfile(path):
            raise SystemExit(f"[!] No existe el WAV: {path}")

    df = run_sweep(args)
    plot_global_bars(df, os.path.join(args.out_dir, "global_bars.png"))
    plot_isir_curves(df, os.path.join(args.out_dir, "isir_curves.png"))
    print(f"\n[ok] {args.out_dir}")


if __name__ == "__main__":
    main()
