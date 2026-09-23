"""
aro12_plot_results.py
=====================
Graficos de la corrida de `aro12_ofb_auto_real.py` sobre la captura del aro de
12 mics. Lee lo que el benchmark ya dejo en --out-dir (diagnostics_real.csv +
los WAV de cada salida + input_trimmed.wav) y genera tres PNG:

  1. metrics_dotplot.png   metricas no intrusivas, una panel por metrica, con la
                           linea del MIC CRUDO como referencia -> se lee la
                           MEJORA sobre la entrada, que es lo unico que importa
                           sin referencia limpia.
  2. spectrograms_all.png  espectrogramas de las 6 senales en una grilla, misma
                           escala de dB.
  3. channel_levels.png    RMS/pico por canal de los 12 mics del aro (chequeo de
                           hardware: canales flojos o mal cableados).

OJO: los WAV de salida estan normalizados a pico por `save_wav_normalized`, asi
que los niveles ABSOLUTOS no se comparan entre archivos. Lo que si se compara es
la estructura interna de cada uno (piso de ruido RELATIVO a la voz), que es
justo lo que muestran los espectrogramas.

Uso
---
    conda activate tesis_beam
    python tests/aro12_plot_results.py
    python tests/aro12_plot_results.py --out-dir <dir de la corrida>
"""
import os
import argparse

import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal as sig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "tests", "real_benchmark_out", "aro12_ofb_auto")

# Paleta validada (dataviz): azul = slot 1, naranja = slot 2, gris = neutro.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e6e5e1"
BLUE = "#2a78d6"      # el resto de los procesadores
ORANGE = "#eb6834"    # el procesador bajo prueba
GRAY = "#8a8985"      # el mic crudo (la referencia, no es un procesador)

UNDER_TEST = "NM_MVDR_OFB_AUTO"
REFERENCE = "ref_mic_raw"

# Orden FIJO (crudo arriba, despues por familia). El color sigue a la entidad,
# nunca al ranking: reordenar esta lista no repinta nada.
ORDER = [REFERENCE, "dtln_mono", "dtln_souden_mvdr", "dtln_souden_ban_mvdr",
         "dtln_souden_ban_then_dtln", UNDER_TEST]

LABELS = {
    REFERENCE: "mic crudo (ch6)",
    "dtln_mono": "DTLN mono",
    "dtln_souden_mvdr": "Souden MVDR",
    "dtln_souden_ban_mvdr": "Souden + BAN",
    "dtln_souden_ban_then_dtln": "BAN -> DTLN",
    UNDER_TEST: "NM_MVDR_OFB_AUTO",
}

# (columna, titulo, subtitulo con el rango real de la metrica)
METRICS = [
    ("DNSMOS_SIG", "SIG · calidad de la voz", "MOS 1-5, mas alto mejor"),
    ("DNSMOS_BAK", "BAK · supresion del fondo", "MOS 1-5, mas alto mejor"),
    ("DNSMOS_OVRL", "OVRL · calidad global", "MOS 1-5, mas alto mejor"),
    ("DNSMOS_P808", "P.808 · opinion global", "MOS 1-5, mas alto mejor"),
    ("segSNR_est_db", "segSNR estimado", "dB, mas alto mejor"),
]


def color_for(name):
    if name == UNDER_TEST:
        return ORANGE
    if name == REFERENCE:
        return GRAY
    return BLUE


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_metrics(df, out_png):
    """
    Dot plot por metrica. Dot plot y NO barras a proposito: el MOS del DNSMOS no
    tiene cero (arranca en 1), y una barra desde un cero falso exagera
    diferencias. La linea vertical punteada es el mic crudo = el punto de
    partida; lo que se lee es cuanto se corrio cada procesador respecto de el.
    """
    rows = [r for r in ORDER if r in set(df["senal"])]
    y = np.arange(len(rows))[::-1]

    fig = plt.figure(figsize=(13, 6.2), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 6, hspace=0.75, wspace=1.5,
                          left=0.14, right=0.975, top=0.80, bottom=0.10)
    axes = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]),
            fig.add_subplot(gs[0, 4:6]), fig.add_subplot(gs[1, 1:3]),
            fig.add_subplot(gs[1, 3:5])]

    for ax, (col, title, sub) in zip(axes, METRICS):
        style_axes(ax)
        vals = [float(df.loc[df["senal"] == r, col].iloc[0]) for r in rows]
        ref_val = float(df.loc[df["senal"] == REFERENCE, col].iloc[0])

        ax.axvline(ref_val, color=GRAY, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
        for yi, (r, v) in zip(y, zip(rows, vals)):
            c = color_for(r)
            # Tallo desde la referencia: la LONGITUD es la mejora, el punto el valor.
            ax.plot([ref_val, v], [yi, yi], color=c, linewidth=2, alpha=0.35, zorder=2,
                    solid_capstyle="round")
            ax.plot([v], [yi], marker="o", markersize=9, color=c, zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=2)
            ax.annotate(f"{v:.2f}", (v, yi), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8, color=INK, zorder=4)

        lo, hi = min(vals + [ref_val]), max(vals + [ref_val])
        pad = max(0.12, (hi - lo) * 0.28)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.set_yticks(y)
        ax.set_yticklabels([LABELS.get(r, r) for r in rows], fontsize=8.5, color=INK)
        for lbl, r in zip(ax.get_yticklabels(), rows):
            if r == UNDER_TEST:
                lbl.set_color(INK); lbl.set_fontweight("bold")
            elif r == REFERENCE:
                lbl.set_color(INK_MUTED)
        ax.set_title(title, fontsize=10, color=INK, fontweight="bold", loc="left", pad=16)
        ax.annotate(sub, xy=(0, 1.0), xycoords="axes fraction",
                    xytext=(0, 6), textcoords="offset points",
                    fontsize=8, color=INK_2, ha="left")

    # Ocultar las etiquetas de fila repetidas (solo la 1a columna de cada fila).
    for ax in (axes[1], axes[2], axes[4]):
        ax.set_yticklabels([])

    fig.suptitle("Aro de 12 mics · metricas SIN referencia (relativas al mic crudo)",
                 fontsize=13, color=INK, fontweight="bold", x=0.014, ha="left", y=0.965)
    fig.text(0.014, 0.905,
             "Linea punteada = mic crudo. Naranja = procesador bajo prueba. "
             "Sin referencia limpia estas metricas son cualitativas: DNSMOS premia el fondo callado, no la fidelidad.",
             fontsize=8.5, color=INK_2, ha="left")
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"[*] {out_png}")


def plot_spectrograms(out_dir, out_png, fmax=8000):
    """Grilla de espectrogramas, misma escala de dB (80 dB de rango)."""
    names = [r for r in ORDER if os.path.isfile(os.path.join(out_dir, f"{r}.wav"))]
    specs = []
    for n in names:
        x, fs = sf.read(os.path.join(out_dir, f"{n}.wav"), dtype="float64", always_2d=True)
        f, t, Z = sig.spectrogram(x[:, 0], fs=fs, nperseg=512, noverlap=384,
                                  window="hann", scaling="spectrum", mode="magnitude")
        specs.append((n, f, t, 20 * np.log10(Z + 1e-12)))

    vmax = max(s[3].max() for s in specs)
    vmin = vmax - 80

    fig, axs = plt.subplots(2, 3, figsize=(14, 6.6), facecolor=SURFACE, sharex=True, sharey=True)
    for ax, (n, f, t, S) in zip(axs.ravel(), specs):
        band = f <= fmax
        im = ax.pcolormesh(t, f[band] / 1000.0, S[band], vmin=vmin, vmax=vmax,
                           cmap="magma", shading="auto", rasterized=True)
        ax.set_facecolor(SURFACE)
        color = ORANGE if n == UNDER_TEST else (INK_MUTED if n == REFERENCE else INK)
        ax.set_title(LABELS.get(n, n), fontsize=10, color=color,
                     fontweight="bold" if n in (UNDER_TEST, REFERENCE) else "normal",
                     loc="left", pad=6)
        ax.tick_params(colors=INK_2, labelsize=8, length=0)
        for s in ax.spines.values():
            s.set_color(GRID)
    for ax in axs[1, :]:
        ax.set_xlabel("tiempo (s)", fontsize=8.5, color=INK_2)
    for ax in axs[:, 0]:
        ax.set_ylabel("kHz", fontsize=8.5, color=INK_2)

    fig.subplots_adjust(left=0.05, right=0.90, top=0.845, bottom=0.09, hspace=0.30, wspace=0.10)
    cax = fig.add_axes([0.915, 0.09, 0.013, 0.755])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("dB (rel. al maximo de la grilla)", fontsize=8.5, color=INK_2)
    cb.ax.tick_params(colors=INK_2, labelsize=8, length=0)
    cb.outline.set_edgecolor(GRID)

    fig.suptitle("Aro de 12 mics · espectrogramas de cada salida",
                 fontsize=13, color=INK, fontweight="bold", x=0.014, ha="left", y=0.965)
    fig.text(0.014, 0.905,
             "Cada WAV esta normalizado a pico: compara la estructura interna (piso de ruido vs voz), no el nivel absoluto.",
             fontsize=8.5, color=INK_2, ha="left")
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"[*] {out_png}")


def plot_channel_levels(in_wav, out_png):
    """RMS y pico por canal del aro: el chequeo de que los 12 esten parejos."""
    x, fs = sf.read(in_wav, dtype="float64", always_2d=True)
    M = x.shape[1]
    rms = np.array([20 * np.log10(np.sqrt(np.mean(x[:, m] ** 2)) + 1e-12) for m in range(M)])
    pk = np.array([20 * np.log10(np.max(np.abs(x[:, m])) + 1e-12) for m in range(M)])
    mean_rms = rms.mean()

    fig, ax = plt.subplots(figsize=(11, 4.2), facecolor=SURFACE)
    style_axes(ax)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    xs = np.arange(M)

    # Banda de +-3 dB alrededor de la media = la tolerancia de sensibilidad del mic.
    ax.axhspan(mean_rms - 3, mean_rms + 3, color=BLUE, alpha=0.07, zorder=0)
    ax.axhline(mean_rms, color=GRAY, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)

    out_of_band = np.abs(rms - mean_rms) > 3
    for m in xs:
        c = ORANGE if out_of_band[m] else BLUE
        ax.plot([m, m], [rms[m], pk[m]], color=c, linewidth=2, alpha=0.30,
                zorder=2, solid_capstyle="round")
        ax.plot([m], [pk[m]], marker="o", markersize=7, color=c, alpha=0.45, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.plot([m], [rms[m]], marker="o", markersize=10, color=c, zorder=4,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"{rms[m]:.1f}", (m, rms[m]), textcoords="offset points",
                    xytext=(0, -15), ha="center", fontsize=7.5,
                    color=INK if out_of_band[m] else INK_2)

    ax.annotate("pico", (xs[-1], pk[-1]), textcoords="offset points", xytext=(12, 0),
                fontsize=8, color=INK_2, va="center")
    ax.annotate("RMS", (xs[-1], rms[-1]), textcoords="offset points", xytext=(12, -9),
                fontsize=8, color=INK_2, va="center")
    ax.annotate(f"media +-3 dB\n(tolerancia del mic)", (0.0, mean_rms + 3),
                textcoords="offset points", xytext=(4, 4), fontsize=8, color=INK_2)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"ch{m}" for m in xs], fontsize=8.5, color=INK)
    for lbl, bad in zip(ax.get_xticklabels(), out_of_band):
        if bad:
            lbl.set_color(ORANGE); lbl.set_fontweight("bold")
    ax.set_xlim(-0.6, M - 0.1)
    # Aire abajo para que las etiquetas de valor no se coman el eje de canales.
    ax.set_ylim(rms.min() - 3.0, pk.max() + 1.5)
    ax.set_ylabel("dBFS", fontsize=8.5, color=INK_2)
    fig.subplots_adjust(left=0.07, right=0.94, top=0.72, bottom=0.14)

    fig.suptitle("Aro de 12 mics · nivel por canal en la toma procesada",
                 fontsize=13, color=INK, fontweight="bold", x=0.014, ha="left", y=0.965)
    fig.text(0.014, 0.855,
             f"Dispersion de RMS: {rms.max() - rms.min():.1f} dB. En naranja, los canales que se van de +-3 dB "
             f"respecto de la media.\nEn un aro eso puede ser GEOMETRIA (lado opuesto a la fuente) y no un canal "
             f"flojo: se distingue girando la fuente.",
             fontsize=8.5, color=INK_2, ha="left", va="top", linespacing=1.5)
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"[*] {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=DEFAULT_OUT,
                    help="directorio de la corrida de aro12_ofb_auto_real.py")
    args = ap.parse_args()

    csv_path = os.path.join(args.out_dir, "diagnostics_real.csv")
    if not os.path.isfile(csv_path):
        raise SystemExit(f"[!] No existe {csv_path}. Corre primero tests/aro12_ofb_auto_real.py")
    df = pd.read_csv(csv_path)

    plot_metrics(df, os.path.join(args.out_dir, "metrics_dotplot.png"))
    plot_spectrograms(args.out_dir, os.path.join(args.out_dir, "spectrograms_all.png"))
    trimmed = os.path.join(args.out_dir, "input_trimmed.wav")
    if os.path.isfile(trimmed):
        plot_channel_levels(trimmed, os.path.join(args.out_dir, "channel_levels.png"))
    else:
        print(f"[!] Sin {trimmed}: salteo el grafico de niveles por canal.")


if __name__ == "__main__":
    main()
