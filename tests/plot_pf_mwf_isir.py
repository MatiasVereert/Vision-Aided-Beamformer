"""
Figura: metricas del post-filtro MWF en funcion del iSIR.

Lee tests/dataset_out/pf_mwf/all_rt_isir.csv (barrido RT60 x iSIR de
pf_mwf_rt_isir_sweep.py) y promedia sobre las 5 combinaciones array x RT60 para
dejar el iSIR como unico eje. Estilo alineado con visualization/figuras_tesis.ipynb
(paleta Okabe-Ito, trazo #333, sans humanista).
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CSV = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "pf_mwf", "all_rt_isir.csv")
OUT_PNG = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "pf_mwf", "pf_mwf_vs_isir.png")

OKABE_ITO = {
    "naranja": "#E69F00", "celeste": "#56B4E9", "verde": "#009E73",
    "azul": "#0072B2", "bermellon": "#D55E00", "purpura": "#CC79A7",
}
TRAZO = "#333333"
TEXTWIDTH_IN = 441.02 / 72.27

_SANS = ["Inter", "Lato", "Roboto", "Open Sans", "Liberation Sans", "DejaVu Sans"]
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": _SANS,
    "mathtext.fontset": "custom", "mathtext.rm": _SANS[0],
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": TRAZO, "axes.labelcolor": TRAZO, "text.color": TRAZO,
    "xtick.color": TRAZO, "ytick.color": TRAZO,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "grid.linestyle": "--", "grid.linewidth": 0.5, "grid.alpha": 0.35,
    "lines.linewidth": 1.3, "lines.markersize": 4.0, "figure.dpi": 120,
})

# (etiqueta, color, marker, ancho) -- el MWF elegido va destacado
SERIES = [
    ("NM-MVDR",            TRAZO,                  "o", 1.0, "--"),
    ("PF_050",             OKABE_ITO["azul"],      "s", 1.3, "-"),
    ("PF_033",             OKABE_ITO["celeste"],   "^", 1.3, "-"),
    ("PF050+MWF_g6_osf05", OKABE_ITO["naranja"],   "v", 1.3, "-"),
    ("PF050+MWF_g6_osf03", OKABE_ITO["bermellon"], "D", 2.0, "-"),
]

# (columna, etiqueta, agregacion sobre celdas). El SIR/SAR de BSS-Eval se agrega
# por MEDIANA, no por media: a RT60 bajo la separacion es casi perfecta y el SIR
# se dispara (valores enormes o inf->NaN), asi que la media de 5 celdas la domina
# una sola. Es la misma convencion que el resto del analisis de la tesis.
PANELS = [
    ("Delta_tot_PESQ_early",   r"$\Delta$PESQ",         "mean"),
    ("Delta_tot_STOI_early",   r"$\Delta$STOI",         "mean"),
    ("Delta_tot_SI-SDR_early", r"$\Delta$SI-SDR [dB]",  "mean"),
    ("Delta_tot_SIR_early",    r"$\Delta$SIR [dB]",     "median"),
]


def main():
    df = pd.read_csv(CSV)
    n_cells = df.groupby("processor")["isir_db"].count().iloc[0] // df["isir_db"].nunique()

    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH_IN, 4.9), layout="constrained")

    for ax, (col, ylabel, how) in zip(axes.ravel(), PANELS):
        # SOPORTE COMUN: si una celda quedo sin valor para algun procesador (el SIR
        # infinito a RT60 bajo se guarda como NaN), se descarta la celda ENTERA. Si
        # no, cada curva se promedia sobre un subconjunto distinto de escenas y el
        # panel compara cosas que no son comparables.
        sub = df[["array", "rt60", "isir_db", "processor", col]].copy()
        wide = sub.pivot_table(index=["array", "rt60", "isir_db"],
                               columns="processor", values=col)
        wide = wide.dropna(axis=0, how="any")
        piv = wide.groupby(level="isir_db").agg(how)
        n_used = len(wide) // max(wide.index.get_level_values("isir_db").nunique(), 1)
        if n_used < n_cells:
            ax.set_title(f"{n_used}/{n_cells} celdas con soporte comun", fontsize=7.5,
                         color=TRAZO, loc="right")
        for name, color, marker, lw, ls in SERIES:
            if name not in piv.columns:
                continue
            ax.plot(piv.index, piv[name], marker=marker, color=color, lw=lw, ls=ls,
                    label=name, zorder=3 if "MWF_g6_osf03" in name else 2)
        ax.set_xlabel("iSIR [dB]")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(df["isir_db"].unique()))
        ax.grid(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle(f"Post-filtro MWF vs iSIR  ·  agregado sobre {n_cells} celdas "
                 r"(array $\times$ RT$_{60}$); SIR por mediana", fontsize=9.5)

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"[ok] {OUT_PNG}")


if __name__ == "__main__":
    main()
