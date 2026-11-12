"""
DTLN int8 vs float con METRICAS NO INTRUSIVAS (DNSMOS P.835 + SQUIM).

Complemento de tests/dtln_quant_vs_float_check.py. Ese test mide con las
metricas intrusivas del motor (PESQ-WB, STOI, BSS) y da que des-cuantizar
mueve ~+0.01 PESQ. Pero el artefacto que se escucha en hardware con la etapa 2
int8 puede no estar pesado por PESQ: aca se puntuan los MISMOS wavs con los
predictores sin referencia de evaluation/nonintrusive.py, que son los del
benchmark real (full_benchmark_real.py).

No re-simula nada: lee <dir>/audio/ que deja `dtln_quant_vs_float_check.py
--audio` y las celdas de <dir>/dtln_quant_vs_float.csv (el motor numera los
wavs en el mismo orden que las filas de cada procesador).

Cada senal se recorta desde --start (default 5 s, la misma ventana que ignoran
las metricas intrusivas del motor) y se normaliza a -26 dBFS RMS antes de
puntuar: los wavs estan a la escala fisica del simulador y DNSMOS depende del
nivel, asi que sin esto la comparacion mezclaria ganancia con calidad.

SQUIM necesita torchaudio; si no esta, esas columnas quedan en NaN.

USO
---
    conda activate tesis_beam
    python tests/dtln_quant_vs_float_check.py --audio        # una vez
    python tests/dtln_quant_vs_float_nonintrusive.py
"""

import os
import glob
import argparse

import pandas as pd
import soundfile as sf

from evaluation.nonintrusive import score_nonintrusive, NONINTRUSIVE_KEYS, TARGET_DBFS

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RUN_DIR = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "dtln_quant_vs_float")

ORDER = ["input_ref", "NM-MVDR[Q]", "NM-MVDR[F]",
         "mono[QQ]", "mono[FF]", "mono[FQ]", "mono[QF]"]


def score(path, start_s):
    # Mismo preproceso que el motor con nonintrusive=True (recorte + -26 dBFS).
    x, fs = sf.read(path, dtype="float64")
    return score_nonintrusive(x, fs, start_s=start_s)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=str, default=RUN_DIR)
    ap.add_argument("--start", type=float, default=5.0,
                    help="segundos iniciales que no se puntuan (default 5)")
    args = ap.parse_args()

    audio = os.path.join(args.run_dir, "audio")
    cells = pd.read_csv(os.path.join(args.run_dir, "dtln_quant_vs_float.csv"))
    cells = cells[cells["processor"] == "mono[QQ]"].reset_index(drop=True)
    cells = cells[["rt60", "source", "N_interferences", "isir_db"]]
    cells["source"] = cells["source"].str.slice(0, 4)

    rows = []
    for f in sorted(glob.glob(os.path.join(audio, "c*_input_ref.wav"))):
        tag = os.path.basename(f).replace("_input_ref.wav", "")
        c = int(tag[1:3])
        for name in ORDER:
            print(f"\r[*] {tag} {name:12s}", end="", flush=True)
            r = score(os.path.join(audio, f"{tag}_{name}.wav"), args.start)
            rows.append(dict(cells.iloc[c], cell=c, processor=name, **r))
    print()
    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.run_dir, "nonintrusive.csv")
    df.to_csv(out_csv, index=False)
    summarize(df)
    print(f"\n[ok] {out_csv}")


def summarize(df):
    keys = [k for k in NONINTRUSIVE_KEYS if k in df.columns and df[k].notna().any()]
    by = {p: df[df["processor"] == p].set_index("cell")[keys] for p in ORDER}

    print("\n" + "=" * 78)
    print(f"MEDIANA ABSOLUTA sobre {len(by['input_ref'])} celdas "
          f"(desde {TARGET_DBFS:.0f} dBFS RMS)")
    print("=" * 78)
    print(pd.DataFrame({p: by[p].median() for p in ORDER}).T.round(3).to_string())

    def paired(a, b):
        d = by[a] - by[b]
        return "  ".join(f"{k.replace('DNSMOS_', '').replace('SQUIM_', 'sq')}:"
                         f"{d[k].median():+.3f}({int((d[k] > 0).sum())}/{len(d)})"
                         for k in keys)

    print("\n--- Efecto de des-cuantizar (F - Q, pareado): mediana (celdas donde F gana) ---")
    for lbl, a, b in (("NM-MVDR", "NM-MVDR[F]", "NM-MVDR[Q]"),
                      ("mono FF-QQ", "mono[FF]", "mono[QQ]"),
                      ("  solo etapa1 (FQ-QQ)", "mono[FQ]", "mono[QQ]"),
                      ("  solo etapa2 (QF-QQ)", "mono[QF]", "mono[QQ]")):
        print(f"  {lbl:22s} {paired(a, b)}")

    print("\n--- Brecha NM-MVDR - mono: mediana (celdas donde gana el sistema) ---")
    for lbl, a, b in (("Fase 2 vieja (Q vs QQ)", "NM-MVDR[Q]", "mono[QQ]"),
                      ("Fase 2 nueva (F vs FF)", "NM-MVDR[F]", "mono[FF]")):
        print(f"  {lbl:22s} {paired(a, b)}")

    if "DNSMOS_OVRL" in keys:
        print("\n--- DNSMOS OVRL por celda ---")
        t = df.pivot_table(index=["cell", "rt60", "source", "N_interferences", "isir_db"],
                           columns="processor", values="DNSMOS_OVRL")[ORDER]
        print(t.round(2).to_string())


if __name__ == "__main__":
    main()
