"""
La curva de trade-off del banco de calibracion, y el ajuste que DOMINA a NM_MVDR_SUB.

DE DONDE VIENE
--------------
El primer ajuste (eta=1) recupero SAR, STOI y SDR contra NM_MVDR_SUB pero PERDIO
~1.7 dB de SIR. Eso no es un error del ajuste: eta pondera la distorsion tan
fuerte como el SINR, y el optimizador -- que TIENE a (nu=1, gamma=0) == NM_MVDR_SUB
como punto de su propia grilla -- lo evaluo y lo descarto por puntaje. O sea que
"arrancar desde SUB" no cambiaria nada: hay que cambiar el CRITERIO.

Este script corre las dos formas de hacerlo, sobre los snapshots ya cacheados
por tests/scm_calibration_run.py (no recalcula escenas ni mascaras DTLN):

  1. BARRIDO DE eta. Ajusta la familia para varios eta y traza la curva de
     Pareto (L_sinr vs L_dist), con NM_MVDR y NM_MVDR_SUB dibujados como puntos
     en el MISMO plano. eta -> 0 empuja el ajuste hacia puro SINR; como la
     familia contiene a SUB, ahi tiene que alcanzarlo o superarlo. La curva es
     el resultado que sirve para la tesis: muestra que el compromiso es una
     ELECCION y donde cae cada core sobre ella.

  2. AJUSTE RESTRINGIDO. Minimiza L sujeto a  L_sinr <= L_sinr(SUB), banda por
     banda. Es la respuesta directa a "que no pierda SIR contra SUB": en vez de
     un compromiso, busca DOMINACION -- igualar el SINR de SUB y cobrar la
     mejora entera en distorsion. Si no existe tal punto en alguna banda, esa
     banda se queda con el mejor SINR disponible.

Cada ajuste se guarda como un .npz que tests/scm_calib_benchmark.py puede correr
directo con --calib, asi que el paso siguiente es medirlos en PESQ/STOI/SDR/SIR.

USO
---
    python tests/scm_calib_eta_sweep.py
    python tests/scm_calib_eta_sweep.py --eta-grid 0 0.25 0.5 1 2 4
    python tests/scm_calib_eta_sweep.py --split none      # ajustar con las 8 escenas

Salida: tests/dataset_out/scm_calib_eta/
    eta_sweep.csv          L_sinr / L_dist en train y test por eta y por modo
    eta_sweep_bands.csv    nu_k, gamma_k por banda para cada ajuste
    eta_pareto.png         la curva, con NM_MVDR y NM_MVDR_SUB ubicados
    params_eta<X>.npz          calibracion sin restriccion
    params_eta<X>_domsub.npz   calibracion restringida a no perder contra SUB
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from propagation.mird_loader import MirdDatasetProvider
from beamforming.mask.scm_calibration import (
    make_bands, fit_bands, band_objective, oracle_bound, bands_to_bin_params,
)
from scm_calibration_run import scene_grid, build_and_prepare, PROJECT_ROOT

OUT_DIR = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "scm_calib_eta")

SUB_POINT = (1.0, 0.0)      # NM_MVDR_SUB dentro de la familia
BASE_POINT = (0.0, 0.0)     # NM_MVDR


def evaluate(scenes, K, nu, gamma, mu, min_loading):
    """L_sinr / L_dist / L agregados sobre TODOS los bins y escenas."""
    return band_objective(scenes, np.arange(K), nu, gamma, mu=mu, eta=1.0,
                          min_loading=min_loading, detail=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eta-grid", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--sinr-tol-db", type=float, default=0.0,
                    help="holgura de la restriccion contra SUB (0 = dominacion estricta)")
    # --- escenas (mismos defaults que scm_calibration_run.py: usa su cache) ---
    ap.add_argument("--rt60", type=float, nargs="+", default=[0.360, 0.610])
    ap.add_argument("--interf-angles", type=float, nargs="+", default=[45, 90])
    ap.add_argument("--isir", type=float, nargs="+", default=[0, 10])
    ap.add_argument("--spacing", type=str, default="3-3-3-8-3-3-3")
    ap.add_argument("--target-angle", type=float, default=0)
    ap.add_argument("--target-dist", type=float, default=1.0)
    ap.add_argument("--interf-dist", type=float, default=1.0)
    ap.add_argument("--snr-db", type=float, default=30.0)
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--alpha", type=float, default=0.99)
    ap.add_argument("--sharpen-exp", type=float, default=4.0)
    ap.add_argument("--n-eval", type=int, default=16)
    ap.add_argument("--min-loading", type=float, default=1e-9)
    ap.add_argument("--snr-floor-db", type=float, default=-20.0)
    ap.add_argument("--field", type=str, default="spherical")
    ap.add_argument("--no-geometry", action="store_true")
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--n-bands", type=int, default=20)
    ap.add_argument("--f-min", type=float, default=60.0)
    ap.add_argument("--f-max", type=float, default=7000.0)
    ap.add_argument("--nu-grid", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    ap.add_argument("--gamma-grid", type=float, nargs="+",
                    default=[0.0, 0.15, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--split", type=str, default="rt60",
                    choices=["interleave", "rt60", "isir", "angle", "none"])
    ap.add_argument("--no-cache", dest="cache", action="store_false")
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cfg = {
        "fs": 16000, "duration": args.duration, "t_early": 0.050,
        "array_center": [3.0, 3.0, 1.2], "mird_spacing": args.spacing,
        "snr_db": args.snr_db,
        "source_path": f"{PROJECT_ROOT}/tools/data/signals/p002_emo_adoration_sentences.wav",
        "interf_paths": [f"{PROJECT_ROOT}/tools/data/signals/techno_gated commune.wav"],
        "stft_window": 512, "stft_overlap": 384,
        "dtln_model_path": f"{PROJECT_ROOT}/src/dnn_denoise/models/model_quant_1.tflite",
    }
    provider = MirdDatasetProvider(root_dir=f"{PROJECT_ROOT}/tools/data/rirs/mird")

    specs = scene_grid(args.rt60, args.interf_angles, args.isir)
    scenes = [build_and_prepare(s, cfg, provider, args) for s in specs]
    freqs = scenes[0]["freqs"]
    K = len(freqs)

    if args.split == "rt60":
        tr = [i for i, s in enumerate(specs) if s["rt60"] == args.rt60[0]]
    elif args.split == "isir":
        tr = [i for i, s in enumerate(specs) if s["isir"] == args.isir[0]]
    elif args.split == "angle":
        tr = [i for i, s in enumerate(specs) if s["interf_angle"] == args.interf_angles[0]]
    elif args.split == "interleave":
        tr = [i for i in range(len(specs)) if i % 2 == 0]
    else:
        tr = list(range(len(specs)))
    te = [i for i in range(len(specs)) if i not in tr] or tr
    train = [scenes[i] for i in tr]
    test = [scenes[i] for i in te]
    print(f"[*] split={args.split}  train={len(train)}  test={len(test)}")

    _, _, bands = make_bands(freqs, n_bands=args.n_bands, f_min=args.f_min,
                             f_max=args.f_max)
    kw = dict(min_loading=args.min_loading)

    # --- los dos cores existentes, como puntos fijos del plano ---------------
    fixed = {}
    for tag, (nu, gam) in (("NM_MVDR", BASE_POINT), ("NM_MVDR_SUB", SUB_POINT)):
        fixed[tag] = {
            "train": evaluate(train, K, nu, gam, args.mu, args.min_loading),
            "test": evaluate(test, K, nu, gam, args.mu, args.min_loading),
        }
    ob = oracle_bound(scenes, min_loading=args.min_loading)
    print(f"[control] oracle: L_sinr={ob['L_sinr']:.3f} dB  L_dist={ob['L_dist']:.3f} dB")
    for tag, v in fixed.items():
        print(f"  {tag:12s} test: L_sinr={v['test']['L_sinr']:6.2f}  "
              f"L_dist={v['test']['L_dist']:5.2f}")

    rows, band_rows = [], []
    for tag, v in fixed.items():
        for split_tag in ("train", "test"):
            rows.append({"mode": "core", "eta": np.nan, "name": tag,
                         "split": split_tag, **v[split_tag]})

    # --- barrido de eta, sin y con restriccion contra SUB --------------------
    for eta in args.eta_grid:
        for mode, ref in (("free", None), ("domsub", SUB_POINT)):
            print(f"\n[*] eta={eta:g}  modo={mode}")
            res = fit_bands(train, freqs, bands, args.nu_grid, args.gamma_grid,
                            mu=args.mu, eta=eta, refine=True, verbose=False,
                            ref_point=ref, sinr_tol_db=args.sinr_tol_db, **kw)
            nu_k, gam_k = bands_to_bin_params(res, K)
            name = f"eta{eta:g}" + ("_domsub" if mode == "domsub" else "")

            ev = {"train": evaluate(train, K, nu_k, gam_k, args.mu, args.min_loading),
                  "test": evaluate(test, K, nu_k, gam_k, args.mu, args.min_loading)}
            for split_tag in ("train", "test"):
                rows.append({"mode": mode, "eta": eta, "name": name,
                             "split": split_tag, **ev[split_tag]})

            t = ev["test"]
            s = fixed["NM_MVDR_SUB"]["test"]
            n_forced = sum(1 for r in res if ref is not None and not r["constrained"])
            print(f"    test: L_sinr={t['L_sinr']:6.2f} ({t['L_sinr']-s['L_sinr']:+.2f} vs SUB)"
                  f"   L_dist={t['L_dist']:5.2f} ({t['L_dist']-s['L_dist']:+.2f} vs SUB)"
                  + (f"   [{n_forced} bandas sin punto factible]" if n_forced else ""))

            for r in res:
                band_rows.append({"name": name, "mode": mode, "eta": eta,
                                  "band": r["band"], "f_lo": r["f_lo"],
                                  "f_hi": r["f_hi"], "nu": r["nu"],
                                  "gamma": r["gamma"],
                                  "constrained": r.get("constrained", False)})

            np.savez(os.path.join(args.out_dir, f"params_{name}.npz"),
                     freqs=freqs, nu_k=nu_k, gamma_k=gam_k, eta=eta, mu=args.mu,
                     alpha=args.alpha, field=args.field, mode=mode,
                     train=[s_["name"] for s_ in train],
                     test=[s_["name"] for s_ in test])

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "eta_sweep.csv"), index=False)
    pd.DataFrame(band_rows).to_csv(os.path.join(args.out_dir, "eta_sweep_bands.csv"),
                                   index=False)

    # --- la curva -------------------------------------------------------------
    te_df = df[df.split == "test"]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for mode, style, lab in (("free", "o-", "calibrado (libre)"),
                             ("domsub", "s--", "calibrado (no pierde SINR vs SUB)")):
        m = te_df[te_df["mode"] == mode].sort_values("eta")
        if len(m):
            ax.plot(m["L_dist"], m["L_sinr"], style, label=lab)
            for _, r in m.iterrows():
                ax.annotate(f"η={r['eta']:g}", (r["L_dist"], r["L_sinr"]),
                            fontsize=7, xytext=(4, 3), textcoords="offset points")
    for tag, mk, col in (("NM_MVDR", "X", "crimson"), ("NM_MVDR_SUB", "P", "darkorange")):
        p = fixed[tag]["test"]
        ax.plot(p["L_dist"], p["L_sinr"], mk, ms=13, color=col, label=tag)
    ax.plot(ob["L_dist"], ob["L_sinr"], "*", ms=16, color="green", label="oracle")
    ax.set_xlabel("L_dist  [dB]   (distorsion -> PESQ / SAR)")
    ax.set_ylabel("L_sinr  [dB]   (perdida de SINR -> SIR)")
    ax.set_title("Trade-off de la calibracion (test)\nabajo-izquierda = mejor")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "eta_pareto.png"), dpi=130)
    plt.close(fig)

    # --- resumen --------------------------------------------------------------
    print("\n" + "=" * 78)
    print("TEST -- L_sinr / L_dist [dB], menor es mejor")
    print("=" * 78)
    show = te_df[["name", "mode", "eta", "L_sinr", "L_dist", "L"]].round(3)
    print(show.to_string(index=False))

    s = fixed["NM_MVDR_SUB"]["test"]
    dom = te_df[(te_df["L_sinr"] <= s["L_sinr"] + 1e-9) &
                (te_df["L_dist"] <= s["L_dist"] + 1e-9) &
                (te_df["mode"] != "core")]
    print("\nAjustes que DOMINAN a NM_MVDR_SUB en test (mejor o igual en AMBOS terminos):")
    print("  " + (", ".join(dom["name"]) if len(dom) else "ninguno"))
    print(f"\n[ok] {args.out_dir}")


if __name__ == "__main__":
    main()
