"""
La curva de trade-off de la calibracion de MASCARA: barrido de eta y de a_n.

DE DONDE VIENE
--------------
La mascara calibrada (a_s=1, a_n~2, b_n=-8) gana ~1.1 dB de SDR y ~1.3 dB de SAR
sobre NM_MVDR_SUB pero PIERDE ~0.7 dB de SIR. Eso NO es "otro algoritmo": corre
sobre el MISMO beamformer (nu=1 == el core de sustraccion); lo unico que cambia
es la estimacion de Phi_NN. Una Phi_NN mejor estimada hace que el MVDR se acerque
a su solucion distortionless real: menos deformacion de la voz, nulos menos
agresivos contra el interferente.

La perdida de SIR es entonces consecuencia del CRITERIO, no del metodo:

    L = L_sinr + eta * L_dist

y se ajusto con eta = 1, que pondera la distorsion tanto como el SINR. Este
script mide si eta permite elegir el punto de operacion -- a diferencia de la
etapa 1 (post-hoc), donde el paisaje era plano y eta no movia nada.

DOS BARRIDOS
------------
  1. a_n CONSTANTE. La variante recomendada tiene UN solo grado de libertad
     (a_n, con a_s=1 y b_n=-8 fijos), asi que barrerlo traza la curva de Pareto
     DIRECTAMENTE, sin optimizar nada. Es el resultado mas contable: una sola
     constante recorre todo el compromiso SIR <-> distorsion.
  2. AJUSTE POR BANDA a varios eta. Ubica donde caen las soluciones ajustadas
     sobre esa misma curva, y si eta efectivamente las mueve.

Los dos se evaluan en TEST reportando L_sinr y L_dist POR SEPARADO, que son
cantidades independientes de eta -- comparar valores de L entre distintos eta no
tendria sentido.

USO
---
    python tests/scm_mask_eta_sweep.py                    # a_n + eta {0, 0.25, 4}
    python tests/scm_mask_eta_sweep.py --no-fit           # solo el barrido de a_n (rapido)
    python tests/scm_mask_eta_sweep.py --eta-grid 0 0.5 2

Salida: tests/dataset_out/scm_mask_eta/
    an_sweep.csv     L_sinr / L_dist por a_n (la curva de Pareto)
    eta_fits.csv     idem para cada ajuste por banda
    mask_pareto.png  la curva, con la mascara actual y los ajustes ubicados
    params_eta<X>.npz
"""

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from propagation.mird_loader import MirdDatasetProvider
from beamforming.mask.scm_calibration import (
    make_bands, band_objective_mask, fit_band_mask, bands_to_bin_params,
)
from scm_calibration_run import scene_grid, PROJECT_ROOT
from scm_mask_calibration_run import prepare_scene_full

OUT_DIR = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "scm_mask_eta")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--an-grid", type=float, nargs="+",
                    default=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
    ap.add_argument("--eta-grid", type=float, nargs="+", default=[0.0, 0.25, 4.0])
    ap.add_argument("--b-n", type=float, default=-8.0)
    ap.add_argument("--no-fit", action="store_true",
                    help="omite los ajustes por banda (solo la curva de a_n)")
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
    ap.add_argument("--agg", type=str, default="wmedian")
    ap.add_argument("--nu", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--n-bands", type=int, default=12)
    ap.add_argument("--maxiter", type=int, default=150)
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
    args._provider = MirdDatasetProvider(root_dir=f"{PROJECT_ROOT}/tools/data/rirs/mird")

    specs = scene_grid(args.rt60, args.interf_angles, args.isir)
    scenes = [prepare_scene_full(s, cfg, args) for s in specs]
    train = [sc for sc, sp in zip(scenes, specs) if sp["rt60"] == args.rt60[0]]
    test = [sc for sc, sp in zip(scenes, specs) if sp["rt60"] != args.rt60[0]] or train
    freqs = scenes[0]["freqs"]
    K = len(freqs)
    allb = np.arange(K)
    print(f"[*] train={len(train)} escenas  test={len(test)} escenas (rt60 no visto)")

    ev = dict(nu=args.nu, gamma=args.gamma, mu=args.mu, min_loading=args.min_loading,
              how=args.agg)

    rows = []

    # --- referencias: la mascara ACTUAL sobre el mismo beamformer -------------
    for tag, nu_ in (("mascara actual + SUB (nu=1)", args.nu),
                     ("mascara actual + base (nu=0)", 0.0)):
        d = band_objective_mask(test, allb, None, eta=1.0, detail=True,
                                masks=[(sc["mask_cur"][0], sc["mask_cur"][1]) for sc in test],
                                **{**ev, "nu": nu_})
        rows.append({"kind": "ref", "name": tag, "a_n": np.nan, "eta": np.nan,
                     "L_sinr": d["L_sinr"], "L_dist": d["L_dist"]})
        print(f"  {tag:32s} L_sinr={d['L_sinr']:6.3f}  L_dist={d['L_dist']:6.3f}")

    # --- barrido de a_n: la curva de Pareto con UN solo parametro -------------
    print(f"\n[*] barrido de a_n (a_s=1, b_s=0, b_n={args.b_n:g})")
    t0 = time.time()
    for an in args.an_grid:
        d = band_objective_mask(test, allb, (1.0, 0.0, an, args.b_n), eta=1.0,
                                detail=True, **ev)
        rows.append({"kind": "a_n", "name": f"a_n={an:g}", "a_n": an, "eta": np.nan,
                     "L_sinr": d["L_sinr"], "L_dist": d["L_dist"]})
        print(f"    a_n={an:4.2f}  L_sinr={d['L_sinr']:6.3f}  L_dist={d['L_dist']:6.3f}")
    print(f"    ({time.time()-t0:.0f}s)")

    # --- ajustes por banda a distintos eta ------------------------------------
    if not args.no_fit:
        _, _, bands = make_bands(freqs, n_bands=args.n_bands, f_min=60.0, f_max=7000.0)
        bands = [b for b in bands if b.size]
        for eta in args.eta_grid:
            print(f"\n[*] ajuste por banda con eta={eta:g}")
            t0 = time.time()
            th_bins = {k: np.zeros(K) for k in ("a_s", "b_s", "a_n", "b_n")}
            th_bins["a_s"] += 1.0; th_bins["a_n"] += 1.0
            for bi in bands:
                r = fit_band_mask(train, bi, eta=eta, maxiter=args.maxiter, **ev)
                for k, v in zip(("a_s", "b_s", "a_n", "b_n"), r["theta"]):
                    th_bins[k][bi] = v
            d = band_objective_mask(test, allb,
                                    (th_bins["a_s"], th_bins["b_s"],
                                     th_bins["a_n"], th_bins["b_n"]),
                                    eta=1.0, detail=True, **ev)
            rows.append({"kind": "fit", "name": f"fit eta={eta:g}", "a_n": np.nan,
                         "eta": eta, "L_sinr": d["L_sinr"], "L_dist": d["L_dist"]})
            print(f"    L_sinr={d['L_sinr']:6.3f}  L_dist={d['L_dist']:6.3f}  "
                  f"(a_n mediana {np.median(th_bins['a_n']):.2f}, {time.time()-t0:.0f}s)")
            np.savez(os.path.join(args.out_dir, f"params_eta{eta:g}.npz"),
                     freqs=freqs, eta=eta, nu=args.nu, gamma=args.gamma, **th_bins)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "an_sweep.csv"), index=False)

    # --- curva ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 6))
    m = df[df.kind == "a_n"].sort_values("a_n")
    ax.plot(m["L_dist"], m["L_sinr"], "o-", label="a_n constante (barrido)")
    for _, r in m.iterrows():
        ax.annotate(f"{r['a_n']:g}", (r["L_dist"], r["L_sinr"]), fontsize=7,
                    xytext=(4, 3), textcoords="offset points")
    for _, r in df[df.kind == "fit"].iterrows():
        ax.plot(r["L_dist"], r["L_sinr"], "^", ms=11, label=r["name"])
    for _, r in df[df.kind == "ref"].iterrows():
        ax.plot(r["L_dist"], r["L_sinr"], "X", ms=13, label=r["name"])
    ax.set_xlabel("L_dist [dB]   (distorsion -> SAR / PESQ)")
    ax.set_ylabel("L_sinr [dB]   (perdida de SINR -> SIR)")
    ax.set_title("Trade-off de la calibracion de mascara (test)\nabajo-izquierda = mejor")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "mask_pareto.png"), dpi=130)
    plt.close(fig)

    print("\n" + "=" * 70)
    print(df[["kind", "name", "L_sinr", "L_dist"]].round(3).to_string(index=False))
    print(f"\n[ok] {args.out_dir}")


if __name__ == "__main__":
    main()
