"""
Banco de CALIBRACION de la transformacion mascara -> covarianza, sobre MIRD.

PREGUNTA
--------
La estimacion de las SCM a partir de la mascara del DTLN esta hecha de
decisiones arbitrarias. Con las SCM oracle como referencia, ¿cuanto del techo
que separa al sistema del oracle se recupera CALIBRANDO esa transformacion, sin
tocar la red?

QUE AJUSTA
----------
Dos parametros por banda de frecuencia, que actuan DESPUES de la acumulacion
recursiva (ver `beamforming/mask/scm_calibration.py` para el desarrollo):

    gamma_k : shrinkage de Phi_NN hacia la coherencia difusa Gamma(k), que sale
              SOLO de la geometria del arreglo (Ledoit-Wolf con un target
              fisicamente correcto en lugar de la identidad).
    nu_k    : escala de la sustraccion, Phi_SS = Phi_XX - nu_k Phi_NN. Corrige
              que Phi_XX (condicionada a frames de voz) y Phi_NN (condicionada a
              frames de ruido) NO esten en la misma escala.

La parametrizacion CONTIENE a los dos cores actuales:
    (nu=0, gamma=0) == NM_MVDR        (core base)
    (nu=1, gamma=0) == NM_MVDR_SUB    (sustraccion, mu=0)
asi que el reporte dice directamente cuanto gana el ajuste sobre cada uno.

FUNCION DE COSTO
----------------
    L = 10log10(SINR_max/SINR(w)) + eta * 10log10(1 + |w^H a - 1|^2)     [dB]

medida contra las SCM oracle del MISMO (bin, frame). NO es una distancia entre
matrices: Souden es invariante a la escala global de ambas covarianzas, asi que
una loss de Frobenius gastaria parametros en algo que el filtro no ve. Los dos
terminos se mapean sobre el benchmark: L_sinr <-> SIR, L_dist <-> PESQ. Barrer
eta traza la curva de trade-off.

QUE MIRAR EN LA SALIDA
----------------------
  1. `oracle bound`: L_sinr del beamformer calculado con las SCM oracle. TIENE
     que dar ~0 dB. Si no, la loss o las referencias estan mal y todo lo demas
     es basura.
  2. `gain_vs_base` por banda: cuanto recupera la calibracion sobre el sistema
     actual, en dB de SINR, y en que bandas.
  3. train vs test: una calibracion fija que solo funciona en las escenas de
     ajuste no sirve. El split cruza RT60 / iSIR.
  4. (con --per-scene) el gap entre el ajuste GLOBAL y el ajuste POR ESCENA
     (sobreajustado a proposito). Si son parecidos, la familia parametrica esta
     SATURADA: el gap que queda contra el oracle es la mascara, no la
     calibracion, y no se cierra por esta via.

USO
---
    python tests/scm_calibration_run.py                   # grilla default (8 escenas)
    python tests/scm_calibration_run.py --eta 0.5 --n-bands 16
    python tests/scm_calibration_run.py --per-scene       # test de saturacion
    python tests/scm_calibration_run.py --no-cache        # recalcula todo

Salida: tests/dataset_out/scm_calib/
    scm_calib_bands.csv    optimo, grilla y baselines por banda (train y test)
    scm_calib_scenes.csv   loss por escena de base / sub / fit / oracle
    scm_calib_landscape.csv  el paisaje L(nu, gamma) por banda
    scm_calib_params.npz   nu_k, gamma_k por BIN, listos para enchufar en un core
    scm_calib.png          panel de 4 graficos
"""

import os
import sys
import argparse
import itertools

import numpy as np
import pandas as pd
import scipy.signal as sig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from propagation.mird_loader import MirdDatasetProvider
from beamforming.mask.dtln_masks import get_dtln_masks_sharpen
from beamforming.mask.scm_calibration import (
    prepare_scene, make_bands, fit_bands, fit_band, band_objective,
    oracle_bound, bands_to_bin_params, diffuse_coherence,
)
from lowfreq_diagnostic_run import build_scene          # misma construccion de escena

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT_DIR = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "scm_calib")
CACHE_DIR = os.path.join(OUT_DIR, "cache")


# =====================================================================
# Escenas
# =====================================================================
def scene_grid(rt60s, angles, isirs):
    """Producto cartesiano de los ejes; cada escena es un dict + un tag legible."""
    out = []
    for rt60, ang, isir in itertools.product(rt60s, angles, isirs):
        out.append({"rt60": float(rt60), "interf_angle": float(ang), "isir": float(isir),
                    "tag": f"rt{rt60:g}_ang{ang:g}_isir{isir:g}"})
    return out


def build_and_prepare(spec, cfg, provider, args):
    """Escena MIRD -> STFTs -> mascaras DTLN -> snapshots de SCM (con cache)."""
    # La clave del cache incluye TODO lo que cambia el contenido de las SCM. Si
    # falta un eje, un barrido posterior reusa snapshots viejos en silencio.
    cache_path = os.path.join(
        CACHE_DIR, f"{spec['tag']}_{args.spacing}_a{args.alpha:g}_E{args.n_eval}_"
                   f"sh{args.sharpen_exp:g}_snr{args.snr_db:g}_d{args.duration:g}_"
                   f"ta{args.target_angle:g}_td{args.target_dist:g}_"
                   f"id{args.interf_dist:g}.npz")
    if args.cache and os.path.exists(cache_path):
        z = np.load(cache_path)
        print(f"[cache] {spec['tag']}")
        Gamma = diffuse_coherence(z["mic_coords"], z["freqs"], field=args.field) \
            if not args.no_geometry else None
        from beamforming.mask.scm_calibration import oracle_references
        refs = oracle_references(z["Phi_S"], z["Phi_N"], int(z["ref_mic"]),
                                 min_loading=args.min_loading,
                                 snr_floor_db=args.snr_floor_db)
        return {"name": spec["tag"], "Phi_XX": z["Phi_XX"], "Phi_NN": z["Phi_NN"],
                "Phi_S": z["Phi_S"], "Phi_N": z["Phi_N"], "Gamma": Gamma,
                "refs": refs, "ref_mic": int(z["ref_mic"]),
                "eval_frames": z["eval_frames"], "freqs": z["freqs"]}

    print(f"[*] construyendo escena {spec['tag']}")
    mic_coords, mixture, oracle_target, oracle_noise, _ = build_scene(
        dict(cfg), provider, spec["rt60"], args.target_angle, args.target_dist,
        [(spec["interf_angle"], args.interf_dist)], spec["isir"], args.snr_db)

    nperseg, noverlap = cfg["stft_window"], cfg["stft_overlap"]
    hop = nperseg - noverlap
    M = mixture.shape[0]
    ref_mic = M // 2                         # ref_mic_mode default del benchmark MIRD

    def _stft(x):
        f_, _, Z = sig.stft(x, fs=cfg["fs"], window="hamming",
                            nperseg=nperseg, noverlap=noverlap, nfft=nperseg)
        return f_, np.transpose(Z, (1, 2, 0))

    freqs, X_stft = _stft(mixture)
    _, S_stft = _stft(oracle_target)
    _, N_stft = _stft(oracle_noise)

    mask_s, mask_n = get_dtln_masks_sharpen(
        mixture, ref_mic, cfg["dtln_model_path"],
        block_len=nperseg, block_shift=hop, sharpen_exp=args.sharpen_exp)

    eval_start_s = min(5.0, args.duration * 0.3)
    start_frame = int(eval_start_s * cfg["fs"] / hop)

    sc = prepare_scene(X_stft, S_stft, N_stft, mask_s, mask_n, mic_coords, freqs,
                       ref_mic, alpha=args.alpha, n_eval=args.n_eval,
                       start_frame=start_frame, min_loading=args.min_loading,
                       snr_floor_db=args.snr_floor_db, field=args.field,
                       use_geometry=not args.no_geometry, name=spec["tag"])

    if args.cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(cache_path, Phi_XX=sc["Phi_XX"], Phi_NN=sc["Phi_NN"],
                            Phi_S=sc["Phi_S"], Phi_N=sc["Phi_N"],
                            eval_frames=sc["eval_frames"], freqs=freqs,
                            mic_coords=mic_coords, ref_mic=ref_mic)
    return sc


# =====================================================================
# Reporte
# =====================================================================
def scene_table(scenes, rows, eta, mu, min_loading, how="wmedian"):
    """Loss por escena de los tres puntos de interes + el piso oracle."""
    nu_k, gam_k = bands_to_bin_params(rows, len(scenes[0]["freqs"]))
    out = []
    for sc in scenes:
        rec = {"scene": sc["name"]}
        for label, nu, gam in (("base", 0.0, 0.0), ("sub", 1.0, 0.0),
                               ("fit", nu_k, gam_k)):
            d = band_objective([sc], np.arange(len(sc["freqs"])), nu, gam, mu=mu,
                               eta=eta, min_loading=min_loading, detail=True, how=how)
            rec[f"L_{label}"] = d["L"]
            rec[f"Lsinr_{label}"] = d["L_sinr"]
            rec[f"Ldist_{label}"] = d["L_dist"]
        ob = oracle_bound([sc], eta=eta, mu=mu, min_loading=min_loading, how=how)
        rec["L_oracle"] = ob["L"]
        rec["Lsinr_oracle"] = ob["L_sinr"]
        out.append(rec)
    return pd.DataFrame(out)


def plot_panel(freqs, rows, nu_k, gam_k, out_path, eta):
    f_lo = np.array([r["f_c"] for r in rows])   # centro geometrico (eje log)
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))

    ax[0, 0].semilogx(f_lo, [r["nu"] for r in rows], "o-", label=r"$\nu^*$")
    ax[0, 0].axhline(0.0, color="gray", ls=":", lw=1, label="NM_MVDR (nu=0)")
    ax[0, 0].axhline(1.0, color="crimson", ls="--", lw=1, label="NM_MVDR_SUB (nu=1)")
    ax[0, 0].set_ylabel(r"$\nu$ (escala de la sustraccion)")
    ax[0, 0].set_title("Escala optima de la sustraccion")

    ax[0, 1].semilogx(f_lo, [r["gamma"] for r in rows], "s-", color="teal")
    ax[0, 1].set_ylim(-0.05, 1.05)
    ax[0, 1].set_ylabel(r"$\gamma$ (shrinkage a $\Gamma_{dif}$)")
    ax[0, 1].set_title("Peso del modelo difuso (geometria)")

    ax[1, 0].semilogx(f_lo, [r["L_base"] for r in rows], "o-", label="NM_MVDR")
    ax[1, 0].semilogx(f_lo, [r["L_sub"] for r in rows], "s-", label="NM_MVDR_SUB")
    ax[1, 0].semilogx(f_lo, [r["L_fit"] for r in rows], "^-", label="calibrado")
    ax[1, 0].set_ylabel(f"L = L_sinr + {eta:g}*L_dist  [dB]")
    ax[1, 0].set_title("Perdida contra el oracle (menor = mejor)")
    ax[1, 0].legend(fontsize=8)

    ax[1, 1].semilogx(f_lo, [r["gain_vs_base"] for r in rows], "o-", label="vs NM_MVDR")
    ax[1, 1].semilogx(f_lo, [r["gain_vs_sub"] for r in rows], "s-", label="vs NM_MVDR_SUB")
    ax[1, 1].axhline(0.0, color="gray", lw=1)
    ax[1, 1].set_ylabel("recuperado [dB]")
    ax[1, 1].set_title("Ganancia de la calibracion")
    ax[1, 1].legend(fontsize=8)

    for a in ax.ravel():
        a.grid(True, which="both", alpha=0.3)
        a.set_xlabel("frecuencia [Hz]")
    ax[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# =====================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # --- escenas ---
    ap.add_argument("--rt60", type=float, nargs="+", default=[0.360, 0.610])
    ap.add_argument("--interf-angles", type=float, nargs="+", default=[45, 90])
    ap.add_argument("--isir", type=float, nargs="+", default=[0, 10])
    ap.add_argument("--spacing", type=str, default="3-3-3-8-3-3-3")
    ap.add_argument("--target-angle", type=float, default=0)
    ap.add_argument("--target-dist", type=float, default=1.0)
    ap.add_argument("--interf-dist", type=float, default=1.0)
    ap.add_argument("--snr-db", type=float, default=30.0,
                    help="ruido propio de los microfonos (30 = MEMS realista)")
    ap.add_argument("--duration", type=float, default=12.0)
    # --- estimacion ---
    ap.add_argument("--alpha", type=float, default=0.99)
    ap.add_argument("--sharpen-exp", type=float, default=4.0)
    ap.add_argument("--n-eval", type=int, default=16,
                    help="frames donde se congelan las SCM (con alpha=0.99 la "
                         "memoria es ~100 frames: mas frames no agregan muestras)")
    ap.add_argument("--min-loading", type=float, default=1e-9)
    ap.add_argument("--field", type=str, default="spherical",
                    choices=["spherical", "cylindrical"])
    ap.add_argument("--no-geometry", action="store_true",
                    help="ablacion: desactiva el shrinkage difuso (solo ajusta nu)")
    # --- loss / ajuste ---
    ap.add_argument("--eta", type=float, default=1.0,
                    help="peso del termino de distorsion (PESQ) frente al de SINR (SIR)")
    ap.add_argument("--mu", type=float, default=0.0, help="trade-off PMWF, fijo")
    ap.add_argument("--snr-floor-db", type=float, default=-20.0,
                    help="descarta celdas (bin,frame) con SNR oracle local menor")
    ap.add_argument("--agg", type=str, default="wmedian",
                    choices=["wmedian", "median", "wmean", "mean"],
                    help="agregacion de la loss sobre (escena,bin,frame). wmedian "
                         "(default) pondera por la potencia del target oracle, que "
                         "es como agregan las metricas globales; median es el "
                         "criterio historico y SOBRE-REPORTA la ganancia ~20x.")
    ap.add_argument("--compare-npz", type=str, default=None,
                    help="evalua tambien un .npz de calibracion previo BAJO LA "
                         "AGREGACION ACTUAL (para comparar ajustes de distinto criterio)")
    ap.add_argument("--n-bands", type=int, default=20)
    ap.add_argument("--f-min", type=float, default=60.0)
    ap.add_argument("--f-max", type=float, default=7000.0)
    ap.add_argument("--nu-grid", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
                    help="el refinamiento no sale de este rango: si el optimo cae "
                         "en el borde, ampliarlo")
    ap.add_argument("--gamma-grid", type=float, nargs="+",
                    default=[0.0, 0.15, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--no-refine", action="store_true",
                    help="solo grilla, sin Nelder-Mead")
    # --- protocolo ---
    ap.add_argument("--split", type=str, default="interleave",
                    choices=["interleave", "rt60", "isir", "angle", "none"],
                    help="eje de generalizacion train/test")
    ap.add_argument("--per-scene", action="store_true",
                    help="ademas ajusta cada escena por separado (cota superior "
                         "de la familia -> test de SATURACION)")
    ap.add_argument("--no-cache", dest="cache", action="store_false")
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cfg = {
        "fs": 16000,
        "duration": args.duration,
        "t_early": 0.050,
        "array_center": [3.0, 3.0, 1.2],
        "mird_spacing": args.spacing,
        "snr_db": args.snr_db,
        "source_path": f"{PROJECT_ROOT}/tools/data/signals/p002_emo_adoration_sentences.wav",
        "interf_paths": [f"{PROJECT_ROOT}/tools/data/signals/techno_gated commune.wav"],
        "stft_window": 512,
        "stft_overlap": 384,
        "dtln_model_path": f"{PROJECT_ROOT}/src/dnn_denoise/models/model_quant_1.tflite",
    }
    provider = MirdDatasetProvider(root_dir=f"{PROJECT_ROOT}/tools/data/rirs/mird")

    specs = scene_grid(args.rt60, args.interf_angles, args.isir)
    print(f"[*] {len(specs)} escenas: rt60={args.rt60} x ang={args.interf_angles} "
          f"x iSIR={args.isir}")
    scenes = [build_and_prepare(s, cfg, provider, args) for s in specs]

    freqs = scenes[0]["freqs"]
    K = len(freqs)

    # --- split train / test --------------------------------------------------
    if args.split == "interleave":
        tr = [i for i in range(len(specs)) if i % 2 == 0]
    elif args.split == "rt60":
        tr = [i for i, s in enumerate(specs) if s["rt60"] == args.rt60[0]]
    elif args.split == "isir":
        tr = [i for i, s in enumerate(specs) if s["isir"] == args.isir[0]]
    elif args.split == "angle":
        tr = [i for i, s in enumerate(specs) if s["interf_angle"] == args.interf_angles[0]]
    else:
        tr = list(range(len(specs)))
    te = [i for i in range(len(specs)) if i not in tr] or tr
    train = [scenes[i] for i in tr]
    test = [scenes[i] for i in te]
    print(f"[*] split={args.split}  train={[s['name'] for s in train]}")
    print(f"                        test ={[s['name'] for s in test]}")

    # --- control de sanidad: el piso oracle ---------------------------------
    ob = oracle_bound(scenes, eta=args.eta, mu=args.mu, min_loading=args.min_loading,
                      how=args.agg)
    print(f"\n[control] oracle bound: L_sinr = {ob['L_sinr']:.3f} dB  "
          f"L_dist = {ob['L_dist']:.3f} dB  L = {ob['L']:.3f} dB")
    if ob["L_sinr"] > 0.5:
        print("  !! L_sinr del oracle deberia ser ~0 dB. Revisar loss/referencias "
              "ANTES de creerle a cualquier numero de abajo.")

    # --- ajuste --------------------------------------------------------------
    edges, band_of_bin, bands = make_bands(freqs, n_bands=args.n_bands,
                                           f_min=args.f_min, f_max=args.f_max)
    kw = dict(min_loading=args.min_loading, how=args.agg)
    print(f"\n[*] ajustando {sum(b.size > 0 for b in bands)} bandas sobre "
          f"{len(train)} escenas de train (eta={args.eta:g}, mu={args.mu:g})")
    rows = fit_bands(train, freqs, bands, args.nu_grid, args.gamma_grid,
                     mu=args.mu, eta=args.eta, refine=not args.no_refine, **kw)

    nu_k, gam_k = bands_to_bin_params(rows, K)

    # --- evaluacion en test ---------------------------------------------------
    print(f"\n[*] evaluando en test ({len(test)} escenas)")
    for r in rows:
        bi = r["_bins"]
        r["L_base_test"] = band_objective(test, bi, 0.0, 0.0, mu=args.mu, eta=args.eta, **kw)
        r["L_sub_test"] = band_objective(test, bi, 1.0, 0.0, mu=args.mu, eta=args.eta, **kw)
        r["L_fit_test"] = band_objective(test, bi, r["nu"], r["gamma"], mu=args.mu,
                                         eta=args.eta, **kw)
        r["gain_vs_base_test"] = r["L_base_test"] - r["L_fit_test"]
        r["gain_vs_sub_test"] = r["L_sub_test"] - r["L_fit_test"]
        if args.per_scene:
            # ajuste sobreajustado por escena: cota superior de la familia
            per = [fit_band([sc], bi, args.nu_grid, args.gamma_grid, mu=args.mu,
                            eta=args.eta, refine=False, **kw)["L"] for sc in test]
            r["L_perscene_test"] = float(np.nanmedian(per))
            r["saturation_gap"] = r["L_fit_test"] - r["L_perscene_test"]

    # --- comparacion contra un ajuste previo, BAJO LA MISMA AGREGACION -------
    # Es la unica forma honesta de comparar dos calibraciones hechas con
    # criterios distintos: los valores de L no son comparables entre modos de
    # agregacion, asi que hay que re-evaluar los dos con el mismo how.
    cmp_rows = []
    if args.compare_npz:
        zc = np.load(args.compare_npz, allow_pickle=True)
        nu_c = np.interp(freqs, zc["freqs"], zc["nu_k"])
        gam_c = np.interp(freqs, zc["freqs"], zc["gamma_k"])
        all_bins = np.arange(K)
        cands = [("NM_MVDR", 0.0, 0.0), ("NM_MVDR_SUB", 1.0, 0.0),
                 (f"previo ({os.path.basename(args.compare_npz)})", nu_c, gam_c),
                 (f"ajuste actual (agg={args.agg})", nu_k, gam_k)]
        print(f"\n[*] comparacion en TEST bajo agg={args.agg} "
              f"(los tres evaluados con el MISMO criterio)")
        for tag, nu_, gam_ in cands:
            d = band_objective(test, all_bins, nu_, gam_, mu=args.mu, eta=args.eta,
                               **kw, detail=True)
            cmp_rows.append({"name": tag, "agg": args.agg, **d})
            print(f"    {tag:38s} L_sinr={d['L_sinr']:6.3f}  "
                  f"L_dist={d['L_dist']:6.3f}  L={d['L']:6.3f}")
        pd.DataFrame(cmp_rows).to_csv(
            os.path.join(args.out_dir, "scm_calib_compare.csv"), index=False)

    # --- salidas --------------------------------------------------------------
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in rows])
    df.to_csv(os.path.join(args.out_dir, "scm_calib_bands.csv"), index=False)

    land = []
    for r in rows:
        for i, nu in enumerate(args.nu_grid):
            for j, g in enumerate(args.gamma_grid):
                land.append({"band": r["band"], "f_lo": r["f_lo"], "f_hi": r["f_hi"],
                             "nu": nu, "gamma": g, "L": float(r["_grid"][i, j])})
    pd.DataFrame(land).to_csv(os.path.join(args.out_dir, "scm_calib_landscape.csv"),
                              index=False)

    df_sc = scene_table(scenes, rows, args.eta, args.mu, args.min_loading, how=args.agg)
    df_sc.to_csv(os.path.join(args.out_dir, "scm_calib_scenes.csv"), index=False)

    np.savez(os.path.join(args.out_dir, "scm_calib_params.npz"),
             freqs=freqs, nu_k=nu_k, gamma_k=gam_k, eta=args.eta, mu=args.mu,
             alpha=args.alpha, field=args.field, agg=args.agg,
             train=[s["name"] for s in train], test=[s["name"] for s in test])

    plot_panel(freqs, rows, nu_k, gam_k,
               os.path.join(args.out_dir, "scm_calib.png"), args.eta)

    # --- resumen --------------------------------------------------------------
    print("\n" + "=" * 78)
    print("RESUMEN (mediana sobre bandas, en dB de perdida contra el oracle)")
    print("=" * 78)
    for tag, cb, cs, cf in (("TRAIN", "L_base", "L_sub", "L_fit"),
                            ("TEST ", "L_base_test", "L_sub_test", "L_fit_test")):
        print(f"{tag}   NM_MVDR {df[cb].median():6.2f} | NM_MVDR_SUB {df[cs].median():6.2f} "
              f"| calibrado {df[cf].median():6.2f}   "
              f"(recupera {df[cb].median() - df[cf].median():+.2f} dB sobre el core actual)")
    print(f"        piso oracle {ob['L']:6.2f}")

    lf = df[df["f_hi"] <= 300.0]
    if len(lf):
        print(f"\nBanda GRAVE (< 300 Hz, la que PESQ no ve):")
        print(f"        NM_MVDR {lf['L_base'].median():6.2f} | "
              f"NM_MVDR_SUB {lf['L_sub'].median():6.2f} | "
              f"calibrado {lf['L_fit'].median():6.2f}   "
              f"nu={lf['nu'].median():.2f} gamma={lf['gamma'].median():.2f}")

    if args.per_scene and "saturation_gap" in df:
        gap = df["saturation_gap"].median()
        print(f"\nTEST DE SATURACION: gap ajuste-global vs ajuste-por-escena = "
              f"{gap:+.2f} dB")
        print("  gap ~ 0  -> la familia esta saturada: lo que falta contra el oracle")
        print("              es la MASCARA, no la calibracion. No seguir por aca.")
        print("  gap >> 0 -> hay margen: conviene hacer los parametros adaptativos.")

    print(f"\n[ok] {args.out_dir}")


if __name__ == "__main__":
    main()
