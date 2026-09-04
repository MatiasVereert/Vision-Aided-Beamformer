"""
¿Que warp de mascara le conviene al NM_MVDR_DSM_BLIND?

POR QUE HAY QUE RE-BARRER
-------------------------
El warp calibrado (mask_s = sigma(a_s logit(m) + b_s), mask_n = sigma(a_n
logit(1-m) + b_n)) se ajusto sobre mascaras calculadas en el canal de referencia
CRUDO, y dio a_s=1 (identidad) con a_n~2, b_n~-8. El DSM_BLIND alimenta al DTLN
con la salida del front-end ciego, que tiene MEJOR SNR: la mascara resultante es
mas confiada (mas masa cerca de 0 y de 1), asi que su distribucion es otra y el
a_n optimo no tiene por que ser el mismo. Transplantar 2.0 seria usar el
parametro fuera de su dominio de calibracion.

QUE COMPARA
-----------
Sobre la mascara del FRONT-END CIEGO (la de la segunda pasada, que es la que
alimenta al beamformer), con el core de sustraccion (nu=1) fijo:

    actual   : stretch_sharpen(m_raw)  -- stretch min-max GLOBAL + **4.
               Es lo que hace hoy el wrapper. NO ES CAUSAL: el stretch necesita
               el min/max de todo el archivo.
    warp a_n : masks_from_raw(m_raw, 1, 0, a_n, b_n), barriendo a_n.
               Causal, sin estado global.

Se reporta L_sinr y L_dist POR SEPARADO contra las SCM oracle (la loss del banco;
ver beamforming/mask/scm_calibration.py), agregando por mediana ponderada por la
potencia del target -- el criterio que sigue a las metricas globales.

AVISO SOBRE EL PROXY
--------------------
Esta loss sigue bien al SDR pero NO al SIR ni a PESQ (verificado: la mejora de
SDR/SAR se traduce, la de SIR no). Sirve para LOCALIZAR a_n barato; el valor
elegido hay que confirmarlo despues en el benchmark real.

USO
---
    python tests/dsm_blind_an_sweep.py
    python tests/dsm_blind_an_sweep.py --an-grid 1 2 3 4 --b-n -6
    python tests/dsm_blind_an_sweep.py --also-ref     # ademas, la mascara del canal crudo

Salida: tests/dataset_out/dsm_blind_an/
    an_sweep.csv     L_sinr / L_dist por a_n, train y test
    an_sweep.png     las curvas, con el post-proceso actual marcado
"""

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import scipy.signal as sig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from propagation.mird_loader import MirdDatasetProvider
from beamforming.mask.dtln_masks import get_dtln_masks_sharpen, get_dtln_masks_soft
from beamforming.mask.ds_mask import blind_bf_signal, stretch_sharpen
from beamforming.mask.scm_calibration import (
    band_objective_mask, eval_frame_indices, snapshot_scms_oracle,
    oracle_references, diffuse_coherence,
)
from scm_calibration_run import scene_grid, PROJECT_ROOT
from lowfreq_diagnostic_run import build_scene

OUT_DIR = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "dsm_blind_an")
CACHE_DIR = os.path.join(OUT_DIR, "cache")


def prepare_blind_scene(spec, cfg, args):
    """
    Escena con la mascara del FRONT-END CIEGO, reproduciendo exactamente los
    pasos 1-2 de NM_MVDR_DSM_BLIND.process:

        mascara(1) = DTLN(x_ref)  [stretch+sharpen, como el wrapper]
        y_fix      = blind_bf_signal(x, mascara(1))       <- RTF ciega
        m_raw      = DTLN_soft(y_fix)                     <- la que se barre

    Guarda ademas la mascara ACTUAL de la segunda pasada (stretch_sharpen(m_raw)),
    que es el baseline a batir.
    """
    tag = spec["tag"]
    cache = os.path.join(CACHE_DIR, f"{tag}_{args.spacing}_a{args.alpha:g}_"
                                    f"ra{args.rtf_alpha:g}_rl{args.rtf_loading:g}_"
                                    f"{args.rtf_mode}_{args.w_mode}_"
                                    f"E{args.n_eval}_snr{args.snr_db:g}_"
                                    f"d{args.duration:g}.npz")
    hop = cfg["stft_window"] - cfg["stft_overlap"]
    start_frame = int(min(5.0, args.duration * 0.3) * cfg["fs"] / hop)

    if args.cache and os.path.exists(cache):
        z = np.load(cache)
        print(f"[cache] {tag}")
        X_stft = z["X_stft"].astype(np.complex128)
        m_raw, m_ref = z["mask_raw"], z["mask_ref_raw"]
        Phi_S, Phi_N = z["Phi_S"], z["Phi_N"]
        ev, mic_coords, freqs = z["eval_frames"], z["mic_coords"], z["freqs"]
    else:
        print(f"[*] construyendo {tag} (front-end ciego)")
        mic_coords, mixture, o_tgt, o_noi, _ = build_scene(
            dict(cfg), args._provider, spec["rt60"], args.target_angle,
            args.target_dist, [(spec["interf_angle"], args.interf_dist)],
            spec["isir"], args.snr_db)
        nperseg, noverlap = cfg["stft_window"], cfg["stft_overlap"]
        ref_mic = mixture.shape[0] // 2

        def _stft(x):
            f_, _, Z = sig.stft(x, fs=cfg["fs"], window="hamming",
                                nperseg=nperseg, noverlap=noverlap, nfft=nperseg)
            return f_, np.transpose(Z, (1, 2, 0))

        freqs, X_stft = _stft(mixture)
        _, S_stft = _stft(o_tgt)
        _, N_stft = _stft(o_noi)

        # --- PASADA 1: mascara del canal crudo (igual que el wrapper) --------
        ms1, mn1 = get_dtln_masks_sharpen(
            mixture, ref_mic, cfg["dtln_model_path"], block_len=nperseg,
            block_shift=hop, sharpen_exp=args.sharpen_exp)
        m_ref, _ = get_dtln_masks_soft(mixture, ref_mic, cfg["dtln_model_path"],
                                       block_len=nperseg, block_shift=hop)

        # --- FRONT-END CIEGO + PASADA 2 --------------------------------------
        y_fix = blind_bf_signal(
            mixture, ms1, mn1, cfg["fs"], ref_mic_idx=ref_mic,
            nperseg=nperseg, noverlap=noverlap, window="hamming",
            rtf_alpha=args.rtf_alpha, rtf_loading=args.rtf_loading,
            rtf_mode=args.rtf_mode, w_mode=args.w_mode,
            bf_loading=args.bf_loading)
        m_raw, _ = get_dtln_masks_soft(y_fix[None, :], 0, cfg["dtln_model_path"],
                                       block_len=nperseg, block_shift=hop)

        T0 = min(X_stft.shape[1], S_stft.shape[1], m_raw.shape[1], m_ref.shape[1])
        ev = eval_frame_indices(T0, args.n_eval, start_frame=start_frame)
        Phi_S, Phi_N = snapshot_scms_oracle(S_stft[:, :T0], N_stft[:, :T0], ev,
                                            alpha=args.alpha)
        if args.cache:
            os.makedirs(CACHE_DIR, exist_ok=True)
            np.savez(cache, X_stft=X_stft.astype(np.complex64), mask_raw=m_raw,
                     mask_ref_raw=m_ref, Phi_S=Phi_S, Phi_N=Phi_N,
                     eval_frames=ev, mic_coords=mic_coords, freqs=freqs)

    ref_mic = X_stft.shape[2] // 2
    T = min(X_stft.shape[1], m_raw.shape[1], m_ref.shape[1])
    cur_s, cur_n = stretch_sharpen(m_raw[:, :T], sharpen_exp=args.sharpen_exp)
    return {
        "name": tag, "freqs": freqs, "ref_mic": ref_mic,
        "X_stft": X_stft[:, :T], "mask_raw": m_raw[:, :T],
        "mask_ref_raw": m_ref[:, :T], "mask_cur": (cur_s, cur_n),
        "Phi_S": Phi_S, "Phi_N": Phi_N, "eval_frames": ev, "alpha": args.alpha,
        "Gamma": diffuse_coherence(mic_coords, freqs, field="spherical"),
        "refs": oracle_references(Phi_S, Phi_N, ref_mic,
                                  min_loading=args.min_loading,
                                  snr_floor_db=args.snr_floor_db),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--an-grid", type=float, nargs="+",
                    default=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
    ap.add_argument("--b-n", type=float, default=-8.0)
    ap.add_argument("--a-s", type=float, default=1.0)
    ap.add_argument("--b-s", type=float, default=0.0)
    ap.add_argument("--also-ref", action="store_true",
                    help="agrega la curva con la mascara del canal CRUDO, para "
                         "ver cuanto del cambio de optimo lo causa el front-end")
    # --- front-end ciego (mismos defaults que NM_MVDR_DSM_BLIND) -------------
    ap.add_argument("--rtf-alpha", type=float, default=0.999)
    ap.add_argument("--rtf-loading", type=float, default=1e-2)
    ap.add_argument("--rtf-mode", type=str, default="cs")
    ap.add_argument("--w-mode", type=str, default="ds")
    ap.add_argument("--bf-loading", type=float, default=1e-6)
    # --- escenas --------------------------------------------------------------
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
    ap.add_argument("--agg", type=str, default="wmedian")
    ap.add_argument("--nu", type=float, default=1.0)
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
    scenes = [prepare_blind_scene(s, cfg, args) for s in specs]
    train = [sc for sc, sp in zip(scenes, specs) if sp["rt60"] == args.rt60[0]]
    test = [sc for sc, sp in zip(scenes, specs) if sp["rt60"] != args.rt60[0]] or train
    K = len(scenes[0]["freqs"])
    allb = np.arange(K)
    ev = dict(nu=args.nu, gamma=0.0, mu=0.0, min_loading=args.min_loading,
              how=args.agg)
    print(f"[*] train={len(train)}  test={len(test)} (rt60 no visto)")

    rows = []

    def _add(kind, name, a_n, split, d):
        rows.append({"kind": kind, "name": name, "a_n": a_n, "split": split,
                     "L_sinr": d["L_sinr"], "L_dist": d["L_dist"]})

    # --- baseline: el post-proceso ACTUAL del wrapper -------------------------
    for split, sc_ in (("train", train), ("test", test)):
        d = band_objective_mask(sc_, allb, None, eta=1.0, detail=True,
                                masks=[(s["mask_cur"][0], s["mask_cur"][1]) for s in sc_],
                                **ev)
        _add("actual", "stretch+sharpen (actual)", np.nan, split, d)
        print(f"  [{split}] mascara ACTUAL del front-end   "
              f"L_sinr={d['L_sinr']:6.3f}  L_dist={d['L_dist']:6.3f}")

    # --- barrido de a_n sobre la mascara del FRONT-END ------------------------
    print(f"\n[*] barrido de a_n sobre la mascara del front-end ciego "
          f"(a_s={args.a_s:g}, b_s={args.b_s:g}, b_n={args.b_n:g})")
    t0 = time.time()
    for an in args.an_grid:
        th = (args.a_s, args.b_s, an, args.b_n)
        for split, sc_ in (("train", train), ("test", test)):
            d = band_objective_mask(sc_, allb, th, eta=1.0, detail=True, **ev)
            _add("warp", f"a_n={an:g}", an, split, d)
        r = [x for x in rows if x["kind"] == "warp" and x["a_n"] == an]
        tr = [x for x in r if x["split"] == "train"][0]
        te = [x for x in r if x["split"] == "test"][0]
        print(f"    a_n={an:4.2f}  train L_sinr={tr['L_sinr']:6.3f} "
              f"L_dist={tr['L_dist']:5.3f}   |   test L_sinr={te['L_sinr']:6.3f} "
              f"L_dist={te['L_dist']:5.3f}")
    print(f"    ({time.time()-t0:.0f}s)")

    # --- opcional: la misma curva con la mascara del canal CRUDO --------------
    if args.also_ref:
        print("\n[*] control: misma curva con la mascara del canal de referencia CRUDO")
        for sc in scenes:
            sc["_blind"] = sc["mask_raw"]
            sc["mask_raw"] = sc["mask_ref_raw"]
        for an in args.an_grid:
            th = (args.a_s, args.b_s, an, args.b_n)
            d = band_objective_mask(test, allb, th, eta=1.0, detail=True, **ev)
            _add("warp_ref", f"ref a_n={an:g}", an, "test", d)
            print(f"    a_n={an:4.2f}  test L_sinr={d['L_sinr']:6.3f} "
                  f"L_dist={d['L_dist']:5.3f}")
        for sc in scenes:
            sc["mask_raw"] = sc["_blind"]

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "an_sweep.csv"), index=False)

    # --- grafico --------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for i, met in enumerate(("L_sinr", "L_dist")):
        for split, mk in (("train", "o--"), ("test", "s-")):
            m = df[(df.kind == "warp") & (df.split == split)].sort_values("a_n")
            ax[i].plot(m["a_n"], m[met], mk, label=f"warp ({split})")
        if args.also_ref:
            m = df[(df.kind == "warp_ref")].sort_values("a_n")
            ax[i].plot(m["a_n"], m[met], "^:", color="gray",
                       label="warp sobre canal crudo (test)")
        for split, col in (("train", "tab:orange"), ("test", "crimson")):
            v = df[(df.kind == "actual") & (df.split == split)][met]
            if len(v):
                ax[i].axhline(float(v.iloc[0]), ls="--", color=col, lw=1.2,
                              label=f"stretch+sharpen actual ({split})")
        ax[i].set_xlabel("$a_n$"); ax[i].set_ylabel(f"{met} [dB]")
        ax[i].set_title(f"{met} vs $a_n$ (front-end ciego)")
        ax[i].grid(alpha=0.3); ax[i].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "an_sweep.png"), dpi=130)
    plt.close(fig)

    te = df[(df.kind == "warp") & (df.split == "test")]
    best = te.loc[te["L_sinr"].idxmin()]
    act = df[(df.kind == "actual") & (df.split == "test")].iloc[0]
    print("\n" + "=" * 70)
    print(f"MEJOR a_n en test (por L_sinr): {best['a_n']:g}  "
          f"-> L_sinr={best['L_sinr']:.3f}  L_dist={best['L_dist']:.3f}")
    print(f"contra el post-proceso ACTUAL:      L_sinr={act['L_sinr']:.3f}  "
          f"L_dist={act['L_dist']:.3f}")
    print(f"                            delta:  {best['L_sinr']-act['L_sinr']:+.3f} / "
          f"{best['L_dist']-act['L_dist']:+.3f}   (negativo = mejor)")
    print(f"\n[ok] {args.out_dir}")


if __name__ == "__main__":
    main()
