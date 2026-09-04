"""
ETAPA 2 del banco: calibrar los parametros de la MASCARA (no los post-hoc).

POR QUE ESTA ETAPA
------------------
La familia post-hoc (nu, gamma) quedo SATURADA: da ~0.5 dB sobre el core base,
empata con NM_MVDR_SUB, y no la mueve ni eta, ni una restriccion, ni corregir la
agregacion. Lo que esa familia NO puede tocar es QUE SE ACUMULA -- y ahi esta el
camino que nunca se optimizo contra nada:

    m_raw  = DTLN(x_ref)                                      # ya en [0,1]
    m      = (m_raw - min) / (max - min)                       <-- STRETCH GLOBAL
    mask_s = m ** 4        mask_n = (1 - m) ** 4               <-- SHARPEN

El STRETCH usa el min/max de TODO el archivo y de TODAS las frecuencias: es NO
CAUSAL (para el frame 5 mira el futuro -> no implementable en el sistema online
que se lleva a HLS), depende del archivo entero, y acopla todos los bins. El
SHARPEN ata las dos ramas: mask_n = (1-m)**4 con el mismo exponente que mask_s.

QUE SE AJUSTA
-------------
Un warp fijo POR BANDA, causal y sin estado global, con las dos ramas
INDEPENDIENTES (ver `scm_calibration.warp_mask`):

    mask_s = sigma(a_s * logit(m_raw) + b_s)
    mask_n = sigma(a_n * logit(1 - m_raw) + b_n)

(a=1, b=0) es la identidad. La hipotesis que motiva desacoplar: Phi_NN es la que
se INVIERTE, asi que sus errores pesan mas, y lo que necesita no es "1 menos
probabilidad de voz" sino un detector de DOMINANCIA DE RUIDO, optimamente mucho
mas conservador que el complemento.

COSTO
-----
Estos parametros actuan ANTES de la acumulacion, asi que cada evaluacion del
objetivo REHACE LA RECURSION (~100x mas caro que la etapa 1). Se acota corriendo
la recursion solo sobre los bins de la banda que se esta ajustando.

LAS CUATRO FILAS QUE HAY QUE COMPARAR
-------------------------------------
    cur_base : mascara ACTUAL (stretch + **4), nu=0      -> NM_MVDR
    cur_sub  : mascara ACTUAL,                 nu=1      -> NM_MVDR_SUB
    raw_sub  : mascara CRUDA, warp identidad,  nu=1      -> ablacion: cuanto
               aporta realmente el stretch + sharpen (puede ser NEGATIVO)
    fit_sub  : warp AJUSTADO,                  nu=1      -> el resultado

nu/gamma quedan FIJOS en el punto de NM_MVDR_SUB para aislar la contribucion de
la mascara. Con --joint se reajustan tambien.

USO
---
    python tests/scm_mask_calibration_run.py --quick     # 4 bandas, para probar
    python tests/scm_mask_calibration_run.py
    python tests/scm_mask_calibration_run.py --n-bands 10 --maxiter 200

Salida: tests/dataset_out/scm_mask_calib/
    mask_calib_bands.csv   theta ajustado y las cuatro filas, por banda
    mask_calib.png         panel: a_s vs a_n, b_s vs b_n, y L por banda
    mask_calib_params.npz  a_s, b_s, a_n, b_n POR BIN
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
from beamforming.mask.dtln_masks import (get_dtln_masks_soft,
                                         get_dtln_masks_sharpen,
                                         align_mask_frames)
from evaluation.bf_wrappers import resolve_stft_window
from beamforming.mask.scm_calibration import (
    make_bands, band_objective_mask, fit_band_mask, masks_from_raw, MASK_THETA0,
    eval_frame_indices, snapshot_scms_oracle, oracle_references, diffuse_coherence,
)
from scm_calibration_run import scene_grid, PROJECT_ROOT
from lowfreq_diagnostic_run import build_scene

OUT_DIR = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "scm_mask_calib")
CACHE_DIR = os.path.join(OUT_DIR, "cache")


def prepare_scene_full(spec, cfg, args):
    """
    Escena lista para la etapa 2. A diferencia de `prepare_scene` (etapa 1), aca
    hay que conservar X_stft COMPLETO y la mascara CRUDA, porque la recursion se
    rehace en cada evaluacion del objetivo.

    NO se guardan Phi_XX / Phi_NN: dependen de la mascara, que es justamente lo
    que se esta ajustando. Del oracle alcanza con los snapshots Phi_S / Phi_N,
    que NO dependen de la mascara y se calculan una sola vez.

    El cache guarda X_stft en complex64 (la recursion promedia en float64 igual;
    el error relativo ~1e-7 queda muy por debajo de las diferencias de loss que
    se comparan, del orden de 0.01 dB) para no dejar cientos de MB por escena.
    """
    tag = spec["tag"]
    cache = os.path.join(CACHE_DIR, f"{tag}_{args.spacing}_a{args.alpha:g}_"
                                    f"E{args.n_eval}_snr{args.snr_db:g}_"
                                    f"d{args.duration:g}_w{args.win}_"
                                    f"s{args.mask_shift}.npz")
    hop = cfg["stft_window"] - cfg["stft_overlap"]
    start_frame = int(min(5.0, args.duration * 0.3) * cfg["fs"] / hop)

    if args.cache and os.path.exists(cache):
        z = np.load(cache)
        print(f"[cache] {tag}")
        X_stft = z["X_stft"].astype(np.complex128)
        m_raw, m_cur_s, m_cur_n = z["mask_raw"], z["mask_cur_s"], z["mask_cur_n"]
        Phi_S, Phi_N = z["Phi_S"], z["Phi_N"]
        ev, mic_coords, freqs = z["eval_frames"], z["mic_coords"], z["freqs"]
    else:
        print(f"[*] construyendo {tag}")
        mic_coords, mixture, o_tgt, o_noi, _ = build_scene(
            dict(cfg), args._provider, spec["rt60"], args.target_angle,
            args.target_dist, [(spec["interf_angle"], args.interf_dist)],
            spec["isir"], args.snr_db)
        nperseg, noverlap = cfg["stft_window"], cfg["stft_overlap"]

        # VENTANA DEL BANCO. Tiene que ser la MISMA que la del beamformer que
        # va a consumir la calibracion: el warp compensa, entre otras cosas, el
        # desacople mascara(rect) -> SCM(ventana). --win rect calibra el sistema
        # TODO RECTANGULAR (una sola FFT, el que se quiere en HW).
        win_spec = resolve_stft_window({}, args.win, nperseg)

        def _stft(x):
            f_, _, Z = sig.stft(x, fs=cfg["fs"], window=win_spec,
                                nperseg=nperseg, noverlap=noverlap, nfft=nperseg)
            return f_, np.transpose(Z, (1, 2, 0))

        freqs, X_stft = _stft(mixture)
        _, S_stft = _stft(o_tgt)
        _, N_stft = _stft(o_noi)
        ref_mic = mixture.shape[0] // 2

        # mascara CRUDA (salida del DTLN, sin stretch ni sharpen)
        m_raw, _ = get_dtln_masks_soft(mixture, ref_mic, cfg["dtln_model_path"],
                                       block_len=nperseg, block_shift=hop)
        # mascara ACTUAL (stretch global + **4): el baseline a batir
        m_cur_s, m_cur_n = get_dtln_masks_sharpen(
            mixture, ref_mic, cfg["dtln_model_path"], block_len=nperseg,
            block_shift=hop, sharpen_exp=args.sharpen_exp)

        # El bloque i del buffer del DTLN es el frame i-1 de scipy.stft. El banco
        # historico NO lo corregia (por eso NM_MVDR_CAL pinea mask_shift=0);
        # --mask-shift 1 (default) calibra sobre el sistema YA corregido.
        if args.mask_shift:
            m_raw = align_mask_frames(m_raw, args.mask_shift)
            m_cur_s, m_cur_n = align_mask_frames((m_cur_s, m_cur_n), args.mask_shift)

        T0 = min(X_stft.shape[1], S_stft.shape[1], m_raw.shape[1], m_cur_s.shape[1])
        ev = eval_frame_indices(T0, args.n_eval, start_frame=start_frame)
        Phi_S, Phi_N = snapshot_scms_oracle(S_stft[:, :T0], N_stft[:, :T0], ev,
                                            alpha=args.alpha)
        if args.cache:
            os.makedirs(CACHE_DIR, exist_ok=True)
            np.savez(cache, X_stft=X_stft.astype(np.complex64), mask_raw=m_raw,
                     mask_cur_s=m_cur_s, mask_cur_n=m_cur_n, Phi_S=Phi_S,
                     Phi_N=Phi_N, eval_frames=ev, mic_coords=mic_coords, freqs=freqs)

    ref_mic = X_stft.shape[2] // 2
    T = min(X_stft.shape[1], m_raw.shape[1], m_cur_s.shape[1])
    return {
        "name": tag, "freqs": freqs, "ref_mic": ref_mic,
        "X_stft": X_stft[:, :T], "mask_raw": m_raw[:, :T],
        "mask_cur": (m_cur_s[:, :T], m_cur_n[:, :T]),
        "Phi_S": Phi_S, "Phi_N": Phi_N, "eval_frames": ev, "alpha": args.alpha,
        "Gamma": diffuse_coherence(mic_coords, freqs, field=args.field),
        "refs": oracle_references(Phi_S, Phi_N, ref_mic,
                                  min_loading=args.min_loading,
                                  snr_floor_db=args.snr_floor_db),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
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
    ap.add_argument("--win", type=str, default="hamming",
                    help="ventana de la STFT del banco: hamming (historico) | rect | hann")
    ap.add_argument("--mask-shift", type=int, default=1,
                    help="corrimiento mascara<->STFT (0 = banco historico)")
    ap.add_argument("--n-eval", type=int, default=16)
    ap.add_argument("--min-loading", type=float, default=1e-9)
    ap.add_argument("--snr-floor-db", type=float, default=-20.0)
    ap.add_argument("--field", type=str, default="spherical")
    ap.add_argument("--agg", type=str, default="wmedian",
                    choices=["wmedian", "median", "wmean", "mean"])
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--nu", type=float, default=1.0,
                    help="nu FIJO durante el ajuste de mascara (1 = punto de NM_MVDR_SUB)")
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--n-bands", type=int, default=12)
    ap.add_argument("--f-min", type=float, default=60.0)
    ap.add_argument("--f-max", type=float, default=7000.0)
    ap.add_argument("--maxiter", type=int, default=120)
    ap.add_argument("--split", type=str, default="rt60",
                    choices=["interleave", "rt60", "isir", "angle", "none"])
    ap.add_argument("--quick", action="store_true",
                    help="4 bandas y 1 escena de train: solo para medir tiempos")
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
    if args.quick:
        specs = specs[:2]
    scenes = [prepare_scene_full(s, cfg, args) for s in specs]
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
    print(f"[*] split={args.split}  train={[s['name'] for s in train]}")
    print(f"                        test ={[s['name'] for s in test]}")

    _, _, bands = make_bands(freqs, n_bands=args.n_bands, f_min=args.f_min,
                             f_max=args.f_max)
    bands = [b for b in bands if b.size]
    if args.quick:
        bands = bands[:4]

    kw = dict(mu=args.mu, eta=args.eta, min_loading=args.min_loading, how=args.agg)

    def cur_masks(scenes_, bi):
        return [(sc["mask_cur"][0][bi], sc["mask_cur"][1][bi]) for sc in scenes_]

    rows = []
    t0 = time.time()
    for n, bi in enumerate(bands):
        tb = time.time()
        # baselines con la mascara ACTUAL
        cb = band_objective_mask(train, bi, None, nu=0.0, gamma=0.0, detail=True,
                                 masks=cur_masks(train, bi), **kw)
        cs = band_objective_mask(train, bi, None, nu=args.nu, gamma=args.gamma,
                                 detail=True, masks=cur_masks(train, bi), **kw)
        # ablacion: mascara CRUDA sin warp (sin stretch, sin sharpen)
        rs = band_objective_mask(train, bi, MASK_THETA0, nu=args.nu,
                                 gamma=args.gamma, detail=True, **kw)
        # ajuste
        fit = fit_band_mask(train, bi, nu=args.nu, gamma=args.gamma,
                            maxiter=args.maxiter, **kw)

        # test
        cs_te = band_objective_mask(test, bi, None, nu=args.nu, gamma=args.gamma,
                                    detail=True, masks=cur_masks(test, bi), **kw)
        fit_te = band_objective_mask(test, bi, fit["theta"], nu=args.nu,
                                     gamma=args.gamma, detail=True, **kw)

        a_s, b_s, a_n, b_n = fit["theta"]
        rows.append({
            "band": n, "f_lo": float(freqs[bi[0]]), "f_hi": float(freqs[bi[-1]]),
            "f_c": float(np.sqrt(max(freqs[bi[0]], 0.5 * freqs[1]) * max(freqs[bi[-1]], freqs[1]))),
            "n_bins": int(bi.size),
            "a_s": a_s, "b_s": b_s, "a_n": a_n, "b_n": b_n,
            "L_cur_base": cb["L"], "L_cur_sub": cs["L"], "L_raw_sub": rs["L"],
            "L_fit_sub": fit["L"],
            "L_cur_sub_test": cs_te["L"], "L_fit_sub_test": fit_te["L"],
            "gain_train": cs["L"] - fit["L"],
            "gain_test": cs_te["L"] - fit_te["L"],
            "Lsinr_cur_sub": cs["L_sinr"], "Lsinr_fit_sub": fit["L_sinr"],
            "Ldist_cur_sub": cs["L_dist"], "Ldist_fit_sub": fit["L_dist"],
            "n_eval": fit["n_eval"], "_bins": bi,
        })
        print(f"  banda {n:2d} {rows[-1]['f_lo']:6.0f}-{rows[-1]['f_hi']:6.0f} Hz  "
              f"a_s={a_s:4.2f} b_s={b_s:+5.2f} | a_n={a_n:4.2f} b_n={b_n:+5.2f}   "
              f"L: actual {cs['L']:6.2f} | cruda {rs['L']:6.2f} | fit {fit['L']:6.2f}"
              f"   test {cs_te['L']:6.2f}->{fit_te['L']:6.2f} ({rows[-1]['gain_test']:+.2f})"
              f"   [{fit['n_eval']} evals, {time.time()-tb:.0f}s]")

    print(f"\n[*] tiempo total de ajuste: {(time.time()-t0)/60:.1f} min")

    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in rows])
    df.to_csv(os.path.join(args.out_dir, "mask_calib_bands.csv"), index=False)

    # parametros por BIN
    out = {k: np.zeros(K) for k in ("a_s", "b_s", "a_n", "b_n")}
    out["a_s"] += 1.0; out["a_n"] += 1.0
    for r in rows:
        for k in ("a_s", "b_s", "a_n", "b_n"):
            out[k][r["_bins"]] = r[k]
    np.savez(os.path.join(args.out_dir, "mask_calib_params.npz"), freqs=freqs,
             nu=args.nu, gamma=args.gamma, eta=args.eta, agg=args.agg,
             alpha=args.alpha, train=[s["name"] for s in train],
             test=[s["name"] for s in test], **out)

    # --- panel ---------------------------------------------------------------
    fc = df["f_c"].values
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    ax[0].semilogx(fc, df["a_s"], "o-", label="$a_s$ (voz)")
    ax[0].semilogx(fc, df["a_n"], "s-", label="$a_n$ (ruido)")
    ax[0].axhline(1.0, color="gray", ls=":", label="identidad")
    ax[0].set_ylabel("contraste $a$"); ax[0].set_title("Contraste por rama")
    ax[1].semilogx(fc, df["b_s"], "o-", label="$b_s$ (voz)")
    ax[1].semilogx(fc, df["b_n"], "s-", label="$b_n$ (ruido)")
    ax[1].axhline(0.0, color="gray", ls=":")
    ax[1].set_ylabel("umbral $b$"); ax[1].set_title("Corrimiento de umbral")
    ax[2].semilogx(fc, df["L_cur_sub"], "o-", label="mascara actual (NM_MVDR_SUB)")
    ax[2].semilogx(fc, df["L_raw_sub"], "d-", label="mascara cruda (sin stretch/sharpen)")
    ax[2].semilogx(fc, df["L_fit_sub"], "^-", label="warp ajustado")
    ax[2].set_ylabel("L [dB]"); ax[2].set_title("Perdida contra el oracle (train)")
    for a in ax:
        a.set_xlabel("frecuencia [Hz]"); a.grid(True, which="both", alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "mask_calib.png"), dpi=130)
    plt.close(fig)

    # --- resumen --------------------------------------------------------------
    print("\n" + "=" * 78)
    print("RESUMEN (mediana sobre bandas, dB de perdida contra el oracle)")
    print("=" * 78)
    print(f"TRAIN  mascara actual (nu=0)   {df['L_cur_base'].median():6.2f}")
    print(f"       mascara actual (nu={args.nu:g})   {df['L_cur_sub'].median():6.2f}")
    print(f"       mascara CRUDA sin warp  {df['L_raw_sub'].median():6.2f}   "
          f"<- lo que aporta el stretch+sharpen: "
          f"{df['L_raw_sub'].median() - df['L_cur_sub'].median():+.2f} dB")
    print(f"       warp AJUSTADO           {df['L_fit_sub'].median():6.2f}   "
          f"({df['gain_train'].median():+.2f} dB)")
    print(f"TEST   mascara actual          {df['L_cur_sub_test'].median():6.2f}")
    print(f"       warp AJUSTADO           {df['L_fit_sub_test'].median():6.2f}   "
          f"({df['gain_test'].median():+.2f} dB)")
    print(f"\nDESACOPLE DE RAMAS: a_n - a_s mediana = "
          f"{(df['a_n'] - df['a_s']).median():+.2f}  "
          f"(bandas con a_n > a_s: {(df['a_n'] > df['a_s']).sum()}/{len(df)})")
    print(f"\n[ok] {args.out_dir}")


if __name__ == "__main__":
    main()
