"""
CALIBRACION DEL WARP DE MASCARA PARA NM_MVDR_DSM_BLIND, CADENA CAUSAL Y RECT.

Por que no sirve la calibracion de scm_mask_calibration_run.py
-------------------------------------------------------------
Aquella se ajusto (a) con nu=0, el punto del core BASE, y el ciego corre
core="subtract" (nu=1); y (b) sobre la mascara del canal de referencia CRUDO,
mientras que el ciego alimenta al DTLN con la salida del front-end apuntado por
la RTF estimada -- mejor SNR, otra distribucion de la mascara. Transplantar el
theta seria usarlo fuera de su dominio de calibracion.

Que se ajusta aca
-----------------
El warp logit-afin de `scm_calibration.masks_from_raw` sobre la mascara de la
SEGUNDA PASADA (la del front-end, que es la que alimenta al beamformer), con
nu=1 fijo, contra las SCM oracle, por banda, con split train/test por RT60.

TODA LA CADENA ES CAUSAL
------------------------
  * ventana de ANALISIS rectangular (una sola FFT) -- la de sintesis no entra en
    este banco, que trabaja en el dominio STFT;
  * DTLN con escala FIJA en vez de la normalizacion por el pico global del
    archivo (peak_norm=1.0);
  * sin el stretch min-max global, ni en la pasada 1 ni en la 2;
  * align_mask_frames(1): es un arreglo de indexado, no un adelanto temporal
    (el bloque i+1 del buffer y el frame i de scipy terminan en la misma
    muestra), asi que no rompe causalidad.

Las cuatro filas que se comparan, por banda:
    prod    : cadena de PRODUCCION completa (pico global + stretch + **e). No causal.
    causal  : misma cadena sin las dos etapas globales (solo la potencia).
    raw     : mascara cruda, warp identidad (ablacion).
    fit     : warp ajustado.

Uso:
    python tests/window_mismatch/dsm_blind_warp_calib.py [--quick] [--n-bands 12]
"""

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import scipy.signal as sig

ROOT = "/home/matias/Documents/Tesis/Vision-Aided-Beamformer"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

from propagation.mird_loader import MirdDatasetProvider                      # noqa: E402
from beamforming.mask.dtln_masks import (get_dtln_masks_soft,                # noqa: E402
                                         get_dtln_masks_sharpen,
                                         align_mask_frames)
from beamforming.mask.ds_mask import blind_bf_signal, stretch_sharpen        # noqa: E402
from beamforming.mask.scm_calibration import (                               # noqa: E402
    make_bands, band_objective_mask, fit_band_mask, MASK_THETA0,
    eval_frame_indices, snapshot_scms_oracle, oracle_references,
    diffuse_coherence,
)
from evaluation.bf_wrappers import resolve_stft_window                       # noqa: E402
from scm_calibration_run import scene_grid                                   # noqa: E402
from lowfreq_diagnostic_run import build_scene                               # noqa: E402

OUT_DIR = os.path.join(ROOT, "tests", "dataset_out", "dsm_blind_warp_calib")
CACHE_DIR = os.path.join(OUT_DIR, "cache")
MODEL_1 = f"{ROOT}/src/dnn_denoise/models/model_quant_1.tflite"


def _fit_T(m, T):
    m = np.asarray(m, dtype=np.float64)
    if m.shape[1] >= T:
        return m[:, :T]
    return np.concatenate([m, np.repeat(m[:, -1:], T - m.shape[1], axis=1)], axis=1)


def _front_end(mixture, cfg, args, ref, win, causal):
    """Pasada 1 -> RTF ciega -> y_fix -> pasada 2. Devuelve la mascara CRUDA."""
    L, nov = cfg["stft_window"], cfg["stft_overlap"]
    H = L - nov
    kw1 = {'peak_norm': 1.0, 'stretch': False} if causal else {}
    m1_s, m1_n = get_dtln_masks_sharpen(mixture, ref, cfg["dtln_model_path"],
                                        block_len=L, block_shift=H,
                                        sharpen_exp=args.sharpen_exp, **kw1)
    if causal:      # sin stretch, la potencia se aplica igual (puntual)
        pass
    m1_s, m1_n = align_mask_frames((m1_s, m1_n), 1)

    y_fix = blind_bf_signal(mixture, m1_s, m1_n, cfg["fs"], ref_mic_idx=ref,
                            nperseg=L, noverlap=nov, window=win,
                            rtf_alpha=args.rtf_alpha, rtf_loading=args.rtf_loading,
                            rtf_mode="cs", w_mode="ds", bf_loading=1e-6)

    kw2 = {'peak_norm': 1.0} if causal else {}
    m_raw, _ = get_dtln_masks_soft(y_fix[None, :], 0, cfg["dtln_model_path"],
                                   block_len=L, block_shift=H, **kw2)
    return align_mask_frames(m_raw, 1)


def prepare_scene(spec, cfg, args):
    tag = spec["tag"]
    cache = os.path.join(CACHE_DIR, f"{tag}_{args.spacing}_a{args.alpha:g}_"
                                    f"E{args.n_eval}_snr{args.snr_db:g}_"
                                    f"d{args.duration:g}_w{args.win}_"
                                    f"e{args.sharpen_exp:g}.npz")
    L, nov = cfg["stft_window"], cfg["stft_overlap"]
    hop = L - nov
    start_frame = int(min(5.0, args.duration * 0.3) * cfg["fs"] / hop)

    if args.cache and os.path.exists(cache):
        z = np.load(cache)
        print(f"[cache] {tag}")
        X_stft = z["X_stft"].astype(np.complex128)
        m_raw, m_prod, m_causal = z["m_raw"], z["m_prod"], z["m_causal"]
        Phi_S, Phi_N = z["Phi_S"], z["Phi_N"]
        ev, mic_coords, freqs = z["eval_frames"], z["mic_coords"], z["freqs"]
    else:
        print(f"[*] construyendo {tag}")
        mic_coords, mixture, o_tgt, o_noi, _ = build_scene(
            dict(cfg), args._provider, spec["rt60"], args.target_angle,
            args.target_dist, [(spec["interf_angle"], args.interf_dist)],
            spec["isir"], args.snr_db)
        win = resolve_stft_window({}, args.win, L)

        def _stft(x):
            f_, _, Z = sig.stft(x, fs=cfg["fs"], window=win, nperseg=L,
                                noverlap=nov, nfft=L)
            return f_, np.transpose(Z, (1, 2, 0))

        freqs, X_stft = _stft(mixture)
        _, S_stft = _stft(o_tgt)
        _, N_stft = _stft(o_noi)
        ref = mixture.shape[0] // 2

        # mascara del front-end en los DOS caminos
        m_raw = _front_end(mixture, cfg, args, ref, win, causal=True)
        m_raw_prod = _front_end(mixture, cfg, args, ref, win, causal=False)
        m_prod = np.stack(stretch_sharpen(m_raw_prod, args.sharpen_exp))
        mc = np.clip(m_raw, 0.0, 1.0)
        m_causal = np.stack((mc ** args.sharpen_exp,
                             (1.0 - mc) ** args.sharpen_exp))

        T0 = min(X_stft.shape[1], S_stft.shape[1], m_raw.shape[1],
                 m_prod.shape[2], m_causal.shape[2])
        ev = eval_frame_indices(T0, args.n_eval, start_frame=start_frame)
        Phi_S, Phi_N = snapshot_scms_oracle(S_stft[:, :T0], N_stft[:, :T0], ev,
                                            alpha=args.alpha)
        if args.cache:
            os.makedirs(CACHE_DIR, exist_ok=True)
            np.savez(cache, X_stft=X_stft.astype(np.complex64), m_raw=m_raw,
                     m_prod=m_prod, m_causal=m_causal, Phi_S=Phi_S, Phi_N=Phi_N,
                     eval_frames=ev, mic_coords=mic_coords, freqs=freqs)

    ref = X_stft.shape[2] // 2
    T = min(X_stft.shape[1], m_raw.shape[1], m_prod.shape[2], m_causal.shape[2])
    return {
        "name": tag, "freqs": freqs, "ref_mic": ref,
        "X_stft": X_stft[:, :T], "mask_raw": _fit_T(m_raw, T),
        "mask_prod": (_fit_T(m_prod[0], T), _fit_T(m_prod[1], T)),
        "mask_causal": (_fit_T(m_causal[0], T), _fit_T(m_causal[1], T)),
        "Phi_S": Phi_S, "Phi_N": Phi_N, "eval_frames": ev, "alpha": args.alpha,
        "Gamma": diffuse_coherence(mic_coords, freqs, field=args.field),
        "refs": oracle_references(Phi_S, Phi_N, ref, min_loading=args.min_loading,
                                  snr_floor_db=args.snr_floor_db),
    }


def main():
    ap = argparse.ArgumentParser()
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
    ap.add_argument("--sharpen-exp", type=float, default=8.0)
    ap.add_argument("--win", type=str, default="rect")
    ap.add_argument("--rtf-alpha", type=float, default=0.999)
    ap.add_argument("--rtf-loading", type=float, default=1e-2)
    ap.add_argument("--n-eval", type=int, default=16)
    ap.add_argument("--min-loading", type=float, default=1e-9)
    ap.add_argument("--snr-floor-db", type=float, default=-20.0)
    ap.add_argument("--field", type=str, default="spherical")
    ap.add_argument("--agg", type=str, default="wmedian")
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--nu", type=float, default=1.0, help="1 = core subtract del ciego")
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--n-bands", type=int, default=12)
    ap.add_argument("--maxiter", type=int, default=120)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-cache", dest="cache", action="store_false")
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = {
        "fs": 16000, "duration": args.duration, "t_early": 0.050,
        "array_center": [3.0, 3.0, 1.2], "mird_spacing": args.spacing,
        "snr_db": args.snr_db,
        "source_path": f"{ROOT}/tools/data/signals/p002_emo_adoration_sentences.wav",
        "interf_paths": [f"{ROOT}/tools/data/signals/techno_gated commune.wav"],
        "stft_window": 512, "stft_overlap": 384, "dtln_model_path": MODEL_1,
    }
    args._provider = MirdDatasetProvider(root_dir=f"{ROOT}/tools/data/rirs/mird")

    specs = scene_grid(args.rt60, args.interf_angles, args.isir)
    if args.quick:
        specs = specs[:2]
    scenes = [prepare_scene(s, cfg, args) for s in specs]
    freqs = scenes[0]["freqs"]
    K = len(freqs)

    tr = [i for i, s in enumerate(specs) if s["rt60"] == args.rt60[0]]
    te = [i for i in range(len(specs)) if i not in tr] or tr
    train, test = [scenes[i] for i in tr], [scenes[i] for i in te]
    print(f"[*] train={[s['name'] for s in train]}")
    print(f"    test ={[s['name'] for s in test]}")

    _, _, bands = make_bands(freqs, n_bands=args.n_bands, f_min=60.0, f_max=7000.0)
    bands = [b for b in bands if b.size]
    if args.quick:
        bands = bands[:4]

    kw = dict(mu=args.mu, eta=args.eta, min_loading=args.min_loading, how=args.agg)

    def masks_of(scenes_, key, bi):
        return [(sc[key][0][bi], sc[key][1][bi]) for sc in scenes_]

    rows, t0 = [], time.time()
    for n, bi in enumerate(bands):
        tb = time.time()
        prod = band_objective_mask(train, bi, None, nu=args.nu, gamma=args.gamma,
                                   detail=True, masks=masks_of(train, "mask_prod", bi), **kw)
        caus = band_objective_mask(train, bi, None, nu=args.nu, gamma=args.gamma,
                                   detail=True, masks=masks_of(train, "mask_causal", bi), **kw)
        raw = band_objective_mask(train, bi, MASK_THETA0, nu=args.nu,
                                  gamma=args.gamma, detail=True, **kw)
        fit = fit_band_mask(train, bi, nu=args.nu, gamma=args.gamma,
                            maxiter=args.maxiter, **kw)
        prod_te = band_objective_mask(test, bi, None, nu=args.nu, gamma=args.gamma,
                                      detail=True, masks=masks_of(test, "mask_prod", bi), **kw)
        caus_te = band_objective_mask(test, bi, None, nu=args.nu, gamma=args.gamma,
                                      detail=True, masks=masks_of(test, "mask_causal", bi), **kw)
        fit_te = band_objective_mask(test, bi, fit["theta"], nu=args.nu,
                                     gamma=args.gamma, detail=True, **kw)
        a_s, b_s, a_n, b_n = fit["theta"]
        rows.append({"band": n, "f_lo": float(freqs[bi[0]]), "f_hi": float(freqs[bi[-1]]),
                     "n_bins": int(bi.size), "a_s": a_s, "b_s": b_s, "a_n": a_n, "b_n": b_n,
                     "L_prod": prod["L"], "L_causal": caus["L"], "L_raw": raw["L"],
                     "L_fit": fit["L"], "L_prod_test": prod_te["L"],
                     "L_causal_test": caus_te["L"], "L_fit_test": fit_te["L"],
                     "_bins": bi})
        print(f"  banda {n:2d} {rows[-1]['f_lo']:6.0f}-{rows[-1]['f_hi']:6.0f} Hz  "
              f"a_s={a_s:4.2f} b_s={b_s:+5.2f} | a_n={a_n:4.2f} b_n={b_n:+6.2f}   "
              f"L: prod {prod['L']:6.2f} | causal {caus['L']:6.2f} | cruda {raw['L']:6.2f} "
              f"| fit {fit['L']:6.2f}   TEST prod {prod_te['L']:6.2f} causal "
              f"{caus_te['L']:6.2f} fit {fit_te['L']:6.2f}  [{time.time()-tb:.0f}s]")

    print(f"\n[*] tiempo de ajuste: {(time.time()-t0)/60:.1f} min")
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    df.to_csv(os.path.join(args.out_dir, "warp_bands.csv"), index=False)

    out = {k: np.zeros(K) for k in ("a_s", "b_s", "a_n", "b_n")}
    out["a_s"] += 1.0; out["a_n"] += 1.0
    for r in rows:
        for k in ("a_s", "b_s", "a_n", "b_n"):
            out[k][r["_bins"]] = r[k]
    np.savez(os.path.join(args.out_dir, "warp_params.npz"), freqs=freqs, nu=args.nu,
             gamma=args.gamma, sharpen_exp=args.sharpen_exp, win=args.win,
             train=[s["name"] for s in train], test=[s["name"] for s in test], **out)

    print("\n" + "=" * 78)
    print("RESUMEN (mediana sobre bandas, dB de perdida contra el oracle)")
    print("=" * 78)
    for lbl, ktr, kte in (("produccion (pico global + stretch + pot)", "L_prod", "L_prod_test"),
                          ("causal    (escala fija + pot, sin stretch)", "L_causal", "L_causal_test"),
                          ("warp AJUSTADO (causal)                   ", "L_fit", "L_fit_test")):
        print(f"  {lbl}  train {df[ktr].median():6.2f}   test {df[kte].median():6.2f}")
    print(f"  mascara cruda sin warp (train)              {df['L_raw'].median():6.2f}")
    print(f"\n  costo de hacerla CAUSAL   : {df['L_causal_test'].median() - df['L_prod_test'].median():+.2f} dB")
    print(f"  ganancia del warp (causal): {df['L_causal_test'].median() - df['L_fit_test'].median():+.2f} dB")
    print(f"  fit causal vs produccion  : {df['L_prod_test'].median() - df['L_fit_test'].median():+.2f} dB")
    print(f"\n[ok] {args.out_dir}")


if __name__ == "__main__":
    main()
