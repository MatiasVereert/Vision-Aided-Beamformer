"""
¿Estimar la mascara sobre la salida de un BEAMFORMER FIJO mejora las SCM?

LA PROPUESTA
------------
Hoy la mascara sale de correr el DTLN sobre UN canal crudo, o sea que el
estimador ve la escena con la SNR de un microfono solo mientras que el
beamformer que consume esa mascara tiene M. Este banco mide que pasa si antes
del DTLN se mete un filtro FIJO apuntado al target (delay-and-sum, o
superdirectivo): la senal que entra al DTLN gana hasta 10 log10(M) dB de SNR
sin estimar nada de la senal -- y por lo tanto sin poder realimentar errores de
la propia mascara.

LA PREGUNTA DE LA PROYECCION HACIA ATRAS
----------------------------------------
La mascara NO filtra ninguna senal en estos cores: PONDERA un promedio de outer
products de la senal multicanal. Por eso un factor por bin sobre la mascara se
CANCELA EXACTO (numerador y denominador llevan la misma mascara) y no hace falta
ninguna "prediccion por canal": la mascara ya es un escalar por (k,t) para los M
canales. Lo unico que si cambia es el PUNTO DE OPERACION -- el DTLN ve una SNR
mejorada en AG(k) dB y devuelve una mascara optimista, con m_n = 1 - m_s
demasiado chica -> Phi_NN estimada con menos frames efectivos. La correccion
tiene que ser NO LINEAL, y la natural es en el dominio de la SNR:

    logit(m_ref) = logit(m_fix) - beta ln AG(k)

que es `warp_mask(a=1, b=-beta ln AG)`: un punto de la familia que ya ajusta el
banco de calibracion, pero con b_k fijado por la GEOMETRIA. `beta` se barre
(0.5 = mascara leida como razon de amplitudes, 1.0 = como ganancia de Wiener).
Ver `beamforming/mask/ds_mask.py`.

QUE SE MIDE (esta corrida NO toca el benchmark: mide la ESTIMACION)
-------------------------------------------------------------------
Contra las SCM oracle (senales limpias multicanal, mismo alpha), por banda:

    leak_db  SNR de las celdas que selecciona mask_n. MAS BAJO = MEJOR: es la
             fuga de voz dentro de Phi_NN, el mecanismo de auto-cancelacion.
    pick_db  SNR de las celdas que selecciona mask_s. MAS ALTO = MEJOR.
    cmd_NN   Correlation Matrix Distance entre Phi_NN estimada y Phi_N oracle.
    cmd_XX   idem entre Phi_XX enmascarada y Phi_S oracle.
    rtf_deg  angulo entre el autovector principal de Phi_XX y la RTF oracle.
    L        perdida del beamformer contra el oracle (L_sinr + eta L_dist), la
             unica que se traduce a las metricas. Se reporta para nu=0 (core
             base NM_MVDR) y nu=1 (NM_MVDR_SUB).

Las cuatro primeras son diagnostico de la SCM; L es el veredicto. Si leak_db
baja pero L no se mueve, la mascara mejoro donde no importaba.

FILAS
-----
    ref          mascara ACTUAL (DTLN sobre el canal de referencia)  <- baseline
    ds_bBETA     DTLN sobre el delay-and-sum, proyeccion con ese beta
    sd_bBETA     idem con el superdirectivo cargado (--with-sd)
y cada una con el post-proceso ACTUAL (stretch min-max + **4) y con la mascara
CRUDA (identidad), para que no se confunda el efecto del front-end con el del
post-proceso.

USO
---
    python tests/ds_mask_scm_run.py --quick
    python tests/ds_mask_scm_run.py
    python tests/ds_mask_scm_run.py --with-sd --beta 0 0.5 1.0

Salida: tests/dataset_out/ds_mask/
    ds_mask_bands.csv    todas las metricas por banda y condicion
    ds_mask_global.csv   resumen agregado sobre todo el eje
    ds_mask.png          panel de diagnostico
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
from beamforming.mask.dtln_masks import get_dtln_masks_soft
from beamforming.mask.oracle_masks import _stft_mag_blocks
from beamforming.mask.ds_mask import (
    fixed_bf_signal, array_gain, backproject_mask, stretch_sharpen,
)
from beamforming.mask.scm_calibration import (
    make_bands, eval_frame_indices, snapshot_scms_masked, snapshot_scms_oracle,
    oracle_references, parametric_weights, weight_loss, scm_fidelity,
    mask_separation_db, diffuse_coherence, _aggregate,
)
from scm_calibration_run import scene_grid, PROJECT_ROOT
from lowfreq_diagnostic_run import build_scene

OUT_DIR = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "ds_mask")
CACHE_DIR = os.path.join(OUT_DIR, "cache")


# =====================================================================
# Helpers
# =====================================================================
def _irm_blocks(tgt, noi, block_len, block_shift):
    """
    IRM en potencia (|S|^2 / (|S|^2 + |N|^2)) y peso por celda (|S|^2 + |N|^2),
    calculadas con el framing EXACTO del DTLN (bloques rectangulares, mismo
    buffer deslizante). Es la mascara ideal en el dominio de la senal que se le
    pasa: para el canal de referencia es el objetivo del sistema; para la salida
    de un filtro fijo es lo que el DTLN "deberia" devolver ahi.
    """
    ps = _stft_mag_blocks(np.ascontiguousarray(tgt, dtype=np.float32),
                          block_len, block_shift).astype(np.float64) ** 2
    pn = _stft_mag_blocks(np.ascontiguousarray(noi, dtype=np.float32),
                          block_len, block_shift).astype(np.float64) ** 2
    return ps / (ps + pn + 1e-30), ps + pn


def _shift(m, s):
    """
    Corre la mascara `s` frames hacia atras en el tiempo (repitiendo el ultimo
    frame). s = 1 compensa el desfasaje medido entre el framing del DTLN y el de
    la STFT del beamformer: el bloque i del DTLN cubre las mismas muestras que
    el frame i-1 de scipy.stft, pero el pipeline los aparea en el MISMO indice.
    """
    if s == 0:
        return m
    return np.concatenate([m[:, s:], np.repeat(m[:, -1:], s, axis=1)], axis=1)


# =====================================================================
# Escena
# =====================================================================
def prepare_scene(spec, cfg, args):
    """
    Escena MIRD -> STFTs -> las TRES mascaras crudas (canal de referencia, DS,
    superdirectivo) -> snapshots oracle. Se cachea todo lo caro.

    Las mascaras crudas se guardan SIN post-proceso: el stretch/sharpen y la
    proyeccion hacia atras son baratos y se aplican despues, asi que un solo
    cache sirve para todos los beta.
    """
    tag = spec["tag"]
    cache = os.path.join(CACHE_DIR, f"{tag}_{args.spacing}_a{args.alpha:g}_"
                                    f"E{args.n_eval}_snr{args.snr_db:g}_"
                                    f"d{args.duration:g}_L{args.sd_loading:g}_v3.npz")
    hop = cfg["stft_window"] - cfg["stft_overlap"]
    start_frame = int(min(5.0, args.duration * 0.3) * cfg["fs"] / hop)

    if args.cache and os.path.exists(cache):
        z = np.load(cache)
        print(f"[cache] {tag}")
        d = {k: z[k] for k in z.files}
        d["X_stft"] = d["X_stft"].astype(np.complex128)
    else:
        print(f"[*] construyendo {tag}")
        scene_cfg = dict(cfg)
        mic_coords, mixture, o_tgt, o_noi, _ = build_scene(
            scene_cfg, args._provider, spec["rt60"], args.target_angle,
            args.target_dist, [(spec["interf_angle"], args.interf_dist)],
            spec["isir"], args.snr_db)
        source_pos = np.asarray(scene_cfg["source_pos"]).reshape(1, 3)
        nperseg, noverlap = cfg["stft_window"], cfg["stft_overlap"]
        M = mixture.shape[0]
        ref_mic = M // 2

        def _stft(x):
            f_, _, Z = sig.stft(x, fs=cfg["fs"], window="hamming",
                                nperseg=nperseg, noverlap=noverlap, nfft=nperseg)
            return f_, np.transpose(Z, (1, 2, 0))

        freqs, X_stft = _stft(mixture)
        _, S_stft = _stft(o_tgt)
        _, N_stft = _stft(o_noi)

        # --- mascaras crudas: una por front-end ------------------------------
        # Se guarda ademas la potencia de target y de ruido A LA SALIDA de cada
        # front-end: es la SNR que el DTLN realmente recibe, y sin ese numero no
        # se puede interpretar nada de lo que sigue. Como el filtro es LINEAL,
        # aplicarselo por separado a las componentes oracle es exacto.
        masks_raw, ags, wngs, sp, npw = {}, {}, {}, {}, {}
        masks_raw["ref"], _ = get_dtln_masks_soft(
            mixture, ref_mic, cfg["dtln_model_path"], block_len=nperseg,
            block_shift=hop)
        ags["ref"] = np.ones(len(freqs))
        wngs["ref"] = np.ones(len(freqs))
        sp["ref"] = np.sum(np.abs(S_stft[:, :, ref_mic]) ** 2, axis=1)
        npw["ref"] = np.sum(np.abs(N_stft[:, :, ref_mic]) ** 2, axis=1)

        # IRM por front-end, en el framing NATIVO del DTLN (bloques rectangulares):
        # es contra esto que hay que comparar la mascara para saber si el DTLN
        # estima bien "en su dominio" o si lo que falla es el cambio de dominio.
        irm = {}
        irm["ref"] = _irm_blocks(o_tgt[ref_mic], o_noi[ref_mic], nperseg, hop)

        for mode in ("ds", "sd"):
            y_fix, w_fix, f_fix = fixed_bf_signal(
                mixture, mic_coords, source_pos, cfg["fs"], ref_mic_idx=ref_mic,
                nperseg=nperseg, noverlap=noverlap, mode=mode,
                loading=args.sd_loading, field=args.field)
            ag, wng = array_gain(w_fix, mic_coords, f_fix, field=args.field)
            ags[mode] = np.clip(ag, 1.0, float(M))
            wngs[mode] = wng
            sp[mode] = np.sum(np.abs(np.einsum("km,ktm->kt", w_fix.conj(), S_stft)) ** 2, axis=1)
            npw[mode] = np.sum(np.abs(np.einsum("km,ktm->kt", w_fix.conj(), N_stft)) ** 2, axis=1)
            masks_raw[mode], _ = get_dtln_masks_soft(
                y_fix[None, :], 0, cfg["dtln_model_path"], block_len=nperseg,
                block_shift=hop)
            # el mismo filtro sobre las componentes limpias (es lineal: exacto)
            yt = fixed_bf_signal(o_tgt, mic_coords, source_pos, cfg["fs"],
                                 ref_mic_idx=ref_mic, nperseg=nperseg,
                                 noverlap=noverlap, mode=mode,
                                 loading=args.sd_loading, field=args.field)[0]
            yn = fixed_bf_signal(o_noi, mic_coords, source_pos, cfg["fs"],
                                 ref_mic_idx=ref_mic, nperseg=nperseg,
                                 noverlap=noverlap, mode=mode,
                                 loading=args.sd_loading, field=args.field)[0]
            irm[mode] = _irm_blocks(yt, yn, nperseg, hop)

        T0 = min([X_stft.shape[1], S_stft.shape[1]] +
                 [m.shape[1] for m in masks_raw.values()])
        ev = eval_frame_indices(T0, args.n_eval, start_frame=start_frame)
        Phi_S, Phi_N = snapshot_scms_oracle(S_stft[:, :T0], N_stft[:, :T0], ev,
                                            alpha=args.alpha)

        d = {"X_stft": X_stft[:, :T0], "Phi_S": Phi_S, "Phi_N": Phi_N,
             "eval_frames": ev, "freqs": freqs, "mic_coords": mic_coords,
             "ref_mic": ref_mic,
             "ps": np.abs(S_stft[:, :T0, ref_mic]) ** 2,
             "pn": np.abs(N_stft[:, :T0, ref_mic]) ** 2}
        for k in masks_raw:
            d[f"m_{k}"] = masks_raw[k][:, :T0]
            d[f"ag_{k}"] = ags[k]
            d[f"wng_{k}"] = wngs[k]
            d[f"sp_{k}"] = sp[k]
            d[f"np_{k}"] = npw[k]
            d[f"irm_{k}"] = irm[k][0][:, :T0].astype(np.float32)
            d[f"iw_{k}"] = irm[k][1][:, :T0].astype(np.float32)
        if args.cache:
            os.makedirs(CACHE_DIR, exist_ok=True)
            np.savez(cache, **{**d, "X_stft": d["X_stft"].astype(np.complex64)})

    ref_mic = int(d["ref_mic"])
    d["name"] = tag
    d["refs"] = oracle_references(d["Phi_S"], d["Phi_N"], ref_mic,
                                  min_loading=args.min_loading,
                                  snr_floor_db=args.snr_floor_db)
    d["Gamma"] = diffuse_coherence(d["mic_coords"], d["freqs"], field=args.field)
    return d


# =====================================================================
# Evaluacion de una condicion sobre una escena
# =====================================================================
def evaluate(sc, mask_s, mask_n, args):
    """
    Corre la recursion con esas mascaras y devuelve, por celda (bin, frame de
    evaluacion), todo lo que se va a agregar despues.
    """
    ref_mic = int(sc["ref_mic"])
    Phi_XX, Phi_NN = snapshot_scms_masked(sc["X_stft"], mask_s, mask_n,
                                          sc["eval_frames"], alpha=args.alpha)
    out = {}
    fid_x = scm_fidelity(Phi_XX, sc["Phi_S"])
    fid_n = scm_fidelity(Phi_NN, sc["Phi_N"])
    out["cmd_XX"] = fid_x["cmd"]
    out["cmd_NN"] = fid_n["cmd"]
    out["rtf_deg"] = fid_x["evec_deg"]

    # nu = 0 (core base NM_MVDR) y nu = 1 (NM_MVDR_SUB): la mascara es la misma,
    # cambia solo que se le pasa a la formula de Souden.
    for nu in (0.0, 1.0):
        W, _ = parametric_weights(Phi_XX, Phi_NN, None, ref_mic, nu=nu, gamma=0.0,
                                  mu=args.mu, min_loading=args.min_loading)
        L = weight_loss(W, sc["Phi_S"], sc["Phi_N"], sc["refs"], eta=args.eta)
        tag = "nu0" if nu == 0.0 else "nu1"
        out[f"L_{tag}"] = L["L"]
        out[f"Lsinr_{tag}"] = L["L_sinr"]
        out[f"Ldist_{tag}"] = L["L_dist"]

    # separacion de la mascara: numerador/denominador SIN dividir, para poder
    # agregar por banda sumando energias.
    S_ref = np.sqrt(sc["ps"])[:, :, None]      # (K,T,1) magnitudes, canal ref
    N_ref = np.sqrt(sc["pn"])[:, :, None]
    out["pick_num"], out["pick_den"] = mask_separation_db(mask_s, S_ref, N_ref, 0,
                                                          return_parts=True)
    out["leak_num"], out["leak_den"] = mask_separation_db(mask_n, S_ref, N_ref, 0,
                                                          return_parts=True)
    out["pow_S"] = sc["refs"]["pow_S"]
    out["bin_pow"] = np.sum(sc["ps"], axis=1)
    return out


def build_conditions(sc, args):
    """
    (nombre, mask_s, mask_n) para cada front-end x beta x post-proceso.

    El front-end HIBRIDO ("mix") toma mask_s del filtro fijo y mask_n del canal
    de referencia. No es una propuesta: es la ABLACION que separa las dos ramas.
    Phi_XX y Phi_NN no piden lo mismo -- una quiere celdas donde domina la voz,
    la otra celdas donde domina el ruido -- y un front-end que sube la SNR de
    entrada las ayuda de forma OPUESTA: le da mas contraste a la rama de voz y
    le esconde el ruido a la rama de ruido. Si el hibrido gana, el problema es
    de rama, no del front-end.
    """
    conds = []
    fronts = ["ref", "ds"] + (["sd"] if args.with_sd else []) + ["mix", "oracle"]
    combos = []
    for fe in fronts:
        betas = list(args.beta) if fe in ("ds", "sd") else [0.0]
        combos += [(fe, b, args.mask_shift) for b in betas]
    if args.shift_probe:
        # misma mascara, corrida 1 frame: aisla el desfasaje DTLN vs STFT del
        # resto del experimento (afecta por igual a todas las condiciones).
        combos += [(fe, 0.0, args.mask_shift + 1) for fe in ("ref", "ds")]

    for fe, b, sh in combos:
            if fe == "mix":
                m_s = sc["m_ds"]
                m_n = 1.0 - sc["m_ref"]
            elif fe == "oracle":
                m_s = sc["irm_ref"].astype(np.float64)
                m_n = 1.0 - m_s
            else:
                m_s = backproject_mask(sc[f"m_{fe}"], sc[f"ag_{fe}"], beta=b)
                m_n = 1.0 - m_s
            m_s, m_n = _shift(m_s, sh), _shift(m_n, sh)
            base = fe if b == 0.0 else f"{fe}_b{b:g}"
            if sh != args.mask_shift:
                base = f"{base}_s{sh}"
            if args.post in ("cur", "both"):
                # stretch_sharpen replica el camino ACTUAL (stretch global + **4);
                # en el hibrido hay que aplicarlo por rama porque las dos vienen
                # de mascaras crudas distintas.
                ms = stretch_sharpen(m_s, sharpen_exp=args.sharpen_exp)[0]
                mn = stretch_sharpen(1.0 - m_n, sharpen_exp=args.sharpen_exp)[1]
                conds.append((f"{base}|cur", ms, mn))
            if args.post in ("raw", "both"):
                conds.append((f"{base}|raw", m_s, m_n))
    return conds


# =====================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rt60", type=float, nargs="+", default=[0.360, 0.610])
    ap.add_argument("--interf-angles", type=float, nargs="+", default=[45, 90])
    ap.add_argument("--isir", type=float, nargs="+", default=[0])
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
    ap.add_argument("--agg", type=str, default="wmedian",
                    choices=["wmedian", "median", "wmean", "mean"])
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--beta", type=float, nargs="+", default=[0.0, 0.5, 1.0],
                    help="fuerza de la proyeccion hacia atras (0 = sin correccion)")
    ap.add_argument("--post", type=str, default="both", choices=["cur", "raw", "both"])
    ap.add_argument("--mask-shift", type=int, default=0,
                    help="corrimiento de la mascara en frames (0 = pipeline actual)")
    ap.add_argument("--no-shift-probe", dest="shift_probe", action="store_false",
                    help="no agregar las filas con la mascara corrida 1 frame")
    ap.add_argument("--with-sd", action="store_true",
                    help="agrega el front-end superdirectivo cargado")
    ap.add_argument("--sd-loading", type=float, default=1e-2)
    ap.add_argument("--n-bands", type=int, default=12)
    ap.add_argument("--f-min", type=float, default=60.0)
    ap.add_argument("--f-max", type=float, default=7000.0)
    ap.add_argument("--quick", action="store_true", help="1 escena, 6 s")
    ap.add_argument("--no-cache", dest="cache", action="store_false")
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    args = ap.parse_args()

    if args.quick:
        args.duration = 6.0
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
        specs = specs[:1]
    t0 = time.time()
    scenes = [prepare_scene(s, cfg, args) for s in specs]
    freqs = scenes[0]["freqs"]
    print(f"[*] {len(scenes)} escenas listas en {time.time()-t0:.0f}s")

    _, _, bands = make_bands(freqs, n_bands=args.n_bands, f_min=args.f_min,
                             f_max=args.f_max)
    bands = [b for b in bands if b.size]

    # --- acuerdo de cada mascara con la IRM, en los DOS dominios ------------
    # mae_ref  : contra la IRM del canal de REFERENCIA (lo que el sistema quiere)
    # mae_own  : contra la IRM del dominio DONDE se estimo (lo que el DTLN ve)
    # Si mae_own es bueno y mae_ref malo, el DTLN funciona y lo que falla es el
    # cambio de dominio -> tiene sentido buscar una proyeccion hacia atras. Si
    # los dos empeoran, el que se degrada es el estimador y no hay proyeccion
    # que lo arregle.
    mae = {}
    import copy as _copy
    args_raw = _copy.copy(args)
    args_raw.post = "raw"        # el acuerdo con la IRM se mide sobre la mascara
                                 # SIN post-proceso (m**4 no es comparable a una IRM)
    for sc in scenes:
        for name, ms, mn in build_conditions(sc, args_raw):
            base = name.split("|")[0]
            if base in mae and len(mae[base]) > len(scenes) - 1:
                continue
            fe = base.split("_")[0]
            probe = ms if name.endswith("|raw") else None
            if probe is None:
                continue
            acc = mae.setdefault(base, [])
            out = {}
            for dom in ("ref", fe if fe in ("ds", "sd") else "ref"):
                irm = sc[f"irm_{dom}"].astype(np.float64)
                wgt = sc[f"iw_{dom}"].astype(np.float64)
                Tn = min(probe.shape[1], irm.shape[1])
                e = np.abs(probe[:, :Tn] - irm[:, :Tn]) * wgt[:, :Tn]
                key = "ref" if dom == "ref" and "own" in out else dom
                out.setdefault("num_" + ("ref" if dom == "ref" else "own"),
                               np.sum(e, axis=1))
                out.setdefault("den_" + ("ref" if dom == "ref" else "own"),
                               np.sum(wgt[:, :Tn], axis=1))
            if "num_own" not in out:
                out["num_own"], out["den_own"] = out["num_ref"], out["den_ref"]
            acc.append(out)

    # --- evaluar cada condicion sobre cada escena ---------------------------
    results = {}                       # nombre -> lista de dicts (una por escena)
    names = [c[0] for c in build_conditions(scenes[0], args)]
    for name in names:
        results[name] = []
    for sc in scenes:
        for name, ms, mn in build_conditions(sc, args):
            t1 = time.time()
            results[name].append(evaluate(sc, ms, mn, args))
            print(f"\r  {sc['name']:24s} {name:16s} [{time.time()-t1:.0f}s]")

    # --- agregacion ---------------------------------------------------------
    CELL = ["cmd_XX", "cmd_NN", "rtf_deg", "L_nu0", "Lsinr_nu0", "Ldist_nu0",
            "L_nu1", "Lsinr_nu1", "Ldist_nu1"]

    def agg_rows(bin_idx, label, f_lo, f_hi, f_c, sel=None):
        """`sel` = subconjunto de indices de escena (None = todas)."""
        sel = list(range(len(scenes))) if sel is None else list(sel)
        out = []
        for name in names:
            row = {"cond": name, "band": label, "f_lo": f_lo, "f_hi": f_hi,
                   "f_c": f_c, "n_bins": int(np.size(bin_idx))}
            res = [results[name][i] for i in sel]
            for key in CELL:
                vals = [r[key][bin_idx] for r in res]
                wts = [r["pow_S"][bin_idx] for r in res]
                row[key] = _aggregate(vals, args.agg, wts)
            base = name.split("|")[0]
            acc = [mae[base][i] for i in sel] if base in mae else []
            for tag in ("ref", "own"):
                nm = sum(float(np.sum(a[f"num_{tag}"][bin_idx])) for a in acc)
                dn = sum(float(np.sum(a[f"den_{tag}"][bin_idx])) for a in acc)
                row[f"mae_{tag}"] = nm / (dn + 1e-30) if dn > 0 else np.nan
            fe = name.split("|")[0].split("_b")[0].split("_s")[0]
            fe_s = "ds" if fe == "mix" else ("ref" if fe == "oracle" else fe)
            num = sum(float(np.sum(scenes[i][f"sp_{fe_s}"][bin_idx])) for i in sel)
            den = sum(float(np.sum(scenes[i][f"np_{fe_s}"][bin_idx])) for i in sel)
            row["snr_in_db"] = 10.0 * np.log10((num + 1e-30) / (den + 1e-30))
            for tag in ("pick", "leak"):
                num = sum(float(np.sum(r[f"{tag}_num"][bin_idx])) for r in res)
                den = sum(float(np.sum(r[f"{tag}_den"][bin_idx])) for r in res)
                row[f"{tag}_db"] = 10.0 * np.log10((num + 1e-30) / (den + 1e-30))
            out.append(row)
        return out

    rows = []
    for n, bi in enumerate(bands):
        rows += agg_rows(bi, n, float(freqs[bi[0]]), float(freqs[bi[-1]]),
                         float(np.sqrt(max(freqs[bi[0]], 0.5 * freqs[1]) *
                                       max(freqs[bi[-1]], freqs[1]))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "ds_mask_bands.csv"), index=False)

    allbins = np.arange(len(freqs))
    gdf = pd.DataFrame(agg_rows(allbins, -1, float(freqs[0]), float(freqs[-1]), np.nan))
    gdf.to_csv(os.path.join(args.out_dir, "ds_mask_global.csv"), index=False)

    # --- resumen ------------------------------------------------------------
    base = gdf[gdf["cond"] == f"ref|{'cur' if args.post != 'raw' else 'raw'}"].iloc[0]
    print("\n" + "=" * 108)
    print(f"GLOBAL (agg={args.agg}, eta={args.eta}, escenas={[s['name'] for s in scenes]})")
    print("=" * 108)
    print(f"{'condicion':18s} {'snr_in':>7s} {'leak_db':>8s} {'pick_db':>8s} {'mae_ref':>8s} {'mae_own':>8s} "
          f"{'cmd_NN':>8s} {'rtf_deg':>8s} {'L_nu0':>8s} {'dL_nu0':>8s} "
          f"{'L_nu1':>8s} {'dL_nu1':>8s}")
    for _, r in gdf.iterrows():
        print(f"{r['cond']:18s} {r['snr_in_db']:7.2f} {r['leak_db']:8.2f} {r['pick_db']:8.2f} "
              f"{r['mae_ref']:8.4f} {r['mae_own']:8.4f} "
              f"{r['cmd_NN']:8.4f} {r['rtf_deg']:8.2f} "
              f"{r['L_nu0']:8.3f} {base['L_nu0']-r['L_nu0']:+8.3f} "
              f"{r['L_nu1']:8.3f} {base['L_nu1']-r['L_nu1']:+8.3f}")
    print("  snr_in: SNR de la senal que recibe el DTLN (lo que promete el front-end)")
    print("  leak_db: MAS BAJO mejor (fuga de voz en Phi_NN) | pick_db: mas alto mejor")
    print("  mae_ref/mae_own: error contra la IRM del canal de referencia / del propio dominio")
    print("  cmd_*: 0 = SCM identica al oracle salvo escala | dL: ganancia contra 'ref' (dB, + mejor)")

    # --- panel --------------------------------------------------------------
    show = [n for n in names if n.endswith("|cur")] or names
    # estilos EXPLICITOS: con >10 series el ciclo de colores de matplotlib se
    # repite y dos condiciones distintas quedan del mismo color (se leen mal).
    cmap = plt.get_cmap("tab20")
    style = {}
    for i, n in enumerate(show):
        if n.startswith("oracle"):
            style[n] = dict(color="k", ls="--", lw=2.0, marker="")
        elif n.startswith("ref") and "_s" not in n:
            style[n] = dict(color="tab:red", ls="-", lw=2.0, marker="o", ms=3)
        else:
            style[n] = dict(color=cmap(i % 20), ls="-" if "_b" not in n else ":",
                            lw=1.2, marker="o", ms=3)
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax = ax.ravel()
    for fe, sty in (("ds", "-"), ("sd", "--")):
        if f"ag_{fe}" in scenes[0]:
            ax[0].semilogx(freqs[1:], 10 * np.log10(scenes[0][f"ag_{fe}"][1:]),
                           sty, label=f"AG difusa {fe.upper()}")
            ax[0].semilogx(freqs[1:], 10 * np.log10(scenes[0][f"wng_{fe}"][1:]),
                           sty, alpha=0.4, label=f"WNG {fe.upper()}")
    ax[0].axhline(10 * np.log10(scenes[0]["X_stft"].shape[2]), color="gray", ls=":",
                  label="10 log M")
    ax[0].set_title("Ganancia del filtro fijo (lo que gana el DTLN)")
    ax[0].set_ylabel("dB")
    for key, a, ttl in (("leak_db", ax[1], "Fuga de voz en $\\Phi_{NN}$ (mas bajo mejor)"),
                        ("cmd_NN", ax[2], "CMD($\\Phi_{NN}$, oracle)"),
                        ("L_nu0", ax[3], "Perdida contra el oracle, nu=0")):
        for name in show:
            d = df[df["cond"] == name]
            a.semilogx(d["f_c"], d[key], label=name, **style[name])
        a.set_title(ttl)
        a.set_ylabel(key)
    for a in ax:
        a.set_xlabel("frecuencia [Hz]")
        a.grid(True, which="both", alpha=0.3)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "ds_mask.png"), dpi=130)
    plt.close(fig)
    # --- por escena: ¿la ganancia es sistematica o se la come la varianza? ---
    srows = []
    for i, sc in enumerate(scenes):
        for r in agg_rows(allbins, -1, float(freqs[0]), float(freqs[-1]), np.nan, sel=[i]):
            r["scene"] = sc["name"]
            srows.append(r)
    sdf = pd.DataFrame(srows)
    sdf.to_csv(os.path.join(args.out_dir, "ds_mask_by_scene.csv"), index=False)

    base_name = f"ref|{'cur' if args.post != 'raw' else 'raw'}"
    piv = sdf.pivot(index="cond", columns="scene", values="L_nu0")
    piv = piv.loc[[c for c in names if c in piv.index]]
    dpiv = piv.loc[base_name] - piv
    print("\nPOR ESCENA -- ganancia de L_nu0 contra el baseline [dB, + mejor]")
    print(dpiv.round(2).to_string())
    print(f"\n[ok] {args.out_dir}")


if __name__ == "__main__":
    main()
