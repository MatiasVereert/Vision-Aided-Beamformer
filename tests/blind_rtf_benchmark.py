"""
¿Se puede recuperar la ganancia del front-end DS SIN el DOA?

EL PLANTEO
----------
NM_MVDR_DSM (mascara estimada sobre un beamformer fijo) gana en 16/16 celdas,
pero para armar ese filtro fijo necesita la POSICION de la fuente: el estimador
de mascara deja de ser ciego. NM_MVDR_DSM_BLIND cierra el lazo: la RTF con la
que se apunta el front-end sale de la SCM de senal que la propia cadena ya
estima (Phi_SS = Phi_XX - Phi_NN), sin geometria ni DOA.

    mascara(1) = DTLN(x_ref)  ->  Phi_SS  ->  d  ->  y_fix  ->  mascara(2)  ->  BF

Los dos seguros del lazo son parametros nuevos y son lo que hay que barrer:
    rtf_loading  carga diagonal (relativa al nivel de ruido) de la matriz que se
                 usa para ESTIMAR. Grande = conservador: donde no hay senal
                 confiable, d -> e_ref y el front-end degrada al canal crudo, o
                 sea al sistema actual (piso garantizado).
    rtf_alpha    factor de olvido de esa recursion, aparte del alpha del BF.

FILAS
-----
    NM_MVDR_SUB    sin front-end (ciego)                    <- el piso a batir
    DSM_SUB_DS     front-end con DOA + geometria            <- el techo con DOA
    BLIND_CS/EVD/CW  front-end ciego, tres estimadores de RTF
    BLIND_MVDR     idem "cw" pero con front-end MVDR en vez de DS
    BLIND_IT2      dos vueltas del lazo

Todas comparten el resto de la cadena EXACTAMENTE (mismo core `subtract` con
mu=0, mismo alpha, mismo ref_mic, mismo framing, mismo sharpen).

USO
---
    python tests/blind_rtf_benchmark.py --quick
    python tests/blind_rtf_benchmark.py
    python tests/blind_rtf_benchmark.py --eps-sweep 1e-3 1e-2 1e-1
"""

import os
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

from propagation.mird_loader import MirdDatasetProvider
from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search
from evaluation.bf_wrappers import (NM_MVDR_SUB, NM_MVDR_DSM, NM_MVDR_DSM_BLIND,
                                    NM_MVDR_DSM_BLIND_PF)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT_DIR = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "blind_rtf")

ORDER = ["NM_MVDR_SUB", "DSM_SUB_DS", "BLIND_CS", "BLIND_EVD", "BLIND_CW",
         "BLIND_MVDR", "BLIND_IT2",
         # con post-filtro de sustraccion espectral (smooth=0.33)
         "SUB_PF", "BLIND_PF", "BLIND_PF_REF",
         # BAN (normalizacion analitica ciega) y su cruce con el post-filtro
         "DSM_SUB_DS_BAN", "BLIND_BAN", "BLIND_PF_BAN",
         # BAN sobre el core BASE, que es el regimen para el que fue pensado
         "BLIND_BASE", "BLIND_BASE_BAN"]


def build_processors(args):
    common = dict(min_loading=1e-9, alpha=0.99)
    sub = dict(mu=0.0)
    blind = dict(core="subtract", rtf_alpha=args.rtf_alpha,
                 rtf_loading=args.rtf_loading, **common, **sub)
    p = {
        "NM_MVDR_SUB": NM_MVDR_SUB(**common, **sub),
        "DSM_SUB_DS":  NM_MVDR_DSM(core="subtract", bf_mode="ds", **common, **sub),
        "BLIND_CS":    NM_MVDR_DSM_BLIND(rtf_mode="cs", **blind),
        "BLIND_EVD":   NM_MVDR_DSM_BLIND(rtf_mode="evd", **blind),
        "BLIND_CW":    NM_MVDR_DSM_BLIND(rtf_mode="cw", **blind),
        "BLIND_MVDR":  NM_MVDR_DSM_BLIND(rtf_mode="cw", w_mode="mvdr", **blind),
        "BLIND_IT2":   NM_MVDR_DSM_BLIND(rtf_mode="cs", n_iter=2, **blind),
        # --- post-filtro de sustraccion espectral, a igual smooth ---------
        # SUB_PF aisla el post-filtro (misma ganancia, sin front-end); BLIND_PF
        # lo alimenta con la mascara del front-end ciego; BLIND_PF_REF con la
        # del canal crudo (== el gate de NM_MVDR_PF sobre este beamformer).
        "SUB_PF":      NM_MVDR_SUB(smooth=args.smooth, **common, **sub),
        "BLIND_PF":    NM_MVDR_DSM_BLIND_PF(rtf_mode="cs", smooth=args.smooth, **blind),
        "BLIND_PF_REF": NM_MVDR_DSM_BLIND(rtf_mode="cs", smooth=args.smooth,
                                          pf_mask_src="ref", **blind),
        # --- BAN, solo y cruzado con el post-filtro -----------------------
        # Las cuatro esquinas del 2x2 (PF x BAN) sobre el mismo front-end ciego
        # son BLIND_CS / BLIND_PF / BLIND_BAN / BLIND_PF_BAN.
        "DSM_SUB_DS_BAN": NM_MVDR_DSM(core="subtract", bf_mode="ds", ban=True,
                                      **common, **sub),
        "BLIND_BAN":    NM_MVDR_DSM_BLIND(rtf_mode="cs", ban=True, **blind),
        "BLIND_PF_BAN": NM_MVDR_DSM_BLIND_PF(rtf_mode="cs", smooth=args.smooth,
                                             ban=True, **blind),
        # BAN nacio para corregir la escala del core ESTANDAR (el que degenera a
        # u/M en graves). El core `subtract` ya arregla eso por otra via, asi que
        # el par honesto para juzgar BAN es este, sobre core="base".
        "BLIND_BASE":     NM_MVDR_DSM_BLIND(rtf_mode="cs", core="base",
                                            rtf_alpha=args.rtf_alpha,
                                            rtf_loading=args.rtf_loading, **common),
        "BLIND_BASE_BAN": NM_MVDR_DSM_BLIND(rtf_mode="cs", core="base", ban=True,
                                            rtf_alpha=args.rtf_alpha,
                                            rtf_loading=args.rtf_loading, **common),
    }
    # Barrido de la carga de la estimacion: es EL parametro del lazo.
    if args.eps_sweep:
        for eps in args.eps_sweep:
            name = f"BLIND_CS_e{eps:g}"
            p[name] = NM_MVDR_DSM_BLIND(
                rtf_mode=args.rtf_mode, core="subtract", rtf_alpha=args.rtf_alpha,
                rtf_loading=eps, **common, **sub)
            if name not in ORDER:
                ORDER.append(name)
    if args.procs:
        missing = [n for n in args.procs if n not in p]
        if missing:
            raise SystemExit(f"procesadores desconocidos: {missing}\n"
                             f"disponibles: {list(p)}")
        p = {n: p[n] for n in args.procs}
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    ap.add_argument("--quick", action="store_true", help="2 celdas (plomeria)")
    ap.add_argument("--procs", type=str, nargs="+", default=None)
    ap.add_argument("--rtf-alpha", type=float, default=0.999)
    ap.add_argument("--rtf-loading", type=float, default=1e-2)
    ap.add_argument("--rtf-mode", type=str, default="cs")
    ap.add_argument("--smooth", type=float, default=0.33,
                    help="piso del post-filtro espectral (1.0 = sin filtro)")
    ap.add_argument("--eps-sweep", type=float, nargs="+", default=None,
                    help="valores de rtf_loading a barrer (agrega filas)")
    ap.add_argument("--duration", type=float, default=15)
    ap.add_argument("--no-catalog", dest="save_catalog", action="store_false")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    interpreter_1 = tf.lite.Interpreter(
        model_path=f"{PROJECT_ROOT}/src/dnn_denoise/models/model_quant_1.tflite")
    interpreter_1.allocate_tensors()
    interpreter_2 = tf.lite.Interpreter(
        model_path=f"{PROJECT_ROOT}/src/dnn_denoise/models/model_quant_2.tflite")
    interpreter_2.allocate_tensors()

    provider = MirdDatasetProvider(root_dir=f"{PROJECT_ROOT}/tools/data/rirs/mird")

    # Misma escena base que tests/ds_mask_benchmark.py, para poder comparar
    # numero contra numero con la corrida del front-end geometrico.
    base_config = {
        'fs': 16000,
        'duration': args.duration,
        't_early': 0.050,
        'array_center': [3.0, 3.0, 1.2],
        'mird_spacing': "3-3-3-8-3-3-3",
        'snr_db': 30.0,
        'source_path': f"{PROJECT_ROOT}/tools/data/signals/p002_emo_adoration_sentences.wav",
        'interf_paths': [f"{PROJECT_ROOT}/tools/data/signals/techno_gated commune.wav"],

        'wpe_taps': 7, 'wpe_delay': 2, 'wpe_alpha': 0.9999,
        'wpe_stft_size': 512, 'wpe_stft_shift': 128,
        'wpe_fixed_bits': None, 'wpe_fixed_round': 'nearest', 'wpe_backend': 'cov',
        'wpe_block_L': 512, 'wpe_block_shift': 2, 'wpe_block_iters': 2,
        'wpe_block_reg': 1e-6, 'wpe_block_solver': 'cholesky', 'wpe_block_mode': 'resolve',

        'stft_window': 512,
        'stft_overlap': 384,
        'eval_references': ['anechoic', 'early', 'reverberant'],
        'dtln_model_path': f"{PROJECT_ROOT}/src/dnn_denoise/models/model_quant_1.tflite",
    }

    param_grid = {
        'rt60': [0.360] if args.quick else [0.360, 0.610],
        'target_angle': [0],
        'target_dist': [1.0],
        'interf_configs': [[(45, 1.0)], [(90, 1.0)]] if args.quick
                          else [[(45, 1.0)], [(90, 1.0)]],
        'isir_db': [0] if args.quick else [-5, 0, 5, 10],
        'mismatch_gain': [0], 'mismatch_phase': [0],
        'use_wpe': [False], 'wpe_method': ['online'], 'wpe_taps': [7], 'wpe_delay': [2],
        'error_angle_deg': [0.0], 'error_distance_m': [0.0],
    }

    processors_dict = build_processors(args)
    print(f"[*] procesadores: {list(processors_dict)}")

    df = run_mird_grid_search(
        grid_params=param_grid,
        dataset_provider=provider,
        processors=processors_dict,
        scene_base_config=base_config,
        output_dir=args.out_dir,
        interpreter_1=interpreter_1,
        interpreter_2=interpreter_2,
        apply_dtln_post=False,
        save_catalog=args.save_catalog,
    )

    cols = [c for c in ["Delta_tot_PESQ_early", "Delta_tot_STOI_early",
                        "Delta_tot_SDR_early", "Delta_tot_SIR_early",
                        "Delta_tot_SAR_early", "Delta_tot_SINR_early"]
            if c in df.columns]
    if not cols:
        print("[!] no hay columnas Delta_tot_*_early en el resultado")
        return

    order = [p for p in ORDER if p in df["processor"].unique()]
    base_name = "NM_MVDR_SUB"

    def _show(title, sub):
        if not len(sub):
            return
        # MEDIANA: SIR/SAR tienen colas enormes a RT bajo.
        t = sub.groupby("processor")[cols].median().round(3)
        t = t.reindex([p for p in order if p in t.index])
        print(f"\n=== {title}  ({sub['processor'].value_counts().iloc[0]} celdas) ===")
        print(t.to_string())
        if base_name in t.index:
            print(f"\n  -- delta contra {base_name} (ciego, sin front-end) --")
            for p_ in t.index:
                if p_ == base_name:
                    continue
                d = (t.loc[p_] - t.loc[base_name]).round(3)
                print(f"  {p_:18s} " + "  ".join(
                    f"{c.replace('Delta_tot_','').replace('_early',''):5s}{d[c]:+7.3f}"
                    for c in cols))

    print("\n" + "=" * 78)
    print("MEDIANA sobre las celdas (referencia early)")
    print("=" * 78)
    _show("TODAS LAS CELDAS", df)
    if "rt60" in df.columns:
        for rt in sorted(df["rt60"].unique()):
            _show(f"rt60 = {rt}", df[df["rt60"] == rt])
    if "isir_db" in df.columns and not args.quick:
        for isir in sorted(df["isir_db"].unique()):
            _show(f"iSIR = {isir} dB", df[df["isir_db"] == isir])

    if base_name in df["processor"].unique():
        key = [c for c in ("rt60", "interf_configs", "isir_db") if c in df.columns]
        base = df[df["processor"] == base_name].set_index(key)
        print("\n" + "=" * 78)
        print(f"CELDAS GANADAS contra {base_name}")
        print("=" * 78)
        for p_ in order:
            if p_ == base_name:
                continue
            cur = df[df["processor"] == p_].set_index(key)
            wins = {c.replace("Delta_tot_", "").replace("_early", ""):
                    int((cur[c] > base[c].reindex(cur.index)).sum()) for c in cols}
            print(f"  {p_:18s} n={len(cur):2d}  " +
                  "  ".join(f"{k}:{v}" for k, v in wins.items()))

    print(f"\n[ok] {args.out_dir}")


if __name__ == "__main__":
    main()
