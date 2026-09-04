"""
¿El post-filtro recupera el SIR que pierde la mascara calibrada?

DE DONDE VIENE
--------------
La calibracion de mascara (NM_MVDR_MCAL_C: voz = mascara CRUDA del DTLN, ruido =
odds-ratio recortado sigma(2*logit(1-m) - 8)) corre sobre EL MISMO beamformer que
NM_MVDR_SUB (nu=1 == core de sustraccion); lo unico que cambia es la estimacion
de Phi_NN. Medido sobre 16 celdas MIRD, celdas no vistas, contra NM_MVDR_SUB:

    PESQ +0.000   STOI +0.008   SDR +1.14   SIR -1.18   SAR +1.49

Se probo recuperar ese SIR desde adentro del beamformer y NO se puede: el
barrido de eta no mueve el optimo (0.03 dB entre eta=0 y eta=4) y el barrido de
a_n muestra que los dos terminos de la loss se minimizan en el MISMO punto
(a_n ~ 2-2.5) -- no hay frontera de Pareto que recorrer, hay un minimo conjunto.

El SIR que se pierde es supresion de INTERFERENTE RESIDUAL, y eso es
exactamente lo que una ganancia espectral por bin recupera barato. De ahi esta
prueba.

DOS POST-FILTROS, EN CASCADA
----------------------------
  1. Post-filtro ESPECTRAL del propio wrapper (`smooth`): G = smooth +
     (1-smooth)*mask_suave, aplicado sobre la salida del beamformer. Es la etapa
     de produccion (la de NM_MVDR_PF).
  2. DTLN COMPLETO como post-filtro (`apply_dtln_post=True` del benchmark): los
     dos nucleos de la red sobre la senal de salida. Da las columnas
     `Delta_tot_pipeline_*` ademas de las `Delta_tot_*` del beamformer solo.

O sea que cada procesador se mide en DOS puntos de la cadena, y el CSV permite
ver si el post-filtrado cierra el hueco de SIR y a que costo en SAR/PESQ.

PROCESADORES (6)
----------------
    NM_MVDR          sistema actual (sin sustraccion)          -- referencia baja
    NM_MVDR_SUB      sustraccion, mascara actual               -- el mejor en SIR
    NM_MVDR_SUB_PF   idem + post-filtro espectral
    MCAL_C           sustraccion, mascara CALIBRADA            -- la propuesta
    MCAL_C_PF        idem + post-filtro espectral
    ORACLE_SCM       covarianzas de las senales limpias        -- cota superior

Las comparaciones que deciden:
    MCAL_C_PF  vs  NM_MVDR_SUB_PF   (con post-filtro, ¿sigue el hueco de SIR?)
    *_pipeline vs  *                (¿cuanto agrega el DTLN completo encima?)

PESOS
-----
Se guardan los pesos w(k,t,m) del beamformer (ANTES de cualquier post-filtro: la
ganancia espectral no esta en w) para las celdas de `--weights-cells`,
submuestreados a 16 fps y en complex64. Sirven para el diagnostico narrowband de
`evaluation/lowfreq_diagnostic.py` sin volver a correr el barrido.

USO
---
    python tests/postfilter_mask_calib_run.py
    python tests/postfilter_mask_calib_run.py --quick          # 2 celdas
    python tests/postfilter_mask_calib_run.py --smooth 0.2
    python tests/postfilter_mask_calib_run.py --weights-cells 0 8

Salida: tests/dataset_out/pf_mask_calib/
    mird_benchmark_metrics.{csv,parquet}
    weights/exp<NNN>_<procesador>_rt<X>_isir<Y>.npz
"""

import os
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

from propagation.mird_loader import MirdDatasetProvider
from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search
from evaluation.bf_wrappers import (
    NM_MVDR, NM_MVDR_SUB, NM_MVDR_MCAL, SOUDEN_ORACLE_SCM,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT_DIR = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "pf_mask_calib")

# Celdas que el ajuste de mascara USO para entrenar (rt60=0.360, iSIR in {0,10}).
FIT_RT60 = {0.360}
FIT_ISIR = {0.0, 10.0}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smooth", type=float, default=0.33,
                    help="piso del post-filtro espectral: G = smooth + (1-smooth)*mask. "
                         "0.33 (~-9.6 dB) es el default de produccion; mas bajo = "
                         "mas supresion y mas riesgo de ruido musical")
    ap.add_argument("--const-a-n", type=float, default=2.0)
    ap.add_argument("--b-n", type=float, default=-8.0)
    ap.add_argument("--alpha", type=float, default=0.99)
    ap.add_argument("--min-loading", type=float, default=1e-9)
    ap.add_argument("--no-dtln-post", dest="dtln_post", action="store_false",
                    help="omite el DTLN completo de salida (solo el post-filtro espectral)")
    ap.add_argument("--weights-cells", type=int, nargs="*", default=[0],
                    help="indices de celda cuyos pesos guardar; vacio = ninguna, "
                         "'all' no existe (usar un rango explicito)")
    ap.add_argument("--weights-fps", type=int, default=16)
    ap.add_argument("--no-weights", action="store_true")
    ap.add_argument("--quick", action="store_true", help="2 celdas en vez de 16")
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    w_dir = None if args.no_weights else os.path.join(args.out_dir, "weights")

    interpreter_1 = tf.lite.Interpreter(
        model_path=f"{PROJECT_ROOT}/src/dnn_denoise/models/model_quant_1.tflite")
    interpreter_1.allocate_tensors()
    interpreter_2 = tf.lite.Interpreter(
        model_path=f"{PROJECT_ROOT}/src/dnn_denoise/models/model_quant_2.tflite")
    interpreter_2.allocate_tensors()

    provider = MirdDatasetProvider(root_dir=f"{PROJECT_ROOT}/tools/data/rirs/mird")

    base_config = {
        'fs': 16000,
        'duration': 15,
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
        'interf_configs': [[(45, 1.0)]] if args.quick else [[(45, 1.0)], [(90, 1.0)]],
        'isir_db': [0, 5] if args.quick else [-5, 0, 5, 10],
        'mismatch_gain': [0], 'mismatch_phase': [0],
        'use_wpe': [False], 'wpe_method': ['online'], 'wpe_taps': [7], 'wpe_delay': [2],
        'error_angle_deg': [0.0], 'error_distance_m': [0.0],
    }

    kw = dict(min_loading=args.min_loading, alpha=args.alpha, mu=0.0)
    mcal = dict(nu=1.0, gamma=0.0, const_a_n=args.const_a_n, b_n_const=args.b_n)
    processors_dict = {
        "NM_MVDR":        NM_MVDR(min_loading=args.min_loading, alpha=args.alpha),
        "NM_MVDR_SUB":    NM_MVDR_SUB(**kw),
        "NM_MVDR_SUB_PF": NM_MVDR_SUB(**kw, smooth=args.smooth),
        "MCAL_C":         NM_MVDR_MCAL(**kw, **mcal),
        "MCAL_C_PF":      NM_MVDR_MCAL(**kw, **mcal, smooth=args.smooth),
        "ORACLE_SCM":     SOUDEN_ORACLE_SCM(min_loading=args.min_loading,
                                            alpha=args.alpha),
    }

    print(f"[*] smooth={args.smooth:g}  a_n={args.const_a_n:g}  b_n={args.b_n:g}  "
          f"DTLN post={'SI' if args.dtln_post else 'NO'}")
    print(f"[*] pesos -> {w_dir if w_dir else 'no se guardan'} "
          f"(celdas {args.weights_cells}, {args.weights_fps} fps)")

    df = run_mird_grid_search(
        grid_params=param_grid,
        dataset_provider=provider,
        processors=processors_dict,
        scene_base_config=base_config,
        output_dir=args.out_dir,
        interpreter_1=interpreter_1,
        interpreter_2=interpreter_2,
        apply_dtln_post=args.dtln_post,
        save_catalog=False,
        save_weights_dir=w_dir,
        save_weights_cells=(args.weights_cells or None) if w_dir else None,
        save_weights_fps=args.weights_fps,
    )

    order = ["NM_MVDR", "NM_MVDR_SUB", "NM_MVDR_SUB_PF", "MCAL_C", "MCAL_C_PF",
             "ORACLE_SCM"]
    mets = ["PESQ", "STOI", "SDR", "SIR", "SAR"]

    def _table(sub, prefix):
        cols = [f"{prefix}{m}_early" for m in mets]
        cols = [c for c in cols if c in sub.columns]
        if not cols:
            return None
        t = sub.groupby("processor")[cols].median().round(3)
        t.columns = [c.replace(prefix, "").replace("_early", "") for c in cols]
        return t.reindex([p for p in order if p in t.index])

    def _show(title, sub):
        if not len(sub):
            return
        print(f"\n{'='*74}\n{title}   ({len(sub)//len(processors_dict)} celdas)\n{'='*74}")
        t1 = _table(sub, "Delta_tot_")
        if t1 is not None:
            print("\n-- beamformer (+ post-filtro espectral donde aplica) --")
            print(t1.to_string())
            for a, b in (("MCAL_C", "NM_MVDR_SUB"), ("MCAL_C_PF", "NM_MVDR_SUB_PF")):
                if a in t1.index and b in t1.index:
                    print(f"   {a} - {b}: {(t1.loc[a]-t1.loc[b]).round(3).to_dict()}")
        t2 = _table(sub, "Delta_tot_pipeline_")
        if t2 is not None:
            print("\n-- + DTLN COMPLETO de salida --")
            print(t2.to_string())
            for a, b in (("MCAL_C", "NM_MVDR_SUB"), ("MCAL_C_PF", "NM_MVDR_SUB_PF")):
                if a in t2.index and b in t2.index:
                    print(f"   {a} - {b}: {(t2.loc[a]-t2.loc[b]).round(3).to_dict()}")

    _show("TODAS LAS CELDAS", df)
    if "rt60" in df.columns and "isir_db" in df.columns:
        seen = df["rt60"].isin(FIT_RT60) & df["isir_db"].isin(FIT_ISIR)
        _show("CELDAS NO VISTAS POR EL AJUSTE DE MASCARA", df[~seen])

    if w_dir and os.path.isdir(w_dir):
        n = len(os.listdir(w_dir))
        print(f"\n[*] {n} archivos de pesos en {w_dir}")
    print(f"\n[ok] {args.out_dir}")


if __name__ == "__main__":
    main()
