"""
¿La calibracion de las SCM sobrevive a las METRICAS REALES?

El banco (tests/scm_calibration_run.py) ajusto nu_k / gamma_k minimizando una
loss PROXY: perdida de SINR y de respuesta al target contra las SCM oracle, por
bin. Recupero ~0.7 dB de mediana por banda. Esta corrida contesta la unica
pregunta que importa despues de eso: si ese proxy se traduce en PESQ / STOI /
SDR / SIR sobre la grilla MIRD, o si se diluye.

PROCESADORES
------------
    NM_MVDR          core base = el sistema actual            (nu=0, gamma=0)
    NM_MVDR_SUB      sustraccion de covarianza, mu=0          (nu=1, gamma=0)
    NM_MVDR_CAL      calibrado: nu_k y gamma_k del banco
    NM_MVDR_CAL_NG   ABLACION: solo nu_k, gamma forzado a 0. Separa cuanto
                     aporta la escala de la sustraccion y cuanto la GEOMETRIA
                     (el shrinkage hacia la coherencia difusa).
    ORACLE_SCM       covarianzas de las senales limpias: la cota superior

Los cuatro primeros comparten TODA la cadena (misma mascara DTLN, mismo framing,
mismo ref_mic, mismo alpha): la unica diferencia es (nu_k, gamma_k).

GENERALIZACION -- LEER ANTES DE INTERPRETAR
-------------------------------------------
La calibracion se ajusto SOLO sobre rt60=0.360 con iSIR in {0, 10} (4 escenas).
Esta grilla corre 16 celdas, asi que 12 de las 16 son condiciones NO VISTAS:
todo rt60=0.610 y todos los iSIR -5 y 5. El resumen separa las celdas vistas de
las no vistas -- si la ganancia solo aparece en las vistas, es sobreajuste y hay
que descartarla.

AVISO SOBRE PESQ
----------------
La ganancia mas grande que midio el banco esta entre 2.7 y 8 kHz, donde PESQ
pesa poco, y en graves debajo de 300 Hz, donde P.862 directamente no evalua. Es
esperable que el efecto se vea antes en SIR/SDR que en PESQ. Que PESQ no se
mueva NO invalida la calibracion; que SIR tampoco se mueva, si.

USO
---
    python tests/scm_calib_benchmark.py
    python tests/scm_calib_benchmark.py --quick        # 2 escenas, para probar
    python tests/scm_calib_benchmark.py --calib <ruta al .npz>

Salida: tests/dataset_out/scm_calib_bench/mird_benchmark_metrics.{csv,parquet}
"""

import os
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

from propagation.mird_loader import MirdDatasetProvider
from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search
from evaluation.bf_wrappers import (NM_MVDR, NM_MVDR_SUB, NM_MVDR_CAL,
                                    NM_MVDR_MCAL, SOUDEN_ORACLE_SCM)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT_DIR = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "scm_calib_bench")
CALIB_NPZ = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "scm_calib",
                         "scm_calib_params.npz")
MASK_NPZ = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "scm_mask_calib_w",
                        "mask_calib_params.npz")

# Celdas que el banco USO para ajustar (rt60=0.360, iSIR in {0,10}). Todo lo demas
# de la grilla es condicion no vista.
FIT_RT60 = {0.360}
FIT_ISIR = {0.0, 10.0}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", type=str, default=CALIB_NPZ)
    ap.add_argument("--mask-calib", type=str, default=MASK_NPZ,
                    help="tabla por banda de la calibracion de MASCARA (etapa 2)")
    ap.add_argument("--const-a-n", type=float, default=2.0,
                    help="a_n de la variante SIN tabla por banda")
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    ap.add_argument("--quick", action="store_true",
                    help="2 escenas en vez de 16 (prueba de plomeria)")
    args = ap.parse_args()

    for pth in (args.calib, args.mask_calib):
        if not os.path.exists(pth):
            raise SystemExit(f"No existe {pth}. Correr antes los runners de calibracion.")
    if not os.path.exists(args.calib):
        raise SystemExit(f"No existe {args.calib}. Correr antes "
                         f"tests/scm_calibration_run.py")
    z = np.load(args.calib, allow_pickle=True)
    print(f"[*] calibracion: {args.calib}")
    print(f"    ajustada en train={list(z['train'])}")
    print(f"    eta={float(z['eta']):g}  mu={float(z['mu']):g}  alpha={float(z['alpha']):g}")
    print(f"    nu_k    en [{z['nu_k'].min():.2f}, {z['nu_k'].max():.2f}]")
    print(f"    gamma_k en [{z['gamma_k'].min():.2f}, {z['gamma_k'].max():.2f}]")

    interpreter_1 = tf.lite.Interpreter(
        model_path=f"{PROJECT_ROOT}/src/dnn_denoise/models/model_quant_1.tflite")
    interpreter_1.allocate_tensors()
    interpreter_2 = tf.lite.Interpreter(
        model_path=f"{PROJECT_ROOT}/src/dnn_denoise/models/model_quant_2.tflite")
    interpreter_2.allocate_tensors()

    provider = MirdDatasetProvider(root_dir=f"{PROJECT_ROOT}/tools/data/rirs/mird")

    # Misma configuracion de escena que tests/sub_grid_sweep.py Y que la que uso
    # el banco para calibrar (snr_db=30, spacing 3-3-3-8-3-3-3, sin WPE).
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

    processors_dict = {
        "NM_MVDR":       NM_MVDR(min_loading=1e-9, alpha=0.99),
        "NM_MVDR_SUB":   NM_MVDR_SUB(min_loading=1e-9, alpha=0.99, mu=0.0),
        # etapa 1: calibracion POST-HOC (nu_k, gamma_k). Empata con SUB.
        "NM_MVDR_CAL":   NM_MVDR_CAL(min_loading=1e-9, alpha=0.99, mu=0.0,
                                     calib_path=args.calib),
        # etapa 2: calibracion de la MASCARA, tabla por banda.
        "NM_MVDR_MCAL":  NM_MVDR_MCAL(min_loading=1e-9, alpha=0.99, mu=0.0,
                                      calib_path=args.mask_calib, nu=1.0, gamma=0.0),
        # etapa 2 SIN tabla: mascara cruda en la rama de voz + odds-ratio
        # recortado con a_n constante en la de ruido. Es la version que se puede
        # implementar en hardware sin tabla de coeficientes por banda.
        "NM_MVDR_MCAL_C": NM_MVDR_MCAL(min_loading=1e-9, alpha=0.99, mu=0.0,
                                       const_a_n=args.const_a_n, b_n_const=-8.0,
                                       nu=1.0, gamma=0.0),
        "ORACLE_SCM":    SOUDEN_ORACLE_SCM(min_loading=1e-9, alpha=0.99),
    }

    df = run_mird_grid_search(
        grid_params=param_grid,
        dataset_provider=provider,
        processors=processors_dict,
        scene_base_config=base_config,
        output_dir=args.out_dir,
        interpreter_1=interpreter_1,
        interpreter_2=interpreter_2,
        apply_dtln_post=False,
        save_catalog=False,
    )

    cols = [c for c in ["Delta_tot_PESQ_early", "Delta_tot_STOI_early",
                        "Delta_tot_SDR_early", "Delta_tot_SIR_early",
                        "Delta_tot_SAR_early", "Delta_tot_SINR_early"]
            if c in df.columns]
    if not cols:
        print("[!] no hay columnas Delta_tot_*_early en el resultado")
        return

    order = ["NM_MVDR", "NM_MVDR_SUB", "NM_MVDR_CAL", "NM_MVDR_MCAL",
             "NM_MVDR_MCAL_C", "ORACLE_SCM"]

    def _show(title, sub):
        if not len(sub):
            return
        t = sub.groupby("processor")[cols].median().round(3)
        t = t.reindex([p for p in order if p in t.index])
        print(f"\n=== {title}  ({sub['processor'].value_counts().iloc[0]} celdas) ===")
        print(t.to_string())
        for p_ in ("NM_MVDR_MCAL", "NM_MVDR_MCAL_C"):
            if p_ in t.index and "NM_MVDR_SUB" in t.index:
                print(f"  {p_} - NM_MVDR_SUB : "
                      f"{(t.loc[p_] - t.loc['NM_MVDR_SUB']).round(3).to_dict()}")

    print("\n" + "=" * 78)
    print("MEDIANA sobre las celdas (referencia early)")
    print("=" * 78)
    _show("TODAS LAS CELDAS", df)

    if "rt60" in df.columns and "isir_db" in df.columns:
        seen = df["rt60"].isin(FIT_RT60) & df["isir_db"].isin(FIT_ISIR)
        _show("CELDAS VISTAS EN EL AJUSTE (rt60=0.360, iSIR 0/10)", df[seen])
        _show("CELDAS NO VISTAS (el test que vale)", df[~seen])
    else:
        print("\n[!] sin columnas rt60/isir_db: no se pudo separar visto/no visto")

    print(f"\n[ok] {args.out_dir}")


if __name__ == "__main__":
    main()
