"""
¿CUANTO CUESTA ESTIMAR EL iSIR EN VEZ DE SABERLO?

`NM_MVDR_OFB_AUTO` es la configuracion ganadora del lazo de salida con la
agenda del post-filtro alimentada por un estimador CIEGO del iSIR (ver
`ISIRTracker` y tests/ofb_isir_estimator_check.py, que valida el estimador
contra el iSIR verdadero sin correr el beamformer). Este barrido cierra el
argumento midiendo lo unico que el test barato no puede: el efecto sobre las
metricas.

FILAS
-----
    PF_OUT        pf_mask='out'   -- el extremo bueno con iSIR BAJO
    PF_BACK       pf_mask='back'  -- el extremo bueno con iSIR ALTO
    OFB_LEGACY    la agenda con el estimador VIEJO ('sum', pf_isir=(0,2) en el
                  dominio crudo del estimador). Es la mejor configuracion
                  conocida hasta ahora: el numero a igualar.
    OFB_AUTO      la clase nueva: estimador 'band' calibrado en dB reales.
    OFB_AUTO_ORC  la clase nueva con el iSIR VERDADERO (oraculo). Como la
                  agenda esta en dB reales, corre EXACTAMENTE la misma curva
                  que OFB_AUTO: la diferencia es SOLO el error del estimador.

Las cinco comparten todo lo demas (mismo core, mismo alpha, mismo sharpen,
mismo fuse='mean' para el SCM, mismo smooth=0.5).

Lo que hay que leer: OFB_AUTO tiene que (a) quedar entre PF_OUT y PF_BACK y por
arriba de los dos en promedio, (b) empatar a OFB_LEGACY, y (c) quedar a menos
de un ruido de medicion de OFB_AUTO_ORC.

RESULTADO (20 celdas, Delta PESQ promediado sobre salas y angulos)
-----------------------------------------------------------------
    iSIR           -5      0      5     10     15   | media
    PF_OUT        0.618  0.900  1.039  1.040  0.850 | 0.889
    PF_BACK       0.559  0.898  1.087  1.136  0.964 | 0.929
    OFB_LEGACY    0.615  0.911  1.085  1.129  0.958 | 0.940
    OFB_AUTO      0.613  0.910  1.084  1.130  0.960 | 0.939
    OFB_AUTO_ORC  0.614  0.908  1.084  1.130  0.962 | 0.940

Las tres cosas se cumplen: la agenda toma el mejor extremo en cada iSIR en vez
de promediarlos, el oraculo cuesta +0.0004 de media (+0.006 en la peor celda, y
el ciego gana en 9 de 20), y el estimador nuevo empata al viejo. OJO con esa
ultima linea: este barrido corre con UN SOLO locutor, que es exactamente la
nuisance a la que el estimador viejo es sensible. Lo que justifica el cambio
esta medido en tests/ofb_isir_estimator_check.py, no aca.

USO
---
    conda activate tesis_beam
    python tests/ofb_auto_benchmark.py --quick
    python tests/ofb_auto_benchmark.py
"""

import os
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

from propagation.mird_loader import MirdDatasetProvider
from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search
from evaluation.bf_wrappers import NM_MVDR_OFB, NM_MVDR_OFB_AUTO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT_DIR = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "ofb_auto")

ORDER = ["PF_OUT", "PF_BACK", "OFB_LEGACY", "OFB_AUTO", "OFB_AUTO_ORC"]


def build_processors(smooth):
    common = dict(win_type='rect', synth='hann', sharpen_exp=8.0, alpha=0.99,
                  block_update=1, leak=0.0, smooth=smooth, fuse='mean')
    return {
        "PF_OUT":  NM_MVDR_OFB(pf_mask='out', **common),
        "PF_BACK": NM_MVDR_OFB(pf_mask='back', **common),
        # El estimador viejo ('sum') con su calibracion vieja: pf_isir esta en
        # el dominio CRUDO del estimador, no en dB reales.
        "OFB_LEGACY": NM_MVDR_OFB(pf_mask='isir', pf_isir=(0.0, 2.0),
                                  pf_isir_alpha=0.995, **common),
        "OFB_AUTO": NM_MVDR_OFB_AUTO(smooth=smooth),
        "OFB_AUTO_ORC": NM_MVDR_OFB_AUTO(smooth=smooth, isir_db='scene'),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    ap.add_argument("--quick", action="store_true", help="2 celdas (plomeria)")
    ap.add_argument("--smooth", type=float, default=0.5)
    ap.add_argument("--duration", type=float, default=15)
    ap.add_argument("--procs", type=str, nargs="+", default=None)
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

    # La MISMA escena base que el benchmark principal (snr_db=60), para que los
    # numeros sean comparables fila por fila con mird_benchmark_metrics.
    base_config = {
        'fs': 16000,
        'duration': args.duration,
        't_early': 0.050,
        'array_center': [3.0, 3.0, 1.2],
        'mird_spacing': "3-3-3-8-3-3-3",
        'snr_db': 60.0,
        'source_path': f"{PROJECT_ROOT}/tools/data/signals/p002_emo_adoration_sentences.wav",
        'interf_paths': [f"{PROJECT_ROOT}/tools/data/signals/techno_gated commune.wav"],

        'wpe_taps': 7, 'wpe_delay': 2, 'wpe_alpha': 0.9999,
        'wpe_stft_size': 512, 'wpe_stft_shift': 128,
        'wpe_fixed_bits': None, 'wpe_fixed_round': 'nearest', 'wpe_backend': 'cov',
        'wpe_block_L': 512, 'wpe_block_shift': 2, 'wpe_block_iters': 2,
        'wpe_block_reg': 1e-6, 'wpe_block_solver': 'cholesky', 'wpe_block_mode': 'resolve',

        'stft_window': 512,
        'stft_overlap': 384,
        'eval_references': ['early'],
        'dtln_model_path': f"{PROJECT_ROOT}/src/dnn_denoise/models/model_quant_1.tflite",
    }

    param_grid = {
        'rt60': [0.610] if args.quick else [0.360, 0.610],
        'target_angle': [0],
        'target_dist': [1.0],
        'interf_configs': [[(45, 1.0)]] if args.quick else [[(45, 1.0)], [(90, 1.0)]],
        'isir_db': [0] if args.quick else [-5, 0, 5, 10, 15],
        'mismatch_gain': [0], 'mismatch_phase': [0],
        'use_wpe': [False], 'wpe_method': ['online'], 'wpe_taps': [7], 'wpe_delay': [2],
        'error_angle_deg': [0.0], 'error_distance_m': [0.0],
    }

    processors_dict = build_processors(args.smooth)
    if args.procs:
        processors_dict = {n: processors_dict[n] for n in args.procs}
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

    summarize(df, args.out_dir)


def summarize(df, out_dir):
    cols = [c for c in ["Delta_bf_PESQ_early", "Delta_bf_STOI_early",
                        "Delta_bf_SDR_early", "Delta_bf_SIR_early",
                        "Delta_bf_SAR_early"] if c in df.columns]
    order = [p for p in ORDER if p in df["processor"].unique()]

    print("\n" + "=" * 78)
    print("MEDIANA sobre las celdas (referencia early)")
    print("=" * 78)
    t = df.groupby("processor")[cols].median().round(3).reindex(order)
    print(t.to_string())

    print("\n--- Delta PESQ por iSIR (media sobre salas y angulos) ---")
    p = df.pivot_table(index="processor", columns="isir_db",
                       values="Delta_bf_PESQ_early").reindex(order)
    print(p.round(3).to_string())

    print("\n--- Delta STOI por iSIR ---")
    print(df.pivot_table(index="processor", columns="isir_db",
                         values="Delta_bf_STOI_early").reindex(order).round(3).to_string())

    key = [c for c in ("rt60", "interf_configs", "isir_db") if c in df.columns]
    if "OFB_AUTO" in order:
        auto = df[df["processor"] == "OFB_AUTO"].set_index(key)
        print("\n--- celdas ganadas por OFB_AUTO ---")
        for p_ in order:
            if p_ == "OFB_AUTO":
                continue
            cur = df[df["processor"] == p_].set_index(key)
            wins = {c.replace("Delta_bf_", "").replace("_early", ""):
                    int((auto[c] > cur[c].reindex(auto.index)).sum()) for c in cols}
            print(f"  vs {p_:14s} n={len(cur):2d}  " +
                  "  ".join(f"{k}:{v}" for k, v in wins.items()))
    if {"OFB_AUTO", "OFB_AUTO_ORC"} <= set(order):
        a = df[df["processor"] == "OFB_AUTO"].set_index(key)["Delta_bf_PESQ_early"]
        o = df[df["processor"] == "OFB_AUTO_ORC"].set_index(key)["Delta_bf_PESQ_early"]
        d = (o - a.reindex(o.index))
        print(f"\n--- COSTO DEL ESTIMADOR (oraculo - ciego, Delta PESQ) ---")
        print(f"    media {d.mean():+.4f}   mediana {d.median():+.4f}   "
              f"peor celda {d.max():+.4f}   celdas donde el ciego gana: "
              f"{int((d < 0).sum())}/{len(d)}")

    print(f"\n[ok] {out_dir}")


if __name__ == "__main__":
    main()
