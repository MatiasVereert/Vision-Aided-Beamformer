"""
¿El front-end fijo para la mascara y la correccion del desfasaje sobreviven a
las METRICAS REALES?

Dos cambios llegan hasta aca desde el banco de SCM (tests/ds_mask_scm_run.py),
que los midio contra las covarianzas oracle sobre 4 escenas MIRD:

  1. FRONT-END FIJO PARA LA MASCARA. El DTLN deja de ver el canal de referencia
     crudo y pasa a ver la salida de un beamformer FIJO apuntado al target
     (delay-and-sum, o superdirectivo cargado). Recupero +1.03 dB (DS) y
     +1.71 dB (SD) de perdida contra el oracle, en 4/4 escenas, sobre un techo
     de 3.57 dB que marca la mascara ideal. La ganancia esta arriba de 1 kHz y
     entra casi entera por la rama de RUIDO (Phi_NN).

  2. DESFASAJE DE 1 FRAME. El bloque i del DTLN cubre EXACTAMENTE las mismas 512
     muestras que el frame i-1 de scipy.stft, pero el pipeline los apareaba en el
     mismo indice: a cada frame se le aplicaba la mascara del anterior. Corregirlo
     dio +0.63 dB en 4/4 escenas y NO cuesta latencia (los dos terminan en la
     misma muestra: era un error de indexado, no un adelanto temporal).

Los dos son ADITIVOS entre si (+1.74 dB juntos) y con la sustraccion de
covarianza (+0.57 dB), que actua en otra etapa.

POR QUE HACE FALTA ESTA CORRIDA
-------------------------------
La loss del banco es un PROXY: perdida de SINR y de respuesta al target por
celda. Ya paso una vez que un proxy no se tradujo (la calibracion (nu,gamma)
recuperaba 0.7 dB de proxy y empataba con NM_MVDR_SUB en PESQ/SIR). Aca se mide
lo unico que decide: PESQ / STOI / SDR / SIR / SAR sobre la grilla MIRD.

FILAS
-----
    NM_MVDR_OLD      sistema historico EXACTO (mask_shift=0)   <- baseline
    NM_MVDR          idem + desfasaje corregido    -> aisla el fix (2)
    NM_MVDR_SUB_OLD  sustraccion, mask_shift=0
    NM_MVDR_SUB      sustraccion + desfasaje corregido
    DSM_DS           core base   + mascara sobre el DS         -> aisla el fix (1)
    DSM_SUB_DS       sustraccion + mascara sobre el DS         -> la combinacion
    DSM_SUB_SD       sustraccion + mascara sobre el superdirectivo
    ORACLE_SCM       covarianzas de las senales limpias: la cota superior

Todas comparten la MISMA cadena salvo lo que dice su nombre (mismo alpha, mismo
ref_mic, mismo framing, mismo sharpen, sin WPE, sin post-filtro DTLN).

AVISO SOBRE LAS CALIBRACIONES YA AJUSTADAS
------------------------------------------
Los .npz de scm_calibration_run / scm_mask_calibration_run se ajustaron CON el
desfasaje puesto. NM_MVDR_CAL / NM_MVDR_MCAL no entran en esta grilla justamente
por eso: si el fix se adopta, hay que rehacer esos ajustes antes de volver a
compararlos.

USO
---
    python tests/ds_mask_benchmark.py --quick      # 2 celdas, prueba de plomeria
    python tests/ds_mask_benchmark.py              # 16 celdas
    python tests/ds_mask_benchmark.py --procs NM_MVDR_OLD NM_MVDR DSM_SUB_DS

Salida: tests/dataset_out/ds_mask_bench/
    mird_benchmark_metrics.{csv,parquet}   metricas por celda y procesador
    mird_benchmark_catalog.h5              AUDIOS del mejor/peor caso por
                                           (procesador, metrica), para escuchar
"""

import os
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

from propagation.mird_loader import MirdDatasetProvider
from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search
from evaluation.bf_wrappers import (NM_MVDR, NM_MVDR_SUB, NM_MVDR_DSM,
                                    SOUDEN_ORACLE_SCM)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT_DIR = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "ds_mask_bench")

# Orden de lectura del resumen: de menos a mas cambios sobre el sistema actual.
ORDER = ["NM_MVDR_OLD", "NM_MVDR", "NM_MVDR_SUB_OLD", "NM_MVDR_SUB",
         "DSM_DS", "DSM_SUB_DS", "DSM_SUB_SD", "ORACLE_SCM"]


def build_processors(args):
    """
    Todos con min_loading=1e-9 y alpha=0.99, que es el punto de operacion con el
    que se midio el banco de SCM y el resto de los barridos recientes.
    """
    common = dict(min_loading=1e-9, alpha=0.99)
    sub = dict(mu=0.0)
    p = {
        # mask_shift=0 -> pipeline historico EXACTO (mascara 1 frame tarde)
        "NM_MVDR_OLD":     NM_MVDR(mask_shift=0, **common),
        "NM_MVDR":         NM_MVDR(**common),
        "NM_MVDR_SUB_OLD": NM_MVDR_SUB(mask_shift=0, **common, **sub),
        "NM_MVDR_SUB":     NM_MVDR_SUB(**common, **sub),
        "DSM_DS":          NM_MVDR_DSM(core="base", bf_mode="ds", **common),
        "DSM_SUB_DS":      NM_MVDR_DSM(core="subtract", bf_mode="ds", **common, **sub),
        "DSM_SUB_SD":      NM_MVDR_DSM(core="subtract", bf_mode="sd",
                                       sd_loading=args.sd_loading, **common, **sub),
        "ORACLE_SCM":      SOUDEN_ORACLE_SCM(**common),
    }
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
    ap.add_argument("--quick", action="store_true",
                    help="2 celdas en vez de 16 (prueba de plomeria)")
    ap.add_argument("--procs", type=str, nargs="+", default=None,
                    help="subconjunto de procesadores a correr")
    ap.add_argument("--sd-loading", type=float, default=1e-2)
    ap.add_argument("--duration", type=float, default=15)
    # El catalogo H5 (mird_benchmark_catalog.h5) guarda los AUDIOS del mejor y
    # del peor caso por (procesador, metrica) -- es lo que se escucha en el
    # dashboard. Cuesta tiempo (calcula ademas la respuesta polar por caso) y
    # espacio, pero sin el la corrida no se puede auditar de oido, que es la
    # unica forma de ver artefactos que ninguna metrica marca.
    ap.add_argument("--no-catalog", dest="save_catalog", action="store_false",
                    help="corrida solo-metricas, sin audios (mas rapida)")
    ap.add_argument("--dtln-post", action="store_true",
                    help="agrega el DTLN mono como post-filtro y su audio al H5")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    interpreter_1 = tf.lite.Interpreter(
        model_path=f"{PROJECT_ROOT}/src/dnn_denoise/models/model_quant_1.tflite")
    interpreter_1.allocate_tensors()
    interpreter_2 = tf.lite.Interpreter(
        model_path=f"{PROJECT_ROOT}/src/dnn_denoise/models/model_quant_2.tflite")
    interpreter_2.allocate_tensors()

    provider = MirdDatasetProvider(root_dir=f"{PROJECT_ROOT}/tools/data/rirs/mird")

    # Misma escena base que tests/scm_calib_benchmark.py y que el banco de SCM:
    # snr_db=30, spacing 3-3-3-8-3-3-3, sin WPE.
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
        'interf_configs': [[(45, 1.0)]] if args.quick else [[(45, 1.0)], [(90, 1.0)]],
        'isir_db': [0] if args.quick else [-5, 0, 5, 10],
        'mismatch_gain': [0], 'mismatch_phase': [0],
        'use_wpe': [False], 'wpe_method': ['online'], 'wpe_taps': [7], 'wpe_delay': [2],
        'error_angle_deg': [0.0], 'error_distance_m': [0.0],
    }
    if args.quick:
        param_grid['interf_configs'] = [[(45, 1.0)], [(90, 1.0)]]

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
        apply_dtln_post=args.dtln_post,
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

    def _show(title, sub):
        if not len(sub):
            return
        # MEDIANA, no media: SIR/SAR tienen colas enormes a RT bajo (ver el
        # criterio ya adoptado en el resto de los barridos).
        t = sub.groupby("processor")[cols].median().round(3)
        t = t.reindex([p for p in order if p in t.index])
        print(f"\n=== {title}  ({sub['processor'].value_counts().iloc[0]} celdas) ===")
        print(t.to_string())
        if "NM_MVDR_OLD" in t.index:
            print("\n  -- delta contra NM_MVDR_OLD (el sistema tal como estaba) --")
            for p_ in t.index:
                if p_ == "NM_MVDR_OLD":
                    continue
                d = (t.loc[p_] - t.loc["NM_MVDR_OLD"]).round(3)
                print(f"  {p_:16s} " + "  ".join(
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

    # ¿la mejora es sistematica por celda, o la sostienen dos celdas?
    if "NM_MVDR_OLD" in df["processor"].unique():
        key = [c for c in ("rt60", "interf_configs", "isir_db") if c in df.columns]
        base = df[df["processor"] == "NM_MVDR_OLD"].set_index(key)
        print("\n" + "=" * 78)
        print("CELDAS GANADAS contra NM_MVDR_OLD (de N celdas)")
        print("=" * 78)
        for p_ in order:
            if p_ == "NM_MVDR_OLD":
                continue
            cur = df[df["processor"] == p_].set_index(key)
            wins = {c.replace("Delta_tot_", "").replace("_early", ""):
                    int((cur[c] > base[c].reindex(cur.index)).sum()) for c in cols}
            print(f"  {p_:16s} n={len(cur):2d}  " +
                  "  ".join(f"{k}:{v}" for k, v in wins.items()))

    print(f"\n[ok] {args.out_dir}")


if __name__ == "__main__":
    main()
