"""
Barrido de grilla MIRD del core con SUSTRACCION DE COVARIANZA.

Objetivo: confirmar que la mejora de NM_MVDR_SUB no es de UNA escena afortunada.
Hasta aca todo se midio sobre rt60=0.61 / iSIR=0 / interferente a 45 grados.

Procesadores (los cuatro pedidos):
    NM_MVDR      core base (sistema actual)
    NM_MVDR_SUB  Phi_SS = Phi_XX - Phi_NN, normaliza por lambda_S = lambda - M
    NM_MVDR_PF   core base + post-filtro de sustraccion espectral (produccion)
    ORACLE_SCM   covarianzas de las senales limpias (cota superior)

Ejes: rt60 x iSIR x angulo del interferente = 2 x 4 x 2 = 16 escenas.
El spacing 3-3-3-8-3-3-3 solo existe en el dataset para rt60 0.360 y 0.610.

apply_dtln_post=False: la cascada DTLN-completo-como-post-filtro esta descartada
del sistema y duplicaria el tiempo de corrida.

Salida: tests/dataset_out/sub_grid/mird_benchmark_metrics.{csv,parquet}
"""

import os

import pandas as pd
import tensorflow as tf

from propagation.mird_loader import MirdDatasetProvider
from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search
from evaluation.bf_wrappers import NM_MVDR, NM_MVDR_SUB, NM_MVDR_PF, SOUDEN_ORACLE_SCM

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT_DIR = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "sub_grid")

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
    'snr_db': 30.0,          # ruido propio de microfono realista (no el 60 del default)
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
    'rt60': [0.360, 0.610],
    'target_angle': [0],
    'target_dist': [1.0],
    'interf_configs': [
        [(45, 1.0)],
        [(90, 1.0)],
    ],
    'isir_db': [-5, 0, 5, 10],
    'mismatch_gain': [0],
    'mismatch_phase': [0],
    'use_wpe': [False],
    'wpe_method': ['online'],
    'wpe_taps': [7],
    'wpe_delay': [2],
    'error_angle_deg': [0.0],
    'error_distance_m': [0.0],
}

processors_dict = {
    "NM_MVDR":     NM_MVDR(min_loading=1e-9, alpha=0.99),
    "NM_MVDR_SUB": NM_MVDR_SUB(min_loading=1e-9, alpha=0.99, mu=0.0),
    "NM_MVDR_PF":  NM_MVDR_PF(min_loading=1e-6, alpha=0.99, smooth=0.33),
    "ORACLE_SCM":  SOUDEN_ORACLE_SCM(min_loading=1e-9, alpha=0.99),
}

if __name__ == "__main__":
    df = run_mird_grid_search(
        grid_params=param_grid,
        dataset_provider=provider,
        processors=processors_dict,
        scene_base_config=base_config,
        output_dir=OUT_DIR,
        interpreter_1=interpreter_1,
        interpreter_2=interpreter_2,
        apply_dtln_post=False,
        save_catalog=False,
    )

    print("\n=== PROMEDIO SOBRE LAS 16 ESCENAS (referencia early) ===")
    cols = [c for c in ["Delta_tot_PESQ_early", "Delta_tot_STOI_early",
                        "Delta_tot_SDR_early", "Delta_tot_SIR_early",
                        "Delta_tot_SAR_early", "Delta_tot_SINR_early"]
            if c in df.columns]
    if cols:
        print(df.groupby("processor")[cols].mean().round(3).to_string())
