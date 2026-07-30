"""
oracle_mask_benchmark.py
========================
Experimento Oracle vs DTLN.

Corre el benchmark MIRD mask-based con EXACTAMENTE la misma configuracion que el
bloque __main__ de src/evaluation/full_benchmark_test_dtln_mird.py (mismos
base_config, param_grid y processors_dict, que ya incluyen los procesadores
Oracle y DTLN apareados por alpha), pero guardando TODOS los resultados en una
ruta nueva e independiente:

    tests/dataset_out/oracle_mask_benchmark/
        - mird_benchmark_catalog.h5     (best/worst cases + audio + resp. espacial)
        - mird_benchmark_metrics.parquet
        - mird_benchmark_metrics.csv

USO:
    conda activate tesis_beam
    python tests/oracle_mask_benchmark.py
"""

import os
import sys

# --- Alinear el path de imports con el layout del repo (paquetes bajo src/) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import tensorflow as tf

from propagation.mird_loader import MirdDatasetProvider
from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search
from evaluation.bf_wrappers import (
    NM_MVDR,
    DTLN_MB_MVDR_SOUDEN_SLOW,
    NM_MVDR_PF,
    ORACLE_MB_MVDR_SOUDEN,
)

# Modelos DTLN (rutas absolutas, no dependen del CWD)
DTLN_MODEL_1 = "/home/matias/Documents/Tesis/Vision-Aided-Beamformer/src/dnn_denoise/models/model_quant_1.tflite"
DTLN_MODEL_2 = "/home/matias/Documents/Tesis/Vision-Aided-Beamformer/src/dnn_denoise/models/model_quant_2.tflite"

# Raiz del dataset MIRD
ROOT_MIRD_DIR = "/home/matias/Documents/Tesis/Vision-Aided-Beamformer/tools/data/rirs/mird"

# Ruta NUEVA de salida (absoluta)
OUTPUT_DIR = os.path.join(REPO_ROOT, "tests", "dataset_out", "oracle_mask_benchmark")


def main():
    # --- Interpretes DTLN (identico al __main__ del benchmark) ---
    try:
        interpreter_1 = tf.lite.Interpreter(model_path=DTLN_MODEL_1)
        interpreter_1.allocate_tensors()

        interpreter_2 = tf.lite.Interpreter(model_path=DTLN_MODEL_2)
        interpreter_2.allocate_tensors()
        print("[*] DTLN TF-Lite interpreters successfully allocated.")
    except Exception as e:
        print(f"[!] Warning: Could not initialize DTLN models automatically. Running without neural enhancement. Details: {e}")
        interpreter_1, interpreter_2 = None, None

    # --- Provider MIRD ---
    print("[*] Initializing automated MIRD dataset provider...")
    provider = MirdDatasetProvider(root_dir=os.path.abspath(ROOT_MIRD_DIR))

    # =====================================================================
    # base_config: IDENTICO al __main__ de full_benchmark_test_dtln_mird.py
    # =====================================================================
    base_config = {
        'fs': 16000,
        'duration': 15,
        't_early': 0.050,  # (50 ms)
        'array_center': [3.0, 3.0, 1.2],  # Virtual translation anchor for SimAcoustic
        'mird_spacing': "3-3-3-8-3-3-3",  # Target linear array spacing in the dataset

        'snr_db': 60.0,
        'source_path': r"/home/matias/Documents/Tesis/Vision-Aided-Beamformer/tools/data/signals/p002_emo_adoration_sentences.wav",
        'interf_paths': [
            r"/home/matias/Documents/Tesis/Vision-Aided-Beamformer/tools/data/signals/techno_gated commune.wav"
        ],

        'wpe_taps': 7,
        'wpe_delay': 3,
        'wpe_alpha': 0.9999,
        'wpe_stft_size': 512,
        'wpe_stft_shift': 128,

        'stft_window': 512,
        'stft_overlap': 384,

        'eval_references': ['anechoic', 'early', 'reverberant'],
        'dtln_model_path': r"/home/matias/Documents/Tesis/Vision-Aided-Beamformer/src/dnn_denoise/models/model_quant_1.tflite",
    }

    # =====================================================================
    # param_grid: IDENTICO al __main__ del benchmark
    # =====================================================================
    param_grid = {
        'rt60': [0.610],
        'target_angle': [0],
        'target_dist': [1.0],

        'interf_configs': [
            [(45, 1.0)],
        ],

        'isir_db': [-5],
        'mismatch_gain': [0],
        'mismatch_phase': [0],
        'use_wpe': [False],
        'error_angle_deg': [0.0],
        'error_distance_m': [0.0]
    }

    # =====================================================================
    # processors: mismos que el benchmark (Oracle vs DTLN apareados por alpha)
    # =====================================================================
    processors_dict = {
        "NM-MVDR_alpha_1_ref": NM_MVDR(min_loading=1e-6, alpha=1),
        "NM-MVDR_alpha_0.99_ref": NM_MVDR(min_loading=1e-6, alpha=0.99),
        # Cota superior agnostica al modelo: misma cadena Souden pero con mascara ideal.
        # SOFT: mascara ideal suave (sharpen_exp=1.0, IRM continua).
        "Oracle-MVDR_alpha_1": ORACLE_MB_MVDR_SOUDEN(min_loading=1e-6, alpha=1, sharpen_exp=1.0),
        "Oracle-MVDR_alpha_0.99": ORACLE_MB_MVDR_SOUDEN(min_loading=1e-6, alpha=0.99, sharpen_exp=1.0),
        # HARD-EDGE: mascara ideal agudizada (sharpen_exp=4.0, iguala el **4 fijo del DTLN).
        "Oracle-MVDR_hard_alpha_1": ORACLE_MB_MVDR_SOUDEN(min_loading=1e-6, alpha=1, sharpen_exp=4.0),
        "Oracle-MVDR_hard_alpha_0.99": ORACLE_MB_MVDR_SOUDEN(min_loading=1e-6, alpha=0.99, sharpen_exp=4.0),
        "Slow": DTLN_MB_MVDR_SOUDEN_SLOW(),
        "Specsub": NM_MVDR_PF(smooth=1.0, min_loading=1e-6),
    }

    # Aseguramos la ruta nueva y corremos con CWD ahi, de modo que TODOS los
    # artefactos (incluido el CSV que el benchmark escribe con ruta relativa)
    # caigan dentro de OUTPUT_DIR y no contaminen la raiz del repo.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.chdir(OUTPUT_DIR)

    df_final = run_mird_grid_search(
        grid_params=param_grid,
        dataset_provider=provider,
        processors=processors_dict,
        scene_base_config=base_config,
        output_dir=OUTPUT_DIR,
        interpreter_1=interpreter_1,
        interpreter_2=interpreter_2,
    )

    print(f"\n[*] Resultados guardados en: {OUTPUT_DIR}")
    print("\n[Preview] Oracle vs DTLN (metricas contra referencia 'early'):")

    cols_to_show = [
        "processor", "rt60", "target_angle", "interf_configs", "isir_db",
        "Delta_tot_PESQ_early", "Delta_tot_STOI_early", "Delta_tot_SIR_early", "Delta_tot_SDR_early"
    ]
    cols_exist = [c for c in cols_to_show if c in df_final.columns]
    print(df_final[cols_exist].to_string(index=False))

    return df_final


if __name__ == "__main__":
    main()
