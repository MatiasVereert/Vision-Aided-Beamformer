"""
sharpen_sweep_extended.py
=========================
Barrido EXTENDIDO del realce de mascara, alpha=0.99 siempre. Objetivos:

  1. Empujar el exponente del DTLN mas alla de 8 (12, 16, 24, 32) para ver si
     satura o revierte.
  2. Empujar tambien el Oracle a exponente alto (32) para chequear la hipotesis
     de "vaciado" de la mascara de voz (posible caida de SDR).

Referencia: Oracle soft (exp=1).

Resultados en ruta nueva:  tests/dataset_out/sharpen_sweep_extended/

USO:
    conda activate tesis_beam
    python tests/sharpen_sweep_extended.py
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import pandas as pd

from propagation.mird_loader import MirdDatasetProvider
from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search
from evaluation.bf_wrappers import (
    NM_MVDR,
    ORACLE_MB_MVDR_SOUDEN,
)

ROOT_MIRD_DIR = "/home/matias/Documents/Tesis/Vision-Aided-Beamformer/tools/data/rirs/mird"
OUTPUT_DIR = os.path.join(REPO_ROOT, "tests", "dataset_out", "sharpen_sweep_extended")

ALPHA = 0.99
DTLN_EXPS = [8.0, 12.0, 16.0, 24.0, 32.0]


def main():
    print("[*] Initializing automated MIRD dataset provider...")
    provider = MirdDatasetProvider(root_dir=os.path.abspath(ROOT_MIRD_DIR))

    base_config = {
        'fs': 16000,
        'duration': 15,
        't_early': 0.050,
        'array_center': [3.0, 3.0, 1.2],
        'mird_spacing': "3-3-3-8-3-3-3",
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

    param_grid = {
        'rt60': [0.610],
        'target_angle': [0],
        'target_dist': [1.0],
        'interf_configs': [[(45, 1.0)]],
        'isir_db': [-5],
        'mismatch_gain': [0],
        'mismatch_phase': [0],
        'use_wpe': [False],
        'error_angle_deg': [0.0],
        'error_distance_m': [0.0]
    }

    processors_dict = {
        # Referencias Oracle
        "Oracle_soft_exp1": ORACLE_MB_MVDR_SOUDEN(min_loading=1e-6, alpha=ALPHA, sharpen_exp=1.0),
        "Oracle_exp32": ORACLE_MB_MVDR_SOUDEN(min_loading=1e-6, alpha=ALPHA, sharpen_exp=32.0),
    }
    # DTLN a exponentes altos
    for e in DTLN_EXPS:
        processors_dict[f"DTLN_exp{e:g}"] = NM_MVDR(
            min_loading=1e-6, alpha=ALPHA, sharpen_exp=e
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.chdir(OUTPUT_DIR)

    df_final = run_mird_grid_search(
        grid_params=param_grid,
        dataset_provider=provider,
        processors=processors_dict,
        scene_base_config=base_config,
        output_dir=OUTPUT_DIR,
        interpreter_1=None,
        interpreter_2=None,
    )

    print(f"\n[*] Resultados guardados en: {OUTPUT_DIR}")
    print("\n[Preview] Barrido extendido (Delta vs 'early', alpha=0.99):")
    cols = ["processor", "Delta_tot_PESQ_early", "Delta_tot_STOI_early",
            "Delta_tot_SIR_early", "Delta_tot_SDR_early"]
    cols = [c for c in cols if c in df_final.columns]
    with pd.option_context('display.float_format', lambda v: f"{v:.4f}"):
        print(df_final[cols].to_string(index=False))

    return df_final


if __name__ == "__main__":
    main()
