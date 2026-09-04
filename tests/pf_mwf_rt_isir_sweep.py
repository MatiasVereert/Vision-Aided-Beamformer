"""
Barrido RT60 x iSIR del POST-FILTRO MWF (NM-MVDR + specsub + Wiener DD relajado).

Objetivo: decidir si la etapa Wiener (beamforming/MWF/wiener_postfilter.py) mueve el
PUNTO DE OPERACION del post-filtro o solo lo recorre. La comparacion justa no es
"PF vs PF+Wiener" sino "PF+Wiener vs PF con `smooth` mas agresivo": si el Wiener
gana PESQ pagando lo mismo en STOI/SI-SDR que bajar `smooth`, no aporta nada nuevo;
si gana PESQ SIN pagar mas (o pagando menos), es una etapa genuinamente distinta.
Por eso el barrido incluye PF_050 (el actual) y PF_033 (el mismo PF, mas agresivo).

Nota sobre el eje RT60: el dataset MIRD no tiene los tres RT60 para el mismo array.
El spacing 3-3-3-8-3-3-3 (el de la config de trabajo) solo existe en 0.360/0.610 s;
0.160 s solo esta grabado con 8-8-8-8-8-8-8. Por eso el barrido corre en DOS bloques
(una corrida por array), con la columna `array` para no mezclarlos: el de 8 cm da el
eje RT60 completo, el de 3-3-3-8 cruza el resultado sobre el array de la config.

Salida: tests/dataset_out/pf_mwf/<array>/ + all_rt_isir.csv consolidado.
"""

import os
import time

import pandas as pd
import tensorflow as tf

from propagation.mird_loader import MirdDatasetProvider
from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search
from evaluation.bf_wrappers import NM_MVDR, NM_MVDR_PF, NM_MVDR_PF_MWF

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT_DIR = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "pf_mwf")

# --- DTLN ---------------------------------------------------------------------
interpreter_1 = tf.lite.Interpreter(
    model_path=f"{PROJECT_ROOT}/src/dnn_denoise/models/model_quant_1.tflite")
interpreter_1.allocate_tensors()
interpreter_2 = tf.lite.Interpreter(
    model_path=f"{PROJECT_ROOT}/src/dnn_denoise/models/model_quant_2.tflite")
interpreter_2.allocate_tensors()

provider = MirdDatasetProvider(root_dir=f"{PROJECT_ROOT}/tools/data/rirs/mird")

# --- Escena base (identica a full_benchmark_test_dtln_mird.py) -----------------
base_config = {
    'fs': 16000,
    'duration': 15,
    't_early': 0.050,
    'array_center': [3.0, 3.0, 1.2],
    'mird_spacing': "3-3-3-8-3-3-3",          # se sobreescribe por bloque
    'snr_db': 60.0,
    'source_path': f"{PROJECT_ROOT}/tools/data/signals/p002_emo_adoration_sentences.wav",
    'interf_paths': [f"{PROJECT_ROOT}/tools/data/signals/techno_gated commune.wav"],

    'wpe_taps': 7, 'wpe_delay': 3, 'wpe_alpha': 0.9999,
    'wpe_stft_size': 512, 'wpe_stft_shift': 128,
    'wpe_fixed_bits': None, 'wpe_fixed_round': 'nearest', 'wpe_backend': 'cov',
    'wpe_block_L': 512, 'wpe_block_shift': 2, 'wpe_block_iters': 2,
    'wpe_block_reg': 1e-6, 'wpe_block_solver': 'cholesky', 'wpe_block_mode': 'resolve',

    'stft_window': 512, 'stft_overlap': 384,
    'eval_references': ['anechoic', 'early', 'reverberant'],
    'dtln_model_path': f"{PROJECT_ROOT}/src/dnn_denoise/models/model_quant_1.tflite",
}

# --- Ejes del barrido ----------------------------------------------------------
ISIR_AXIS = [-5, 0, 5, 10, 15]

# (spacing, RT60 disponibles para ese spacing)
BLOCKS = [
    ("8-8-8-8-8-8-8", [0.160, 0.360, 0.610]),   # eje RT60 completo
    ("3-3-3-8-3-3-3", [0.360, 0.610]),          # array de la config de trabajo
]


def make_grid(rt60_axis):
    return {
        'rt60': rt60_axis,
        'target_angle': [0],
        'target_dist': [1.0],
        'interf_configs': [[(45, 1.0)]],
        'isir_db': ISIR_AXIS,
        'mismatch_gain': [0], 'mismatch_phase': [0],
        'use_wpe': [False], 'wpe_method': ['online'],
        'wpe_taps': [7], 'wpe_delay': [2],
        'error_angle_deg': [0.0], 'error_distance_m': [0.0],
    }


# --- Procesadores --------------------------------------------------------------
# El PF_033 esta para el head-to-head: es el PF actual llevado a un punto mas
# agresivo, o sea el competidor real de la etapa Wiener.
processors_dict = {
    "NM-MVDR":            NM_MVDR(min_loading=1e-9, alpha=0.99),
    "PF_050":             NM_MVDR_PF(min_loading=1e-9, alpha=0.99, smooth=0.50),
    "PF_033":             NM_MVDR_PF(min_loading=1e-9, alpha=0.99, smooth=0.33),
    "PF050+MWF_g6_osf03": NM_MVDR_PF_MWF(min_loading=1e-9, alpha=0.99, smooth=0.50,
                                         w_gmin_db=-6.0, w_osf=0.3),
    "PF050+MWF_g6_osf05": NM_MVDR_PF_MWF(min_loading=1e-9, alpha=0.99, smooth=0.50,
                                         w_gmin_db=-6.0, w_osf=0.5),
}


if __name__ == "__main__":
    t0 = time.time()
    frames = []

    for spacing, rt60_axis in BLOCKS:
        cfg = dict(base_config)
        cfg['mird_spacing'] = spacing
        n_rows = len(rt60_axis) * len(ISIR_AXIS) * len(processors_dict)
        print(f"\n########## ARRAY {spacing} | RT60 {rt60_axis} | {n_rows} filas ##########",
              flush=True)

        df_blk = run_mird_grid_search(
            grid_params=make_grid(rt60_axis),
            dataset_provider=provider,
            processors=processors_dict,
            scene_base_config=cfg,
            output_dir=os.path.join(OUT_DIR, spacing),
            interpreter_1=interpreter_1,
            interpreter_2=interpreter_2,
            save_catalog=False,
            show_progress=False,
        )
        df_blk = df_blk.copy()
        df_blk.insert(0, "array", spacing)
        frames.append(df_blk)
        print(f"[t] acumulado {(time.time() - t0)/60:.1f} min", flush=True)

    df = pd.concat(frames, ignore_index=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "all_rt_isir.csv"), index=False)
    df.to_parquet(os.path.join(OUT_DIR, "all_rt_isir.parquet"), engine="pyarrow")

    # --- Resumen: PESQ/STOI/SI-SDR/SIR por celda ---------------------------------
    metrics = ["Delta_tot_PESQ_early", "Delta_tot_STOI_early",
               "Delta_tot_SI-SDR_early", "Delta_tot_SIR_early", "Delta_tot_SAR_early"]
    metrics = [m for m in metrics if m in df.columns]
    pd.set_option("display.width", 240)

    print("\n=== BARRIDO RT60 x iSIR (Delta vs referencia early) ===")
    print(df[["array", "rt60", "isir_db", "processor"] + metrics].to_string(index=False))

    print("\n=== PROMEDIO POR PROCESADOR (todas las celdas) ===")
    print(df.groupby("processor")[metrics].mean().to_string())

    print("\n=== PESQ: procesador x RT60 ===")
    print(df.pivot_table(index="processor", columns=["array", "rt60"],
                         values="Delta_tot_PESQ_early").to_string())

    print("\n=== PESQ: procesador x iSIR ===")
    print(df.pivot_table(index="processor", columns="isir_db",
                         values="Delta_tot_PESQ_early").to_string())

    print(f"\n=== TOTAL {(time.time() - t0)/60:.1f} MINUTOS ===")
