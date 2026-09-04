"""
Evaluacion del post-filtro MWF ADAPTATIVO contra los fijos.

Schedule (calibrado con tests/pf_mwf_adaptive_calib.py, umbrales en unidades del
ESTIMADOR ciego, no del iSIR verdadero):

    est <= 2.0  (~5 dB reales)  -> CONSERVADOR  smooth 0.55, g_min -3 dB
    est >= 4.8  (~12 dB reales) -> AGRESIVO     smooth 0.42, g_min -9 dB
    en el medio: rampa lineal

Direccion contra-intuitiva pero medida: el gate es barato en STOI a iSIR ALTO (la
mascara del DTLN acierta y solo recorta ruido) y caro a iSIR BAJO (la mascara falla y
pega sobre habla).

IMPORTANTE -- el barrido corre en iSIR INTERMEDIOS ({-3, 2, 7, 12}), distintos de los
que se usaron para elegir el schedule ({-5, 0, 5, 10, 15}). Si se evaluara en los
mismos puntos, el resultado seria un ajuste al conjunto de test y no diria nada sobre
si el schedule generaliza; aca ademas se pone a prueba la INTERPOLACION de la rampa.
"""

import os
import time

import pandas as pd
import tensorflow as tf

from propagation.mird_loader import MirdDatasetProvider
from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search
from evaluation.bf_wrappers import (
    NM_MVDR, NM_MVDR_PF, NM_MVDR_PF_MWF, NM_MVDR_PF_MWF_ADAPT,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT_DIR = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "pf_mwf_adapt")
ADAPT_LOG = os.path.join(OUT_DIR, "adapt_choices.csv")

os.makedirs(OUT_DIR, exist_ok=True)
if os.path.exists(ADAPT_LOG):
    os.remove(ADAPT_LOG)

interpreter_1 = tf.lite.Interpreter(
    model_path=f"{PROJECT_ROOT}/src/dnn_denoise/models/model_quant_1.tflite")
interpreter_1.allocate_tensors()
interpreter_2 = tf.lite.Interpreter(
    model_path=f"{PROJECT_ROOT}/src/dnn_denoise/models/model_quant_2.tflite")
interpreter_2.allocate_tensors()

provider = MirdDatasetProvider(root_dir=f"{PROJECT_ROOT}/tools/data/rirs/mird")

base_config = {
    'fs': 16000, 'duration': 15, 't_early': 0.050,
    'array_center': [3.0, 3.0, 1.2], 'mird_spacing': "3-3-3-8-3-3-3",
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

param_grid = {
    'rt60': [0.360, 0.610],
    'target_angle': [0], 'target_dist': [1.0],
    'interf_configs': [[(45, 1.0)]],
    'isir_db': [-3, 2, 7, 12],          # PUNTOS NUEVOS (no usados para calibrar)
    'mismatch_gain': [0], 'mismatch_phase': [0],
    'use_wpe': [False], 'wpe_method': ['online'],
    'wpe_taps': [7], 'wpe_delay': [2],
    'error_angle_deg': [0.0], 'error_distance_m': [0.0],
}

processors_dict = {
    "NM-MVDR":   NM_MVDR(min_loading=1e-9, alpha=0.99),
    "PF_050":    NM_MVDR_PF(min_loading=1e-9, alpha=0.99, smooth=0.50),
    "PF_033":    NM_MVDR_PF(min_loading=1e-9, alpha=0.99, smooth=0.33),
    "MWF_fijo":  NM_MVDR_PF_MWF(min_loading=1e-9, alpha=0.99, smooth=0.50,
                                w_gmin_db=-6.0, w_osf=0.3),
    "MWF_adapt": NM_MVDR_PF_MWF_ADAPT(min_loading=1e-9, alpha=0.99,
                                      lo_db=2.0, hi_db=4.8,
                                      smooth_lo=0.55, smooth_hi=0.42,
                                      gmin_lo_db=-3.0, gmin_hi_db=-9.0,
                                      w_osf=0.3, log_path=ADAPT_LOG),
}


if __name__ == "__main__":
    t0 = time.time()
    df = run_mird_grid_search(
        grid_params=param_grid, dataset_provider=provider, processors=processors_dict,
        scene_base_config=base_config, output_dir=OUT_DIR,
        interpreter_1=interpreter_1, interpreter_2=interpreter_2,
        save_catalog=False, show_progress=False,
    )
    df.to_csv(os.path.join(OUT_DIR, "adaptive_eval.csv"), index=False)

    pd.set_option("display.width", 240)
    g = df.groupby("processor")[["proc_PESQ_early", "proc_STOI_early",
                                 "Delta_tot_SI-SDR_early"]].mean()
    ref = g.loc["NM-MVDR"]

    print("\n=== PROMEDIO POR PROCESADOR ===")
    tabla = g.copy()
    tabla["PESQ ganado"] = g["proc_PESQ_early"] - ref["proc_PESQ_early"]
    tabla["STOI cedido"] = ref["proc_STOI_early"] - g["proc_STOI_early"]
    tabla["PESQ/STOI"] = tabla["PESQ ganado"] / tabla["STOI cedido"]
    print(tabla.round(4).to_string())

    print("\n=== PESQ por iSIR ===")
    print(df.pivot_table(index="processor", columns="isir_db",
                         values="proc_PESQ_early").round(4).to_string())
    print("\n=== STOI por iSIR ===")
    print(df.pivot_table(index="processor", columns="isir_db",
                         values="proc_STOI_early").round(4).to_string())

    # Auditoria: que configuracion eligio el schedule en cada escena
    if os.path.exists(ADAPT_LOG):
        ch = pd.read_csv(ADAPT_LOG, header=None,
                         names=["isir_est", "smooth", "g_min_db"])
        sub = df[df.processor == "MWF_adapt"].reset_index(drop=True)
        if len(ch) == len(sub):
            ch.insert(0, "rt60", sub["rt60"].values)
            ch.insert(0, "isir_real", sub["isir_db"].values)
        print("\n=== QUE ELIGIO EL SCHEDULE ===")
        print(ch.round(3).to_string(index=False))

    print(f"\n=== TOTAL {(time.time() - t0)/60:.1f} MINUTOS ===")
