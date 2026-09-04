"""
CALIBRACION del estimador ciego de iSIR que maneja el post-filtro adaptativo.

El schedule de NM_MVDR_PF_MWF_ADAPT decide la agresividad con `estimate_isir_db`,
que pesa la potencia del mic de referencia con la mascara del DTLN. Ese estimador es
SESGADO (la mascara se aplica sobre la mezcla, asi que la potencia "de habla" arrastra
ruido y el rango sale comprimido). Para el schedule eso da igual mientras sea MONOTONO
en el iSIR real: los umbrales lo_db/hi_db se expresan en unidades del estimador.

Este script mide esa curva. Corre UN solo procesador adaptativo con schedule PLANO
(lo_db=hi_db, configuracion constante = PF_050 + Wiener suave) sobre el barrido de
iSIR, y cruza el estimador registrado contra el iSIR verdadero de cada celda.

Salida: tests/dataset_out/pf_mwf_adapt/calib.csv y la curva por consola.
"""

import os
import time

import pandas as pd
import tensorflow as tf

from propagation.mird_loader import MirdDatasetProvider
from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search
from evaluation.bf_wrappers import NM_MVDR_PF_MWF_ADAPT

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT_DIR = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "pf_mwf_adapt")
LOG_PATH = os.path.join(OUT_DIR, "_probe_log.csv")

os.makedirs(OUT_DIR, exist_ok=True)
if os.path.exists(LOG_PATH):
    os.remove(LOG_PATH)

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

# Rejilla de calibracion: iSIR fino, dos RT60, para ver si el estimador se mueve con
# el iSIR y CUANTO lo contamina la reverberacion.
param_grid = {
    'rt60': [0.360, 0.610],
    'target_angle': [0], 'target_dist': [1.0],
    'interf_configs': [[(45, 1.0)]],
    'isir_db': [-5, -2, 0, 3, 5, 8, 10, 13, 15],
    'mismatch_gain': [0], 'mismatch_phase': [0],
    'use_wpe': [False], 'wpe_method': ['online'],
    'wpe_taps': [7], 'wpe_delay': [2],
    'error_angle_deg': [0.0], 'error_distance_m': [0.0],
}

# Schedule PLANO: lo_db == hi_db -> t=1 siempre -> configuracion fija (smooth_hi /
# gmin_hi). Lo unico que interesa de esta corrida es el iSIR registrado.
probe = NM_MVDR_PF_MWF_ADAPT(min_loading=1e-9, alpha=0.99,
                             lo_db=0.0, hi_db=0.0,
                             smooth_lo=0.50, smooth_hi=0.50,
                             gmin_lo_db=-6.0, gmin_hi_db=-6.0,
                             log_path=LOG_PATH)

if __name__ == "__main__":
    t0 = time.time()
    df = run_mird_grid_search(
        grid_params=param_grid, dataset_provider=provider,
        processors={"probe": probe}, scene_base_config=base_config,
        output_dir=OUT_DIR, interpreter_1=interpreter_1, interpreter_2=interpreter_2,
        save_catalog=False, show_progress=False,
    )

    # El wrapper appendea una linea por escena, en el MISMO orden en que el barrido
    # las corre, y hay un solo procesador -> las filas se alinean 1 a 1.
    est = pd.read_csv(LOG_PATH, header=None, names=["isir_est", "smooth", "g_min_db"])
    assert len(est) == len(df), f"desalineado: {len(est)} estimaciones vs {len(df)} filas"
    df = df.reset_index(drop=True)
    df["isir_est"] = est["isir_est"].values
    df.to_csv(os.path.join(OUT_DIR, "calib.csv"), index=False)

    pd.set_option("display.width", 200)
    print("\n=== ESTIMADOR vs iSIR VERDADERO ===")
    piv = df.pivot_table(index="isir_db", columns="rt60", values="isir_est")
    piv["media"] = piv.mean(axis=1)
    print(piv.round(2).to_string())

    d = piv["media"].diff()
    print("\nmonotono creciente:", bool((d.dropna() > 0).all()))
    rango_real = df["isir_db"].max() - df["isir_db"].min()
    rango_est = piv["media"].max() - piv["media"].min()
    print(f"pendiente media: {rango_est/rango_real:.3f} dB_est por dB_real "
          f"(rango estimador {rango_est:.2f} dB vs real {rango_real:.0f} dB)")
    print(f"dispersion entre RT60 (max-min por celda): "
          f"{(piv[0.610] - piv[0.360]).abs().max():.2f} dB")
    print(f"\n=== TOTAL {(time.time() - t0)/60:.1f} MINUTOS ===")
