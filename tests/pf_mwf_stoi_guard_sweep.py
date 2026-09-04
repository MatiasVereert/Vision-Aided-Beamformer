"""
Barrido de las PROTECCIONES DE STOI del post-filtro MWF.

Problema que ataca: el gate espectral cuesta inteligibilidad (STOI absoluto 0.925 con
NM-MVDR pelado -> 0.916 con PF_050 -> 0.909 con el MWF), porque atenua tambien en bins
con habla. Las tres defensas, cada una sobre un mecanismo distinto de esa perdida:

  gmin_mask : piso adaptativo g_min**(1-mask_s) -> el Wiener NO PUEDE atenuar donde el
              DTLN dice habla. Concentra el recorte en los bins que STOI no cuenta.
  smooth_f  : promedio movil de la ganancia en frecuencia -> mata el desgarro espectral.
  smooth_t  : suavizado temporal de la ganancia -> protege la envolvente, que es
              literalmente lo que correlaciona STOI.

La pregunta: se puede quedar con la ganancia de PESQ del MWF al STOI de PF_050?

Escenario reducido respecto de pf_mwf_rt_isir_sweep.py (un solo array, 2 RT60) para que
entre en pocos minutos; el eje iSIR se mantiene completo porque es donde se juega.
"""

import os
import time

import pandas as pd
import tensorflow as tf

from propagation.mird_loader import MirdDatasetProvider
from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search
from evaluation.bf_wrappers import NM_MVDR, NM_MVDR_PF, NM_MVDR_PF_MWF

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT_DIR = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "pf_mwf_stoi")

interpreter_1 = tf.lite.Interpreter(
    model_path=f"{PROJECT_ROOT}/src/dnn_denoise/models/model_quant_1.tflite")
interpreter_1.allocate_tensors()
interpreter_2 = tf.lite.Interpreter(
    model_path=f"{PROJECT_ROOT}/src/dnn_denoise/models/model_quant_2.tflite")
interpreter_2.allocate_tensors()

provider = MirdDatasetProvider(root_dir=f"{PROJECT_ROOT}/tools/data/rirs/mird")

base_config = {
    'fs': 16000, 'duration': 15, 't_early': 0.050,
    'array_center': [3.0, 3.0, 1.2],
    'mird_spacing': "3-3-3-8-3-3-3",
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
    'isir_db': [-5, 0, 5, 10, 15],
    'mismatch_gain': [0], 'mismatch_phase': [0],
    'use_wpe': [False], 'wpe_method': ['online'],
    'wpe_taps': [7], 'wpe_delay': [2],
    'error_angle_deg': [0.0], 'error_distance_m': [0.0],
}

_MWF = dict(min_loading=1e-9, alpha=0.99, smooth=0.50, w_gmin_db=-6.0, w_osf=0.3)

processors_dict = {
    "NM-MVDR":        NM_MVDR(min_loading=1e-9, alpha=0.99),          # techo de STOI
    "PF_050":         NM_MVDR_PF(min_loading=1e-9, alpha=0.99, smooth=0.50),
    "MWF_base":       NM_MVDR_PF_MWF(**_MWF),                          # sin proteccion
    "MWF_gmask":      NM_MVDR_PF_MWF(w_gmin_mask=True, **_MWF),
    "MWF_gmask_sf3":  NM_MVDR_PF_MWF(w_gmin_mask=True, w_smooth_f=3, **_MWF),
    "MWF_gmask_st05": NM_MVDR_PF_MWF(w_gmin_mask=True, w_smooth_t=0.5, **_MWF),
    "MWF_st05":       NM_MVDR_PF_MWF(w_smooth_t=0.5, **_MWF),
}


if __name__ == "__main__":
    t0 = time.time()
    df = run_mird_grid_search(
        grid_params=param_grid, dataset_provider=provider, processors=processors_dict,
        scene_base_config=base_config, output_dir=OUT_DIR,
        interpreter_1=interpreter_1, interpreter_2=interpreter_2,
        save_catalog=False, show_progress=False,
    )
    df.to_csv(os.path.join(OUT_DIR, "stoi_guard.csv"), index=False)

    cols = ["Delta_tot_PESQ_early", "Delta_tot_STOI_early",
            "Delta_tot_SI-SDR_early", "Delta_tot_SIR_early"]
    cols = [c for c in cols if c in df.columns]
    pd.set_option("display.width", 240)

    print("\n=== PROMEDIO POR PROCESADOR ===")
    resumen = df.groupby("processor")[cols + ["proc_STOI_early", "proc_PESQ_early"]].mean()
    print(resumen.to_string())

    # El criterio de exito: PESQ >= MWF_base MANTENIENDO el STOI de PF_050.
    ref_stoi = resumen.loc["PF_050", "proc_STOI_early"]
    ref_pesq = resumen.loc["PF_050", "proc_PESQ_early"]
    print(f"\n=== vs PF_050 (STOI {ref_stoi:.4f} | PESQ {ref_pesq:.4f}) ===")
    delta = pd.DataFrame({
        "dSTOI": resumen["proc_STOI_early"] - ref_stoi,
        "dPESQ": resumen["proc_PESQ_early"] - ref_pesq,
    })
    print(delta.round(4).to_string())

    print("\n=== STOI por iSIR ===")
    print(df.pivot_table(index="processor", columns="isir_db",
                         values="proc_STOI_early").round(4).to_string())
    print("\n=== PESQ por iSIR ===")
    print(df.pivot_table(index="processor", columns="isir_db",
                         values="proc_PESQ_early").round(4).to_string())
    print(f"\n=== TOTAL {(time.time() - t0)/60:.1f} MINUTOS ===")
