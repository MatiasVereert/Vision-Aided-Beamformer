"""
LAZO REALIMENTADO CON UN SOLO DTLN: ¿gana o se desestabiliza?

La cadena de dos pasadas corre el DTLN dos veces y la primera (sobre el canal
crudo) no llega al beamformer: es un bootstrap para poder estimar la RTF. La
hipotesis es que si la mascara(2) es mejor, realimentarla mejora la RTF, que
mejora la mascara, etc. El riesgo es que sin la mascara(1) el lazo se queda sin
ancla.

Son DOS cambios y hay que separarlos, porque no tienen por que aportar lo mismo:

    base   : NM_MVDR_DSM_BLIND causal + smooth = el sistema de hoy.
    spec   : IGUAL, pero la mascara(2) sale de alimentar al DTLN con el ESPECTRO
             conformado en vez de resintetizar y volver a enmarcar.
             -> aisla el efecto de sacar el ida y vuelta al tiempo.
    fb     : UN solo DTLN, lazo cerrado frame a frame.
             -> aisla el efecto de la realimentacion (sobre 'spec').
    fb_gate: el lazo cerrado + el gate de arranque en frio, que es el seguro que
             reemplaza al ancla que se perdio.

Con --sd repite el experimento partiendo del front-end SEMI-CIEGO (superdirectivo
restringido + gate), que es la configuracion puntera del benchmark. Las dos cosas
aprietan el apuntamiento -- una con la geometria, la otra con la RTF estimada --
asi que la ganancia medida sobre 'ds' NO se extrapola.

Uso:
    python tests/window_mismatch/run_dsm_blind_feedback.py [--full] [--sd] [--sd-gate]
"""

import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = "/home/matias/Documents/Tesis/Vision-Aided-Beamformer"
sys.path.insert(0, os.path.join(ROOT, "src"))

from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search   # noqa: E402
from evaluation.bf_wrappers import NM_MVDR_DSM_BLIND, NM_MVDR_DSM_FB        # noqa: E402
from propagation.mird_loader import MirdDatasetProvider                     # noqa: E402

MODEL_1 = f"{ROOT}/src/dnn_denoise/models/model_quant_1.tflite"
MODEL_2 = f"{ROOT}/src/dnn_denoise/models/model_quant_2.tflite"
OUT_DIR = os.environ.get("SWEEP_OUT", "tests/dataset_out/dsm_blind_feedback_run")

# La configuracion de referencia: la variante causal con post-filtro.
CFG = dict(win_type='rect', synth='hann', sharpen_exp=8.0, smooth=0.5, alpha=0.99)


def build_processors():
    return {
        "base":    NM_MVDR_DSM_BLIND(causal=True, **CFG),
        "spec":    NM_MVDR_DSM_FB(mode="spec", **CFG),
        "fb":      NM_MVDR_DSM_FB(mode="fb", **CFG),
        "fb_gate": NM_MVDR_DSM_FB(mode="fb", conf_gate=0.35, conf_alpha=0.99, **CFG),
    }


# El front-end SEMI-CIEGO: superdirectivo restringido (coherencia difusa teorica,
# que sale solo de la geometria del arreglo) + el gate de arranque en frio. Es la
# configuracion PUNTERA del benchmark, no la 'ds' pura de build_processors().
SD = dict(conf_gate=0.35, conf_alpha=0.99, w_mode="sd", sd_eps=0.30)


def build_sd_processors():
    """
    El mismo experimento, pero partiendo del mejor front-end en vez del DS.

    Importa porque las dos cosas atacan lo MISMO desde lados distintos: la
    superdirectividad aprieta el haz con la geometria y la realimentacion aprieta
    el apuntamiento con la RTF estimada. No hay motivo para que se sumen, asi que
    la ganancia del lazo medida sobre 'ds' no se puede extrapolar aca.
    """
    return {
        "sd_base": NM_MVDR_DSM_BLIND(causal=True, **SD, **CFG),
        "sd_spec": NM_MVDR_DSM_FB(mode="spec", **SD, **CFG),
        "sd_fb":   NM_MVDR_DSM_FB(mode="fb", **SD, **CFG),
    }


def build_sd_gate_processors():
    """
    ¿CUANTO VALE EL GATE EN EL FRONT-END SUPERDIRECTIVO?

    Las filas de `build_sd_processors` llevan todas el gate prendido, asi que
    miden el lazo A GATE CONSTANTE. Esto barre el otro eje: el gate ENCENDIDO Y
    APAGADO, para las dos formas de la cadena.

    La hipotesis es que el gate pesa MAS con superdirectividad que con DS. El
    superdirectivo compra directividad achicando el margen de WNG, y eso lo hace
    mas sensible a un error de apuntamiento: una RTF mal estimada, que con DS
    solo desapunta un poco, aca puede caer en un nulo. El gate corta la rama de
    senal justo mientras no hay evidencia, que es cuando la RTF es peor.

    Referencia sobre DS (escena sana, 8 celdas): el gate cuesta -1.75 dB de SIR
    (gana 1/8) y no aporta PESQ. Si la hipotesis vale, aca tiene que dar vuelta
    el signo.
    """
    SD_NG = dict(w_mode="sd", sd_eps=0.30)          # sin conf_gate
    return {
        "sd_base":    NM_MVDR_DSM_BLIND(causal=True, **SD, **CFG),
        "sd_base_ng": NM_MVDR_DSM_BLIND(causal=True, **SD_NG, **CFG),
        "sd_fb":      NM_MVDR_DSM_FB(mode="fb", **SD, **CFG),
        "sd_fb_ng":   NM_MVDR_DSM_FB(mode="fb", **SD_NG, **CFG),
    }


def main():
    try:
        itp1 = tf.lite.Interpreter(model_path=MODEL_1); itp1.allocate_tensors()
        itp2 = tf.lite.Interpreter(model_path=MODEL_2); itp2.allocate_tensors()
    except Exception as e:
        print(f"[!] DTLN no cargado: {e}")
        itp1 = itp2 = None

    provider = MirdDatasetProvider(root_dir=os.path.abspath(f"{ROOT}/tools/data/rirs/mird"))
    base_config = {
        'fs': 16000, 'duration': 15, 't_early': 0.050,
        'array_center': [3.0, 3.0, 1.2], 'mird_spacing': "3-3-3-8-3-3-3",
        'snr_db': 60.0,
        'source_path': f"{ROOT}/tools/data/signals/p002_emo_adoration_sentences.wav",
        'interf_paths': [f"{ROOT}/tools/data/signals/techno_gated commune.wav"],
        'wpe_taps': 7, 'wpe_delay': 3, 'wpe_alpha': 0.9999,
        'wpe_stft_size': 512, 'wpe_stft_shift': 128,
        'wpe_fixed_bits': None, 'wpe_fixed_round': 'nearest', 'wpe_backend': 'cov',
        'wpe_block_L': 512, 'wpe_block_shift': 2, 'wpe_block_iters': 2,
        'wpe_block_reg': 1e-6, 'wpe_block_solver': 'cholesky', 'wpe_block_mode': 'resolve',
        'stft_window': 512, 'stft_overlap': 384,
        'eval_references': ['anechoic', 'early', 'reverberant'],
        'dtln_model_path': MODEL_1, 'dtln_model2_path': MODEL_2,
    }
    full = "--full" in sys.argv
    param_grid = {
        'rt60': [0.360, 0.610], 'target_angle': [0], 'target_dist': [1.0],
        'interf_configs': ([[(45, 1.0)], [(90, 2.0)]] if full else [[(45, 1.0)]]),
        'isir_db': [-5, 0], 'mismatch_gain': [0], 'mismatch_phase': [0],
        'use_wpe': [True], 'wpe_method': ['online'], 'wpe_taps': [7], 'wpe_delay': [2],
        'error_angle_deg': [0.0], 'error_distance_m': [0.0],
    }
    procs = (build_sd_gate_processors() if "--sd-gate" in sys.argv else
             build_sd_processors() if "--sd" in sys.argv else build_processors())
    df = run_mird_grid_search(
        grid_params=param_grid, dataset_provider=provider,
        processors=procs, scene_base_config=base_config,
        output_dir=OUT_DIR, interpreter_1=itp1, interpreter_2=itp2,
        save_catalog=False, apply_dtln_post=False)
    summarize(df)
    return df


def summarize(df, ref='early'):
    metrics = ['PESQ', 'STOI', 'SI-SDR', 'SDR', 'SIR', 'SAR']
    cols = [f"proc_{m}_{ref}" for m in metrics if f"proc_{m}_{ref}" in df.columns]
    pd.set_option('display.width', 220)
    print(f"\n=== lazo realimentado, un solo DTLN (ref '{ref}') ===")
    print(df.groupby('processor')[cols].mean().round(3).to_string())
    procset = set(df.processor)
    pairs = (("sd_base_ng", "sd_base"), ("sd_fb_ng", "sd_fb"),
             ("sd_fb", "sd_base"), ("sd_fb_ng", "sd_base_ng")) \
        if "sd_fb_ng" in procset else \
        (("sd_spec", "sd_base"), ("sd_fb", "sd_spec"), ("sd_fb", "sd_base")) \
        if "sd_base" in procset else \
        (("spec", "base"), ("fb", "spec"), ("fb", "base"),
         ("fb_gate", "fb"), ("fb_gate", "base"))
    for a, b in pairs:
        A = df[df.processor == a].reset_index(drop=True)
        B = df[df.processor == b].reset_index(drop=True)
        if A.empty or B.empty:
            continue
        n = min(len(A), len(B))
        print(f"\n--- {a} - {b} ({n} escenas) ---")
        for c in cols:
            d = A[c].to_numpy()[:n] - B[c].to_numpy()[:n]
            print(f"   {c:20s} media {np.nanmean(d):+7.3f}  mediana {np.nanmedian(d):+7.3f}"
                  f"  gana {int(np.sum(d > 0))}/{n}")


if __name__ == "__main__":
    main()
