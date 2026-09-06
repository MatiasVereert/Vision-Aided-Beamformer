"""
ACTUALIZACION POR BLOQUES: ¿cuanto cuesta desacoplar el calculo de los pesos?

El lazo `NM_MVDR_DSM_FB` es causal, pero no es de latencia minima: para emitir
y(t) hay que ver x(t), actualizar las dos recursiones, descomponer Phi_SS,
invertir Phi_NN y recien ahi multiplicar. Todo eso cae en serie dentro del
periodo de hop (8 ms a hop=128 / 16 kHz).

`block_update=P` corta esa dependencia: el frame t se filtra con los pesos que
quedaron listos en t-1 (o en el ultimo multiplo de P), de modo que el camino
critico se reduce a la FFT, dos productos punto y la sintesis, y lo caro pasa a
ser una etapa de pipeline con P periodos de hop para terminar.

La pregunta es cuanto se pierde. Hay DOS efectos y conviene leerlos separados:

    fb      -> el sistema de hoy (block_update=None). La referencia.
    fb_P1   -> retencion de un frame, pesos recalculados en TODOS los frames.
               Aisla el costo de NO filtrar el frame con sus propios pesos.
    fb_P{k} -> ademas, los pesos del nucleo se recalculan cada k frames.
               Contra fb_P1, aisla el costo de bajar la tasa.

Con alpha=0.99 la constante de tiempo del nucleo es ~100 frames (~0.8 s), asi
que los P chicos deberian ser gratis: no se puede perder informacion que la
propia recursion todavia no incorporo. El barrido llega hasta P=32 (0.26 s)
para encontrar donde empieza a doler.

Uso:
    python tests/window_mismatch/run_dsm_fb_block_update.py [--full] [--fe|--pair|--load]
"""

import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = "/home/matias/Documents/Tesis/Vision-Aided-Beamformer"
sys.path.insert(0, os.path.join(ROOT, "src"))

from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search   # noqa: E402
from evaluation.bf_wrappers import NM_MVDR_DSM_FB                          # noqa: E402
from propagation.mird_loader import MirdDatasetProvider                    # noqa: E402

MODEL_1 = f"{ROOT}/src/dnn_denoise/models/model_quant_1.tflite"
MODEL_2 = f"{ROOT}/src/dnn_denoise/models/model_quant_2.tflite"
OUT_DIR = os.environ.get("SWEEP_OUT", "tests/dataset_out/dsm_fb_block_update")

# La configuracion final del sistema (la misma de tests/dsm_blind_real_run.py).
CFG = dict(mode="fb", win_type='rect', synth='hann', sharpen_exp=8.0,
           smooth=0.5, alpha=0.99)

PERIODS = [1, 2, 4, 8, 16, 32]


def build_processors():
    procs = {"fb": NM_MVDR_DSM_FB(**CFG)}
    for P in PERIODS:
        procs[f"fb_P{P}"] = NM_MVDR_DSM_FB(block_update=P, **CFG)
    return procs


def build_fe_processors():
    """
    EL OTRO EJE: bajar tambien la tasa del FRONT-END de la mascara.

    El front-end es la mitad "barata" por diseño (no invierte nada), pero igual
    descompone Phi_SS para sacar la RTF, asi que en tiempo medido no es barato.
    Esto mide si tambien se le puede bajar la tasa, a P del nucleo fijo en 8.
    """
    procs = {"fb": NM_MVDR_DSM_FB(**CFG)}
    for Pfe in (1, 4, 8, 16):
        procs[f"fb_P8_fe{Pfe}"] = NM_MVDR_DSM_FB(block_update=8, fe_update=Pfe, **CFG)
    return procs


def build_pair_processors():
    """
    EL REPARTO ENTRE LOS DOS EJES.

    El barrido `--fe` mostro que el front-end aguanta el espaciado MUCHO mejor
    que el nucleo: de 1 a 16 cuesta -0.016 PESQ, contra -0.074 del nucleo por el
    mismo factor. No es raro -- `rtf_alpha=0.999` es una constante de tiempo de
    ~1000 frames (8 s) contra los ~100 frames (0.8 s) de `alpha=0.99` -- pero da
    vuelta el reparto: conviene espaciar MAS el front-end y MENOS el nucleo, que
    es al reves de lo que sugiere el costo por llamada.

    Estas combinaciones prueban ese reparto contra el `P=8, fe=4` del barrido
    anterior (0.88 ms, -0.047 PESQ), que es el punto a batir.
    """
    procs = {"fb": NM_MVDR_DSM_FB(**CFG)}
    for P, fe in ((2, 16), (4, 16), (4, 32), (8, 32)):
        procs[f"fb_P{P}_fe{fe}"] = NM_MVDR_DSM_FB(block_update=P, fe_update=fe, **CFG)
    return procs


def build_load_processors():
    """
    ¿CUANTO CUESTA LA REGULARIZACION QUE PIDE float32?

    `min_loading=1e-9` es una carga RELATIVA al nivel de ruido, y en float32
    (eps = 1.2e-7) es invisible: la diagonal cargada es identica a la sin
    cargar, el Cholesky de Phi_NN se va contra un pivote negativo y la cadena
    devuelve NaN. Verificado en el port a C++: con 1e-9 y 1e-7 explota, desde
    1e-6 es estable.

    O sea que pasar el sistema a complex64 no es recompilar: obliga a subir la
    carga ~3 ordenes. Esto mide si esa carga sale gratis en calidad.
    """
    procs = {"fb": NM_MVDR_DSM_FB(**CFG)}                       # min_loading=1e-9
    for ml in (1e-7, 1e-6, 1e-5, 1e-4):
        procs[f"fb_ml{ml:.0e}"] = NM_MVDR_DSM_FB(min_loading=ml, **CFG)
    return procs


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
    procs = (build_load_processors() if "--load" in sys.argv else
             build_pair_processors() if "--pair" in sys.argv else
             build_fe_processors() if "--fe" in sys.argv else build_processors())
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
    print(f"\n=== actualizacion por bloques (ref '{ref}') ===")
    print(df.groupby('processor')[cols].mean().round(3).to_string())

    procset = [p for p in df.processor.unique() if p != "fb"]
    pairs = [(p, "fb") for p in procset]
    if "fb_P1" in procset:
        pairs += [(p, "fb_P1") for p in procset if p != "fb_P1"]
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
