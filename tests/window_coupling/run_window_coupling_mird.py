"""
Barrido MIRD: EFECTO DE LA VENTANA DE LA STFT Y DEL ACOPLE BF+DTLN.

Responde las dos preguntas del experimento con UNA sola corrida del benchmark
(`run_mird_grid_search`, el mismo de full_benchmark_test_dtln_mird.py):

  P1) Cambiar la ventana del core del beamformer (hamming -> rectangular),
      cuanto altera el RESULTADO INTERMEDIO (la salida del BF)?
      -> comparar las filas  NM-MVDR_hamming  vs  NM-MVDR_rect
         en las columnas  proc_* / Delta_bf_*.

  P2) Y el RESULTADO FINAL con el DTLN acoplado en la misma STFT?
      -> comparar  NM-MVDR+DTLN_acoplado_hamming  vs  NM-MVDR+DTLN_acoplado_rect
         en las columnas  proc_* (para estos procesadores, proc_* YA es la salida
         de la cadena completa BF+DTLN).

Ademas sale gratis la referencia DESACOPLADA: las columnas `dtln_post_*` de las
filas NM-MVDR_* son la cascada actual (BF -> iSTFT -> re-framing rectangular
nativo del DTLN -> DTLN). Comparar `dtln_post_*` de NM-MVDR_rect contra `proc_*`
de NM-MVDR+DTLN_acoplado_rect responde: acoplar en una sola STFT, cuesta algo?

NOTA: para las filas del procesador ACOPLADO, las columnas `dtln_post_*` aplican
un SEGUNDO DTLN encima (Node 6 del benchmark corre igual). No tienen sentido
fisico; se ignoran. Se deja apply_dtln_post=True porque es lo que produce la
referencia desacoplada de las filas NM-MVDR_*.

Uso:
    python tests/window_coupling/run_window_coupling_mird.py            # rapido (1 escena, WPE on/off)
    python tests/window_coupling/run_window_coupling_mird.py --full     # 2 RT60 x 2 interferencias x 2 iSIR
"""

import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search   # noqa: E402
from evaluation.bf_wrappers import NM_MVDR, DTLN_MB_MVDR_SOUDEN_BAN        # noqa: E402
from evaluation.coupled_wrappers import NM_MVDR_DTLN, SOUDEN_BAN_DTLN      # noqa: E402
from propagation.mird_loader import MirdDatasetProvider                    # noqa: E402

ROOT = "/home/matias/Documents/Tesis/Vision-Aided-Beamformer"
MODEL_1 = f"{ROOT}/src/dnn_denoise/models/model_quant_1.tflite"
MODEL_2 = f"{ROOT}/src/dnn_denoise/models/model_quant_2.tflite"
OUT_DIR = "tests/dataset_out/window_coupling"

# Poner en True para incluir tambien el core con BAN (duplica el tiempo de corrida)
INCLUDE_BAN = False

# Variante "acople total": la mascara del DTLN tambien sale de la STFT compartida
# (una sola FFT en todo el sistema) y desaparece el corrimiento de un frame entre
# mascara y espectro que arrastra el camino por defecto.
INCLUDE_MASK_FROM_STFT = True

# Tercera ventana de control (hann). El default historico del repo es HAMMING,
# no hanning: si se quiere el par hann/rect explicito, poner esto en True.
INCLUDE_HANN = False


def build_processors():
    procs = {
        # --- P1: salida INTERMEDIA del beamformer, misma cadena, distinta ventana
        "NM-MVDR_hamming": NM_MVDR(min_loading=1e-6, alpha=0.99, win_type='hamming'),
        "NM-MVDR_rect": NM_MVDR(min_loading=1e-6, alpha=0.99, win_type='rect'),

        # --- P2: cadena COMPLETA acoplada en una sola STFT
        "NM-MVDR+DTLN_acoplado_hamming": NM_MVDR_DTLN(
            min_loading=1e-6, alpha=0.99, win_type='hamming',
            model1_path=MODEL_1, model2_path=MODEL_2),
        "NM-MVDR+DTLN_acoplado_rect": NM_MVDR_DTLN(
            min_loading=1e-6, alpha=0.99, win_type='rect',
            model1_path=MODEL_1, model2_path=MODEL_2),
    }
    if INCLUDE_HANN:
        procs["NM-MVDR_hann"] = NM_MVDR(min_loading=1e-6, alpha=0.99, win_type='hann')
        procs["NM-MVDR+DTLN_acoplado_hann"] = NM_MVDR_DTLN(
            min_loading=1e-6, alpha=0.99, win_type='hann',
            model1_path=MODEL_1, model2_path=MODEL_2)

    if INCLUDE_MASK_FROM_STFT:
        procs["NM-MVDR+DTLN_acoplado_rect_maskSTFT"] = NM_MVDR_DTLN(
            min_loading=1e-6, alpha=0.99, win_type='rect', mask_from_stft=True,
            model1_path=MODEL_1, model2_path=MODEL_2)

    if INCLUDE_BAN:
        procs.update({
            "SOUDEN-BAN_hamming": DTLN_MB_MVDR_SOUDEN_BAN(min_loading=1e-6, win_type='hamming'),
            "SOUDEN-BAN_rect": DTLN_MB_MVDR_SOUDEN_BAN(min_loading=1e-6, win_type='rect'),
            "SOUDEN-BAN+DTLN_acoplado_rect": SOUDEN_BAN_DTLN(
                min_loading=1e-6, win_type='rect',
                model1_path=MODEL_1, model2_path=MODEL_2),
        })
    return procs


def main():
    try:
        itp1 = tf.lite.Interpreter(model_path=MODEL_1)
        itp1.allocate_tensors()
        itp2 = tf.lite.Interpreter(model_path=MODEL_2)
        itp2.allocate_tensors()
        print("[*] DTLN TF-Lite interpreters successfully allocated.")
    except Exception as e:
        print(f"[!] No se pudieron cargar los modelos DTLN: {e}")
        itp1, itp2 = None, None

    provider = MirdDatasetProvider(root_dir=os.path.abspath(f"{ROOT}/tools/data/rirs/mird"))

    # Config base: copia de la de full_benchmark_test_dtln_mird.py (misma escena,
    # mismo front-end) para que los numeros sean comparables con las corridas previas.
    base_config = {
        'fs': 16000,
        'duration': 15,
        't_early': 0.050,
        'array_center': [3.0, 3.0, 1.2],
        'mird_spacing': "3-3-3-8-3-3-3",

        'snr_db': 60.0,
        'source_path': f"{ROOT}/tools/data/signals/p002_emo_adoration_sentences.wav",
        'interf_paths': [f"{ROOT}/tools/data/signals/techno_gated commune.wav"],

        'wpe_taps': 7,
        'wpe_delay': 3,
        'wpe_alpha': 0.9999,
        'wpe_stft_size': 512,
        'wpe_stft_shift': 128,
        'wpe_fixed_bits': None,
        'wpe_fixed_round': 'nearest',
        'wpe_backend': 'cov',
        'wpe_block_L': 512,
        'wpe_block_shift': 2,
        'wpe_block_iters': 2,
        'wpe_block_reg': 1e-6,
        'wpe_block_solver': 'cholesky',
        'wpe_block_mode': 'resolve',

        'stft_window': 512,
        'stft_overlap': 384,

        'eval_references': ['anechoic', 'early', 'reverberant'],
        'dtln_model_path': MODEL_1,
        'dtln_model2_path': MODEL_2,
        # OJO: 'stft_win_type' NO se fija aca a proposito. Cada wrapper trae su
        # propia ventana via win_type=..., que tiene prioridad. Si se pusiera aca,
        # solo afectaria a los wrappers que NO la especifican.
    }

    full = "--full" in sys.argv
    param_grid = {
        # Solo hay RIRs con spacing 3-3-3-8-3-3-3 para RT60 0.360 y 0.610.
        'rt60': [0.360, 0.610] if full else [0.610],
        'target_angle': [0],
        'target_dist': [1.0],
        'interf_configs': ([[(45, 1.0)], [(90, 2.0)]] if full else [[(45, 1.0)]]),
        'isir_db': [-5, 0] if full else [-5],
        'mismatch_gain': [0],
        'mismatch_phase': [0],
        'use_wpe': [True] if full else [True, False],
        'wpe_method': ['online'],
        'wpe_taps': [7],
        'wpe_delay': [2],
        'error_angle_deg': [0.0],
        'error_distance_m': [0.0],
    }

    df = run_mird_grid_search(
        grid_params=param_grid,
        dataset_provider=provider,
        processors=build_processors(),
        scene_base_config=base_config,
        output_dir=OUT_DIR,
        interpreter_1=itp1,
        interpreter_2=itp2,
        save_catalog=False,          # no hace falta el catalogo H5/polares para esto
        apply_dtln_post=True,        # da la referencia DESACOPLADA de las filas NM-MVDR_*
    )

    summarize(df)
    return df


def summarize(df, ref='early'):
    """Tabla compacta: salida del BF (proc_*) y cascada desacoplada (dtln_post_*)."""
    metrics = ['PESQ', 'STOI', 'SI-SDR', 'SIR', 'SAR']
    cols = ['processor', 'use_wpe']
    for m in metrics:
        for pre in ('proc', 'dtln_post'):
            c = f"{pre}_{m}_{ref}"
            if c in df.columns:
                cols.append(c)
    cols = [c for c in cols if c in df.columns]

    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 50)
    print(f"\n=== VENTANA / ACOPLE -- referencia '{ref}' ===")
    print(df[cols].round(3).to_string(index=False))

    print("\n--- P1: efecto de la ventana en la SALIDA DEL BF (rect - hamming) ---")
    _delta(df, 'NM-MVDR_rect', 'NM-MVDR_hamming', 'proc', metrics, ref)

    print("\n--- P2: efecto de la ventana en la SALIDA FINAL ACOPLADA (rect - hamming) ---")
    _delta(df, 'NM-MVDR+DTLN_acoplado_rect', 'NM-MVDR+DTLN_acoplado_hamming',
           'proc', metrics, ref)

    print("\n--- P3: acoplado(rect) - desacoplado(rect) [misma cadena, una sola STFT] ---")
    _delta_cross(df, ('NM-MVDR+DTLN_acoplado_rect', 'proc'),
                 ('NM-MVDR_rect', 'dtln_post'), metrics, ref)

    print("\n--- P4: acoplado(rect) - desacoplado(hamming) [vs la cadena ACTUAL] ---")
    _delta_cross(df, ('NM-MVDR+DTLN_acoplado_rect', 'proc'),
                 ('NM-MVDR_hamming', 'dtln_post'), metrics, ref)

    if 'NM-MVDR+DTLN_acoplado_rect_maskSTFT' in set(df['processor']):
        print("\n--- P5: acople TOTAL (mascara desde la STFT compartida) - acoplado(rect) ---")
        _delta(df, 'NM-MVDR+DTLN_acoplado_rect_maskSTFT',
               'NM-MVDR+DTLN_acoplado_rect', 'proc', metrics, ref)

    print("\n=== PROMEDIO SOBRE TODAS LAS ESCENAS (mediana) ===")
    med_cols = [c for c in df.columns
                if any(c == f"{pre}_{m}_{ref}" for pre in ('proc', 'dtln_post') for m in metrics)]
    print(df.groupby('processor')[med_cols].median().round(3).to_string())


# Columnas que identifican la escena EN EL DATAFRAME. Ojo: run_mird_grid_search
# NO guarda la geometria de las interferencias (solo 'N_interferences'), asi que
# dos escenas que difieren unicamente en el angulo del interferente son
# indistinguibles por columnas -> el emparejamiento se hace por ORDEN de aparicion
# dentro de cada procesador (el barrido recorre los experimentos en orden fijo).
LABEL_COLS = ['rt60', 'isir_db', 'use_wpe']


def _delta(df, proc_a, proc_b, prefix, metrics, ref):
    _delta_cross(df, (proc_a, prefix), (proc_b, prefix), metrics, ref)


def _delta_cross(df, a, b, metrics, ref):
    """
    Diferencia POR ESCENA (mismo indice de experimento en los dos procesadores)
    y mediana sobre escenas. La mediana es la que hay que mirar: una sola escena
    no alcanza para separar el efecto de la ventana del ruido de la escena.
    """
    (proc_a, pre_a), (proc_b, pre_b) = a, b
    ca = {m: f"{pre_a}_{m}_{ref}" for m in metrics}
    cb = {m: f"{pre_b}_{m}_{ref}" for m in metrics}
    metrics = [m for m in metrics if ca[m] in df.columns and cb[m] in df.columns]

    da = df[df['processor'] == proc_a].reset_index(drop=True)
    db = df[df['processor'] == proc_b].reset_index(drop=True)
    if da.empty or db.empty:
        print(f"  (faltan filas: {proc_a} o {proc_b})")
        return
    n = min(len(da), len(db))

    deltas = {m: (da[ca[m]].to_numpy()[:n] - db[cb[m]].to_numpy()[:n]) for m in metrics}
    for i in range(n):
        tag = " ".join(f"{k}={da.iloc[i][k]}" for k in LABEL_COLS if k in da.columns)
        parts = [f"{m} {deltas[m][i]:+.3f}" for m in metrics]
        print(f"  esc{i:02d} {tag:32s} | " + "  ".join(parts))
    if n > 1:
        parts = []
        for m in metrics:
            v = np.asarray(deltas[m], dtype=float)
            v = v[np.isfinite(v)]
            parts.append(f"{m} {np.median(v):+.3f}" if v.size else f"{m}   n/a")
        print(f"  {'MEDIANA (' + str(n) + ' escenas)':38s} | " + "  ".join(parts))


if __name__ == "__main__":
    main()
