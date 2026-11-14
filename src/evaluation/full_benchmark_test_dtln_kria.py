#!/usr/bin/env python3
"""full_benchmark_test_dtln_kria.py -- el banco, sobre RIRs MEDIDAS con el aro.

MISMO MOTOR, OTRO DATASET. No duplica nada: importa run_mird_grid_search() de
full_benchmark_test_dtln_mird.py y le pasa un KriaRirProvider. El motor son
1280 lineas con cacheo por nodos, emulacion de punto fijo del WPE, metricas
intrusivas y no intrusivas y catalogo H5; copiarlo para cambiar el origen de
las RIRs habria dejado dos copias para mantener en sincronia, y la segunda
siempre se atrasa.

QUE CAMBIA RESPECTO DE LA CORRIDA MIRD

  LA GRILLA SE LEE, NO SE ELIGE. El MIRD es una grilla densa: 13 angulos cada
  15 grados, 2 distancias, 3 T60. Esta base es DISCRETA y chica -- son las
  posiciones donde efectivamente se puso el parlante -- asi que los ejes salen
  de las mediciones que hay. Pedir 45 grados porque "queda lindo en la figura"
  seria pedir una interpolacion que no existe; el provider avisa cuanto tuvo
  que moverse y ese aviso no hay que ignorarlo.

  LOS EJES CAMBIAN DE SIGNIFICADO. 'rt60' pasa a ser el nombre de la SALA y
  'mird_spacing' el de la posicion del ARREGLO. Se dejan con esos nombres a
  proposito: son los ejes que el motor ya sabe barrer, cachear y etiquetar, y
  renombrarlos obligaria a tocar el motor entero sin ganar nada.

  EL MICROFONO DE REFERENCIA ES ch6, NO M // 2. Con los 11 canales de hoy
  M // 2 cae en ch5. Lo pone ref_mic_mode, que sale del provider.

  LA SALIDA LLEVA OTRO PREFIJO ('kria'), para no pisar los resultados del MIRD
  en el mismo directorio.

QUE NO CAMBIA: los procesadores, las metricas, el cacheo, el DTLN. Si una
comparacion da distinto entre MIRD y la Kria, la diferencia esta en las RIRs y
en la geometria, que es exactamente lo que se quiere medir.

USO
    conda run -n tesis_beam python src/evaluation/full_benchmark_test_dtln_kria.py --list
    conda run -n tesis_beam python src/evaluation/full_benchmark_test_dtln_kria.py \\
        --room lab --array c1 --target f_r --interf f_l --isir -5 0 5
"""
import argparse
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
for _p in (SRC_DIR, SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from propagation.kria_rir_loader import KriaRirProvider      # noqa: E402
from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search  # noqa: E402
from evaluation.bf_wrappers import NM_MVDR_OFB_AUTO          # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
SIGNALS = os.path.join(PROJECT_ROOT, "tools", "data", "signals")
ARRAY_CENTER = [3.0, 3.0, 1.2]


def build_config(prov, room, array, dims_default=(5.0, 4.0, 2.6)):
    """base_config con lo que el dataset medido impone."""
    prov.load_rir(room, array, source=prov.sources(room, array)[0])
    dims = prov.room_dims(room)
    if dims is None:
        dims = dims_default
        print(f"[kria] OJO: la sala '{room}' no tiene dimensiones cargadas; "
              f"se usan {dims} solo para el catalogo H5. No afecta metricas, "
              f"pero si queres la distancia critica, cargalas con "
              f"`measure.py room add --slug {room} --dims X Y Z`.")
    geom = prov.array_geometry(room, array)
    ref = prov.ref_mic_index(room, array)
    act = prov.active_channels(room, array)
    print(f"[kria] arreglo: M = {len(act)} canales {act}, "
          f"referencia indice {ref} (ch{act[ref]}), "
          f"radio {np.linalg.norm(geom[0, :2]) * 1e3:.1f} mm")
    return {
        'fs': prov.get_current_fs(),
        'duration': 15,
        't_early': 0.050,
        'array_center': ARRAY_CENTER,
        'mird_spacing': array,          # <- el eje "arreglo"
        'out_prefix': 'kria',
        'room_dims': list(dims),
        'ref_mic_mode': ref,            # <- ch6, no M // 2
        'snr_db': 60.0,
        'source_path': os.path.join(
            SIGNALS, "p002_emo_adoration_sentences.wav"),
        'interf_paths': [os.path.join(SIGNALS, "techno_gated commune.wav")],
        'wpe_taps': 7, 'wpe_delay': 3, 'wpe_alpha': 0.9999,
        'wpe_stft_size': 512, 'wpe_stft_shift': 128,
        'wpe_fixed_bits': None, 'wpe_fixed_round': 'nearest',
        'wpe_backend': 'cov',
        'wpe_block_L': 512, 'wpe_block_shift': 2, 'wpe_block_iters': 2,
        'wpe_block_reg': 1e-6, 'wpe_block_solver': 'cholesky',
        'wpe_block_mode': 'resolve',
        'stft_window': 512, 'stft_overlap': 384,
        'eval_references': ['early'],
        'dtln_model_path': os.path.join(
            SRC_DIR, "dnn_denoise", "models", "model_quant_1.tflite"),
    }


def build_grid(prov, room, array, target, interf, isir):
    """La grilla, con las posiciones REALES de las mediciones."""
    tab = {s: (d, az, el) for s, d, az, el, _ in prov.scene_table(room, array)}
    faltan = [s for s in ([target] + list(interf)) if s not in tab]
    if faltan:
        raise SystemExit(
            f"[kria] estas fuentes no existen o no tienen posicion: {faltan}\n"
            f"        hay: {sorted(tab)}\n"
            f"        (una fuente sin posicion medida ni estimable no se puede "
            f"ubicar en la escena)")
    td, taz, _ = tab[target]
    print(f"[kria] objetivo '{target}': dist {td:.2f} m, az {taz:.1f} deg")
    cfgs = []
    for s in interf:
        d, az, _ = tab[s]
        print(f"[kria] interferente '{s}': dist {d:.2f} m, az {az:.1f} deg")
        cfgs.append((az, d))
    return {
        'rt60': [room],                 # <- el eje "sala"
        'target_angle': [taz],
        'target_dist': [td],
        'interf_configs': [cfgs] if cfgs else [[]],
        'isir_db': list(isir),
        'mismatch_gain': [0], 'mismatch_phase': [0],
        'use_wpe': [False], 'wpe_method': ['online'],
        'wpe_taps': [7], 'wpe_delay': [2],
        'error_angle_deg': [0.0], 'error_distance_m': [0.0],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=None, help="raiz de data/rirs")
    ap.add_argument("--list", action="store_true",
                    help="mostrar las mediciones y salir")
    ap.add_argument("--room"); ap.add_argument("--array")
    ap.add_argument("--target"); ap.add_argument("--interf", nargs="*", default=[])
    ap.add_argument("--isir", type=float, nargs="+", default=[0.0])
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "results"))
    ap.add_argument("--no-dtln", action="store_true",
                    help="correr sin la etapa neuronal")
    a = ap.parse_args(argv)

    prov = KriaRirProvider(**({} if a.root is None else {"root": a.root}))
    if a.list or not (a.room and a.array and a.target):
        print(prov.describe())
        print("\nPara correr hace falta --room --array --target "
              "(y opcionalmente --interf).")
        for r in prov.rooms():
            for arr in prov.arrays(r):
                print(f"  ej:  --room {r} --array {arr} "
                      f"--target {prov.sources(r, arr)[0]}")
        return 0

    base = build_config(prov, a.room, a.array)
    grid = build_grid(prov, a.room, a.array, a.target, a.interf, a.isir)
    processors = {
        "NM_MVDR_OFB_AUTO": NM_MVDR_OFB_AUTO(smooth=0.2, sharpen_exp=8.0),
        "NM_MVDR_OFB_AUTO_BAN": NM_MVDR_OFB_AUTO(smooth=0.2, sharpen_exp=8.0,
                                                 ban=True),
    }
    # Los interpretes del DTLN se crean aca y se PASAN: el motor decide si
    # corre la etapa neuronal con `interpreter_1 is not None`, asi que si no se
    # los pasa se corre sin DTLN y en silencio. Mismo bloque que el main() de
    # la corrida MIRD, para que las dos sean comparables.
    itp1 = itp2 = None
    if not a.no_dtln:
        try:
            import tensorflow as tf
            models = os.path.join(SRC_DIR, "dnn_denoise", "models")
            itp1 = tf.lite.Interpreter(
                model_path=os.path.join(models, "model_quant_1.tflite"))
            itp1.allocate_tensors()
            itp2 = tf.lite.Interpreter(
                model_path=os.path.join(models, "model_quant_2.tflite"))
            itp2.allocate_tensors()
            print("[kria] interpretes DTLN TF-Lite listos.")
        except Exception as e:                               # noqa: BLE001
            print(f"[kria] sin DTLN ({e}). Las columnas *_dtln* no van a salir.")
            itp1 = itp2 = None

    os.makedirs(a.out, exist_ok=True)
    df = run_mird_grid_search(grid, prov, processors, base, output_dir=a.out,
                              interpreter_1=itp1, interpreter_2=itp2)
    print(f"\n[kria] {len(df)} filas -> {a.out}/kria_benchmark_metrics.csv")
    cols = [c for c in ("processor", "rt60", "target_angle", "isir_db",
                        "Delta_tot_PESQ_early", "Delta_tot_SIR_early",
                        "Delta_tot_STOI_early") if c in df.columns]
    if cols:
        print(df[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
