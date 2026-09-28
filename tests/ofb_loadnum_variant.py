"""
ofb_loadnum_variant.py
======================
EL NUMERADOR, ¿CON LA MATRIZ CARGADA O SIN CARGAR?

`_solve_chol` le pone carga diagonal a Phi_SS para que su Cholesky exista
SIEMPRE (al arranque Phi_SS tiene rango 1). Pero despues extrae la columna del
numerador, c = Phi_SS e_ref, de la matriz SIN cargar:

    A  = Phi_SS + I*(min_loading*tr/M + 1e-30)
    LA = chol(A)            ->  lambda = ||LB^-1 LA||_F^2     <- A CARGADA
    y  = LB^-1 Phi_SS[:,ref]                                  <- SIN cargar

El port en C++ usa la cargada en los dos lugares, porque no guarda una copia
sin cargar. Este test mide si esa diferencia se ve en las metricas.

OJO CON min_loading: el default de Python es 1e-9, que en float64 hace la
diferencia invisible. El PORT corre en float32 con min_loading=1e-5 (el 1e-9
queda debajo del epsilon de float). Asi que la comparacion que importa es a
1e-5, no a 1e-9; se corren las dos.

USO
---
    conda activate tesis_beam
    python tests/ofb_loadnum_variant.py            # A + B (rapido)
    python tests/ofb_loadnum_variant.py --grid     # + la grilla MIRD
"""

import os
import sys
import argparse

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for d in (os.path.join(PROJECT_ROOT, "src"), os.path.join(PROJECT_ROOT, "tests")):
    if d not in sys.path:
        sys.path.insert(0, d)

from beamforming.mask.ofb import SoudenCore                          # noqa: E402
from evaluation.bf_wrappers import OFB_MVDR                          # noqa: E402
from propagation.mird_loader import MirdDatasetProvider              # noqa: E402
from ofb_refactor_equivalence import (build_scene, metrics, si_sdr, FS,
                                      MIRD_ROOT, MODEL_1, SPACING,
                                      SOURCE_WAV, INTERF_WAV)        # noqa: E402

MODES = ("chol", "chol_loadnum")


def part_a(args):
    """Cuanto se separan los pesos, por bin, en funcion de min_loading."""
    rng = np.random.default_rng(0)
    K, M = 257, args.M
    print(f"\n--- A) separacion de los pesos por bin (K={K}, M={M}, {args.frames} frames) ---")
    print(f"{'min_loading':>12} {'err rel max':>13} {'err rel mediano':>17}")
    for ml in (1e-9, 1e-7, 1e-5, 1e-4, 1e-3):
        cores = {m: SoudenCore(K, M, M // 2, min_loading=ml, solve_mode=m) for m in MODES}
        r = np.random.default_rng(0)
        for _ in range(args.frames):
            X = (r.standard_normal((K, M)) + 1j * r.standard_normal((K, M))) / np.sqrt(2)
            ms, mn = r.random(K) ** 8, (1 - r.random(K)) ** 8
            for c in cores.values():
                c.update(X, ms, mn)
        w = {m: cores[m].solve() for m in MODES}
        d = np.linalg.norm(w["chol_loadnum"] - w["chol"], axis=1)
        n = np.linalg.norm(w["chol"], axis=1)
        rel = d / np.maximum(n, 1e-30)
        print(f"{ml:12.0e} {rel.max():13.2e} {np.median(rel):17.2e}")
    del rng


def part_b(args):
    """End-to-end sobre una escena MIRD, con el min_loading del PORT."""
    prov = MirdDatasetProvider(root_dir=MIRD_ROOT)
    for ml in args.min_loading:
        print(f"\n--- B) escena MIRD rt60={args.rt60} iSIR={args.isir:g}  min_loading={ml:g} ---")
        mic, ref, cfg = build_scene(prov, args.rt60, 0, 45, args.isir, args.dur)
        out = {}
        for m in MODES:
            y, _ = OFB_MVDR(smooth=args.smooth, solve_mode=m, min_loading=ml,
                            return_weights=False).process(mic, cfg)
            out[m] = y
            pq, st, sd = metrics(y, ref)
            print(f"  {m:<14} PESQ {pq:.4f}  STOI {st:.4f}  SI-SDR {sd:+.3f}")
        d = out["chol_loadnum"] - out["chol"]
        print(f"  max|dif| de la senal de salida: {np.max(np.abs(d)):.3e}")
        print(f"  SI-SDR entre las dos salidas:   {si_sdr(out['chol_loadnum'], out['chol']):.1f} dB")


def part_c(args):
    """La grilla MIRD: 2 rt60 x 4 iSIR = 8 celdas, las dos variantes."""
    import tensorflow as tf
    from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search

    ml = args.min_loading[-1]
    out_dir = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "ofb_loadnum")
    os.makedirs(out_dir, exist_ok=True)
    itp = []
    for path in (MODEL_1, MODEL_1.replace("_1.tflite", "_2.tflite")):
        i = tf.lite.Interpreter(model_path=path)
        i.allocate_tensors()
        itp.append(i)
    base_config = {
        'fs': FS, 'duration': args.dur, 't_early': 0.050,
        'array_center': [3.0, 3.0, 1.2], 'mird_spacing': SPACING, 'snr_db': 60.0,
        'source_path': SOURCE_WAV, 'interf_paths': [INTERF_WAV],
        'wpe_taps': 7, 'wpe_delay': 2, 'wpe_alpha': 0.9999,
        'wpe_stft_size': 512, 'wpe_stft_shift': 128,
        'wpe_fixed_bits': None, 'wpe_fixed_round': 'nearest', 'wpe_backend': 'cov',
        'wpe_block_L': 512, 'wpe_block_shift': 2, 'wpe_block_iters': 2,
        'wpe_block_reg': 1e-6, 'wpe_block_solver': 'cholesky', 'wpe_block_mode': 'resolve',
        'stft_window': 512, 'stft_overlap': 384, 'eval_references': ['early'],
        'dtln_model_path': MODEL_1,
    }
    param_grid = {
        'rt60': args.rt60s, 'target_angle': [0], 'target_dist': [1.0],
        'interf_configs': [[(a, 1.0)] for a in args.angles], 'isir_db': args.isirs,
        'mismatch_gain': [0], 'mismatch_phase': [0],
        'use_wpe': [False], 'wpe_method': ['online'], 'wpe_taps': [7], 'wpe_delay': [2],
        'error_angle_deg': [0.0], 'error_distance_m': [0.0],
    }
    df = run_mird_grid_search(
        grid_params=param_grid, dataset_provider=MirdDatasetProvider(root_dir=MIRD_ROOT),
        processors={m: OFB_MVDR(smooth=args.smooth, solve_mode=m, min_loading=ml,
                                return_weights=False) for m in MODES},
        scene_base_config=base_config, output_dir=out_dir,
        interpreter_1=itp[0], interpreter_2=itp[1],
        apply_dtln_post=False, save_catalog=False)
    cols = [c for c in ["Delta_bf_PESQ_early", "Delta_bf_STOI_early",
                        "Delta_bf_SDR_early", "Delta_bf_SIR_early"] if c in df.columns]
    print(f"\n--- C) grilla MIRD, min_loading={ml:g}: Delta PESQ por iSIR ---")
    print(df.pivot_table(index="processor", columns="isir_db",
                         values="Delta_bf_PESQ_early").round(4).to_string())
    print("\n--- mediana sobre las 8 celdas ---")
    print(df.groupby("processor")[cols].median().round(4).to_string())
    k = [c for c in ("rt60", "isir_db") if c in df.columns]
    for c in cols:
        a = df[df.processor == "chol"].set_index(k)[c]
        b = df[df.processor == "chol_loadnum"].set_index(k)[c]
        d = (b - a.reindex(b.index)).dropna()
        gana = int((d > 0).sum())
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        print(f"(loadnum - chol) {c:<24} media {d.mean():+.5f} +- {se:.5f} (e.e.)"
              f"  |max| {np.abs(d).max():.5f}  gana {gana}/{len(d)}"
              f"  {'<- distinguible de 0' if abs(d.mean()) > 2*se else ''}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--M", type=int, default=12)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--dur", type=float, default=10.0)
    ap.add_argument("--smooth", type=float, default=0.2)
    ap.add_argument("--rt60", type=float, default=0.610)
    ap.add_argument("--isir", type=float, default=0.0)
    ap.add_argument("--min-loading", type=float, nargs="+", default=[1e-9, 1e-5],
                    dest="min_loading")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--rt60s", type=float, nargs="+", default=[0.360, 0.610])
    ap.add_argument("--isirs", type=float, nargs="+", default=[-5.0, 0.0, 5.0, 15.0])
    ap.add_argument("--angles", type=float, nargs="+", default=[45.0])
    args = ap.parse_args()
    part_a(args)
    part_b(args)
    if args.grid:
        part_c(args)


if __name__ == "__main__":
    main()
