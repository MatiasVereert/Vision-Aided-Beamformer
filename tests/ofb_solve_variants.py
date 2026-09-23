"""
ofb_solve_variants.py
=====================
LAS DOS FORMAS DE RESOLVER EL MISMO SISTEMA, COMPARADAS.

El nucleo calcula w = B e_ref / tr(B) con B = Phi_NN^-1 Phi_XX. Hay dos maneras
de llegar al mismo numero:

  'direct' (historica, y la rapida en numpy)
      B = solve(Phi_NN, Phi_XX)        <- las M columnas
      lambda = tr(B) ;  w = B[:,ref]/lambda
      Calcula las M columnas de B solo para sumarle la diagonal, y despues usa
      UNA. Las otras M-1 se tiran.          chol M^3/6 + M RHS M^3 = 1.17 M^3

  'chol' (la del PORT)
      Phi_NN = LB LB^H ,  Phi_XX = LA LA^H
      lambda = || LB^-1 LA ||_F^2      <- identidad: tr(B^-1A) = ||LB^-1 LA||_F^2
      w = Phi_NN^-1 Phi_XX[:,ref] / lambda        <- UN solo lado derecho
                                            0.50 M^3 + O(M^2)  -> 2.3x menos

La identidad existe SOLO porque el nucleo no resta: pide el Cholesky de Phi_XX,
o sea que Phi_XX sea definida positiva. Con Phi_SS = Phi_XX - Phi_NN eso era
falso en el 99.6 % de los bins.

QUE MIDE ESTE TEST
------------------
  A) equivalencia por bin sobre estado sintetico, en funcion de cuantos frames
     se acumularon (el transitorio de rango: Phi_XX arranca en rango 1).
  B) equivalencia END-TO-END sobre una escena MIRD: muestra a muestra y en
     PESQ/STOI/SI-SDR.
  C) el harness del benchmark, 8 celdas, las dos formas.

USO
---
    conda activate tesis_beam
    python tests/ofb_solve_variants.py
    python tests/ofb_solve_variants.py --no-mird
"""

import os
import sys
import time
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


def part_a(args):
    """Equivalencia por bin, y el transitorio de rango de Phi_XX."""
    rng = np.random.default_rng(0)
    K, M = 257, args.M
    print(f"\n--- A) equivalencia por bin (K={K}, M={M}) ---")
    print(f"{'frames':>7} {'error rel. max':>16}   (Phi_XX alcanza rango completo en M frames)")
    for nf in (1, 4, M // 2, M, 2 * M, 200):
        c1 = SoudenCore(K, M, M // 2)
        c2 = SoudenCore(K, M, M // 2, solve_mode="chol")
        for _ in range(nf):
            X = (rng.standard_normal((K, M)) + 1j * rng.standard_normal((K, M))) / np.sqrt(2)
            ms, mn = rng.random(K) ** 8, (1 - rng.random(K)) ** 8
            c1.update(X, ms, mn)
            c2.update(X, ms, mn)
        w1, w2 = c1.solve(), c2.solve()
        print(f"{nf:7d} {np.max(np.abs(w1 - w2)) / np.max(np.abs(w1)):16.2e}")

    print(f"\n--- costo en numpy (NO es el del port: numpy hace 'direct' en LAPACK) ---")
    c = {m: SoudenCore(K, M, M // 2, solve_mode=m) for m in ("direct", "chol")}
    X = (rng.standard_normal((K, M)) + 1j * rng.standard_normal((K, M))) / np.sqrt(2)
    for m in c:
        for _ in range(50):
            c[m].update(X, np.full(K, .3), np.full(K, .7))
        c[m].solve()
        t0 = time.perf_counter()
        for _ in range(20):
            c[m].solve()
        print(f"  {m:<7} {1e3*(time.perf_counter()-t0)/20:7.3f} ms")
    print("  (en el port la relacion se da vuelta: 1.17 M^3 contra 0.50 M^3)")


def part_b(args):
    """End-to-end sobre una escena MIRD: muestra a muestra y en metricas."""
    prov = MirdDatasetProvider(root_dir=MIRD_ROOT)
    print(f"\n--- B) end-to-end, escena MIRD rt60={args.rt60} iSIR={args.isir:g} ---")
    mic, ref, cfg = build_scene(prov, args.rt60, 0, 45, args.isir, args.dur)
    out = {}
    for m in ("direct", "chol"):
        t0 = time.perf_counter()
        y, _ = OFB_MVDR(smooth=args.smooth, solve_mode=m, return_weights=False).process(mic, cfg)
        out[m] = y
        pq, st, sd = metrics(y, ref)
        print(f"  {m:<7} PESQ {pq:.4f}  STOI {st:.4f}  SI-SDR {sd:+.3f}   "
              f"({time.perf_counter()-t0:.1f} s)")
    d = out["chol"] - out["direct"]
    n = int(0.2 * FS)                      # el transitorio de rango son M frames
    print(f"  max|dif| global      {np.max(np.abs(d)):.3e}")
    print(f"  max|dif| tras 200 ms {np.max(np.abs(d[n:])):.3e}   <- pasado el transitorio")
    print(f"  SI-SDR entre las dos salidas: {si_sdr(out['chol'], out['direct']):.1f} dB")


def part_c(args):
    """El harness del benchmark, las dos formas."""
    import tensorflow as tf
    from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search

    out_dir = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "ofb_solve_variants")
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
        'rt60': [0.360, 0.610], 'target_angle': [0], 'target_dist': [1.0],
        'interf_configs': [[(45, 1.0)]], 'isir_db': [-5.0, 0.0, 5.0, 15.0],
        'mismatch_gain': [0], 'mismatch_phase': [0],
        'use_wpe': [False], 'wpe_method': ['online'], 'wpe_taps': [7], 'wpe_delay': [2],
        'error_angle_deg': [0.0], 'error_distance_m': [0.0],
    }
    df = run_mird_grid_search(
        grid_params=param_grid, dataset_provider=MirdDatasetProvider(root_dir=MIRD_ROOT),
        processors={m: OFB_MVDR(smooth=args.smooth, solve_mode=m, return_weights=False)
                    for m in ("direct", "chol")},
        scene_base_config=base_config, output_dir=out_dir,
        interpreter_1=itp[0], interpreter_2=itp[1],
        apply_dtln_post=False, save_catalog=False)
    cols = [c for c in ["Delta_bf_PESQ_early", "Delta_bf_STOI_early",
                        "Delta_bf_SDR_early", "Delta_bf_SIR_early"] if c in df.columns]
    print("\n--- C) harness MIRD, Delta PESQ por iSIR ---")
    print(df.pivot_table(index="processor", columns="isir_db",
                         values="Delta_bf_PESQ_early").round(4).to_string())
    print("\n--- mediana sobre las celdas ---")
    print(df.groupby("processor")[cols].median().round(4).to_string())
    k = [c for c in ("rt60", "isir_db") if c in df.columns]
    a = df[df.processor == "direct"].set_index(k)["Delta_bf_PESQ_early"]
    b = df[df.processor == "chol"].set_index(k)["Delta_bf_PESQ_early"]
    d = b - a.reindex(b.index)
    print(f"\n(chol - direct) Delta PESQ: media {d.mean():+.5f}  |max| {np.abs(d).max():.5f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--M", type=int, default=12)
    ap.add_argument("--dur", type=float, default=10.0)
    ap.add_argument("--smooth", type=float, default=0.2)
    ap.add_argument("--rt60", type=float, default=0.610)
    ap.add_argument("--isir", type=float, default=0.0)
    ap.add_argument("--no-mird", action="store_true")
    args = ap.parse_args()
    part_a(args)
    part_b(args)
    if not args.no_mird:
        part_c(args)


if __name__ == "__main__":
    main()
