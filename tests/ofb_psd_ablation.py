"""
ofb_psd_ablation.py
===================
CUANTO PESA LA PROYECCION PSD, Y CUANTO CUESTA SACARLA.

El nucleo de Souden calcula Phi_SS = Phi_XX - Phi_NN, y esa resta de dos
estimaciones ruidosas NO tiene por que dar una matriz semidefinida positiva
aunque las dos lo sean. `psd_project=True` la proyecta al cono PSD:

    evals, evecs = eigh(Phi_SS);  evals <- max(evals, 0);  Phi_SS <- V diag(e) V^H

Es la unica DESCOMPOSICION de todo el sistema (el resto es un Cholesky y
sustituciones) y, en un port a mano sobre ARM, el kernel mas caro y mas
incomodo: un Jacobi hermitiano de 12x12 por bin, 257 bins, cada P frames.

Este test pregunta si hace falta. Esta clavada en True desde que se heredo del
camino viejo y NUNCA se barrio en este esquema (el lazo de salida), que es
justamente donde Phi_NN se alimenta de la mascara fundida y podria estar mejor
condicionada que en los esquemas anteriores.

TRES PARTES
-----------
  A (--mird)  el harness del benchmark, 2 salas x 4 iSIR: Delta PESQ/STOI/SDR
              con y sin proyeccion. Es el criterio que decide.
  B (--aro12) la captura real del aro de 12 mics: no intrusivas (DNSMOS),
              divergencia entre las dos salidas, y los WAV para escuchar. M=12
              es el M del port, y el costo del eigh crece con M^3.
  C (--cost)  micro-benchmark del `solve` con y sin proyeccion: cuanto del
              frame se va en el eigh. Es el numero que le pone precio a sacarla.

USO
---
    conda activate tesis_beam
    python tests/ofb_psd_ablation.py                 # las tres
    python tests/ofb_psd_ablation.py --only-cost
"""

import os
import sys
import time
import argparse

import numpy as np
import scipy.signal as sig

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
TESTS_DIR = os.path.join(PROJECT_ROOT, "tests")
for d in (SRC_DIR, TESTS_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import beamforming.mask.ofb as ofb_mod                              # noqa: E402
from beamforming.mask.blind_feedback import SoudenSubtractCore      # noqa: E402
from beamforming.mask.ofb import SoudenCore as _ORIG_CORE           # noqa: E402
from beamforming.mask.ofb import OutputFeedbackMVDR                 # noqa: E402
from evaluation.bf_wrappers import OFB_MVDR, ola_taper              # noqa: E402
from propagation.mird_loader import MirdDatasetProvider             # noqa: E402
from ofb_refactor_equivalence import (build_scene, metrics, si_sdr, FS,
                                      MIRD_ROOT, MODEL_1, SPACING,
                                      SOURCE_WAV, INTERF_WAV)       # noqa: E402
from ofb_block_update_sweep import ARO12_WAV                        # noqa: E402

OUT_ARO = os.path.join(PROJECT_ROOT, "tests", "real_benchmark_out", "ofb_psd_ablation")


class _CoreNoPSD(SoudenSubtractCore):
    """El mismo nucleo con la proyeccion apagada."""
    def __init__(self, *a, **k):
        k['psd_project'] = False
        super().__init__(*a, **k)


class OFB_MVDR_NOPSD(OFB_MVDR):
    """
    `OFB_MVDR` sin la proyeccion PSD. La cristalizacion no expone el knob a
    proposito (es una decision tomada, no un eje de barrido), asi que la
    ablacion se hace cambiando la clase del nucleo SOLO durante esta corrida.
    """
    def process(self, mic_signals, scene_config):
        ofb_mod.SoudenCore = _CoreNoPSD
        try:
            return super().process(mic_signals, scene_config)
        finally:
            ofb_mod.SoudenCore = _ORIG_CORE


def part_a(args):
    """El harness del benchmark: el criterio que decide."""
    import tensorflow as tf
    from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search

    out_dir = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "ofb_psd_ablation")
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
        'interf_configs': [[(45, 1.0)]], 'isir_db': args.isir,
        'mismatch_gain': [0], 'mismatch_phase': [0],
        'use_wpe': [False], 'wpe_method': ['online'], 'wpe_taps': [7], 'wpe_delay': [2],
        'error_angle_deg': [0.0], 'error_distance_m': [0.0],
    }
    df = run_mird_grid_search(
        grid_params=param_grid, dataset_provider=MirdDatasetProvider(root_dir=MIRD_ROOT),
        processors={"PSD": OFB_MVDR(smooth=args.smooth, return_weights=False),
                    "NO_PSD": OFB_MVDR_NOPSD(smooth=args.smooth, return_weights=False)},
        scene_base_config=base_config, output_dir=out_dir,
        interpreter_1=itp[0], interpreter_2=itp[1],
        apply_dtln_post=False, save_catalog=False)

    order = ["PSD", "NO_PSD"]
    cols = [c for c in ["Delta_bf_PESQ_early", "Delta_bf_STOI_early",
                        "Delta_bf_SDR_early", "Delta_bf_SIR_early"] if c in df.columns]
    print("\n--- A) MIRD, Delta PESQ por iSIR (media sobre salas) ---")
    print(df.pivot_table(index="processor", columns="isir_db",
                         values="Delta_bf_PESQ_early").reindex(order).round(3).to_string())
    print("\n--- Delta STOI por iSIR ---")
    print(df.pivot_table(index="processor", columns="isir_db",
                         values="Delta_bf_STOI_early").reindex(order).round(3).to_string())
    print("\n--- mediana sobre las 8 celdas ---")
    print(df.groupby("processor")[cols].median().reindex(order).round(4).to_string())
    key = [c for c in ("rt60", "isir_db") if c in df.columns]
    a = df[df.processor == "PSD"].set_index(key)["Delta_bf_PESQ_early"]
    b = df[df.processor == "NO_PSD"].set_index(key)["Delta_bf_PESQ_early"]
    d = (b - a.reindex(b.index))
    print(f"\n(NO_PSD - PSD) Delta PESQ: media {d.mean():+.4f}  "
          f"peor {d.min():+.4f}  celdas donde NO_PSD gana: {int((d > 0).sum())}/{len(d)}")
    df.to_parquet(os.path.join(out_dir, "psd_ablation.parquet"))


def part_b(args):
    """La captura real del aro: M=12, sin referencia limpia, con WAVs."""
    import soundfile as sf
    from evaluation.full_benchmark_real import (load_multichannel_wav, energy_vad,
                                                segmental_snr_estimate, rms_db)
    os.makedirs(OUT_ARO, exist_ok=True)
    mic_all, fs = load_multichannel_wav(args.aro_wav, expected_channels=12)
    i0 = int(round(args.aro_skip * fs))
    i1 = min(mic_all.shape[1], i0 + int(round(args.aro_dur * fs)))
    mic = np.ascontiguousarray(mic_all[:, i0:i1])
    ref_mic = mic.shape[0] // 2
    cfg = {'fs': fs, 'stft_window': 512, 'stft_overlap': 384,
           'dtln_model_path': MODEL_1, 'ref_mic_idx': ref_mic}

    outs = {}
    for name, cls in (("PSD", OFB_MVDR), ("NO_PSD", OFB_MVDR_NOPSD)):
        t0 = time.perf_counter()
        y, _ = cls(smooth=args.smooth, return_weights=False).process(mic, cfg)
        outs[name] = y
        print(f"[*] aro12 {name}: {time.perf_counter()-t0:.1f} s de proceso")

    x_ref = mic[ref_mic]
    peak = max(max(np.max(np.abs(y)) for y in outs.values()), np.max(np.abs(x_ref)))
    sc = 0.95 / (peak + 1e-12)
    sf.write(os.path.join(OUT_ARO, "ref_mic_raw.wav"),
             (x_ref * sc).astype(np.float32), fs, subtype="PCM_16")
    for name, y in outs.items():
        sf.write(os.path.join(OUT_ARO, f"ofb_{name}.wav"),
                 (y * sc).astype(np.float32), fs, subtype="PCM_16")

    try:
        from evaluation.nonintrusive import compute_nonintrusive, NONINTRUSIVE_KEYS
    except Exception:
        compute_nonintrusive, NONINTRUSIVE_KEYS = None, []
    print("\n--- B) aro12 (M=12), no intrusivas ---")
    print(f"{'senal':<10} {'RMS dBFS':>9} {'segSNR':>7}" +
          "".join(f"{k.replace('DNSMOS_','').replace('SQUIM_','SQ_'):>9}"
                  for k in NONINTRUSIVE_KEYS))
    for name, x in [("ref_mic", x_ref)] + list(outs.items()):
        ni = compute_nonintrusive(x, fs) if compute_nonintrusive else {}
        seg = segmental_snr_estimate(x, energy_vad(x, fs))
        print(f"{name:<10} {rms_db(x):9.1f} {seg:7.2f}" +
              "".join(f"{ni.get(k, np.nan):9.2f}" for k in NONINTRUSIVE_KEYS))
    print(f"\nSI-SDR NO_PSD vs PSD: {si_sdr(outs['NO_PSD'], outs['PSD']):.1f} dB "
          f"(cuanto se aparta una salida de la otra)")
    print(f"[*] WAVs (misma escala): {OUT_ARO}")


def part_c(args):
    """Cuanto del frame se va en el eigh: el precio de la proyeccion."""
    K, M, reps = 257, args.cost_m, args.cost_reps
    rng = np.random.default_rng(0)
    X = (rng.standard_normal((K, M)) + 1j * rng.standard_normal((K, M))) / np.sqrt(2)
    print(f"\n--- C) costo del `solve` (K={K}, M={M}, {reps} repeticiones) ---")
    res = {}
    for name, psd in (("con PSD", True), ("sin PSD", False)):
        core = SoudenSubtractCore(K, M, M // 2, psd_project=psd)
        for _ in range(30):                      # estado realista
            core.update(X, np.full(K, 0.3), np.full(K, 0.7))
        core.solve()                             # calentar
        t0 = time.perf_counter()
        for _ in range(reps):
            core.solve()
        res[name] = 1e3 * (time.perf_counter() - t0) / reps
        print(f"  {name:<9} {res[name]:7.3f} ms por solve")
    d = res["con PSD"] - res["sin PSD"]
    print(f"  el eigh son {d:.3f} ms, o sea el {100*d/res['con PSD']:.0f}% del solve")
    print(f"  sacarlo lo deja {res['con PSD']/res['sin PSD']:.2f}x mas rapido")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dur", type=float, default=12.0)
    ap.add_argument("--smooth", type=float, default=0.2)
    ap.add_argument("--isir", type=float, nargs="+", default=[-5.0, 0.0, 5.0, 15.0])
    ap.add_argument("--aro-wav", default=ARO12_WAV)
    ap.add_argument("--aro-skip", type=float, default=8.0)
    ap.add_argument("--aro-dur", type=float, default=15.0)
    ap.add_argument("--cost-m", type=int, default=12)
    ap.add_argument("--cost-reps", type=int, default=30)
    ap.add_argument("--only-cost", action="store_true")
    ap.add_argument("--no-mird", action="store_true")
    ap.add_argument("--no-aro12", action="store_true")
    args = ap.parse_args()

    if args.only_cost:
        part_c(args)
        return
    part_c(args)
    if not args.no_mird:
        part_a(args)
    if not args.no_aro12:
        part_b(args)


if __name__ == "__main__":
    main()
