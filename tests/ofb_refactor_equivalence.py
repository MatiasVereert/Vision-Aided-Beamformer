"""
ofb_refactor_equivalence.py
===========================
EL PROCESADOR CRISTALIZADO CONTRA EL DE LOS BARRIDOS.

`OFB_MVDR` (sobre `beamforming/mask/ofb.py`) es la reescritura limpia de
`NM_MVDR_OFB_AUTO` (sobre `beamforming/mask/output_feedback.py`) para el port a
C++. El algoritmo es el mismo; lo que se cayo fueron los ejes de los barridos ya
cerrados. Este test mide que no se haya caido nada mas.

LA UNICA DIFERENCIA DE DISENO ESPERADA
--------------------------------------
En `output_feedback.py` el coeficiente de fuga del canal de referencia arranca
en b = 1 y se suaviza hacia su valor nominal con `leak_smooth=0.5`, asi que AUN
CON `leak=0` la red veia una mezcla con el canal crudo durante los primeros
frames:

    b(t) = 2^-(t+1)   ->   0.5, 0.25, 0.125, ...  ( < 1e-3 desde el frame 10 )

En el frame 0 eso no cambia nada (w = e_ref -> Y = x_ref, los dos terminos de
la mezcla son la misma senal), asi que lo que queda es un transitorio de ~8
frames (~64 ms con hop de 128 a 16 kHz) en la ENTRADA DE LA RED. No es una
defensa: es el residuo de inicializar b = 1. `OFB_MVDR` le da `Y` puro desde el
arranque.

Este script mide cuanto cuesta esa limpieza:
  * `dif` : max |y_viejo - y_nuevo| y el error relativo en RMS. NO tiene que dar
            cero -- da la magnitud del transitorio propagado por el estado LSTM.
  * PESQ / STOI / SI-SDR de los dos, contra el target solo en el mic de
    referencia, en un barrido de iSIR. Es el numero que decide: si el delta
    esta en el ruido de medicion, la limpieza sale gratis.

Con `--mird` corre ademas el barrido de verdad (el harness del benchmark, mismas
celdas que tests/ofb_auto_benchmark.py) y compara Delta PESQ celda por celda.

RESULTADO (2026-09-22): LA LIMPIEZA SALE GRATIS
------------------------------------------------
Escena MIRD rt60=0.61, target 0 deg, interferente 45 deg, 12 s, smooth=0.2,
contra el target solo en el mic de referencia:

    iSIR     dPESQ    dSTOI   dSI-SDR   max|dif|
     -5     +0.000   -0.001    +0.00     7.1e-04
      0     -0.005   +0.008    +0.27     1.1e-03
      5     +0.002   +0.000    -0.04     4.9e-04
     15     +0.001   +0.001    +0.03     1.9e-04
    media   -0.0005  +0.0019   +0.06

y con el harness (8 celdas: 2 salas x 4 iSIR), mediana sobre celdas:

    procesador   Delta PESQ   Delta STOI   Delta SDR
    AUTO (viejo)   1.0155       0.1277       8.619
    OFB_MVDR       1.0173       0.1275       8.825

o sea la diferencia esta en el ruido de medicion -- el mismo orden que el costo
del estimador ciego contra el oraculo (+0.0004 medio, 0.006 en la peor celda).
La diferencia muestra a muestra NO es cero (3e-2 relativo en RMS): el estado
LSTM diverge a partir de esos ~8 frames. Lo que no se traslada es la metrica.

USO
---
    conda activate tesis_beam
    python tests/ofb_refactor_equivalence.py
    python tests/ofb_refactor_equivalence.py --dur 15 --extra-p 2 4
    python tests/ofb_refactor_equivalence.py --mird
"""

import os
import sys
import argparse

import numpy as np
import scipy.signal as sig
import soundfile as sf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from propagation.mird_loader import MirdDatasetProvider          # noqa: E402
from evaluation.bf_wrappers import NM_MVDR_OFB_AUTO, OFB_MVDR    # noqa: E402

FS = 16000
MIRD_ROOT = os.path.join(PROJECT_ROOT, "tools", "data", "rirs", "mird")
SPACING = "3-3-3-8-3-3-3"
MODEL_1 = os.path.join(PROJECT_ROOT, "src", "dnn_denoise", "models", "model_quant_1.tflite")
SOURCE_WAV = os.path.join(PROJECT_ROOT, "tools", "data", "signals",
                          "p002_emo_adoration_sentences.wav")
INTERF_WAV = os.path.join(PROJECT_ROOT, "tools", "data", "signals",
                          "techno_gated commune.wav")


def load_mono(path, n, offset=0):
    """Mono a FS, `n` muestras, repitiendo el archivo si hace falta."""
    x, fs = sf.read(path, always_2d=True)
    x = x[:, 0].astype(np.float64)
    if fs != FS:
        x = sig.resample_poly(x, FS, fs)
    if len(x) < n + offset:
        x = np.tile(x, int(np.ceil((n + offset) / len(x))))
    return x[offset:offset + n]


def build_scene(provider, rt60, angle_t, angle_i, isir_db, dur, snr_db=60.0, seed=0):
    """
    Escena MIRD: target + un interferente, convolucionados con las RIR reales
    del array lineal de 8 mics, mezclados al iSIR pedido en el mic de
    referencia. Devuelve (mic_signals (M,N), target_ref (N,), scene_config).

    Es la misma receta del benchmark (mismas RIR, mismo array, mismos wavs),
    pero armada aca adentro: este test compara DOS procesadores sobre LA MISMA
    entrada, asi que no necesita las referencias early/late ni el cacheo del
    harness grande.
    """
    n = int(dur * FS)
    rng = np.random.default_rng(seed)

    def rir(angle):
        h = np.asarray(provider.load_rir(rt60, SPACING, 1.0, angle), dtype=np.float64)
        fs_nat = provider.get_current_fs()
        if fs_nat != FS:                     # 48 kHz -> 16 kHz
            h = sig.resample_poly(h, FS, fs_nat, axis=0)
        return h                             # (L, M)

    h_t, h_i = rir(angle_t), rir(angle_i)
    M = h_t.shape[1]
    s = load_mono(SOURCE_WAV, n)
    v = load_mono(INTERF_WAV, n, offset=FS)

    tgt = np.stack([sig.fftconvolve(s, h_t[:, m])[:n] for m in range(M)])
    itf = np.stack([sig.fftconvolve(v, h_i[:, m])[:n] for m in range(M)])

    ref = M // 2
    p_t = np.mean(tgt[ref] ** 2)
    itf *= np.sqrt(p_t / (np.mean(itf[ref] ** 2) + 1e-20) * 10.0 ** (-isir_db / 10.0))
    noise = rng.standard_normal((M, n)) * np.sqrt(p_t * 10.0 ** (-snr_db / 10.0))

    scene_config = {
        'fs': FS, 'stft_window': 512, 'stft_overlap': 384,
        'dtln_model_path': MODEL_1, 'ref_mic_idx': ref, 'isir_db': float(isir_db),
    }
    return tgt + itf + noise, tgt[ref], scene_config


def si_sdr(y, ref):
    ref = ref - ref.mean()
    y = y - y.mean()
    a = np.dot(y, ref) / (np.dot(ref, ref) + 1e-20)
    e = y - a * ref
    return 10.0 * np.log10((a * a * np.dot(ref, ref) + 1e-20) / (np.dot(e, e) + 1e-20))


def metrics(y, ref):
    from pesq import pesq as pesq_fn
    from pystoi import stoi as stoi_fn
    # Los primeros 3 s quedan afuera: ahi convergen las SCM, y el transitorio
    # de la fuga que este test mide vive en los primeros 64 ms de esa ventana.
    k = 3 * FS
    a, b = ref[k:], y[k:len(ref)][:len(ref) - k]
    sc = max(np.max(np.abs(a)), np.max(np.abs(b)), 1e-12)
    return (pesq_fn(FS, (a / sc).astype(np.float32), (b / sc).astype(np.float32), 'wb'),
            stoi_fn(a, b, FS, extended=False), si_sdr(b, a))


def run_pair(mic, ref, cfg, smooth, extra_p):
    """Corre el viejo, el nuevo y (opcional) el nuevo con block_update > 1."""
    rows = {}
    y_old, _ = NM_MVDR_OFB_AUTO(smooth=smooth).process(mic, cfg)
    rows["AUTO (viejo)"] = y_old
    y_new, _ = OFB_MVDR(smooth=smooth).process(mic, cfg)
    rows["OFB_MVDR"] = y_new
    for P in extra_p:
        y_p, _ = OFB_MVDR(smooth=smooth, block_update=P).process(mic, cfg)
        rows[f"OFB_MVDR P={P}"] = y_p
    return rows, y_old, y_new


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dur", type=float, default=10.0)
    ap.add_argument("--smooth", type=float, default=0.2)
    ap.add_argument("--rt60", type=float, default=0.610)
    ap.add_argument("--isir", type=float, nargs="+", default=[-5.0, 0.0, 5.0, 15.0])
    ap.add_argument("--extra-p", type=int, nargs="*", default=[],
                    help="corre ademas OFB_MVDR con estos block_update")
    ap.add_argument("--mird", action="store_true",
                    help="ademas, el barrido con el harness del benchmark")
    ap.add_argument("--only-mird", action="store_true",
                    help="solo el harness (saltea la comparacion muestra a muestra)")
    args = ap.parse_args()

    if args.only_mird:
        run_mird(args)
        return

    provider = MirdDatasetProvider(root_dir=MIRD_ROOT)
    print(f"[*] escena MIRD rt60={args.rt60}s spacing={SPACING} target=0 interf=45 "
          f"dur={args.dur}s smooth={args.smooth}\n")

    hdr = f"{'iSIR':>5} {'procesador':<16} {'PESQ':>6} {'STOI':>6} {'SI-SDR':>7}"
    print(hdr + f" {'max|dif|':>9} {'dif RMS':>9}")
    print("-" * (len(hdr) + 20))
    deltas = []
    for isir in args.isir:
        mic, ref, cfg = build_scene(provider, args.rt60, 0, 45, isir, args.dur)
        rows, y_old, y_new = run_pair(mic, ref, cfg, args.smooth, args.extra_p)
        mvals = {k: metrics(y, ref) for k, y in rows.items()}
        d = y_new - y_old
        dif = (np.max(np.abs(d)),
               np.sqrt(np.mean(d ** 2)) / (np.sqrt(np.mean(y_old ** 2)) + 1e-20))
        for k, (pq, st, sd) in mvals.items():
            extra = f" {dif[0]:9.2e} {dif[1]:9.2e}" if k == "OFB_MVDR" else ""
            print(f"{isir:5.0f} {k:<16} {pq:6.3f} {st:6.3f} {sd:7.2f}{extra}")
        pq_o, st_o, sd_o = mvals["AUTO (viejo)"]
        pq_n, st_n, sd_n = mvals["OFB_MVDR"]
        deltas.append((pq_n - pq_o, st_n - st_o, sd_n - sd_o))
        print()

    d = np.array(deltas)
    print("=" * 60)
    print("DELTA (OFB_MVDR - AUTO), o sea el costo de sacar el transitorio de fuga")
    print(f"  PESQ   media {d[:,0].mean():+.4f}   |max| {np.abs(d[:,0]).max():.4f}")
    print(f"  STOI   media {d[:,1].mean():+.4f}   |max| {np.abs(d[:,1]).max():.4f}")
    print(f"  SI-SDR media {d[:,2].mean():+.3f} dB  |max| {np.abs(d[:,2]).max():.3f} dB")

    if args.mird:
        run_mird(args)


def run_mird(args):
    """El barrido con el harness del benchmark (celdas de tests/ofb_auto_benchmark.py)."""
    import tensorflow as tf
    from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search

    out_dir = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "ofb_refactor")
    os.makedirs(out_dir, exist_ok=True)
    itp = []
    for p in (MODEL_1, MODEL_1.replace("_1.tflite", "_2.tflite")):
        i = tf.lite.Interpreter(model_path=p)
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
        processors={"AUTO": NM_MVDR_OFB_AUTO(smooth=args.smooth),
                    "OFB_MVDR": OFB_MVDR(smooth=args.smooth)},
        scene_base_config=base_config, output_dir=out_dir,
        interpreter_1=itp[0], interpreter_2=itp[1],
        apply_dtln_post=False, save_catalog=False)

    cols = [c for c in ["Delta_bf_PESQ_early", "Delta_bf_STOI_early",
                        "Delta_bf_SDR_early"] if c in df.columns]
    print("\n--- harness MIRD: Delta PESQ por iSIR ---")
    print(df.pivot_table(index="processor", columns="isir_db",
                         values="Delta_bf_PESQ_early").round(3).to_string())
    print("\n--- mediana sobre las celdas ---")
    print(df.groupby("processor")[cols].median().round(4).to_string())


if __name__ == "__main__":
    main()
