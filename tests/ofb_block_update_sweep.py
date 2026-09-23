"""
ofb_block_update_sweep.py
=========================
CUANTO SALE ACTUALIZAR LOS PESOS MAS LENTO.

`block_update` (P) es la unica palanca de presupuesto del lazo de salida: la
ESTADISTICA se acumula todos los frames (`core.update`: un producto externo
x x^H por bin y dos recursiones de primer orden), pero los PESOS se recalculan
cada P frames. En el `solve` estan las dos cuentas caras del sistema -- el
`eigh` de la proyeccion PSD de Phi_SS y el sistema M x M por bin -- asi que P es
lo que decide si el sistema entra en el presupuesto del ARM.

Lo que NO cambia con P: el camino critico (Y = w^H x), las dos invocaciones del
DTLN y el post-filtro. O sea que el costo por frame es

    t(P) = t_fijo + t_solve / P

y la ganancia SATURA: el piso es `t_fijo`, y de ahi no se baja subiendo P.

DOS NUMEROS DISTINTOS, Y EL SEGUNDO ES EL QUE MANDA EN TIEMPO REAL
------------------------------------------------------------------
P baja el costo MEDIO por frame, no el PICO: los frames en los que toca
resolver siguen costando lo mismo que con P=1. En un sistema de tiempo real con
un buffer por hop, lo que tiene que entrar en el periodo es el PEOR frame, no el
promedio -- a menos que el port parta el `solve` en pedazos y lo reparta entre
los P frames (que es posible: el `solve` es independiente por bin, asi que se
puede hacer K/P bins por frame). Por eso este script reporta las dos cosas:
media y p99/maximo, y el desglose entre frames que resuelven y frames que no.

SALIDA
------
  Parte A (siempre): una escena, el lazo instrumentado frame a frame. Costo
    medio, p99 y maximo por frame, el reparto solve/no-solve, y PESQ/STOI/SI-SDR
    de la misma corrida para ver que se paga en calidad.
  Parte B (--mird): el barrido con el harness del benchmark (las mismas celdas
    que tests/ofb_auto_benchmark.py), o sea Delta PESQ/STOI/SDR por celda contra
    el mic de referencia, comparable con las tablas ya publicadas.

USO
---
    conda activate tesis_beam
    python tests/ofb_block_update_sweep.py
    python tests/ofb_block_update_sweep.py --p 1 2 4 8 16 32 --dur 12
    python tests/ofb_block_update_sweep.py --mird
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

from propagation.mird_loader import MirdDatasetProvider              # noqa: E402
from beamforming.mask.ofb import OutputFeedbackMVDR                  # noqa: E402
from evaluation.bf_wrappers import ola_taper, OFB_MVDR               # noqa: E402
from ofb_refactor_equivalence import (build_scene, metrics, FS, MIRD_ROOT,
                                      MODEL_1, SPACING, SOURCE_WAV,
                                      INTERF_WAV)                    # noqa: E402

# Captura REAL del aro de 12 mics (mic-array-platform, PASO 5). Es una sola toma
# "mezcla" (voz + ruido juntos): NO hay referencia limpia, asi que de este lado
# las metricas son todas NO INTRUSIVAS y sirven para comparar entre si, no contra
# un absoluto. Ver tests/aro12_ofb_auto_real.py.
ARO12_WAV = "/home/matias/Documents/Tesis/mic-array-platform/data/aro12/aro12_mezcla.wav"
ARO12_OUT = os.path.join(PROJECT_ROOT, "tests", "real_benchmark_out", "ofb_block_update")


def run_instrumented(mic, cfg, P, smooth):
    """
    El lazo corrido a mano para poder cronometrar CADA frame. Es exactamente lo
    que hace `output_feedback_run`, sin el acumulador de pesos (que a K=257/M=8
    son 60 MB por celda y no son parte del algoritmo).
    """
    nperseg, hop = cfg['stft_window'], cfg['stft_window'] - cfg['stft_overlap']
    freqs, _, Zxx = sig.stft(mic, fs=cfg['fs'], window='boxcar', nperseg=nperseg,
                             noverlap=cfg['stft_overlap'], nfft=nperseg)
    X = np.transpose(Zxx, (1, 2, 0))                 # (K, T, M)
    K, T, M = X.shape

    proc = OutputFeedbackMVDR(K, M, cfg['ref_mic_idx'], freqs, cfg['dtln_model_path'],
                              nperseg, block_update=P, smooth=smooth)
    Y = np.zeros((K, T), dtype=np.complex128)
    dt = np.zeros(T)
    solved = np.zeros(T, dtype=bool)
    for t in range(T):
        t0 = time.perf_counter()
        Y[:, t] = proc.step(X[:, t, :])
        dt[t] = time.perf_counter() - t0
        solved[t] = (t % P == 0)
    y = ola_taper(Y, nperseg, hop, 'hann', mic.shape[1])
    return y, dt * 1e3, solved                        # ms por frame


def part_a(args):
    provider = MirdDatasetProvider(root_dir=MIRD_ROOT)
    mic, ref, cfg = build_scene(provider, args.rt60, 0, 45, args.isir_one, args.dur)
    hop_ms = 1000.0 * (cfg['stft_window'] - cfg['stft_overlap']) / cfg['fs']
    print(f"\n[*] Parte A: rt60={args.rt60}s iSIR={args.isir_one} dur={args.dur}s "
          f"smooth={args.smooth} -- periodo de hop = {hop_ms:.1f} ms\n")

    hdr = (f"{'P':>3} {'ms/frame':>9} {'p99':>8} {'max':>8} {'con solve':>10} "
           f"{'sin solve':>10} {'% RT':>6} {'PESQ':>6} {'STOI':>6} {'SI-SDR':>7}")
    print(hdr)
    print("-" * len(hdr))
    base = None
    rows = []
    for P in args.p:
        y, dt, solved = run_instrumented(mic, cfg, P, args.smooth)
        pq, st, sd = metrics(y, ref)
        # El primer frame paga la allocacion perezosa de numpy/tflite: se saca.
        d, s = dt[1:], solved[1:]
        m_on = d[s].mean() if s.any() else float('nan')
        m_off = d[~s].mean() if (~s).any() else float('nan')
        print(f"{P:3d} {d.mean():9.3f} {np.percentile(d, 99):8.3f} {d.max():8.3f} "
              f"{m_on:10.3f} {m_off:10.3f} {100*d.mean()/hop_ms:6.1f} "
              f"{pq:6.3f} {st:6.3f} {sd:7.2f}")
        rows.append((P, d.mean(), np.percentile(d, 99), pq, st, sd))
        if base is None:
            base = (pq, st, sd)

    print(f"\n{'P':>3} {'x mas rapido':>13} {'dPESQ':>7} {'dSTOI':>7} {'dSI-SDR':>8}")
    for P, mean, p99, pq, st, sd in rows:
        print(f"{P:3d} {rows[0][1]/mean:13.2f} {pq-base[0]:+7.3f} "
              f"{st-base[1]:+7.3f} {sd-base[2]:+8.2f}")
    print("\nOJO: 'ms/frame' es la MEDIA. En tiempo real lo que tiene que entrar en el")
    print("periodo es la columna 'con solve', salvo que el port reparta el solve por bins.")


def part_aro12(args):
    """
    EL BARRIDO DE P SOBRE UNA ESCENA REAL (una sola toma del aro de 12 mics).

    Por que aca y no sobre MIRD: el costo por frame depende de M (el `solve` es
    M x M por bin y el `eigh` es M^3), y el array real tiene 12 canales contra
    los 8 de MIRD. El numero que importa para el presupuesto del ARM es el del
    array que se va a portar.

    Lo que NO se puede medir aca: PESQ/STOI de verdad. Es UNA sola toma con la
    voz y el ruido ya mezclados, o sea sin referencia limpia. Quedan (a) las no
    intrusivas -- segSNR estimado por VAD de energia, DNSMOS, SQUIM -- que son
    relativas, y (b) la DIVERGENCIA contra P=1, que es lo unico exacto que hay:
    cuanto se aparta la salida al refrescar los pesos mas lento, medida como
    SI-SDR tomando la corrida P=1 como referencia. Los WAVs quedan en disco
    porque, sin referencia, el oido es el instrumento que queda.
    """
    import soundfile as sf
    from evaluation.full_benchmark_real import (load_multichannel_wav, energy_vad,
                                                segmental_snr_estimate, rms_db,
                                                save_spectrograms)
    from ofb_refactor_equivalence import si_sdr

    if not os.path.isfile(args.aro_wav):
        raise SystemExit(f"[!] no existe la captura: {args.aro_wav}")
    os.makedirs(args.aro_out, exist_ok=True)

    mic_all, fs = load_multichannel_wav(args.aro_wav, expected_channels=12)
    i0 = int(round(args.aro_skip * fs))
    i1 = mic_all.shape[1] if args.aro_dur <= 0 else min(mic_all.shape[1],
                                                        i0 + int(round(args.aro_dur * fs)))
    mic = np.ascontiguousarray(mic_all[:, i0:i1])
    M, N = mic.shape
    ref_mic = M // 2 if args.ref_mic is None else int(args.ref_mic)
    cfg = {'fs': fs, 'stft_window': 512, 'stft_overlap': 384,
           'dtln_model_path': MODEL_1, 'ref_mic_idx': ref_mic}
    hop_ms = 1000.0 * (cfg['stft_window'] - cfg['stft_overlap']) / fs
    print(f"\n[*] aro12: {M} canales, {N/fs:.1f} s procesados (skip {args.aro_skip:g} s), "
          f"ref=ch{ref_mic}, smooth={args.smooth} -- periodo de hop = {hop_ms:.1f} ms\n")

    hdr = (f"{'P':>3} {'ms/frame':>9} {'p99':>8} {'max':>8} {'con solve':>10} "
           f"{'sin solve':>10} {'% RT':>6} {'x mas rap':>10} {'SI-SDR vs P=1':>14}")
    print(hdr)
    print("-" * len(hdr))
    outs, rows = {}, []
    for P in args.p:
        y, dt, solved = run_instrumented(mic, cfg, P, args.smooth)
        d, sv = dt[1:], solved[1:]          # el frame 0 paga la allocacion perezosa
        m_on = d[sv].mean() if sv.any() else float('nan')
        m_off = d[~sv].mean() if (~sv).any() else float('nan')
        base_ms = rows[0][1] if rows else d.mean()
        div = si_sdr(y, outs[f"P={args.p[0]}"]) if outs else float('inf')
        print(f"{P:3d} {d.mean():9.3f} {np.percentile(d, 99):8.3f} {d.max():8.3f} "
              f"{m_on:10.3f} {m_off:10.3f} {100*d.mean()/hop_ms:6.1f} "
              f"{base_ms/d.mean():10.2f} " +
              (f"{div:14.1f}" if np.isfinite(div) else f"{'(ref)':>14}"))
        rows.append((P, d.mean(), np.percentile(d, 99), d.max(), m_on, m_off, div))
        outs[f"P={P}"] = y

    # --- WAVs: MISMA escala para los 5 --------------------------------------
    # Normalizar cada archivo a su propio pico haria incomparables los niveles,
    # y el post-filtro cambia justamente el nivel. Una sola escala, la del pico
    # global, deja el A/B honesto.
    x_ref = mic[ref_mic]
    peak = max(max(np.max(np.abs(y)) for y in outs.values()), np.max(np.abs(x_ref)))
    sc = 0.95 / (peak + 1e-12)
    sf.write(os.path.join(args.aro_out, "ref_mic_raw.wav"),
             (x_ref * sc).astype(np.float32), fs, subtype="PCM_16")
    for name, y in outs.items():
        sf.write(os.path.join(args.aro_out, f"ofb_{name.replace('=', '')}.wav"),
                 (y * sc).astype(np.float32), fs, subtype="PCM_16")
    print(f"\n[*] WAVs (misma escala, pico global 0.95): {args.aro_out}")

    # --- no intrusivas -------------------------------------------------------
    try:
        from evaluation.nonintrusive import compute_nonintrusive, NONINTRUSIVE_KEYS
    except Exception as e:
        compute_nonintrusive, NONINTRUSIVE_KEYS = None, []
        print(f"[!] sin metricas no intrusivas: {e}")

    print(f"\n{'senal':<12} {'RMS dBFS':>9} {'segSNR':>7}" +
          "".join(f"{k.replace('DNSMOS_', '').replace('SQUIM_', 'SQ_'):>9}"
                  for k in NONINTRUSIVE_KEYS))
    diag = []
    for name, x in [("ref_mic", x_ref)] + list(outs.items()):
        ni = compute_nonintrusive(x, fs) if compute_nonintrusive else {}
        row = {"senal": name, "rms_dbfs": rms_db(x),
               "segSNR_est_db": segmental_snr_estimate(x, energy_vad(x, fs))}
        row.update({k: ni.get(k, np.nan) for k in NONINTRUSIVE_KEYS})
        diag.append(row)
        print(f"{name:<12} {row['rms_dbfs']:9.1f} {row['segSNR_est_db']:7.2f}" +
              "".join(f"{row[k]:9.2f}" if np.isfinite(row.get(k, np.nan)) else f"{'--':>9}"
                      for k in NONINTRUSIVE_KEYS))

    import csv
    csv_path = os.path.join(args.aro_out, "block_update_aro12.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["P", "ms_frame_medio", "ms_p99", "ms_max", "ms_con_solve",
                    "ms_sin_solve", "si_sdr_vs_P1_db"])
        w.writerows([[r[0]] + [round(float(v), 4) for v in r[1:]] for r in rows])
        w.writerow([])
        w.writerow(["senal", "rms_dbfs", "segSNR_est_db"] + list(NONINTRUSIVE_KEYS))
        for r in diag:
            w.writerow([r["senal"], round(r["rms_dbfs"], 2), round(r["segSNR_est_db"], 3)] +
                       [round(float(r.get(k, np.nan)), 3) for k in NONINTRUSIVE_KEYS])
    print(f"[*] {csv_path}")
    save_spectrograms({"ref_mic (crudo)": x_ref, **outs}, fs,
                      os.path.join(args.aro_out, "block_update_aro12.png"))

    print("\nOJO: 'ms/frame' es la MEDIA y sale de x86, no del ARM -- lo que se lee")
    print("aca es el ESCALADO con P, no el numero absoluto. En tiempo real lo que")
    print("tiene que entrar en el periodo de hop es la columna 'con solve', salvo")
    print("que el port reparta el solve por bins entre los P frames.")


def part_b(args):
    """El barrido de verdad: Delta PESQ/STOI/SDR por celda con el harness."""
    import tensorflow as tf
    from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search

    out_dir = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "ofb_block_update")
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
        'rt60': args.rt60_grid, 'target_angle': [0], 'target_dist': [1.0],
        'interf_configs': [[(45, 1.0)]], 'isir_db': args.isir,
        'mismatch_gain': [0], 'mismatch_phase': [0],
        'use_wpe': [False], 'wpe_method': ['online'], 'wpe_taps': [7], 'wpe_delay': [2],
        'error_angle_deg': [0.0], 'error_distance_m': [0.0],
    }
    procs = {f"P={P}": OFB_MVDR(smooth=args.smooth, block_update=P,
                                return_weights=False) for P in args.p}
    df = run_mird_grid_search(
        grid_params=param_grid, dataset_provider=MirdDatasetProvider(root_dir=MIRD_ROOT),
        processors=procs, scene_base_config=base_config, output_dir=out_dir,
        interpreter_1=itp[0], interpreter_2=itp[1],
        apply_dtln_post=False, save_catalog=False)

    order = [f"P={P}" for P in args.p]
    cols = [c for c in ["Delta_bf_PESQ_early", "Delta_bf_STOI_early",
                        "Delta_bf_SDR_early", "Delta_bf_SIR_early"] if c in df.columns]
    print("\n--- Delta PESQ por iSIR (media sobre salas) ---")
    print(df.pivot_table(index="processor", columns="isir_db",
                         values="Delta_bf_PESQ_early").reindex(order).round(3).to_string())
    print("\n--- Delta STOI por iSIR ---")
    print(df.pivot_table(index="processor", columns="isir_db",
                         values="Delta_bf_STOI_early").reindex(order).round(3).to_string())
    print("\n--- mediana sobre las celdas ---")
    print(df.groupby("processor")[cols].median().reindex(order).round(4).to_string())
    df.to_parquet(os.path.join(out_dir, "block_update_sweep.parquet"))
    print(f"\n[*] {out_dir}/block_update_sweep.parquet")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--dur", type=float, default=12.0)
    ap.add_argument("--smooth", type=float, default=0.2)
    ap.add_argument("--rt60", type=float, default=0.610, help="escena de la parte A")
    ap.add_argument("--isir-one", type=float, default=0.0, help="iSIR de la parte A")
    ap.add_argument("--rt60-grid", type=float, nargs="+", default=[0.360, 0.610])
    ap.add_argument("--isir", type=float, nargs="+", default=[-5.0, 0.0, 5.0, 15.0])
    ap.add_argument("--mird", action="store_true")
    ap.add_argument("--only-mird", action="store_true")
    ap.add_argument("--aro12", action="store_true",
                    help="el barrido sobre la captura REAL del aro de 12 mics, "
                         "con los WAVs de cada P en disco")
    ap.add_argument("--aro-wav", default=ARO12_WAV)
    ap.add_argument("--aro-out", default=ARO12_OUT)
    ap.add_argument("--aro-skip", type=float, default=8.0,
                    help="segundos iniciales a descartar (transitorio del insmod)")
    ap.add_argument("--aro-dur", type=float, default=15.0, help="0 = hasta el final")
    ap.add_argument("--ref-mic", type=int, default=None, help="default M//2")
    args = ap.parse_args()

    if args.aro12:
        part_aro12(args)
        return
    if not args.only_mird:
        part_a(args)
    if args.mird or args.only_mird:
        part_b(args)


if __name__ == "__main__":
    main()
