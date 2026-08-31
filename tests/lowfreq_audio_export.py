"""
Export de AUDIO + metricas para la cadena de baja frecuencia.

Corre UNA escena MIRD por toda la familia de variantes y escribe un WAV por cada
una, para poder ESCUCHAR la diferencia (que es el punto: PESQ no evalua debajo de
~300 Hz, asi que la mejora en graves es literalmente inaudible para la metrica
principal del benchmark pero no para el oido).

Variantes exportadas
--------------------
    00_ref_clean          target limpio en el mic de referencia (lo ideal)
    01_input_refmic       la mezcla degradada en el mic de referencia (el "antes")
    02_NM_MVDR            el sistema actual
    03_NM_MVDR_PF         el sistema actual + post-filtro (la cadena de produccion)
    04_SUB                sustraccion de covarianza (Phi_SS = Phi_XX - Phi_NN)
    05_SUB_GATE           + gate por confianza (lambda_S/M) con tope de frecuencia
    06_SUB_GATE_PF        + post-filtro  <- la arquitectura propuesta completa
    07_SUB_PF             sustraccion + post-filtro, SIN gate (control: aisla el gate)
    08_ORACLE_SCM         cota superior (covarianzas de las senales limpias)

La comparacion que decide es 06 contra 03 (propuesta vs produccion) y 06 contra
07 (que aporta el gate una vez que el post-filtro se hace cargo de los graves).

Ademas de los WAV escribe:
    metrics.csv      PESQ / STOI / SI-SDR / SDR / SIR / SAR contra las 3 referencias
    lowband.csv      relacion de energia por banda contra el target limpio, que es
                     donde se ve lo que las metricas de banda ancha se pierden

Uso
---
    python tests/lowfreq_audio_export.py
    python tests/lowfreq_audio_export.py --isir 5 --rt60 0.360 --snr-db 30
"""

import os
import argparse

import numpy as np
import pandas as pd
import scipy.signal as sig
import soundfile as sf

from evaluation.bf_wrappers import (
    NM_MVDR, NM_MVDR_PF, NM_MVDR_SUB, SOUDEN_ORACLE_SCM,
)
from evaluation.metrics import evaluate_full_pipeline
from propagation.mird_loader import MirdDatasetProvider
from lowfreq_diagnostic_run import build_scene, PROJECT_ROOT

OUT_DIR = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "lowfreq_audio")

# Bandas del reporte de energia. La primera es la que PESQ NO evalua.
BANDS = [(0, 130), (130, 300), (300, 800), (800, 2000), (2000, 8000)]


def band_energy_ratio(y, ref, fs, nperseg=512):
    """
    Relacion de energia por banda entre la senal procesada y el target limpio,
    en dB. 0 dB = la banda sale con el mismo nivel que en el target; negativo =
    el procesador la atenuo. Es la vista mas directa del colapso de escala.
    """
    n = min(len(y), len(ref))
    f, Py = sig.welch(y[:n], fs=fs, nperseg=nperseg)
    _, Pr = sig.welch(ref[:n], fs=fs, nperseg=nperseg)
    out = {}
    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        out[f"{lo}-{hi}Hz"] = 10 * np.log10(max(Py[m].sum(), 1e-30)
                                            / max(Pr[m].sum(), 1e-30))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rt60", type=float, default=0.610)
    ap.add_argument("--spacing", type=str, default="3-3-3-8-3-3-3")
    ap.add_argument("--target-angle", type=float, default=0)
    ap.add_argument("--target-dist", type=float, default=1.0)
    ap.add_argument("--interf-angle", type=float, default=45)
    ap.add_argument("--interf-dist", type=float, default=1.0)
    ap.add_argument("--isir", type=float, default=0.0)
    ap.add_argument("--snr-db", type=float, default=30.0)
    ap.add_argument("--alpha", type=float, default=0.99)
    ap.add_argument("--min-loading", type=float, default=1e-9)
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--smooth", type=float, default=0.33,
                    help="post-filtro: piso espectral (0.33 = default de NM_MVDR_PF)")
    ap.add_argument("--alpha-lf", type=float, default=0.999,
                    help="alpha para la banda grave (< --alpha-fsplit)")
    ap.add_argument("--alpha-fsplit", type=float, default=300.0)
    ap.add_argument("--gate-thresh", type=float, default=0.3)
    ap.add_argument("--gate-fmax", type=float, default=300.0)
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cfg = {
        'fs': 16000, 'duration': args.duration, 't_early': 0.050,
        'array_center': [3.0, 3.0, 1.2], 'mird_spacing': args.spacing,
        'snr_db': args.snr_db,
        'source_path': f"{PROJECT_ROOT}/tools/data/signals/p002_emo_adoration_sentences.wav",
        'interf_paths': [f"{PROJECT_ROOT}/tools/data/signals/techno_gated commune.wav"],
        'stft_window': 512, 'stft_overlap': 384,
        'dtln_model_path': f"{PROJECT_ROOT}/src/dnn_denoise/models/model_quant_1.tflite",
    }
    fs = cfg['fs']
    nperseg, noverlap = cfg['stft_window'], cfg['stft_overlap']

    provider = MirdDatasetProvider(root_dir=f"{PROJECT_ROOT}/tools/data/rirs/mird")
    print(f"[*] Escena MIRD rt60={args.rt60}  iSIR={args.isir} dB  snr_mic={args.snr_db} dB")
    mic_coords, mixture, oracle_target, oracle_noise, scene_data = build_scene(
        cfg, provider, args.rt60, args.target_angle, args.target_dist,
        [(args.interf_angle, args.interf_dist)], args.isir, args.snr_db)

    M = mixture.shape[0]
    ref_ch = M // 2
    cfg['ref_mic_idx'] = ref_ch
    cfg['oracle_target'] = oracle_target
    cfg['oracle_noise'] = oracle_noise
    cfg['VAD'] = scene_data["VAD"]

    # Referencias del benchmark (mismo criterio: early en el mic de referencia).
    ref_early = scene_data["target_early"][ref_ch]
    refs = {
        'anechoic': scene_data["target_anechoic"][ref_ch],
        'early': ref_early,
        'reverberant': scene_data["target_early"][ref_ch] + scene_data["target_late"][ref_ch],
    }
    eval_start_s = min(5.0, cfg['duration'] * 0.3)

    kw = dict(nperseg=nperseg, noverlap=noverlap, min_loading=args.min_loading,
              alpha=args.alpha)
    processors = {
        "02_NM_MVDR":     NM_MVDR(**kw),
        # ABLACION LIMPIA: alpha(f) sobre el core BASE, sin sustraccion ni gate.
        "11_NM_MVDR_aLF": NM_MVDR(**kw, alpha_lf=args.alpha_lf,
                                  alpha_fsplit_hz=args.alpha_fsplit),
        # min_loading=1e-6 es el default del PF de produccion (core base)
        "03_NM_MVDR_PF":  NM_MVDR_PF(nperseg=nperseg, noverlap=noverlap,
                                     min_loading=1e-6, alpha=args.alpha,
                                     smooth=args.smooth),
        "04_SUB":         NM_MVDR_SUB(**kw, mu=0.0),
        "05_SUB_GATE":    NM_MVDR_SUB(**kw, mu=0.0, gate_thresh=args.gate_thresh,
                                      gate_fmax_hz=args.gate_fmax),
        "06_SUB_GATE_PF": NM_MVDR_SUB(**kw, mu=0.0, gate_thresh=args.gate_thresh,
                                      gate_fmax_hz=args.gate_fmax, smooth=args.smooth),
        "07_SUB_PF":      NM_MVDR_SUB(**kw, mu=0.0, smooth=args.smooth),
        "08_ORACLE_SCM":  SOUDEN_ORACLE_SCM(**kw),
        # alpha dependiente de la frecuencia: mas promediado SOLO en graves.
        "09_SUB_aLF":     NM_MVDR_SUB(**kw, mu=0.0, alpha_lf=args.alpha_lf,
                                      alpha_fsplit_hz=args.alpha_fsplit),
        "10_SUB_aLF_PF":  NM_MVDR_SUB(**kw, mu=0.0, alpha_lf=args.alpha_lf,
                                      alpha_fsplit_hz=args.alpha_fsplit,
                                      smooth=args.smooth),
    }

    # Senales que no salen de un procesador.
    outputs = {
        "00_ref_clean": ref_early,
        "01_input_refmic": mixture[ref_ch],
    }
    for name, proc in processors.items():
        print(f"\n[*] {name}")
        y, _ = proc.process(mixture, cfg)
        outputs[name] = y
        print()

    # --- WAV -----------------------------------------------------------------
    # Escala COMUN a todas las salidas: si se normalizara cada una por su propio
    # pico, se borraria justamente la diferencia de nivel en graves que se quiere
    # escuchar. El pico se toma sobre la entrada.
    peak = np.max(np.abs(outputs["01_input_refmic"])) + 1e-12
    for name, y in outputs.items():
        sf.write(os.path.join(args.out_dir, f"{name}.wav"),
                 (y / peak * 0.9).astype(np.float32), fs)
    print(f"\n[*] {len(outputs)} WAV escritos en {args.out_dir}")

    # --- Metricas ------------------------------------------------------------
    rows, band_rows = [], []
    for name, y in outputs.items():
        if name == "00_ref_clean":
            continue
        print(f"[*] metricas {name}")
        row = {"variant": name}
        for rname, rsig in refs.items():
            m = evaluate_full_pipeline(
                ref_sig=rsig, deg_sig=y, fs=fs,
                interf_early=scene_data["interference_early"][ref_ch],
                interf_late=scene_data["interference_late"][ref_ch],
                target_late=scene_data["target_late"][ref_ch],
                compute_pesq=True, compute_cd=True, eval_start_s=eval_start_s,
                inspection_name=f"{name}_{rname}")
            for k, v in m.items():
                row[f"{k}_{rname}"] = v
        rows.append(row)

        br = {"variant": name}
        br.update(band_energy_ratio(y[int(eval_start_s*fs):],
                                    ref_early[int(eval_start_s*fs):], fs))
        band_rows.append(br)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "metrics.csv"), index=False)
    dfb = pd.DataFrame(band_rows)
    dfb.to_csv(os.path.join(args.out_dir, "lowband.csv"), index=False)

    print("\n=== ENERGIA POR BANDA vs TARGET LIMPIO [dB] ===")
    print("  0 dB = la banda sale al nivel del target. Negativo = atenuada.")
    print("  La columna 0-130Hz es la que PESQ no ve.\n")
    print(dfb.to_string(index=False, float_format=lambda v: f"{v:7.2f}"))

    cols = [c for c in ["variant", "PESQ_early", "STOI_early", "SI-SDR_early",
                        "SDR_early", "SIR_early", "SAR_early"] if c in df.columns]
    if len(cols) > 1:
        print("\n=== METRICAS (referencia 'early') ===")
        print(df[cols].to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
    print(f"\n[*] CSVs en {args.out_dir}")


if __name__ == "__main__":
    main()
