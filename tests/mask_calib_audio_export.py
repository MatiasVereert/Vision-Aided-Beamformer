"""
Export de AUDIO para escuchar la calibracion de mascara.

El benchmark dice que NM_MVDR_MCAL_C gana ~1.1 dB de SDR y ~1.3 dB de SAR sobre
NM_MVDR_SUB en las 16 celdas, pierde ~0.7 dB de SIR, y NO mueve PESQ. Eso es un
cambio de PUNTO DE OPERACION -- menos artefactos a cambio de algo menos de
supresion del interferente -- y es exactamente el tipo de diferencia que hay que
ESCUCHAR, porque ninguna metrica sola la resume.

Variantes por escena
--------------------
    00_ref_clean        target limpio en el mic de referencia (lo ideal)
    01_input_refmic     la mezcla degradada en el mic de referencia (el "antes")
    02_NM_MVDR          el sistema actual
    03_NM_MVDR_SUB      sustraccion de covarianza (el mejor en SIR)
    04_MCAL_C           mascara calibrada, version CONSTANTE  <-- la propuesta
    05_MCAL_tabla       mascara calibrada, tabla por banda (control)
    06_ORACLE_SCM       cota superior (covarianzas de las senales limpias)

QUE ESCUCHAR
------------
  03 vs 04 : el interferente deberia quedar un poco MAS presente en 04, pero con
             menos ruido musical / burbujeo y la voz mas natural (es el
             +1.3 dB de SAR). Si en 04 se escucha mas limpio Y mas suave, la
             mejora de SDR/SAR es real y PESQ simplemente no la ve.
  02 vs 04 : contra el sistema actual, sin trade-off: 04 gana en SDR/SAR/STOI.
  04 vs 06 : cuanto queda hasta el techo con SCM oracle.

Las escenas por default son las NO VISTAS por el ajuste (rt60=0.610).

ESCALA: todos los WAV de una escena comparten el mismo factor (el pico de la
entrada). Normalizar cada uno por su propio pico borraria las diferencias de
nivel, que son justamente parte de lo que se compara.

USO
---
    python tests/mask_calib_audio_export.py
    python tests/mask_calib_audio_export.py --isir -5 0 5 --rt60 0.610
    python tests/mask_calib_audio_export.py --duration 8      # mas corto

Salida: tests/dataset_out/mask_calib_audio/<escena>/*.wav + metrics.csv
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation.bf_wrappers import (
    NM_MVDR, NM_MVDR_SUB, NM_MVDR_MCAL, SOUDEN_ORACLE_SCM,
)
from evaluation.metrics import evaluate_full_pipeline
from propagation.mird_loader import MirdDatasetProvider
from lowfreq_diagnostic_run import build_scene, PROJECT_ROOT

OUT_DIR = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "mask_calib_audio")
MASK_NPZ = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "scm_mask_calib_w",
                        "mask_calib_params.npz")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rt60", type=float, nargs="+", default=[0.610],
                    help="0.610 = condicion NO VISTA por el ajuste")
    ap.add_argument("--isir", type=float, nargs="+", default=[0.0, -5.0])
    ap.add_argument("--interf-angle", type=float, default=45)
    ap.add_argument("--interf-dist", type=float, default=1.0)
    ap.add_argument("--target-angle", type=float, default=0)
    ap.add_argument("--target-dist", type=float, default=1.0)
    ap.add_argument("--spacing", type=str, default="3-3-3-8-3-3-3")
    ap.add_argument("--snr-db", type=float, default=30.0)
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--alpha", type=float, default=0.99)
    ap.add_argument("--min-loading", type=float, default=1e-9)
    ap.add_argument("--const-a-n", type=float, default=2.0)
    ap.add_argument("--b-n", type=float, default=-8.0)
    ap.add_argument("--mask-calib", type=str, default=MASK_NPZ)
    ap.add_argument("--no-metrics", action="store_true",
                    help="solo WAV, sin PESQ/STOI (mas rapido)")
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    args = ap.parse_args()

    provider = MirdDatasetProvider(root_dir=f"{PROJECT_ROOT}/tools/data/rirs/mird")
    all_rows = []

    for rt60 in args.rt60:
        for isir in args.isir:
            tag = f"rt{rt60:g}_ang{args.interf_angle:g}_isir{isir:g}"
            cell = os.path.join(args.out_dir, tag)
            os.makedirs(cell, exist_ok=True)
            print(f"\n{'='*70}\n[*] escena {tag}\n{'='*70}")

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
            mic_coords, mixture, o_tgt, o_noi, scene_data = build_scene(
                cfg, provider, rt60, args.target_angle, args.target_dist,
                [(args.interf_angle, args.interf_dist)], isir, args.snr_db)

            M = mixture.shape[0]
            ref_ch = M // 2
            cfg['ref_mic_idx'] = ref_ch
            cfg['mic_coords'] = mic_coords          # lo pide MCAL si gamma > 0
            cfg['oracle_target'] = o_tgt
            cfg['oracle_noise'] = o_noi

            ref_early = scene_data["target_early"][ref_ch]
            refs = {
                'early': ref_early,
                'reverberant': ref_early + scene_data["target_late"][ref_ch],
            }
            eval_start_s = min(5.0, cfg['duration'] * 0.3)

            kw = dict(nperseg=cfg['stft_window'], noverlap=cfg['stft_overlap'],
                      min_loading=args.min_loading, alpha=args.alpha)
            processors = {
                "02_NM_MVDR":     NM_MVDR(**kw),
                "03_NM_MVDR_SUB": NM_MVDR_SUB(**kw, mu=0.0),
                # LA PROPUESTA: voz = mascara cruda, ruido = odds-ratio recortado
                "04_MCAL_C":      NM_MVDR_MCAL(**kw, mu=0.0, nu=1.0, gamma=0.0,
                                               const_a_n=args.const_a_n,
                                               b_n_const=args.b_n),
                "05_MCAL_tabla":  NM_MVDR_MCAL(**kw, mu=0.0, nu=1.0, gamma=0.0,
                                               calib_path=args.mask_calib),
                "06_ORACLE_SCM":  SOUDEN_ORACLE_SCM(**kw),
            }

            outputs = {"00_ref_clean": ref_early,
                       "01_input_refmic": mixture[ref_ch]}
            for name, proc in processors.items():
                print(f"\n[*] {name}")
                y, _ = proc.process(mixture, cfg)
                outputs[name] = y
                print()

            # ESCALA COMUN a toda la escena (el pico de la entrada).
            peak = np.max(np.abs(outputs["01_input_refmic"])) + 1e-12
            for name, y in outputs.items():
                sf.write(os.path.join(cell, f"{name}.wav"),
                         (y / peak * 0.9).astype(np.float32), fs)
            print(f"[*] {len(outputs)} WAV en {cell}")

            if args.no_metrics:
                continue
            for name, y in outputs.items():
                if name == "00_ref_clean":
                    continue
                row = {"scene": tag, "variant": name}
                for rname, rsig in refs.items():
                    m = evaluate_full_pipeline(
                        ref_sig=rsig, deg_sig=y, fs=fs,
                        interf_early=scene_data["interference_early"][ref_ch],
                        interf_late=scene_data["interference_late"][ref_ch],
                        target_late=scene_data["target_late"][ref_ch],
                        compute_pesq=True, compute_cd=False,
                        eval_start_s=eval_start_s,
                        inspection_name=f"{name}_{rname}")
                    for k, v in m.items():
                        row[f"{k}_{rname}"] = v
                all_rows.append(row)

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(os.path.join(args.out_dir, "metrics.csv"), index=False)
        cols = [c for c in ["PESQ_early", "STOI_early", "SDR_early", "SIR_early",
                            "SAR_early"] if c in df.columns]
        if cols:
            print("\n=== MEDIANA sobre las escenas exportadas (referencia early) ===")
            print(df.groupby("variant")[cols].median().round(3).to_string())
    print(f"\n[ok] {args.out_dir}")


if __name__ == "__main__":
    main()
