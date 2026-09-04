"""
Barrido MIRD con las variantes de ALPHA POR FRECUENCIA + export de audio.

Completa el hueco de tests/sub_grid_sweep.py, que corrio los cuatro procesadores
pedidos pero SIN las variantes de alpha(f) -- justamente las que en escena unica
habian dado el mejor resultado.

Procesadores (6):
    NM_MVDR             core base (sistema actual)
    NM_MVDR_SUB         Phi_SS = Phi_XX - Phi_NN  (normaliza por lambda - M)
    NM_MVDR_SUB_aLF     + alpha=0.999 debajo de 300 Hz
    NM_MVDR_PF          core base + post-filtro espectral (produccion)
    NM_MVDR_SUB_aLF_PF  sustraccion + alpha(f) + post-filtro
    ORACLE_SCM          covarianzas de las senales limpias (cota superior)

Ejes: rt60 {0.360, 0.610} x interferente {45, 90} grados x iSIR {-5, 0, 5}
= 12 escenas. El spacing 3-3-3-8-3-3-3 solo existe en rt60 0.360 y 0.610.

Salida (una carpeta por escena):
    <out>/rt60_<x>_ang<y>_isir<z>/*.wav      audio de las 6 variantes + ref + entrada
    <out>/metrics.csv                        PESQ/STOI/SI-SDR/SDR/SIR/SAR por celda
    <out>/lowband.csv                        energia por banda vs el target limpio
    <out>/summary.txt                        promedios y conteo de victorias

Los WAV de una escena comparten ESCALA (el pico de la entrada): normalizar cada
uno por su propio pico borraria justamente la diferencia de nivel en graves.

Uso
---
    python tests/sub_alpha_grid.py
    python tests/sub_alpha_grid.py --isir -5 0 5 --rt60 0.610
"""

import os
import time
import argparse

import numpy as np
import pandas as pd
import soundfile as sf

from evaluation.bf_wrappers import (
    NM_MVDR, NM_MVDR_SUB, NM_MVDR_PF, SOUDEN_ORACLE_SCM,
)
from evaluation.metrics import evaluate_full_pipeline
from evaluation.lowfreq_diagnostic import _to_db  # noqa: F401  (coherencia de unidades)
from propagation.mird_loader import MirdDatasetProvider
from lowfreq_diagnostic_run import build_scene, PROJECT_ROOT
from lowfreq_audio_export import band_energy_ratio, BANDS

OUT_DIR = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "sub_alpha_grid")


def make_processors(nperseg, noverlap, alpha, min_loading, smooth, alpha_lf, fsplit):
    kw = dict(nperseg=nperseg, noverlap=noverlap, min_loading=min_loading, alpha=alpha)
    return {
        "NM_MVDR":            NM_MVDR(**kw),
        "NM_MVDR_SUB":        NM_MVDR_SUB(**kw, mu=0.0),
        "NM_MVDR_SUB_aLF":    NM_MVDR_SUB(**kw, mu=0.0, alpha_lf=alpha_lf,
                                          alpha_fsplit_hz=fsplit),
        # El PF de produccion usa el core base con su loading historico (1e-6).
        "NM_MVDR_PF":         NM_MVDR_PF(nperseg=nperseg, noverlap=noverlap,
                                         min_loading=1e-6, alpha=alpha, smooth=smooth),
        "NM_MVDR_SUB_aLF_PF": NM_MVDR_SUB(**kw, mu=0.0, alpha_lf=alpha_lf,
                                          alpha_fsplit_hz=fsplit, smooth=smooth),
        "ORACLE_SCM":         SOUDEN_ORACLE_SCM(**kw),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rt60", type=float, nargs="*", default=[0.360, 0.610])
    ap.add_argument("--interf-angle", type=float, nargs="*", default=[45, 90])
    ap.add_argument("--isir", type=float, nargs="*", default=[-5, 0, 5])
    ap.add_argument("--snr-db", type=float, default=30.0)
    ap.add_argument("--alpha", type=float, default=0.99)
    ap.add_argument("--alpha-lf", type=float, default=0.999)
    ap.add_argument("--alpha-fsplit", type=float, default=300.0)
    ap.add_argument("--min-loading", type=float, default=1e-9)
    ap.add_argument("--smooth", type=float, default=0.33)
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--spacing", type=str, default="3-3-3-8-3-3-3")
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    provider = MirdDatasetProvider(root_dir=f"{PROJECT_ROOT}/tools/data/rirs/mird")

    nperseg, noverlap = 512, 384
    rows, band_rows = [], []
    cells = [(r, a, s) for r in args.rt60 for a in args.interf_angle for s in args.isir]
    print(f"[*] {len(cells)} escenas x 6 procesadores  (alpha_lf={args.alpha_lf} "
          f"debajo de {args.alpha_fsplit:g} Hz)")
    t_start = time.time()

    for ci, (rt60, ang, isir) in enumerate(cells, 1):
        tag = f"rt60_{rt60:g}_ang{ang:g}_isir{isir:g}"
        cell_dir = os.path.join(args.out_dir, tag)
        os.makedirs(cell_dir, exist_ok=True)
        print(f"\n[{ci}/{len(cells)}] {tag}")

        cfg = {
            'fs': 16000, 'duration': args.duration, 't_early': 0.050,
            'array_center': [3.0, 3.0, 1.2], 'mird_spacing': args.spacing,
            'snr_db': args.snr_db,
            'source_path': f"{PROJECT_ROOT}/tools/data/signals/p002_emo_adoration_sentences.wav",
            'interf_paths': [f"{PROJECT_ROOT}/tools/data/signals/techno_gated commune.wav"],
            'stft_window': nperseg, 'stft_overlap': noverlap,
            'dtln_model_path': f"{PROJECT_ROOT}/src/dnn_denoise/models/model_quant_1.tflite",
        }
        fs = cfg['fs']

        mic_coords, mixture, oracle_target, oracle_noise, scene_data = build_scene(
            cfg, provider, rt60, 0, 1.0, [(ang, 1.0)], isir, args.snr_db)

        M = mixture.shape[0]
        ref_ch = M // 2
        cfg['ref_mic_idx'] = ref_ch
        cfg['oracle_target'] = oracle_target
        cfg['oracle_noise'] = oracle_noise
        cfg['VAD'] = scene_data["VAD"]

        ref_early = scene_data["target_early"][ref_ch]
        eval_start_s = min(5.0, cfg['duration'] * 0.3)

        outputs = {"00_ref_clean": ref_early, "01_input_refmic": mixture[ref_ch]}
        procs = make_processors(nperseg, noverlap, args.alpha, args.min_loading,
                                args.smooth, args.alpha_lf, args.alpha_fsplit)
        for name, proc in procs.items():
            t0 = time.time()
            y, _ = proc.process(mixture, cfg)
            outputs[name] = y
            print(f"\r    {name:20s} {time.time()-t0:5.1f}s")

        # WAVs con escala COMUN (pico de la entrada) para que la diferencia de
        # nivel en graves sea audible y no quede borrada por la normalizacion.
        peak = np.max(np.abs(outputs["01_input_refmic"])) + 1e-12
        for name, y in outputs.items():
            sf.write(os.path.join(cell_dir, f"{name}.wav"),
                     (y / peak * 0.9).astype(np.float32), fs)

        s0 = int(eval_start_s * fs)
        for name, y in outputs.items():
            if name == "00_ref_clean":
                continue
            m = evaluate_full_pipeline(
                ref_sig=ref_early, deg_sig=y, fs=fs,
                interf_early=scene_data["interference_early"][ref_ch],
                interf_late=scene_data["interference_late"][ref_ch],
                target_late=scene_data["target_late"][ref_ch],
                compute_pesq=True, compute_cd=True, eval_start_s=eval_start_s,
                inspection_name=f"{tag}_{name}")
            row = {"rt60": rt60, "interf_angle": ang, "isir_db": isir,
                   "variant": name}
            row.update(m)
            rows.append(row)

            br = {"rt60": rt60, "interf_angle": ang, "isir_db": isir, "variant": name}
            br.update(band_energy_ratio(y[s0:], ref_early[s0:], fs))
            band_rows.append(br)

        pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "metrics.csv"), index=False)
        pd.DataFrame(band_rows).to_csv(os.path.join(args.out_dir, "lowband.csv"), index=False)

    df = pd.DataFrame(rows)
    dfb = pd.DataFrame(band_rows)
    order = ["01_input_refmic", "NM_MVDR", "NM_MVDR_SUB", "NM_MVDR_SUB_aLF",
             "NM_MVDR_PF", "NM_MVDR_SUB_aLF_PF", "ORACLE_SCM"]
    metric_cols = [c for c in ["PESQ", "STOI", "SI-SDR", "SDR", "SIR", "SAR"]
                   if c in df.columns]

    lines = []
    lines.append(f"{len(cells)} escenas  |  rt60 {args.rt60}  interf {args.interf_angle} deg  "
                 f"iSIR {args.isir} dB  |  snr_mic {args.snr_db} dB")
    lines.append(f"alpha={args.alpha}  alpha_lf={args.alpha_lf} debajo de {args.alpha_fsplit:g} Hz"
                 f"  smooth={args.smooth}\n")
    lines.append("=== PROMEDIO SOBRE TODAS LAS ESCENAS ===")
    g = df.groupby("variant")[metric_cols].mean().reindex(order).dropna(how="all")
    lines.append(g.round(3).to_string())

    lines.append("\n=== ENERGIA POR BANDA vs TARGET LIMPIO [dB] (promedio) ===")
    bcols = [f"{lo}-{hi}Hz" for lo, hi in BANDS]
    gb = dfb.groupby("variant")[bcols].mean().reindex(order).dropna(how="all")
    lines.append(gb.round(2).to_string())

    lines.append("\n=== VICTORIAS contra NM_MVDR (por escena) ===")
    base = df[df.variant == "NM_MVDR"].set_index(["rt60", "interf_angle", "isir_db"])
    for v in order:
        if v in ("01_input_refmic", "NM_MVDR") or v not in set(df.variant):
            continue
        cur = df[df.variant == v].set_index(["rt60", "interf_angle", "isir_db"])
        wins = {m: int((cur[m] > base[m]).sum()) for m in metric_cols}
        lines.append(f"  {v:20s} " + "  ".join(f"{m} {w}/{len(cur)}" for m, w in wins.items()))

    lines.append("\n=== PROMEDIO POR iSIR (PESQ / SIR) ===")
    for m in [c for c in ("PESQ", "SIR") if c in metric_cols]:
        piv = df.pivot_table(index="isir_db", columns="variant", values=m)
        piv = piv[[c for c in order if c in piv.columns]]
        lines.append(f"-- {m} --")
        lines.append(piv.round(3).to_string())

    txt = "\n".join(lines)
    open(os.path.join(args.out_dir, "summary.txt"), "w").write(txt)
    print("\n" + txt)
    print(f"\n[*] {(time.time()-t_start)/60:.1f} min   audio + CSVs en {args.out_dir}")


if __name__ == "__main__":
    main()
