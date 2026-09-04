"""
Barrido de alpha(f) sobre las SENALES REALES mezcladas (secadora direccional).

POR QUE. En la escena simulada de techno alpha_lf quedo topeado en 0.999 porque
0.9999 o 1.0 MUTEABAN la banda grave: ahi Phi_XX convergia exactamente a Phi_NN y
la resta daba cero. Pero en esta grabacion la banda grave es TARGET LIMPIO (el
ruido de la secadora tiene 0.2% de su energia debajo de 130 Hz), asi que ese
riesgo no aplica y se puede empujar mas.

Ademas del alpha se barre el CORTE (alpha_fsplit_hz): el dano medido no se acaba
en 300 Hz -- con alpha_lf=0.999/f300 la banda 130-300 Hz todavia sale -7.4 dB
contra los -0.6 dB del oracle.

El script tambien corre el DIAGNOSTICO NARROWBAND sobre la escena real, que es lo
que dice CUAL de los dos mecanismos domina aca:
  * colapso de escala  -> lambda/M ~ 1 y ruido dentro de Phi_XX alto
  * auto-cancelacion   -> target dentro de Phi_NN alto
En la escena de techno mandaba el primero. Aca, con la banda grave dominada por
el target, podria mandar el segundo, y eso cambia que palanca conviene.

Cada variante se mide sola y ademas en cascada con el DTLN completo, que es la
cadena que se va a entregar.

Salida: tests/dataset_out/real_mix_alpha/
    isir<z>/*.wav      audio de todas las variantes
    metrics.csv        por celda
    lowband.csv        energia por banda vs el target limpio
    diagnostic.csv     narrowband (lambda/M, fugas, TR/AG) a iSIR 0
    summary.txt

Uso
---
    python tests/real_mix_alpha_sweep.py
"""

import os
import time
import argparse

import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal as sig
import tensorflow as tf
from pesq import pesq as pesq_wb

from dnn_denoise.dtln_lite import apply_dtln_post_tflite_realtime
from beamforming.mask.dtln_masks import get_dtln_masks_sharpen
from evaluation.bf_wrappers import NM_MVDR, NM_MVDR_SUB, SOUDEN_ORACLE_SCM
from evaluation.metrics import evaluate_full_pipeline
from evaluation.lowfreq_diagnostic import (
    narrowband_report, mask_leakage_report, souden_lambda_report,
)
from lowfreq_diagnostic_run import PROJECT_ROOT
from lowfreq_audio_export import band_energy_ratio, BANDS
from real_mix_isir_sweep import load_multich, mix_at_isir, CAPTURE, DTLN_1, DTLN_2

OUT_DIR = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "real_mix_alpha")

# (alpha_lf, fsplit_hz). None = sin alpha(f) (la referencia SUB actual).
ALPHA_GRID = [
    (None,   None),    # SUB tal cual
    (0.999,  300),     # el actual
    (0.9999, 300),
    (1.0,    300),
    (0.999,  500),
    (0.9999, 500),
    (0.999,  800),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signal", default=os.path.join(CAPTURE, "senal12.wav"))
    ap.add_argument("--noise", default=os.path.join(CAPTURE, "ruido12.wav"))
    ap.add_argument("--isir", type=float, nargs="*", default=[-5, 0, 5])
    ap.add_argument("--alpha", type=float, default=0.99)
    ap.add_argument("--min-loading", type=float, default=1e-9)
    ap.add_argument("--eval-start-s", type=float, default=3.0)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    target, fs = load_multich(args.signal)
    noise, _ = load_multich(args.noise)
    N = min(target.shape[1], noise.shape[1])
    target, noise = target[:, :N], noise[:, :N]
    M = target.shape[0]
    ref_ch = M // 2
    nperseg, noverlap = 512, 384
    hop = nperseg - noverlap

    it1 = tf.lite.Interpreter(model_path=DTLN_1); it1.allocate_tensors()
    it2 = tf.lite.Interpreter(model_path=DTLN_2); it2.allocate_tensors()

    kw = dict(nperseg=nperseg, noverlap=noverlap, min_loading=args.min_loading,
              alpha=args.alpha)
    rows, band_rows = [], []
    t0_all = time.time()

    for isir in args.isir:
        cell = os.path.join(args.out_dir, f"isir{isir:g}")
        os.makedirs(cell, exist_ok=True)
        print(f"\n[*] === iSIR {isir:+g} dB ===")
        mixture, noise_s = mix_at_isir(target, noise, ref_ch, isir)
        ref_clean = target[ref_ch]

        cfg = {'fs': fs, 'stft_window': nperseg, 'stft_overlap': noverlap,
               'dtln_model_path': DTLN_1, 'ref_mic_idx': ref_ch,
               'oracle_target': target, 'oracle_noise': noise_s}

        outputs = {"00_ref_clean": ref_clean, "01_mixture_refmic": mixture[ref_ch]}
        outputs["02_DTLN_mono"] = apply_dtln_post_tflite_realtime(
            interpreter_1=it1, interpreter_2=it2, audio_mono=mixture[ref_ch])
        outputs["NM_MVDR"] = NM_MVDR(**kw).process(mixture, cfg)[0]

        for a_lf, fsp in ALPHA_GRID:
            name = "SUB" if a_lf is None else f"SUB_a{a_lf:g}_f{fsp:g}"
            extra = {} if a_lf is None else dict(alpha_lf=a_lf, alpha_fsplit_hz=fsp)
            t0 = time.time()
            outputs[name] = NM_MVDR_SUB(**kw, mu=0.0, **extra).process(mixture, cfg)[0]
            print(f"\r    {name:20s} {time.time()-t0:5.1f}s")

        outputs["ORACLE_SCM"] = SOUDEN_ORACLE_SCM(**kw).process(mixture, cfg)[0]

        # Cascada con el DTLN completo: es la cadena que se entrega.
        for src in [k for k in list(outputs) if k.startswith(("NM_MVDR", "SUB"))]:
            outputs[src + "_DTLN"] = apply_dtln_post_tflite_realtime(
                interpreter_1=it1, interpreter_2=it2, audio_mono=outputs[src])

        peak = np.max(np.abs(outputs["01_mixture_refmic"])) + 1e-12
        for name, y in outputs.items():
            sf.write(os.path.join(cell, f"{name}.wav"),
                     (y / peak * 0.9).astype(np.float32), fs)

        s0 = int(args.eval_start_s * fs)
        n_ref = noise_s[ref_ch]; zeros = np.zeros_like(n_ref)
        for name, y in outputs.items():
            if name == "00_ref_clean":
                continue
            m = evaluate_full_pipeline(
                ref_sig=ref_clean, deg_sig=y, fs=fs, interf_early=n_ref,
                interf_late=zeros, target_late=zeros, compute_pesq=True,
                compute_cd=True, eval_start_s=args.eval_start_s,
                inspection_name=f"{isir}_{name}")
            m.pop("PESQ", None)
            nc = min(len(ref_clean), len(y))
            try:
                m["PESQ_wb"] = pesq_wb(fs, ref_clean[s0:nc], y[s0:nc], "wb")
            except Exception:
                m["PESQ_wb"] = np.nan
            rows.append({"isir_db": isir, "variant": name, **m})
            band_rows.append({"isir_db": isir, "variant": name,
                              **band_energy_ratio(y[s0:], ref_clean[s0:], fs)})

        pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "metrics.csv"), index=False)
        pd.DataFrame(band_rows).to_csv(os.path.join(args.out_dir, "lowband.csv"), index=False)

        # --- DIAGNOSTICO NARROWBAND (una sola vez, a iSIR 0) -----------------
        if isir == 0:
            print("    [diagnostico narrowband]")
            def _stft(x):
                f_, _, Z = sig.stft(x, fs=fs, window='hamming', nperseg=nperseg,
                                    noverlap=noverlap, nfft=nperseg)
                return f_, np.transpose(Z, (1, 2, 0))
            freqs, S_st = _stft(target)
            _, N_st = _stft(noise_s)
            _, X_st = _stft(mixture)
            mask_s, mask_n = get_dtln_masks_sharpen(
                mixture, ref_ch, DTLN_1, block_len=nperseg, block_shift=hop,
                sharpen_exp=4.0)
            sf_ = int(args.eval_start_s * fs / hop)
            leak = mask_leakage_report(mask_s, mask_n, S_st, N_st, ref_ch, start_frame=sf_)
            lam = souden_lambda_report(X_st, mask_s, mask_n,
                                       min_loading=args.min_loading, start_frame=sf_)
            _, W = NM_MVDR(**kw).process(mixture, cfg)
            rep = narrowband_report(W, S_st, N_st, ref_ch, start_frame=sf_)
            pd.DataFrame({
                "freq_hz": freqs,
                "SNR_in_dB": 10*np.log10(np.maximum(rep["SNR_in"], 1e-30)),
                "TR_dB": 10*np.log10(np.maximum(rep["TR"], 1e-30)),
                "AG_dB": 10*np.log10(np.maximum(rep["AG"], 1e-30)),
                "lambda_over_M": lam["lambda_over_M"],
                "target_in_NN": leak["leak_NN"],
                "noise_in_XX": leak["cont_XX"],
            }).to_csv(os.path.join(args.out_dir, "diagnostic.csv"), index=False)

    df, dfb = pd.DataFrame(rows), pd.DataFrame(band_rows)
    mean_c = [c for c in ["PESQ_wb", "STOI", "SI-SDR", "SDR"] if c in df]
    med_c = [c for c in ["SIR", "SAR"] if c in df]
    tab = (df.groupby("variant")[mean_c].mean()
           .join(df.groupby("variant")[med_c].median()
                 .rename(columns={c: c + "_med" for c in med_c})))
    L = [f"BARRIDO alpha(f) SOBRE SENALES REALES  |  M={M} ref={ref_ch} fs={fs}",
         f"iSIR {args.isir} dB  |  alpha base={args.alpha}  min_loading={args.min_loading:g}",
         "", "=== PROMEDIO 3 iSIR  (SIR/SAR por MEDIANA) ===",
         tab.round(3).sort_values("SI-SDR", ascending=False).to_string(),
         "", "=== ENERGIA POR BANDA vs TARGET LIMPIO [dB] ===",
         dfb.groupby("variant")[[f"{lo}-{hi}Hz" for lo, hi in BANDS]].mean()
         .round(2).to_string()]
    txt = "\n".join(L)
    open(os.path.join(args.out_dir, "summary.txt"), "w").write(txt)
    print("\n" + txt)
    print(f"\n[*] {(time.time()-t0_all)/60:.1f} min -> {args.out_dir}")


if __name__ == "__main__":
    main()
