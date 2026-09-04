"""
Barrido de iSIR sobre SENALES MEDIDAS REALES, mezcladas de a pares.

A diferencia de tests/real_sub_run.py (una sola grabacion de la mezcla, sin
ground truth), aca hay DOS grabaciones REALES separadas del mismo array:

    senal12.wav   target solo   (12 canales, 16 kHz, 30 s)
    ruido12.wav   ruido solo    (idem)

Como estan separadas se puede:
  * MEZCLARLAS a un iSIR controlado,
  * calcular metricas INTRUSIVAS (PESQ/STOI/SI-SDR/SDR/SIR/SAR) contra
    senal12 en el microfono de referencia,
  * y correr el ORACLE, que necesita target y ruido limpios multicanal
    (imposible con una grabacion de la mezcla sola).

O sea: fisica y ruido 100% reales, con la trazabilidad del benchmark sintetico.

REFERENCIA. El target de referencia es senal12[ref_mic] TAL COMO SE GRABO, o sea
que incluye la reverberacion de la sala. Es el analogo de la referencia
'reverberant' del benchmark MIRD: un procesador que desreverbere seria penalizado,
pero ninguno de los que se comparan aca lo hace, asi que la comparacion es justa.

MEZCLA. El ruido se escala para que la relacion de potencias en el microfono de
referencia, sobre el archivo completo, de el iSIR pedido.

PROCESADORES
    DTLN_mono               DTLN completo sobre el mic de referencia, SIN
                            beamformer (baseline monocanal del sistema)
    NM_MVDR                 core base (sistema actual)
    NM_MVDR_SUB             Phi_SS = Phi_XX - Phi_NN (normaliza por lambda - M)
    NM_MVDR_SUB_aLF         + alpha=0.999 debajo de 300 Hz
    NM_MVDR_PF              core base + post-filtro espectral (produccion)
    NM_MVDR_SUB_aLF_PF      sustraccion + alpha(f) + post-filtro espectral
    NM_MVDR_DTLN            core base + DTLN COMPLETO en cascada   (control)
    NM_MVDR_SUB_aLF_DTLN    sustraccion + alpha(f) + DTLN COMPLETO en cascada
    ORACLE_SCM              covarianzas de las senales limpias (cota superior)

Los dos "_DTLN" pasan la salida del beamformer por el DTLN entero (los dos
nucleos TFLite), no por la ganancia espectral de mascara. El control NM_MVDR_DTLN
existe para poder separar cuanto aporta la cascada y cuanto la correccion, y
DTLN_mono (mismo DTLN, sin beamformer) da el piso monocanal: contra el se mide
cuanto aporta REALMENTE el procesamiento espacial.

METRICAS. PESQ se calcula con la libreria `pesq` DIRECTAMENTE: el facade de
pb_bss que usa evaluate_full_pipeline devuelve NaN sobre estas senales reales
(la libreria suelta funciona bien sobre los mismos WAV). SIR y SAR se agregan
por MEDIANA, no por media: fast_bss_eval devuelve inf/NaN en algunas celdas y
la media se rompe.

Salida
    <out>/isir<z>/*.wav    audio de las 8 variantes + referencia limpia + mezcla
    <out>/metrics.csv      metricas por celda
    <out>/lowband.csv      energia por banda vs el target limpio
    <out>/summary.txt      promedios y conteo de victorias

Uso
---
    python tests/real_mix_isir_sweep.py
    python tests/real_mix_isir_sweep.py --isir -5 0 5 --alpha-lf 0.999
"""

import os
import time
import argparse

import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
from pesq import pesq as pesq_wb

from dnn_denoise.dtln_lite import apply_dtln_post_tflite_realtime
from evaluation.bf_wrappers import (
    NM_MVDR, NM_MVDR_SUB, NM_MVDR_PF, SOUDEN_ORACLE_SCM,
)
from evaluation.metrics import evaluate_full_pipeline
from lowfreq_diagnostic_run import PROJECT_ROOT
from lowfreq_audio_export import band_energy_ratio, BANDS

CAPTURE = "/home/matias/pdm_mic_interface/kria_app/capture/wavs_paso5"
OUT_DIR = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "real_mix_isir")
DTLN_1 = f"{PROJECT_ROOT}/src/dnn_denoise/models/model_quant_1.tflite"
DTLN_2 = f"{PROJECT_ROOT}/src/dnn_denoise/models/model_quant_2.tflite"


def load_multich(path):
    """Lee un WAV multicanal y devuelve (M, N) en float."""
    x, fs = sf.read(path, always_2d=True)
    return x.T.astype(np.float64), fs


def mix_at_isir(target, noise, ref_ch, isir_db):
    """
    Escala el ruido para que la relacion de potencias en el mic de referencia
    (archivo completo) sea exactamente isir_db. Devuelve (mezcla, ruido escalado).
    """
    p_s = np.mean(target[ref_ch] ** 2)
    p_n = np.mean(noise[ref_ch] ** 2)
    g = np.sqrt(p_s / (p_n * 10.0 ** (isir_db / 10.0) + 1e-30))
    noise_scaled = noise * g
    return target + noise_scaled, noise_scaled


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signal", default=os.path.join(CAPTURE, "senal12.wav"))
    ap.add_argument("--noise", default=os.path.join(CAPTURE, "ruido12.wav"))
    ap.add_argument("--isir", type=float, nargs="*", default=[-5, 0, 5])
    ap.add_argument("--alpha", type=float, default=0.99)
    ap.add_argument("--alpha-lf", type=float, default=0.999)
    ap.add_argument("--alpha-fsplit", type=float, default=300.0)
    ap.add_argument("--min-loading", type=float, default=1e-9)
    ap.add_argument("--smooth", type=float, default=0.33)
    ap.add_argument("--eval-start-s", type=float, default=3.0)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    target, fs = load_multich(args.signal)
    noise, fs_n = load_multich(args.noise)
    assert fs == fs_n, f"fs distinta: {fs} vs {fs_n}"
    N = min(target.shape[1], noise.shape[1])
    target, noise = target[:, :N], noise[:, :N]
    M = target.shape[0]
    ref_ch = M // 2

    print(f"[*] senal {target.shape}  ruido {noise.shape}  fs={fs}  M={M}  ref_mic={ref_ch}")
    print(f"[*] piso degenerado de la normalizacion de Souden: 1/M^2 = "
          f"{-20*np.log10(M):.1f} dB")

    # Contenido espectral del RUIDO real: define cuanto puede rendir la correccion
    # (la ganancia escala con la energia que la escena tenga en graves).
    import scipy.signal as sg
    f_w, P = sg.welch(noise[ref_ch], fs=fs, nperseg=1024)
    tot = P.sum()
    dist = {f"{lo}-{hi}Hz": 100 * P[(f_w >= lo) & (f_w < hi)].sum() / tot
            for lo, hi in BANDS}
    print("[*] distribucion espectral del RUIDO real: "
          + "  ".join(f"{k} {v:.1f}%" for k, v in dist.items()))

    interpreter_1 = tf.lite.Interpreter(model_path=DTLN_1); interpreter_1.allocate_tensors()
    interpreter_2 = tf.lite.Interpreter(model_path=DTLN_2); interpreter_2.allocate_tensors()

    nperseg, noverlap = 512, 384
    kw = dict(nperseg=nperseg, noverlap=noverlap, min_loading=args.min_loading,
              alpha=args.alpha)
    a_kw = dict(alpha_lf=args.alpha_lf, alpha_fsplit_hz=args.alpha_fsplit)

    rows, band_rows = [], []
    t_start = time.time()

    for isir in args.isir:
        tag = f"isir{isir:g}"
        cell = os.path.join(args.out_dir, tag)
        os.makedirs(cell, exist_ok=True)
        print(f"\n[*] === iSIR {isir:+g} dB ===")

        mixture, noise_s = mix_at_isir(target, noise, ref_ch, isir)
        ref_clean = target[ref_ch]

        cfg = {
            'fs': fs, 'stft_window': nperseg, 'stft_overlap': noverlap,
            'dtln_model_path': DTLN_1,
            'ref_mic_idx': ref_ch,
            'oracle_target': target,        # target REAL solo -> habilita el oracle
            'oracle_noise': noise_s,        # ruido REAL solo, ya escalado
        }

        procs = {
            "NM_MVDR":            NM_MVDR(**kw),
            "NM_MVDR_SUB":        NM_MVDR_SUB(**kw, mu=0.0),
            "NM_MVDR_SUB_aLF":    NM_MVDR_SUB(**kw, mu=0.0, **a_kw),
            "NM_MVDR_PF":         NM_MVDR_PF(nperseg=nperseg, noverlap=noverlap,
                                             min_loading=1e-6, alpha=args.alpha,
                                             smooth=args.smooth),
            "NM_MVDR_SUB_aLF_PF": NM_MVDR_SUB(**kw, mu=0.0, **a_kw, smooth=args.smooth),
            "ORACLE_SCM":         SOUDEN_ORACLE_SCM(**kw),
        }

        outputs = {"00_ref_clean": ref_clean, "01_mixture_refmic": mixture[ref_ch]}

        # Baseline MONOCANAL: el DTLN completo sobre el mic de referencia, sin
        # ningun procesamiento espacial. Es el piso contra el que hay que medir
        # cuanto aporta el arreglo.
        t0 = time.time()
        outputs["02_DTLN_mono"] = apply_dtln_post_tflite_realtime(
            interpreter_1=interpreter_1, interpreter_2=interpreter_2,
            audio_mono=mixture[ref_ch])
        print(f"\r    {'02_DTLN_mono':22s} {time.time()-t0:5.1f}s")
        for name, proc in procs.items():
            t0 = time.time()
            y, _ = proc.process(mixture, cfg)
            outputs[name] = y
            print(f"\r    {name:22s} {time.time()-t0:5.1f}s")

        # --- Cascadas con el DTLN COMPLETO como post-filtro -------------------
        # Se aplican los dos nucleos TFLite sobre la salida del beamformer. El
        # control NM_MVDR_DTLN permite separar el aporte de la cascada del de la
        # correccion.
        for src, dst in (("NM_MVDR", "NM_MVDR_DTLN"),
                         ("NM_MVDR_SUB_aLF", "NM_MVDR_SUB_aLF_DTLN")):
            t0 = time.time()
            outputs[dst] = apply_dtln_post_tflite_realtime(
                interpreter_1=interpreter_1, interpreter_2=interpreter_2,
                audio_mono=outputs[src])
            print(f"\r    {dst:22s} {time.time()-t0:5.1f}s")

        # WAVs con escala COMUN (pico de la mezcla): normalizar cada uno por su
        # propio pico borraria la diferencia de nivel que se quiere escuchar.
        peak = np.max(np.abs(outputs["01_mixture_refmic"])) + 1e-12
        for name, y in outputs.items():
            sf.write(os.path.join(cell, f"{name}.wav"),
                     (y / peak * 0.9).astype(np.float32), fs)

        s0 = int(args.eval_start_s * fs)
        n_ref = noise_s[ref_ch]
        zeros = np.zeros_like(n_ref)
        for name, y in outputs.items():
            if name == "00_ref_clean":
                continue
            # noise_total = interf_early + interf_late + target_late. Con senales
            # reales no hay split early/late: todo el ruido va en interf_early.
            m = evaluate_full_pipeline(
                ref_sig=ref_clean, deg_sig=y, fs=fs,
                interf_early=n_ref, interf_late=zeros, target_late=zeros,
                compute_pesq=True, compute_cd=True,
                eval_start_s=args.eval_start_s,
                inspection_name=f"{tag}_{name}")
            m.pop("PESQ", None)      # el facade devuelve NaN con estas senales
            n_c = min(len(ref_clean), len(y))
            try:
                m["PESQ_wb"] = pesq_wb(fs, ref_clean[s0:n_c], y[s0:n_c], "wb")
            except Exception as exc:
                print(f"      [!] PESQ fallo en {name}: {type(exc).__name__}: {exc}")
                m["PESQ_wb"] = np.nan
            row = {"isir_db": isir, "variant": name}
            row.update(m)
            rows.append(row)

            br = {"isir_db": isir, "variant": name}
            br.update(band_energy_ratio(y[s0:], ref_clean[s0:], fs))
            band_rows.append(br)

        pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "metrics.csv"), index=False)
        pd.DataFrame(band_rows).to_csv(os.path.join(args.out_dir, "lowband.csv"), index=False)

    df, dfb = pd.DataFrame(rows), pd.DataFrame(band_rows)
    order = ["01_mixture_refmic", "02_DTLN_mono", "NM_MVDR", "NM_MVDR_SUB",
             "NM_MVDR_SUB_aLF", "NM_MVDR_PF", "NM_MVDR_SUB_aLF_PF",
             "NM_MVDR_DTLN", "NM_MVDR_SUB_aLF_DTLN", "ORACLE_SCM"]
    # SIR/SAR por MEDIANA (fast_bss_eval devuelve inf/NaN en algunas celdas).
    mean_cols = [c for c in ["PESQ_wb", "STOI", "SI-SDR", "SDR"] if c in df.columns]
    med_cols = [c for c in ["SIR", "SAR"] if c in df.columns]
    mcols = mean_cols + med_cols

    L = [f"SENALES REALES MEZCLADAS  |  M={M}  ref_mic={ref_ch}  fs={fs}",
         f"  senal: {args.signal}",
         f"  ruido: {args.noise}",
         f"  iSIR {args.isir} dB  |  alpha={args.alpha}, alpha_lf={args.alpha_lf} "
         f"debajo de {args.alpha_fsplit:g} Hz, smooth={args.smooth}",
         "  distribucion espectral del ruido real: "
         + "  ".join(f"{k} {v:.1f}%" for k, v in dist.items()),
         f"  piso degenerado 1/M^2 = {-20*np.log10(M):.1f} dB", ""]

    L.append("=== PROMEDIO SOBRE LOS 3 iSIR  (SIR y SAR por MEDIANA) ===")
    _g = df.groupby("variant")[mean_cols].mean()
    _m = df.groupby("variant")[med_cols].median().rename(
        columns={c: c + "_med" for c in med_cols})
    L.append(_g.join(_m).reindex(order).dropna(how="all").round(3).to_string())

    L.append("\n=== ENERGIA POR BANDA vs TARGET LIMPIO [dB] (promedio) ===")
    bcols = [f"{lo}-{hi}Hz" for lo, hi in BANDS]
    L.append(dfb.groupby("variant")[bcols].mean().reindex(order).dropna(how="all")
             .round(2).to_string())

    L.append("\n=== VICTORIAS contra NM_MVDR (sobre 3 iSIR) ===")
    base = df[df.variant == "NM_MVDR"].set_index("isir_db")
    for v in order:
        if v in ("01_mixture_refmic", "NM_MVDR") or v not in set(df.variant):
            continue
        cur = df[df.variant == v].set_index("isir_db")
        L.append(f"  {v:22s} " + "  ".join(
            f"{m} {int((cur[m] > base[m]).sum())}/"
            f"{int((~cur[m].isna() & ~base[m].isna()).sum())}" for m in mcols))

    for m in mcols:
        piv = df.pivot_table(index="isir_db", columns="variant", values=m)
        piv = piv[[c for c in order if c in piv.columns]]
        L.append(f"\n-- {m} por iSIR --")
        L.append(piv.round(3).to_string())

    txt = "\n".join(L)
    open(os.path.join(args.out_dir, "summary.txt"), "w").write(txt)
    print("\n" + txt)
    print(f"\n[*] {(time.time()-t_start)/60:.1f} min   audio + CSVs en {args.out_dir}")


if __name__ == "__main__":
    main()
