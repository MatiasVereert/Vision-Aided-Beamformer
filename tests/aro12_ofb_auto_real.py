"""
aro12_ofb_auto_real.py
======================
Prueba de PROCESAMIENTO sobre una captura REAL del array de 12 mics montado en
el ARO (mic-array-platform, PASO 5: WAV de 12 canales, PCM int32 entrelazado
ch0..ch11, fs=16000 exacta, ganancia 35 dB lineal e identica en los 12).

Corre el benchmark ciego `src/evaluation/full_benchmark_real.py` agregando como
procesador extra EL SISTEMA: `NM_MVDR_OFB_AUTO` (lazo cerrado sobre la salida +
post-filtro con la mascara agendada por el estimador ciego de iSIR).

Que sale de aca (en --out-dir):
  * WAVs: ref_mic_raw (el canal crudo de referencia, para A/B), dtln_mono,
    dtln_souden_mvdr, dtln_souden_ban_mvdr, dtln_souden_ban_then_dtln y
    NM_MVDR_OFB_AUTO.
  * diagnostics_real.csv + tabla en consola: RMS, pico, segSNR estimado,
    DNSMOS (SIG/BAK/OVRL) y SQUIM (STOI/PESQ/SI-SDR).
  * spectrograms_real.png
  * input_trimmed.wav: lo que realmente se proceso (ver --skip).

OJO CON LAS METRICAS: esta es UNA sola toma "mezcla" (target + ruido juntos),
asi que NO hay referencia limpia -> todo lo de arriba es NO INTRUSIVO, o sea
relativo y cualitativo. Para PESQ/STOI/SI-SDR de verdad hay que grabar DOS tomas
separadas (senal sola + ruido solo, mismas posiciones) y usar
tests/dsm_blind_real_run.py, que las mezcla a un iSNR controlado.

Tampoco se activan beamformers GEOMETRICOS: el mapeo ch<->posicion fisica en el
aro sigue sin medirse (bitacora sec 21.6, caveat). NM_MVDR_OFB_AUTO es CIEGO
(mask-based): no necesita geometria ni VAD ni DOA, asi que corre igual.

Uso
---
    conda activate tesis_beam
    python tests/aro12_ofb_auto_real.py /ruta/a/captura12.wav
    python tests/aro12_ofb_auto_real.py captura12.wav --skip 8 --dur 30 --smooth 0.2
"""
import os
import argparse
import functools

import numpy as np
import soundfile as sf
import tensorflow as tf

import evaluation.full_benchmark_real as fbr
from evaluation.full_benchmark_real import run_real_benchmark, DTLN_MODEL_1, DTLN_MODEL_2
from evaluation.bf_wrappers import NM_MVDR_OFB_AUTO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PLATFORM = "/home/matias/Documents/Tesis/mic-array-platform"
DEFAULT_WAV = os.path.join(PLATFORM, "data", "aro12", "mezcla12.wav")
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "tests", "real_benchmark_out", "aro12_ofb_auto")

CHANNELS = 12


def trim_capture(in_wav, out_wav, skip_s, dur_s):
    """
    Recorta la captura y la deja en `out_wav` (misma profundidad de bits).

    `skip_s` descarta el arranque: el transitorio de la cadena (6-8 s) ocurre al
    insmod del driver, y si la grabacion empieza pegada al insmod ese tramo entra
    en el WAV y le arruina la adaptacion de covarianzas al beamformer.
    """
    info = sf.info(in_wav)
    if info.channels != CHANNELS:
        print(f"[!] AVISO: el WAV tiene {info.channels} canales, se esperaban {CHANNELS}.")
    if info.samplerate != 16000:
        print(f"[!] AVISO: fs = {info.samplerate} Hz, se esperaban 16000 (DTLN asume 16 kHz).")

    start = int(round(skip_s * info.samplerate))
    stop = -1 if dur_s <= 0 else min(info.frames, start + int(round(dur_s * info.samplerate)))
    if start >= info.frames:
        raise SystemExit(f"[!] --skip {skip_s} s >= duracion del WAV ({info.frames / info.samplerate:.2f} s)")

    data, fs = sf.read(in_wav, start=start, stop=stop, dtype="int32", always_2d=True)
    sf.write(out_wav, data, fs, subtype="PCM_32")
    print(f"[*] Recorte: {skip_s:g} s descartados -> {data.shape[0] / fs:.2f} s x {data.shape[1]} ch")
    print(f"[*] Entrada efectiva: {out_wav}")
    return out_wav


def report_channels(wav):
    """Niveles por canal: chequeo rapido de que los 12 del aro esten vivos."""
    x, fs = sf.read(wav, dtype="float64", always_2d=True)
    print("\n=== Niveles por canal (sobre el tramo procesado) ===")
    print(f"{'ch':>3} {'RMS dBFS':>9} {'pico dBFS':>10} {'DC dBFS':>9}")
    for m in range(x.shape[1]):
        ch = x[:, m]
        rms = 20 * np.log10(np.sqrt(np.mean(ch ** 2)) + 1e-12)
        pk = 20 * np.log10(np.max(np.abs(ch)) + 1e-12)
        dc = 20 * np.log10(abs(np.mean(ch)) + 1e-12)
        print(f"{m:>3} {rms:>9.1f} {pk:>10.1f} {dc:>9.1f}")
    rms_all = [20 * np.log10(np.sqrt(np.mean(x[:, m] ** 2)) + 1e-12) for m in range(x.shape[1])]
    print(f"[*] Dispersion de RMS entre canales: {max(rms_all) - min(rms_all):.1f} dB "
          f"(tolerancia del mic: +-3 dB; muy por encima = canal flojo o mal cableado)")
    print(f"[*] Canales al piso (<= -95 dBFS, o sea mudos): "
          f"{[m for m, r in enumerate(rms_all) if r <= -95] or 'ninguno'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav", nargs="?", default=DEFAULT_WAV,
                    help="captura de 12 canales de pdm_record (mezcla: voz + ruido)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--skip", type=float, default=8.0,
                    help="segundos iniciales a descartar (default 8: transitorio del insmod)")
    ap.add_argument("--dur", type=float, default=0.0,
                    help="segundos a procesar tras el recorte (0 = hasta el final)")
    ap.add_argument("--smooth", type=float, default=0.2,
                    help="piso del post-filtro de NM_MVDR_OFB_AUTO, G = smooth + (1-smooth)*m "
                         "(default 0.2; es el unico knob del sistema)")
    ap.add_argument("--ref-mic", type=int, default=None,
                    help="canal de referencia del procesador (default M//2 = ch6, el centrado)")
    ap.add_argument("--no-dtln", action="store_true",
                    help="saltea DTLN mono y la cascada BAN->DTLN (mas rapido)")
    args = ap.parse_args()

    if not os.path.isfile(args.wav):
        raise SystemExit(f"[!] No existe el WAV: {args.wav}\n"
                         f"    Pasa la ruta de la captura: python tests/aro12_ofb_auto_real.py <captura12.wav>")
    os.makedirs(args.out_dir, exist_ok=True)

    trimmed = trim_capture(args.wav, os.path.join(args.out_dir, "input_trimmed.wav"),
                           args.skip, args.dur)
    report_channels(trimmed)

    # El loader del benchmark avisa si no ve 8 canales (quedo del array viejo).
    # Aca la entrada son 12: se le fija el esperado para que el aviso solo salte
    # si de verdad falta un canal.
    fbr.load_multichannel_wav = functools.partial(fbr.load_multichannel_wav,
                                                  expected_channels=CHANNELS)

    if args.no_dtln:
        interp1 = interp2 = None
        print("[*] DTLN mono/cascada desactivados (--no-dtln).")
    else:
        try:
            interp1 = tf.lite.Interpreter(model_path=DTLN_MODEL_1); interp1.allocate_tensors()
            interp2 = tf.lite.Interpreter(model_path=DTLN_MODEL_2); interp2.allocate_tensors()
            print("[*] Interpretes DTLN TF-Lite cargados.")
        except Exception as e:
            print(f"[!] Sin modelos DTLN (sigo sin las cascadas mono): {e}")
            interp1 = interp2 = None

    base_config = {
        "fs": 16000,
        # STFT alineada con el DTLN (block_len=512, block_shift=128 -> overlap=384).
        # NM_MVDR_OFB_AUTO ademas fuerza analisis rectangular + sintesis Hann por
        # dentro: es la definicion del esquema, no una opcion.
        "stft_window": 512,
        "stft_overlap": 384,
        "dtln_model_path": DTLN_MODEL_1,
        "per_channel_norm": False,   # el HW ya aplica 35 dB identicos en los 12
        "souden_sharpen_exp": 4.0,   # baselines Souden/BAN (OFB_AUTO usa 8.0 propio)
        "souden_alpha": 0.99,
    }
    if args.ref_mic is not None:
        base_config["ref_mic_idx"] = args.ref_mic
        print(f"[*] ref_mic_idx del procesador = ch{args.ref_mic} "
              f"(el WAV 'ref_mic_raw' que guarda el benchmark sigue siendo M//2)")

    extra_processors = {"NM_MVDR_OFB_AUTO": NM_MVDR_OFB_AUTO(smooth=args.smooth)}
    print(f"[*] Procesador bajo prueba: NM_MVDR_OFB_AUTO(smooth={args.smooth})")

    run_real_benchmark(
        input_wav=trimmed,
        output_dir=args.out_dir,
        base_config=base_config,
        interpreter_1=interp1,
        interpreter_2=interp2,
        geometric_processors={},     # geometria del aro sin medir -> nada geometrico
        extra_processors=extra_processors,
    )

    print(f"\n[*] Todo en: {args.out_dir}")
    print("    Escucha ref_mic_raw.wav contra NM_MVDR_OFB_AUTO.wav: sin referencia limpia,")
    print("    el oido y los espectrogramas mandan sobre las no-intrusivas.")


if __name__ == "__main__":
    main()
