"""
Prueba del NUEVO procesador NM_MVDR_DSM_BLIND (con y sin post-filtro) sobre
SENALES REALES, usando el benchmark intrusivo (src/evaluation/intrusive_benchmark_real.py).

senal12.wav / ruido12.wav son dos tomas REALES separadas del mismo array (12
canales, 16 kHz): target solo y ruido solo. run_intrusive_benchmark las mezcla
a un SNR controlado y mide PESQ/STOI/SI-SDR/SDR/SAR contra el target limpio
(ground-truth exacto, mismas muestras).

Procesadores (config pedida, la misma fila que en full_benchmark_test_dtln_mird.py):
    NM_MVDR_DSM_BLIND     min_loading=1e-9, alpha=0.99
    NM_MVDR_DSM_BLIND_PF  idem + post-filtro de sustraccion espectral (smooth=0.5)

Los baselines (dtln_souden_mvdr / dtln_souden_ban_mvdr / dtln_mono) los agrega
el propio run_real_benchmark por dentro de run_intrusive_benchmark, asi que
salen gratis en la misma tabla/CSV.

Uso
---
    conda activate tesis_beam
    python tests/dsm_blind_real_run.py
    python tests/dsm_blind_real_run.py --snr 0
"""
import os
import argparse

import tensorflow as tf

from evaluation.intrusive_benchmark_real import (
    run_intrusive_benchmark, default_base_config, DTLN_MODEL_1, DTLN_MODEL_2,
)
from evaluation.bf_wrappers import NM_MVDR_DSM_BLIND

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CAPTURE = "/home/matias/pdm_mic_interface/kria_app/capture/wavs_paso5"
DEFAULT_SENAL = os.path.join(CAPTURE, "senal12.wav")
DEFAULT_RUIDO = os.path.join(CAPTURE, "ruido12.wav")
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "tests", "real_benchmark_out", "dsm_blind")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--senal", default=DEFAULT_SENAL)
    ap.add_argument("--ruido", default=DEFAULT_RUIDO)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--snr", type=float, default=5.0,
                    help="SNR objetivo del mixture en dB en el mic de referencia (default 5)")
    ap.add_argument("--ref-mic", type=int, default=None,
                    help="canal de referencia (default M//2)")
    ap.add_argument("--eval-start", type=float, default=5.0,
                    help="segundos iniciales a descartar en la metrica (default 5)")
    ap.add_argument("--min-loading", type=float, default=1e-9)
    ap.add_argument("--alpha", type=float, default=0.99)
    ap.add_argument("--smooth", type=float, default=0.5,
                    help="piso del post-filtro espectral de la variante _PF (default 0.5)")
    args = ap.parse_args()

    for path in (args.senal, args.ruido):
        if not os.path.isfile(path):
            raise SystemExit(f"[!] No existe el WAV: {path}")

    try:
        interp1 = tf.lite.Interpreter(model_path=DTLN_MODEL_1); interp1.allocate_tensors()
        interp2 = tf.lite.Interpreter(model_path=DTLN_MODEL_2); interp2.allocate_tensors()
        print("[*] Interpretes DTLN TF-Lite cargados.")
    except Exception as e:
        print(f"[!] Sin modelos DTLN (sigo sin las cascadas): {e}")
        interp1, interp2 = None, None

    base_config = default_base_config(fs=16000)  # fs se corrige adentro con el del wav

    extra_processors = {
        "NM_MVDR_DSM_BLIND": NM_MVDR_DSM_BLIND(min_loading=args.min_loading, alpha=args.alpha),
        "NM_MVDR_DSM_BLIND_PF": NM_MVDR_DSM_BLIND(min_loading=args.min_loading, alpha=args.alpha,
                                                   smooth=args.smooth),
    }

    print(f"[*] senal: {args.senal}")
    print(f"[*] ruido: {args.ruido}")
    print(f"[*] procesadores extra: {list(extra_processors)}")

    run_intrusive_benchmark(
        senal_path=args.senal,
        ruido_path=args.ruido,
        output_dir=args.out_dir,
        base_config=base_config,
        interpreter_1=interp1,
        interpreter_2=interp2,
        snr=args.snr,
        ref_mic=args.ref_mic,
        eval_start_s=args.eval_start,
        use_wpe=False,
        extra_processors=extra_processors,
    )


if __name__ == "__main__":
    main()
