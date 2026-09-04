"""
Prueba del core con SUSTRACCION DE COVARIANZA sobre SENALES REALES.

Corre full_benchmark_real.py (grabacion de 8 canales de la interfaz PDM sobre la
Kria KV260) inyectando los procesadores por `extra_processors`:

    NM_MVDR      core base (sistema actual)
    NM_MVDR_SUB  Phi_SS = Phi_XX - Phi_NN, normaliza por lambda_S = lambda - M
    NM_MVDR_PF   core base + post-filtro de sustraccion espectral (produccion)

EL ORACLE NO SE PUEDE CORRER ACA. SOUDEN_ORACLE_SCM estima Phi_SS y Phi_NN a
partir de las senales LIMPIAS multicanal (target-solo y ruido-solo) y una
grabacion real es solo la MEZCLA: no existen esas componentes. Por el mismo
motivo no hay PESQ/STOI/SDR/SIR (no hay ground truth). La evaluacion es
NO INTRUSIVA (DNSMOS / segSNR estimado / RMS) mas la escucha y los
espectrogramas que genera el propio benchmark.

Uso
---
    python tests/real_sub_run.py [wav_8ch] [output_dir]
"""

import os
import sys

import tensorflow as tf

from evaluation.full_benchmark_real import run_real_benchmark
from evaluation.bf_wrappers import NM_MVDR, NM_MVDR_SUB, NM_MVDR_PF

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DTLN_1 = f"{PROJECT_ROOT}/src/dnn_denoise/models/model_quant_1.tflite"
DTLN_2 = f"{PROJECT_ROOT}/src/dnn_denoise/models/model_quant_2.tflite"

DEFAULT_WAV = "/home/matias/pdm_mic_interface/kria_app/capture/wavs_paso4/test_1.wav"
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "tests", "real_benchmark_out", "sub_test")


def main():
    wav = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_WAV
    out = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    if not os.path.isfile(wav):
        print(f"[!] No existe el WAV: {wav}")
        sys.exit(1)

    try:
        it1 = tf.lite.Interpreter(model_path=DTLN_1); it1.allocate_tensors()
        it2 = tf.lite.Interpreter(model_path=DTLN_2); it2.allocate_tensors()
    except Exception as e:
        print(f"[!] Sin DTLN mono: {e}")
        it1 = it2 = None

    base_config = {
        "fs": 16000,
        "stft_window": 512,
        "stft_overlap": 384,
        "dtln_model_path": DTLN_1,
        "per_channel_norm": False,
        "souden_sharpen_exp": 4.0,
        "souden_alpha": 0.99,
    }

    # Mismos hiperparametros que en el benchmark MIRD, para que la comparacion
    # real-vs-simulado sea contra los mismos filtros.
    extra = {
        "NM_MVDR":     NM_MVDR(min_loading=1e-9, alpha=0.99),
        "NM_MVDR_SUB": NM_MVDR_SUB(min_loading=1e-9, alpha=0.99, mu=0.0),
        "NM_MVDR_PF":  NM_MVDR_PF(min_loading=1e-6, alpha=0.99, smooth=0.33),
    }

    print(f"[*] WAV real: {wav}")
    print(f"[*] Procesadores extra: {list(extra)}")
    print("[*] ORACLE_SCM omitido: requiere target/ruido limpios, que una "
          "grabacion real no tiene.")

    run_real_benchmark(
        input_wav=wav,
        output_dir=out,
        base_config=base_config,
        interpreter_1=it1,
        interpreter_2=it2,
        geometric_processors={},
        extra_processors=extra,
    )


if __name__ == "__main__":
    main()
