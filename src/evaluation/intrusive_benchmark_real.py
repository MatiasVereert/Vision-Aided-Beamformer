"""
intrusive_benchmark_real.py
===========================
Metricas INTRUSIVAS (PESQ / STOI / SI-SDR / SDR / SAR) sobre grabaciones REALES
del array de microfonos PDM, SIN necesitar un simulador.

IDEA: con "solo la mezcla" no hay referencia limpia -> solo se pueden metricas
no-intrusivas (DNSMOS). Para tener ground-truth se graban DOS tomas separadas,
con las MISMAS posiciones (mic array + fuentes quietas):
    * senal.wav  = SOLO el target (voz), sin ruido  -> es la REFERENCIA limpia.
    * ruido.wav  = SOLO el ruido/interferente, sin voz.
El script las suma a un SNR objetivo para armar la mezcla (mixture = senal +
g*ruido), corre los beamformers (run_real_benchmark) y compara cada salida
contra la referencia = senal[canal_ref] con metricas referenciadas.
Como el target del mixture es EXACTAMENTE senal[ref] (mismas muestras), el
ground-truth es exacto (alineado muestra a muestra; el delay del DTLN lo absorbe
precise_slice_alignment en metrics.py).

WPE (opcional): con use_wpe=True se aplica dereverberacion WPE multicanal sobre
la MEZCLA (antes de los beamformers), espejando el "NODE 4" del benchmark MIRD.
La mixture.wav que consumen los beamformers queda dereverberada, y se reporta la
metrica de la mixture con WPE (pre-beamformer) para ver cuanto aporta la WPE sola.

USO:
    conda activate tesis_beam
    python src/evaluation/intrusive_benchmark_real.py senal.wav ruido.wav [out_dir] \
           [--snr 5] [--ref-mic 6] [--eval-start 5] [--wpe] [--wpe-bits 24]

Salidas en out_dir: WAVs procesados (via run_real_benchmark) + mixture.wav +
senal_ref.wav + intrusive_metrics.csv, y una tabla por consola.
"""
import os
import sys
import argparse

import numpy as np
import soundfile as sf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import tensorflow as tf
from evaluation.full_benchmark_real import run_real_benchmark, DTLN_MODEL_1, DTLN_MODEL_2
from evaluation.metrics import evaluate_full_pipeline
from dereverberation.nara_wrappers import process_wpe_online, process_wpe_online_with_components
from dereverberation.nara_wrappers_fixed import process_wpe_online_fixed, FixedPointConfig
from evaluation.bf_wrappers import (
    NM_MVDR_PF,
    ORACLE_MB_MVDR_SOUDEN,
    SOUDEN_ORACLE_SCM,
)


def load_multi(path):
    data, fs = sf.read(path, dtype="float64", always_2d=True)  # (N, M)
    return data, fs


def default_base_config(fs):
    """base_config por defecto (mismas claves WPE que el benchmark MIRD)."""
    return {
        "fs": fs,
        "stft_window": 512,
        "stft_overlap": 384,
        "dtln_model_path": DTLN_MODEL_1,
        "per_channel_norm": False,
        "souden_sharpen_exp": 4.0,
        "souden_alpha": 0.99,

        # --- WPE (mismas claves que el MIRD base_config) ---
        "wpe_taps": 7,
        "wpe_delay": 3,
        "wpe_alpha": 0.9999,
        "wpe_stft_size": 512,
        "wpe_stft_shift": 128,
        # None => float. 24/20/18 => emulacion fixed-point FPGA.
        "wpe_fixed_bits": None,
        "wpe_fixed_round": "nearest",
    }


def apply_wpe(mixture_nm, base_config, label="la mezcla multicanal"):
    """Aplica WPE dereverb a una senal multicanal.

    mixture_nm : (N, M)  senal en el dominio del tiempo (como se guarda en el wav).
    WPE espera (M, N) -> transponer, aplicar, transponer de vuelta.
    label : texto para el print (p.ej. 'la mezcla', 'target oracle', 'ruido oracle').
    Devuelve (senal_wpe (N, M), fp_stats_or_None).
    """
    u = mixture_nm.T  # (M, N)
    fixed_bits = base_config.get("wpe_fixed_bits", None)
    if fixed_bits is None:
        print(f"[*] [WPE] Aplicando WPE (float) sobre {label}...")
        y = process_wpe_online(
            u=u,
            taps=base_config["wpe_taps"], delay=base_config["wpe_delay"],
            alpha=base_config["wpe_alpha"], stft_size=base_config["wpe_stft_size"],
            stft_shift=base_config["wpe_stft_shift"],
        )
        fp_stats = None
    else:
        print(f"[*] [WPE] Aplicando WPE (FIXED-POINT {fixed_bits}-bit, emulacion FPGA) sobre {label}...")
        fp_cfg = FixedPointConfig.wordlength(
            fixed_bits, rounding=base_config.get("wpe_fixed_round", "nearest")
        )
        y, fp_stats = process_wpe_online_fixed(
            u=u,
            taps=base_config["wpe_taps"], delay=base_config["wpe_delay"],
            alpha=base_config["wpe_alpha"], stft_size=base_config["wpe_stft_size"],
            stft_shift=base_config["wpe_stft_shift"], fp_cfg=fp_cfg, return_stats=True,
        )
        print(f"    [FP-STATS] overflow={fp_stats.overflow} max|P|={fp_stats.max_absP:.2e} "
              f"max|G|={fp_stats.max_absG:.2e} diverged={fp_stats.diverged}")
    return y.T, fp_stats  # (N, M)


def run_intrusive_benchmark(senal_path, ruido_path, output_dir="intrusive_out",
                            base_config=None, interpreter_1=None, interpreter_2=None,
                            snr=5.0, ref_mic=None, eval_start_s=5.0,
                            use_wpe=False, extra_processors=None,
                            eval_ref_mode="domain"):
    """Nucleo reutilizable del benchmark intrusivo real.

    Parameters
    ----------
    senal_path : WAV multicanal SOLO target (referencia limpia).
    ruido_path : WAV multicanal SOLO ruido/interferente.
    output_dir : carpeta de salida (mixture.wav, senal_ref.wav, intrusive_metrics.csv, WAVs procesados).
    base_config : dict de config para run_real_benchmark + WPE. None => default_base_config(fs).
    interpreter_1, interpreter_2 : interpretes DTLN TF-Lite (None => sin cascadas DTLN).
    snr : SNR objetivo del mixture en dB medido en el mic de ref. None/'nat'/NaN => niveles grabados.
    ref_mic : indice del canal de referencia (None => M//2).
    eval_start_s : segundos iniciales a descartar en la metrica (convergencia).
    use_wpe : si True, aplica WPE a la mezcla ANTES de los beamformers (NODE 4 del MIRD).
    extra_processors : dict {nombre: processor} extra (mask-based / oracle) a correr
        ademas de los beamformers de run_real_benchmark. None => se arman por defecto
        los tres de bf_wrappers: DTLN-Souden-Specsub, Oracle-mask y Oracle-SCM.
        Los oracle usan 'oracle_target'/'oracle_noise' en el MISMO dominio que la mezcla
        que procesan (crudo si no hay WPE front-end; dereverberado si lo hay).
    eval_ref_mode : referencia contra la que se miden los beamformers.
        'domain' (default): cada senal vs su target en el mismo dominio (crudo sin WPE,
            dereverberado con WPE) -> no castiga a la WPE por dereverberar.
        'wpe': el TARGET DEREVERBERADO (WPE_G(target), G de la mezcla) como referencia
            COMUN para ambas ramas (con y sin WPE). Head-to-head hacia voz dereverberada;
            proxy de anecoica-conv-temprano (no disponible aca). El sin-WPE queda con techo
            limitado por la reverb que no removio. Requiere WPE float.

    Returns
    -------
    dict con:
        'rows'          : list[(nombre, metrics_dict)] de cada salida de beamformer.
        'wpe_metrics'   : metrics_dict de la mezcla con WPE (ref-mic) vs limpia (o None).
        'raw_metrics'   : metrics_dict de la mezcla cruda (ref-mic) vs limpia.
        'mixture_path'  : path a mixture.wav.
        'ref_path'      : path a senal_ref.wav.
        'csv_path'      : path a intrusive_metrics.csv.
        'snr_out_db'    : SNR resultante del mixture.
        'ref_mic'       : canal de referencia usado.
        'fp_stats'      : FxStats de la WPE fixed-point (o None).
    """
    sig, fs_s = load_multi(senal_path)
    noi, fs_n = load_multi(ruido_path)
    if fs_s != fs_n:
        raise ValueError(f"fs distintas: senal={fs_s}, ruido={fs_n}")
    fs = fs_s
    if sig.shape[1] != noi.shape[1]:
        raise ValueError(f"distinto numero de canales: senal={sig.shape[1]}, ruido={noi.shape[1]}")

    M = sig.shape[1]
    if ref_mic is None:
        ref_mic = M // 2
    if not (0 <= ref_mic < M):
        raise ValueError(f"ref_mic {ref_mic} fuera de rango (M={M})")

    if base_config is None:
        base_config = default_base_config(fs)
    else:
        # asegurar claves WPE sin pisar lo que el caller haya definido;
        # fs SIEMPRE lo manda el wav (autoritativo para run_real_benchmark).
        base_config = dict(base_config)
        for k, v in default_base_config(fs).items():
            base_config.setdefault(k, v)
        base_config["fs"] = fs

    # 1. igualar longitudes (mismas muestras -> target del mixture == referencia)
    N = min(len(sig), len(noi))
    sig = sig[:N]
    noi = noi[:N]

    # 2. escalar el ruido para el SNR objetivo (medido en el mic de referencia)
    p_sig = float(np.mean(sig[:, ref_mic] ** 2)) + 1e-20
    p_noi = float(np.mean(noi[:, ref_mic] ** 2)) + 1e-20
    snr_nat_db = 10.0 * np.log10(p_sig / p_noi)
    natural = (snr is None) or isinstance(snr, str) or (isinstance(snr, float) and np.isnan(snr))
    if natural:
        g = 1.0
    else:
        # queremos p_sig / (g^2 p_noi) = 10^(snr/10)
        g = float(np.sqrt(p_sig / (p_noi * 10.0 ** (float(snr) / 10.0))))
    mixture = sig + g * noi
    snr_out_db = 10.0 * np.log10(p_sig / ((g ** 2) * p_noi))
    print(f"[*] M={M} canales, fs={fs}, {N/fs:.2f} s. Mic de referencia = ch{ref_mic}.")
    print(f"[*] SNR natural (niveles grabados) = {snr_nat_db:+.1f} dB en ch{ref_mic}.")
    print(f"[*] Ruido escalado x{g:.4f} -> SNR del mixture = {snr_out_db:+.1f} dB.")

    os.makedirs(output_dir, exist_ok=True)

    # referencia limpia (mono, mic de ref) — ground-truth de las metricas intrusivas
    ref = sig[:, ref_mic].astype(np.float64)

    # Componentes limpias tal como aparecen en la mezcla CRUDA (M, N): target y ruido.
    souden_alpha = base_config.get("souden_alpha", 0.99)
    oracle_target = np.ascontiguousarray(sig.T, dtype=np.float64)        # (M, N) crudo
    oracle_noise = np.ascontiguousarray((g * noi).T, dtype=np.float64)   # (M, N) crudo
    fixed_bits = base_config.get("wpe_fixed_bits", None)

    # ref-mic de la mezcla CRUDA (antes de reemplazar 'mixture' por la WPE) para la
    # fila 'mixture_raw'; su metrica se calcula mas abajo contra la eval_ref elegida.
    raw_mix_refmic = mixture[:, ref_mic].astype(np.float64).copy()

    # 3. WPE. Se necesita el target dereverberado si hay WPE front-end o si la ref de
    #    evaluacion es 'wpe' (comun a ambas ramas). Se estima G sobre la mezcla CRUDA y
    #    se aplica el mismo filtro al target/ruido (Opcion B, descomposicion exacta).
    want_wpe_target = use_wpe or (eval_ref_mode == "wpe")
    ref_wpe = None
    tgt_wpe = noi_wpe = mix_wpe = None
    if want_wpe_target:
        if fixed_bits is not None:
            print("[!] [WPE] eval_ref_mode='wpe'/float-only: la descomposicion consistente es "
                  "float; el target de referencia se computa en float aunque el front-end sea fixed.")
        print("[*] [WPE] WPE float con descomposicion consistente (mezcla + target/ruido)...")
        _mix_wpe_mn, (tgt_wpe, noi_wpe) = process_wpe_online_with_components(
            u=mixture.T, components=[oracle_target, oracle_noise],
            taps=base_config["wpe_taps"], delay=base_config["wpe_delay"],
            alpha=base_config["wpe_alpha"], stft_size=base_config["wpe_stft_size"],
            stft_shift=base_config["wpe_stft_shift"],
        )
        mix_wpe = _mix_wpe_mn.T  # (N, M)
        ref_wpe = np.ascontiguousarray(tgt_wpe[ref_mic], dtype=np.float64)

    # Mezcla que consumen los beamformers + refs oracle EN EL MISMO DOMINIO que esa mezcla.
    fp_stats = None
    if use_wpe:
        if fixed_bits is None:
            mixture = mix_wpe
            oracle_target, oracle_noise = tgt_wpe, noi_wpe
        else:
            mixture, fp_stats = apply_wpe(mixture, base_config)  # front-end fixed
            print("[!] [WPE] fixed-point: refs oracle quedan SIN WPE (mismatch con la mezcla "
                  "fixed); interpretar oracle con cuidado.")
    # si no use_wpe: mixture cruda, oracle refs crudas (ya asignadas)

    # Referencia de evaluacion (contra la que se miden los beamformers).
    if eval_ref_mode == "wpe":
        if ref_wpe is None:
            raise ValueError("eval_ref_mode='wpe' requiere WPE float (revisar wpe_fixed_bits).")
        eval_ref = ref_wpe                       # target dereverberado, COMUN a ambas ramas
    elif use_wpe and fixed_bits is None:
        eval_ref = ref_wpe                       # dominio WPE
    else:
        eval_ref = ref                           # dominio crudo

    # 'mixture_raw' = mezcla CRUDA (ref-mic). En modo 'wpe' se mide contra la ref comun
    # (distancia del input crudo a la voz dereverberada); si no, contra el target crudo.
    raw_ref_for_mixraw = eval_ref if eval_ref_mode == "wpe" else ref
    raw_metrics = evaluate_full_pipeline(raw_ref_for_mixraw, raw_mix_refmic,
                                         fs, eval_start_s=eval_start_s)

    wpe_metrics = None
    if use_wpe:
        wpe_metrics = evaluate_full_pipeline(eval_ref, mixture[:, ref_mic].astype(np.float64),
                                             fs, eval_start_s=eval_start_s)

    # guardar mixture (FLOAT, sin perdida) para que lo lea run_real_benchmark
    mix_path = os.path.join(output_dir, "mixture.wav")
    sf.write(mix_path, mixture.astype(np.float32), fs, subtype="FLOAT")
    # guardar la referencia limpia (mono, mic de ref) para escuchar / re-evaluar
    ref_path = os.path.join(output_dir, "senal_ref.wav")
    sf.write(ref_path,
             (ref / (np.max(np.abs(ref)) + 1e-12)).astype(np.float32), fs, subtype="FLOAT")
    if ref_wpe is not None:
        sf.write(os.path.join(output_dir, "senal_ref_wpe.wav"),
                 (ref_wpe / (np.max(np.abs(ref_wpe)) + 1e-12)).astype(np.float32), fs, subtype="FLOAT")

    base_config["oracle_target"] = np.ascontiguousarray(oracle_target, dtype=np.float64)  # (M, N)
    base_config["oracle_noise"] = np.ascontiguousarray(oracle_noise, dtype=np.float64)    # (M, N)

    # Procesadores extra (mask-based / oracle) de bf_wrappers.
    if extra_processors is None:
        extra_processors = {
            "dtln_souden_specsub": NM_MVDR_PF(smooth=0.33, alpha=souden_alpha),
            "oracle_souden_mask": ORACLE_MB_MVDR_SOUDEN(sharpen_exp=1.0, alpha=souden_alpha),
            "oracle_souden_scm": SOUDEN_ORACLE_SCM(alpha=souden_alpha),
        }

    # 4. correr TODOS los beamformers sobre el mixture (reusa el orquestador)
    outputs = run_real_benchmark(
        input_wav=mix_path,
        output_dir=output_dir,
        base_config=base_config,
        interpreter_1=interpreter_1,
        interpreter_2=interpreter_2,
        geometric_processors={},
        extra_processors=extra_processors,
    )

    # 5. metricas INTRUSIVAS: cada salida vs eval_ref.
    if eval_ref_mode == "wpe":
        ref_desc = "target DEREVERBERADO (referencia COMUN a ambas ramas)"
    elif eval_ref is ref:
        ref_desc = "target crudo"
    else:
        ref_desc = "target dereverberado (dominio WPE)"
    print(f"\n=== Metricas INTRUSIVAS (vs {ref_desc}, mayor = mejor) ===")
    cols = ["PESQ", "STOI", "SI-SDR", "SDR", "SAR"]
    print(f"{'senal':<26} " + " ".join(f"{c:>8}" for c in cols))
    print("-" * (26 + 9 * len(cols)))
    rows = []

    # filas extra: mezcla cruda y (si aplica) mezcla con WPE, para dimensionar el aporte de la WPE
    extra_rows = [("mixture_raw", raw_metrics)]
    if use_wpe:
        extra_rows.append(("mixture_wpe", wpe_metrics))
    for name, res in extra_rows:
        print(f"{name:<26} " + " ".join(f"{res.get(c, float('nan')):>8.3f}" for c in cols))

    # ordenar: ref_mic_raw (entrada) primero, despues el resto
    names = list(outputs.keys())
    names.sort(key=lambda n: (n != "ref_mic_raw", n))
    for name in names:
        deg = np.asarray(outputs[name], dtype=np.float64)
        res = evaluate_full_pipeline(eval_ref, deg, fs, eval_start_s=eval_start_s)
        rows.append((name, res))
        print(f"{name:<26} " + " ".join(f"{res.get(c, float('nan')):>8.3f}" for c in cols))

    # 6. CSV
    csv_path = os.path.join(output_dir, "intrusive_metrics.csv")
    with open(csv_path, "w") as f:
        f.write("senal," + ",".join(cols) + ",SIR\n")
        for name, res in extra_rows + rows:
            f.write(name + "," + ",".join(f"{res.get(c, float('nan')):.4f}" for c in cols)
                    + f",{res.get('SIR', float('nan')):.4f}\n")
    print(f"\n[*] CSV: {csv_path}")
    print(f"[*] Mixture: {mix_path}   Referencia: {ref_path}")
    if eval_ref_mode == "wpe":
        print("[*] eval_ref_mode='wpe': TODO se mide vs el target DEREVERBERADO (referencia "
              "COMUN). El sin-WPE queda con techo limitado por la reverb que no removio; "
              "es el head-to-head hacia voz dereverberada (proxy de anecoica*temprano).")
    elif ref_wpe is not None:
        print("[*] WPE (modo 'domain'): beamformers y 'mixture_wpe' vs target dereverberado; "
              "'mixture_raw' vs target crudo. NO restar entre dominios como valor absoluto.")
    print("[*] Comparar cada fila contra 'ref_mic_raw' (la entrada al BF): "
          "el delta PESQ/STOI/SI-SDR es la mejora del procesador.")

    return {
        "rows": rows,
        "wpe_metrics": wpe_metrics,
        "raw_metrics": raw_metrics,
        "mixture_path": mix_path,
        "ref_path": ref_path,
        "csv_path": csv_path,
        "snr_out_db": snr_out_db,
        "ref_mic": ref_mic,
        "fp_stats": fp_stats,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("senal", help="WAV multicanal SOLO target (referencia limpia)")
    ap.add_argument("ruido", help="WAV multicanal SOLO ruido/interferente")
    ap.add_argument("out_dir", nargs="?", default="intrusive_out",
                    help="carpeta de salida (default: intrusive_out)")
    ap.add_argument("--snr", type=float, default=5.0,
                    help="SNR objetivo del mixture en dB, medido en el mic de ref (default 5). "
                         "Usar 'nat' para sumar a los niveles grabados sin escalar.")
    ap.add_argument("--ref-mic", type=int, default=None,
                    help="indice del canal de referencia (default M//2, = el que usa el BF)")
    ap.add_argument("--eval-start", type=float, default=5.0,
                    help="segundos iniciales a descartar en la metrica (convergencia; default 5)")
    ap.add_argument("--wpe", action="store_true",
                    help="aplicar WPE (dereverb) a la mezcla ANTES de los beamformers")
    ap.add_argument("--wpe-bits", type=int, default=None,
                    help="word length fixed-point de la WPE (24/20/18); default None = float")
    ap.add_argument("--wpe-round", default="nearest", choices=["nearest", "floor"],
                    help="modo de redondeo fixed-point de la WPE (default nearest)")
    ap.add_argument("--wpe-taps", type=int, default=7, help="taps de la WPE (default 7)")
    ap.add_argument("--wpe-delay", type=int, default=3, help="delay de la WPE (default 3)")
    ap.add_argument("--eval-ref", default="domain", choices=["domain", "wpe"],
                    help="referencia de evaluacion: 'domain' (cada uno vs su dominio) o "
                         "'wpe' (target dereverberado comun a ambas ramas). default domain")
    args = ap.parse_args()

    # 'nat' llega como NaN por type=float; lo tratamos como natural en la funcion.
    snr = args.snr

    # DTLN interpreters (para DTLN mono + cascadas)
    try:
        interp1 = tf.lite.Interpreter(model_path=DTLN_MODEL_1); interp1.allocate_tensors()
        interp2 = tf.lite.Interpreter(model_path=DTLN_MODEL_2); interp2.allocate_tensors()
        print("[*] Interpretes DTLN TF-Lite cargados.")
    except Exception as e:
        print(f"[!] Sin modelos DTLN (sigo sin las cascadas): {e}")
        interp1, interp2 = None, None

    base_config = default_base_config(fs=16000)  # fs se corrige adentro con el del wav
    base_config["wpe_fixed_bits"] = args.wpe_bits
    base_config["wpe_fixed_round"] = args.wpe_round
    base_config["wpe_taps"] = args.wpe_taps
    base_config["wpe_delay"] = args.wpe_delay

    run_intrusive_benchmark(
        senal_path=args.senal,
        ruido_path=args.ruido,
        output_dir=args.out_dir,
        base_config=base_config,
        interpreter_1=interp1,
        interpreter_2=interp2,
        snr=snr,
        ref_mic=args.ref_mic,
        eval_start_s=args.eval_start,
        use_wpe=args.wpe,
        eval_ref_mode=args.eval_ref,
    )


if __name__ == "__main__":
    main()
