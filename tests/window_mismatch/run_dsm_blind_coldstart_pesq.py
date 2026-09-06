"""
¿COMO SE TRADUCE A PESQ EL ARRANQUE EN FRIO DEL LAZO CIEGO?

El banco de dsm_blind_feedback_diag.py mide apuntamiento y SNR del front-end,
que son observables INTERNOS. Esto mide lo de afuera: PESQ/STOI/SDR de la cadena
completa, con las tres piezas de arranque en frio aisladas.

DOS COSAS QUE HAY QUE FORZAR PARA QUE EL EFECTO SEA VISIBLE
-----------------------------------------------------------
1. `eval_start_s = 0.0`. El default del benchmark descarta los primeros
   min(5, 0.3*duracion) segundos, o sea que TODO el transitorio de convergencia
   cae dentro de la ventana que se tira. Con 0 se mide desde la primera muestra.
2. Una escena que empiece con ruido. La fuente del benchmark arranca hablando a
   los 0.05 s, asi que el modo de falla no aparece nunca. Se sintetizan dos wavs
   que se pasan por el eje de grilla `source_path`:
       p002_delay0.wav : voz desde t=0 (control sano)
       p002_delay8.wav : 8 s de silencio y despues LA MISMA voz, muestra a
                         muestra, asi lo unico que cambia es el prefijo.

OJO AL COMPARAR ENTRE delay0 Y delay8: `mix_and_normalize` fija el iSIR con el
RMS GLOBAL del target, asi que 8 s de silencio en un archivo de 21 s le bajan el
RMS ~2.1 dB y el interferente se escala igual de menos. En las celdas delay8 el
iSIR LOCAL durante la voz es ~2 dB mejor que el nominal. Afecta a todos los
procesadores por igual, asi que la comparacion PROCESADOR CONTRA PROCESADOR
dentro de una celda es valida; la de PESQ absoluto entre delay0 y delay8 no.

SIN WPE. El WPE tiene su propio arranque en frio y con eval_start_s=0 caeria
dentro de la ventana: seria un segundo transitorio mezclado con el que se quiere
medir. Ademas ya no es parte del sistema.

FILAS
-----
    blind_prod    : el DSM_BLIND anterior en su forma de PRODUCCION (hamming,
                    pico global + stretch, sharpen 4). NO causal: es la
                    referencia historica, no una opcion de implementacion.
    blind_causal  : el DSM_BLIND anterior en la cadena CAUSAL (rect + sintesis
                    con taper + sharpen 8, causal=True). Es el control directo:
                    todas las variantes de abajo son esta fila mas una pieza.
    blind_cau_gate: + gate duro por confianza (conf_gate). Las otras dos piezas
                    que se probaron (carga guiada por confianza y schedule de
                    alpha) quedaron descartadas: ver `estimate_rtf_recursive`.

Uso:
    python tests/window_mismatch/run_dsm_blind_coldstart_pesq.py [--quick] [--full]
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search   # noqa: E402
from evaluation.bf_wrappers import (NM_MVDR_DSM_BLIND, NM_MVDR_DSM_FB,   # noqa: E402
                                    NM_MVDR_OFB)
from propagation.mird_loader import MirdDatasetProvider                     # noqa: E402

MODEL_1 = f"{ROOT}/src/dnn_denoise/models/model_quant_1.tflite"
MODEL_2 = f"{ROOT}/src/dnn_denoise/models/model_quant_2.tflite"
OUT_DIR = os.environ.get("SWEEP_OUT", "tests/dataset_out/dsm_blind_coldstart")
SIG_DIR = os.path.join(ROOT, "tests/dataset_out/dsm_blind_pesq/signals")
DURATION = 21.0
CONF_ALPHA = 0.99          # la sombra que mide la confianza es un DETECTOR: corta
CONF_GATE = 0.35


def build_sources(delays=(0.0, 8.0), dur=DURATION):
    """Wavs con el MISMO contenido de voz y distinto prefijo de silencio."""
    os.makedirs(SIG_DIR, exist_ok=True)
    src = os.path.join(ROOT, "tools/data/signals/p002_emo_adoration_sentences.wav")
    x, fs = sf.read(src)
    x = np.asarray(x, dtype=np.float64)
    n = int(dur * fs)
    paths = []
    for d in delays:
        out = os.path.join(SIG_DIR, f"p002_delay{int(d)}.wav")
        nd = int(d * fs)
        body = np.tile(x, int(np.ceil((n - nd) / len(x))))[:n - nd]
        sf.write(out, np.concatenate([np.zeros(nd), body]), fs)
        paths.append(out)
    return paths


def build_processors():
    P = NM_MVDR_DSM_BLIND
    cau = dict(win_type='rect', synth='hann', sharpen_exp=8.0, causal=True)
    return {
        "blind_prod": P(win_type='hamming', sharpen_exp=4.0),
        "blind_causal": P(**cau),
        "blind_cau_gate": P(**cau, conf_gate=CONF_GATE, conf_alpha=CONF_ALPHA),
    }


def build_fb_processors():
    """
    EL LAZO REALIMENTADO CONTRA EL DE DOS PASADAS, EN LA ESCENA QUE MAS DUELE.

    `NM_MVDR_DSM_FB` saca la pasada del canal crudo y realimenta la propia
    mascara al estimador de RTF. Esa pasada era un ANCLA: independiente del lazo,
    imposible de corromper por la realimentacion. Aca no hay ancla, asi que el
    prefijo de ruido es exactamente el caso donde la variante deberia romperse si
    se va a romper -- y donde el gate de confianza pasa de opcional a necesario.

        base    : el de dos pasadas, cadena causal + post-filtro.
        spec    : dos pasadas, pero la mascara(2) sale del ESPECTRO conformado
                  (sin resintetizar). Aisla ese cambio del de la realimentacion.
        fb      : un solo DTLN, lazo cerrado.
        fb_gate : el lazo cerrado + el gate de arranque en frio.
    """
    cfg = dict(win_type='rect', synth='hann', sharpen_exp=8.0, smooth=0.5, alpha=0.99)
    return {
        "base": NM_MVDR_DSM_BLIND(causal=True, **cfg),
        "spec": NM_MVDR_DSM_FB(mode="spec", **cfg),
        "fb": NM_MVDR_DSM_FB(mode="fb", **cfg),
        "fb_gate": NM_MVDR_DSM_FB(mode="fb", conf_gate=CONF_GATE,
                                  conf_alpha=CONF_ALPHA, **cfg),
    }


def build_ofb_processors():
    """
    LA MASCARA SOBRE LA SALIDA DEL BEAMFORMER, EN LA ESCENA QUE MAS DUELE.

    `NM_MVDR_OFB` cierra el lazo un paso mas: saca el front-end (y con el la
    estimacion de RTF) y le da al DTLN la SALIDA del beamformer, aprovechando
    que con la retencion de un frame los pesos ya estan listos. La contra es el
    self-nulling, que es un estado ABSORBENTE -- y el prefijo de ruido es
    exactamente donde se puede caer adentro: mientras no hay voz, la unica
    evidencia que tiene el lazo es la que el mismo fabrica.

        ofb_raw    sin ninguna defensa. Es el control: mide si el peligro es
                   real o teorico.
        ofb_lk05   fuga del canal de referencia (b=0.05) en la ENTRADA de la
                   red: en el peor caso la red ve b*x_ref, o sea el sistema base
                   escalado, y el lazo puede salir del pozo.
        ofb_lk05_g fuga + perro guardian con un estadistico independiente de la
                   mascara (min-tracking del canal crudo).
        ofb_dual   segunda red sobre el canal de referencia, m = max(m_out,
                   m_ref): la defensa fuerte, pero paga la red que este esquema
                   queria ahorrar. Es el TECHO.
    """
    cfg = dict(win_type='rect', synth='hann', sharpen_exp=8.0, smooth=0.5, alpha=0.99)
    return {
        "fb":         NM_MVDR_DSM_FB(mode="fb", **cfg),
        "ofb_raw":    NM_MVDR_OFB(leak=0.0, **cfg),
        "ofb_lk05":   NM_MVDR_OFB(leak=0.05, **cfg),
        "ofb_lk05_g": NM_MVDR_OFB(leak=0.05, guard="snr", **cfg),
        "ofb_dual":   NM_MVDR_OFB(leak=0.0, guard="dual", **cfg),
    }


def build_fb_sd_processors():
    """
    EL GATE EN EL FRONT-END SUPERDIRECTIVO, EN LA ESCENA DONDE PUEDE VALER.

    El barrido en escena SANA (run_dsm_blind_feedback.py --sd-gate) dice que el
    gate no se paga: sd_base_ng - sd_base = +0.006 PESQ (6/8). Pero eso mide solo
    su COSTO: con voz desde t=0 la RTF converge rapido y no hay error de
    apuntamiento del que proteger.

    La hipotesis a testear es CONDICIONAL: el superdirectivo achica el margen de
    WNG para comprar directividad, asi que es mas sensible a una RTF mal
    estimada -- un error que con DS solo desapunta, aca puede caer en un nulo. Si
    eso vale, el gate tiene que rendir MAS con SD que con DS justamente en el
    prefijo de ruido, que es cuando la RTF es peor.

    OJO CON UN DETALLE: con el gate cerrado queda d = e_ref, y con w_mode='sd'
    eso NO es el canal de referencia sino el superdirectivo apuntado al
    BROADSIDE. En esta grilla el target esta a 0 grados, o sea broadside, asi que
    el estado de reposo del gate apunta a la fuente. Con la fuente fuera de eje
    el resultado podria invertirse: esta grilla no lo cubre.
    """
    cfg = dict(win_type='rect', synth='hann', sharpen_exp=8.0, smooth=0.5, alpha=0.99)
    sd = dict(w_mode="sd", sd_eps=0.30)
    gate = dict(conf_gate=CONF_GATE, conf_alpha=CONF_ALPHA)
    return {
        "sd_base":    NM_MVDR_DSM_BLIND(causal=True, **sd, **gate, **cfg),
        "sd_base_ng": NM_MVDR_DSM_BLIND(causal=True, **sd, **cfg),
        "sd_fb":      NM_MVDR_DSM_FB(mode="fb", **sd, **gate, **cfg),
        "sd_fb_ng":   NM_MVDR_DSM_FB(mode="fb", **sd, **cfg),
    }


def build_sd_processors():
    """
    SEMI-CIEGO: el front-end de la mascara pasa de DS a superdirectivo usando la
    coherencia difusa TEORICA del arreglo. Sigue sin saber donde esta la fuente
    (la RTF se estima igual que siempre); lo unico que se agrega es que conoce su
    propia geometria. Todas las filas llevan el gate prendido.

    Calibracion de sd_eps sobre este arreglo (M=8, broadside), DI = ganancia
    contra ruido difuso, WNG = ganancia contra ruido blanco (negativo = AMPLIFICA
    el ruido propio de los microfonos):

        sd_eps    DI 200-500 Hz   WNG 200-500 Hz   DI 500-1k   WNG 500-1k
        1.0 (DS)      +0.5 dB         +9.0 dB       +2.0 dB      +9.0 dB
        0.3           +0.7            +7.7          +2.9         +6.3
        0.1           +1.2            +2.4          +3.5         +2.9
        1e-6          +4.3           -40.7          +5.6        -35.1

    O sea: el superdirectivo SIN restriccion compra ~+3.8 dB de directividad en
    graves a cambio de amplificar el ruido de los microfonos 40 dB. Por eso el
    banco corre tambien con snr_db bajo: con los 60 dB del default el ruido
    termico esta tan abajo que la restriccion de WNG no restringe nada y el
    resultado saldria enganosamente a favor del superdirectivo.
    """
    P = NM_MVDR_DSM_BLIND
    cau = dict(win_type='rect', synth='hann', sharpen_exp=8.0, causal=True,
               conf_gate=CONF_GATE, conf_alpha=CONF_ALPHA)
    return {
        "gate_ds": P(**cau),                                   # w_mode='ds'
        "gate_sd_e30": P(**cau, w_mode="sd", sd_eps=0.30),     # suave
        "gate_sd_e10": P(**cau, w_mode="sd", sd_eps=0.10),     # intermedio
        "gate_sd_free": P(**cau, w_mode="sd", sd_eps=1e-6),    # sin restriccion
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="1 rt60 x 1 iSIR x 2 fuentes = 2 celdas")
    ap.add_argument("--full", action="store_true",
                    help="agrega la segunda geometria de interferente")
    ap.add_argument("--wide", action="store_true",
                    help="grilla ampliada: 2 rt60 x 3 iSIR x 2 geometrias de "
                         "interferente x 2 fuentes = 24 celdas. Apunta a tener "
                         "MAS DE UNA celda donde el prefijo duela: en la grilla "
                         "chica toda la ganancia salia de (rt60 0.36, iSIR 0).")
    ap.add_argument("--fb-sd", action="store_true",
                    help="gate ON/OFF en el front-end superdirectivo, con y sin "
                         "lazo realimentado")
    ap.add_argument("--fb", action="store_true",
                    help="compara el lazo realimentado (un solo DTLN) contra el "
                         "de dos pasadas")
    ap.add_argument("--ofb", action="store_true",
                    help="la mascara sobre la SALIDA del beamformer (sin "
                         "front-end ni RTF) y sus defensas contra el self-nulling")
    ap.add_argument("--sd", action="store_true",
                    help="compara DS vs superdirectivo (semi-ciego), todo con gate")
    ap.add_argument("--snr-db", type=float, default=60.0,
                    help="ruido propio de los microfonos. 60 = default del "
                         "benchmark (deja la superdirectividad sin restriccion "
                         "real); 30 esta mas cerca de un array MEMS.")
    ap.add_argument("--eval-start", type=float, default=0.0,
                    help="segundos iniciales descartados (0 = se mide el transitorio)")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    src0, src8 = build_sources()
    try:
        itp1 = tf.lite.Interpreter(model_path=MODEL_1); itp1.allocate_tensors()
        itp2 = tf.lite.Interpreter(model_path=MODEL_2); itp2.allocate_tensors()
    except Exception as e:
        print(f"[!] DTLN no cargado: {e}")
        itp1 = itp2 = None

    provider = MirdDatasetProvider(root_dir=os.path.abspath(f"{ROOT}/tools/data/rirs/mird"))
    base_config = {
        'fs': 16000, 'duration': DURATION, 't_early': 0.050,
        'eval_start_s': args.eval_start,        # <- la clave de todo esto
        'array_center': [3.0, 3.0, 1.2], 'mird_spacing': "3-3-3-8-3-3-3",
        'snr_db': args.snr_db,
        'source_path': src0,
        'interf_paths': [f"{ROOT}/tools/data/signals/techno_gated commune.wav"],
        # El bloque WPE se lee aunque use_wpe=False; se deja el del runner causal.
        'wpe_taps': 7, 'wpe_delay': 3, 'wpe_alpha': 0.9999,
        'wpe_stft_size': 512, 'wpe_stft_shift': 128,
        'wpe_fixed_bits': None, 'wpe_fixed_round': 'nearest', 'wpe_backend': 'cov',
        'wpe_block_L': 512, 'wpe_block_shift': 2, 'wpe_block_iters': 2,
        'wpe_block_reg': 1e-6, 'wpe_block_solver': 'cholesky', 'wpe_block_mode': 'resolve',
        'stft_window': 512, 'stft_overlap': 384,
        'eval_references': ['early'],
        'dtln_model_path': MODEL_1, 'dtln_model2_path': MODEL_2,
    }
    procs = (build_ofb_processors() if args.ofb else
             build_fb_sd_processors() if args.fb_sd else
             build_fb_processors() if args.fb else
             build_sd_processors() if args.sd else build_processors())
    param_grid = {
        'rt60': [0.610] if args.quick else [0.360, 0.610],
        'target_angle': [0], 'target_dist': [1.0],
        'source_path': [src0, src8],
        'interf_configs': ([[(45, 1.0)], [(90, 2.0)]]
                           if (args.full or args.wide) else [[(45, 1.0)]]),
        'isir_db': [0] if args.quick else ([-5, 0, 5] if args.wide else [-5, 0]),
        'mismatch_gain': [0], 'mismatch_phase': [0],
        'use_wpe': [False], 'wpe_method': ['online'], 'wpe_taps': [7], 'wpe_delay': [2],
        'error_angle_deg': [0.0], 'error_distance_m': [0.0],
    }
    df = run_mird_grid_search(
        grid_params=param_grid, dataset_provider=provider,
        processors=procs, scene_base_config=base_config,
        output_dir=args.out_dir, interpreter_1=itp1, interpreter_2=itp2,
        save_catalog=False, apply_dtln_post=False)
    df["prefijo"] = np.where(df["source"].astype(str).str.contains("delay8"),
                             "ruido 8 s", "voz desde 0")
    df.to_csv(os.path.join(args.out_dir, "coldstart_pesq.csv"), index=False)
    summarize(df)
    return df


def summarize(df, ref='early'):
    metrics = ['PESQ', 'STOI', 'SI-SDR', 'SDR', 'SIR', 'SAR']
    cols = [f"proc_{m}_{ref}" for m in metrics if f"proc_{m}_{ref}" in df.columns]
    order = ["blind_prod", "blind_causal", "blind_cau_gate",
             "gate_ds", "gate_sd_e30", "gate_sd_e10", "gate_sd_free",
             "base", "spec", "fb", "fb_gate",
             "sd_base", "sd_base_ng", "sd_fb", "sd_fb_ng",
             "ofb_raw", "ofb_lk05", "ofb_lk05_g", "ofb_dual"]
    order = [p for p in order if p in set(df.processor)]
    pd.set_option('display.width', 230)

    print(f"\n=== MEDIA POR PREFIJO (ref '{ref}') ===")
    for pref, gp in df.groupby("prefijo"):
        print(f"\n--- prefijo: {pref}  ({gp.experiment_id.nunique() if 'experiment_id' in gp else len(gp)//max(len(order),1)} celdas) ---")
        t = gp.groupby('processor')[cols].mean().reindex(order).round(3)
        print(t.to_string())

    # Deltas contra el control causal, celda a celda (pareado).
    key = [c for c in ("rt60", "isir_db", "source", "interf_configs") if c in df.columns]
    procset = set(df.processor)
    ctrl = ("sd_base" if "sd_base" in procset else
            "base" if "base" in procset else
            "gate_ds" if "gate_ds" in procset else "blind_causal")
    base = df[df.processor == ctrl]
    print(f"\n=== DELTA CONTRA {ctrl} (pareado por celda) ===")
    for pref, gp in df.groupby("prefijo"):
        b = base[base.prefijo == pref].set_index(key)
        print(f"\n--- prefijo: {pref} ---")
        rows = []
        for p in order:
            if p == ctrl:
                continue
            a = gp[gp.processor == p].set_index(key)
            idx = a.index.intersection(b.index)
            if not len(idx):
                continue
            r = {"processor": p, "n": len(idx)}
            for c in cols:
                d = a.loc[idx, c].to_numpy() - b.loc[idx, c].to_numpy()
                r[c.replace(f"proc_", "").replace(f"_{ref}", "")] = np.nanmean(d)
                r[c.replace(f"proc_", "").replace(f"_{ref}", "") + "_gana"] = \
                    f"{int(np.sum(d > 0))}/{len(idx)}"
            rows.append(r)
        if rows:
            print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
