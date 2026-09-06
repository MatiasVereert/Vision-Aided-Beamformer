"""
LA MASCARA SOBRE LA SALIDA DEL BEAMFORMER (y como no matarse con el self-nulling)

`NM_MVDR_DSM_FB` son dos beamformers semi-desacoplados: uno apuntado con la RTF
estimada fabrica la senal que ve el DTLN, y el nucleo de Souden produce la
salida. `NM_MVDR_OFB` deja UNO SOLO: con la retencion de un frame, los pesos del
nucleo ya estan listos antes de correr la red, asi que la red puede comer
directamente la salida del beamformer. Se cae el eigh de Phi_SS del front-end y
el DTLN pasa a ver el iSIR del MVDR completo.

El riesgo es el self-nulling: si la voz se cancela, la mascara la deja de ver, la
voz se va a Phi_NN y el MVDR la anula mas. Estado absorbente. Este script mide
las tres defensas (ver `beamforming/mask/output_feedback.py`):

    ofb_raw    lazo PURO, sin ninguna defensa. Es el control: si esto no se cae,
               el peligro era teorico; si se cae, mide cuanto.
    ofb_lkXX   fuga del canal de referencia en la entrada de la red (b = XX).
    ofb_lk05_g fuga + perro guardian mask-independiente.
    ofb_floor  fuga + piso en la rama de senal.
    ofb_dual   segunda red sobre el canal crudo, m = max(m_out, m_ref). La
               defensa fuerte, pero paga la red que este esquema queria ahorrar:
               es el TECHO contra el que se comparan las baratas.

Referencias: `fb` (el sistema de hoy) y `fb_P1` (el mismo con retencion de un
frame, que es la comparacion justa: OFB tambien retiene un frame).

Uso:
    python tests/window_mismatch/run_output_feedback.py [--full] [--pf] [--isir]
                                                       [--diag] [--poison]
"""

import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = "/home/matias/Documents/Tesis/Vision-Aided-Beamformer"
sys.path.insert(0, os.path.join(ROOT, "src"))

from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search   # noqa: E402
from evaluation.bf_wrappers import NM_MVDR_DSM_FB, NM_MVDR_OFB             # noqa: E402
from propagation.mird_loader import MirdDatasetProvider                    # noqa: E402

MODEL_1 = f"{ROOT}/src/dnn_denoise/models/model_quant_1.tflite"
MODEL_2 = f"{ROOT}/src/dnn_denoise/models/model_quant_2.tflite"
OUT_DIR = os.environ.get("SWEEP_OUT", "tests/dataset_out/output_feedback")

# La configuracion final del sistema (la misma de tests/dsm_blind_real_run.py).
CFG = dict(win_type='rect', synth='hann', sharpen_exp=8.0, smooth=0.5, alpha=0.99)


def build_isir_processors():
    """
    BARRIDO DE iSIR: donde se rompe el lazo, y si el ancla del canal de
    referencia lo arregla.

    El lazo tiene DOS estados absorbentes posibles, no uno:

      * cancelar el TARGET (self-nulling). Verificado que no es alcanzable con
        el core `subtract`: Phi_XX se congela cuando la rama de senal se queda
        sin masa (ver run_output_feedback.py --poison).
      * engancharse al INTERFERENTE. Si el beamformer inicial lo deja pasar, la
        mascara lo marca como voz, Phi_XX lo captura y el lazo lo sostiene. Este
        NO tiene ancla interna, y deberia doler con iSIR bajo.

    Y en el otro extremo hay un problema distinto: con iSIR alto la salida del
    beamformer es casi limpia, el DTLN la ve fuera de su distribucion, la
    mascara satura y con sharpen_exp=8 la rama de RUIDO se queda sin masa
    (m_n = (1-m)^8 -> 0), o sea Phi_NN mal condicionada.

    Las dos cosas las ataca lo mismo: mezclar canal de referencia en la entrada
    de la red (`leak`), que la mantiene en distribucion y le devuelve la
    evidencia de las dos fuentes. Por eso el barrido es sobre `leak`, con
    `dual` como techo.
    """
    ns = {k: v for k, v in CFG.items() if k != 'smooth'}
    return {
        "fb_pf":     NM_MVDR_DSM_FB(mode="fb", block_update=1, smooth=0.5, **ns),
        "ofb_lk00":  NM_MVDR_OFB(leak=0.00, smooth=0.5, **ns),
        "ofb_lk05":  NM_MVDR_OFB(leak=0.05, smooth=0.5, **ns),
        "ofb_lk20":  NM_MVDR_OFB(leak=0.20, smooth=0.5, **ns),
        "ofb_lk40":  NM_MVDR_OFB(leak=0.40, smooth=0.5, **ns),
        "ofb_dual":  NM_MVDR_OFB(leak=0.05, guard="dual", smooth=0.5, **ns),
    }


def build_pf_processors():
    """
    LAS DOS PREGUNTAS DEL POST-FILTRO.

    (1) ¿De donde sale la ganancia de OFB? La mascara del lazo alimenta DOS
    cosas: las SCM del nucleo y el post-filtro espectral G = s + (1-s) m. Y hay
    una razon estructural para esperar que la segunda pese mas: en OFB la
    mascara se estima sobre Y(t) = w(t-1)^H x(t), que es EXACTAMENTE la senal
    que el post-filtro despues multiplica. En `fb` la mascara viene de otra
    senal (la salida del front-end apuntado), asi que el post-filtro aplica una
    ganancia estimada sobre una senal que no es la suya. Prendiendo y apagando
    `smooth` en los dos sistemas se separa:

        fb_nopf / ofb_nopf   sin post-filtro  -> aisla el efecto sobre las SCM
        fb_pf   / ofb_pf     con post-filtro  -> si el gap CRECE, la hipotesis
                                                 del post-filtro se sostiene

    (2) ¿Y si en vez de la ganancia espectral se corre el DTLN ENTERO sobre la
    salida? La etapa 1 ya se esta corriendo para la mascara; la etapa 2 es el
    nucleo de separacion en el TIEMPO, y se le puede dar el bloque conformado.

        ofb_s2    G relajada por smooth=0.5 y despues la etapa 2
        ofb_s2_x  el DTLN tal cual (smooth=None): mascara cruda + etapa 2

    OJO: la etapa 2 NO es una mascara espectral (medido sobre voz limpia,
    |rfft(out_block)|/|E| va de 0.12 a 3.0), asi que no se puede realimentar a
    las SCM; solo sirve en el camino de salida.
    """
    ns = {k: v for k, v in CFG.items() if k != 'smooth'}
    return {
        "fb_nopf":   NM_MVDR_DSM_FB(mode="fb", block_update=1, smooth=None, **ns),
        "fb_pf":     NM_MVDR_DSM_FB(mode="fb", block_update=1, smooth=0.5, **ns),
        "ofb_nopf":  NM_MVDR_OFB(leak=0.05, smooth=None, **ns),
        "ofb_pf":    NM_MVDR_OFB(leak=0.05, smooth=0.5, **ns),
        "ofb_s2":    NM_MVDR_OFB(leak=0.05, smooth=0.5, stage2="pf", **ns),
        "ofb_s2_x":  NM_MVDR_OFB(leak=0.05, smooth=None, stage2="pf", **ns),
    }


def build_processors():
    return {
        "fb":         NM_MVDR_DSM_FB(mode="fb", **CFG),
        "fb_P1":      NM_MVDR_DSM_FB(mode="fb", block_update=1, **CFG),
        "ofb_raw":    NM_MVDR_OFB(leak=0.0, **CFG),
        "ofb_lk05":   NM_MVDR_OFB(leak=0.05, **CFG),
        "ofb_lk20":   NM_MVDR_OFB(leak=0.20, **CFG),
        "ofb_lk05_g": NM_MVDR_OFB(leak=0.05, guard="snr", **CFG),
        "ofb_floor":  NM_MVDR_OFB(leak=0.05, mask_floor=0.05, **CFG),
        "ofb_dual":   NM_MVDR_OFB(leak=0.0, guard="dual", **CFG),
    }


def base_config():
    return {
        'fs': 16000, 'duration': 15, 't_early': 0.050,
        'array_center': [3.0, 3.0, 1.2], 'mird_spacing': "3-3-3-8-3-3-3",
        'snr_db': 60.0,
        'source_path': f"{ROOT}/tools/data/signals/p002_emo_adoration_sentences.wav",
        'interf_paths': [f"{ROOT}/tools/data/signals/techno_gated commune.wav"],
        'wpe_taps': 7, 'wpe_delay': 3, 'wpe_alpha': 0.9999,
        'wpe_stft_size': 512, 'wpe_stft_shift': 128,
        'wpe_fixed_bits': None, 'wpe_fixed_round': 'nearest', 'wpe_backend': 'cov',
        'wpe_block_L': 512, 'wpe_block_shift': 2, 'wpe_block_iters': 2,
        'wpe_block_reg': 1e-6, 'wpe_block_solver': 'cholesky', 'wpe_block_mode': 'resolve',
        'stft_window': 512, 'stft_overlap': 384,
        'eval_references': ['anechoic', 'early', 'reverberant'],
        'dtln_model_path': MODEL_1, 'dtln_model2_path': MODEL_2,
    }


def main():
    try:
        itp1 = tf.lite.Interpreter(model_path=MODEL_1); itp1.allocate_tensors()
        itp2 = tf.lite.Interpreter(model_path=MODEL_2); itp2.allocate_tensors()
    except Exception as e:
        print(f"[!] DTLN no cargado: {e}")
        itp1 = itp2 = None

    provider = MirdDatasetProvider(root_dir=os.path.abspath(f"{ROOT}/tools/data/rirs/mird"))
    full = "--full" in sys.argv
    param_grid = {
        'rt60': [0.360, 0.610], 'target_angle': [0], 'target_dist': [1.0],
        'interf_configs': ([[(45, 1.0)], [(90, 2.0)]] if full else [[(45, 1.0)]]),
        'isir_db': ([-5, 0, 5, 10, 15] if "--isir" in sys.argv else [-5, 0]),
        'mismatch_gain': [0], 'mismatch_phase': [0],
        'use_wpe': [False], 'wpe_method': ['online'], 'wpe_taps': [7], 'wpe_delay': [2],
        'error_angle_deg': [0.0], 'error_distance_m': [0.0],
    }
    df = run_mird_grid_search(
        grid_params=param_grid, dataset_provider=provider,
        processors=(build_isir_processors() if "--isir" in sys.argv
                    else build_pf_processors() if "--pf" in sys.argv
                    else build_processors()),
        scene_base_config=base_config(),
        output_dir=OUT_DIR, interpreter_1=itp1, interpreter_2=itp2,
        save_catalog=False, apply_dtln_post=False)
    summarize(df)
    return df


def summarize(df, ref='early'):
    metrics = ['PESQ', 'STOI', 'SI-SDR', 'SDR', 'SIR', 'SAR']
    cols = [f"proc_{m}_{ref}" for m in metrics if f"proc_{m}_{ref}" in df.columns]
    pd.set_option('display.width', 220)
    print(f"\n=== mascara sobre la salida (ref '{ref}') ===")
    print(df.groupby('processor')[cols].mean().round(3).to_string())

    if 'isir_db' in df.columns and df.isir_db.nunique() > 2:
        print("\n=== PESQ por iSIR ===")
        print(df.pivot_table(index='isir_db', columns='processor',
                             values='proc_PESQ_early').round(3).to_string())
        print("\n=== SDR por iSIR ===")
        print(df.pivot_table(index='isir_db', columns='processor',
                             values='proc_SDR_early').round(2).to_string())
    refs = [b for b in ("fb", "fb_P1", "fb_nopf", "fb_pf", "ofb_nopf", "ofb_pf")
            if b in set(df.processor)]
    for a in [p for p in df.processor.unique() if p.startswith("ofb")]:
        for b in refs:
            if b == a:
                continue
            A = df[df.processor == a].reset_index(drop=True)
            B = df[df.processor == b].reset_index(drop=True)
            if A.empty or B.empty:
                continue
            n = min(len(A), len(B))
            print(f"\n--- {a} - {b} ({n} escenas) ---")
            for c in cols:
                d = A[c].to_numpy()[:n] - B[c].to_numpy()[:n]
                print(f"   {c:20s} media {np.nanmean(d):+7.3f}"
                      f"  mediana {np.nanmedian(d):+7.3f}"
                      f"  gana {int(np.sum(d > 0))}/{n}")


# ---------------------------------------------------------------------------
# DIAGNOSTICO: la trayectoria del lazo en UNA escena
# ---------------------------------------------------------------------------
def diag():
    """
    UNA sola celda, mirando la MASA DE LA MASCARA en la banda de voz: esa es la
    variable de estado del colapso. Si el lazo se auto-anula, la masa cae y NO
    vuelve (estado absorbente); si las defensas funcionan, se sostiene.

    Se corre dentro del mismo barrido para que las senales sean exactamente las
    del benchmark (misma escena, mismo WPE, mismo HW).
    """
    try:
        itp1 = tf.lite.Interpreter(model_path=MODEL_1); itp1.allocate_tensors()
        itp2 = tf.lite.Interpreter(model_path=MODEL_2); itp2.allocate_tensors()
    except Exception:
        itp1 = itp2 = None

    variants = {
        "raw (b=0)":    dict(leak=0.0),
        "leak b=0.05":  dict(leak=0.05),
        "leak+guard":   dict(leak=0.05, guard="snr"),
        "dual":         dict(leak=0.0, guard="dual"),
    }
    sinks, procs = {}, {}
    for name, kw in variants.items():
        sinks[name] = {}
        pr = NM_MVDR_OFB(**kw, **CFG)
        pr.diag_sink = sinks[name]
        procs[name] = pr

    provider = MirdDatasetProvider(root_dir=os.path.abspath(f"{ROOT}/tools/data/rirs/mird"))
    param_grid = {
        'rt60': [0.610], 'target_angle': [0], 'target_dist': [1.0],
        'interf_configs': [[(45, 1.0)]], 'isir_db': [-5],
        'mismatch_gain': [0], 'mismatch_phase': [0],
        'use_wpe': [False], 'wpe_method': ['online'], 'wpe_taps': [7], 'wpe_delay': [2],
        'error_angle_deg': [0.0], 'error_distance_m': [0.0],
    }
    run_mird_grid_search(
        grid_params=param_grid, dataset_provider=provider, processors=procs,
        scene_base_config=base_config(),
        output_dir=os.path.join(OUT_DIR, "diag"),
        interpreter_1=itp1, interpreter_2=itp2, save_catalog=False,
        apply_dtln_post=False)

    print("\n=== masa de la mascara en 300-3400 Hz, por decimo de archivo ===")
    for name, d in sinks.items():
        if not d:
            print(f"  {name:14s} (sin diagnostico)")
            continue
        f = d['freqs']
        gb = (f >= 300.0) & (f <= 3400.0)
        mass = d['m_raw'][gb].mean(axis=0)
        q = np.percentile(mass, [5, 50, 95])
        seg = " ".join(f"{s.mean():.2f}" for s in np.array_split(mass, 10))
        print(f"  {name:14s} p05/p50/p95 {q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f}"
              f"  [{seg}]  abierto {100*np.mean(d['open']):.0f}%")


# ---------------------------------------------------------------------------
# ESTRES: ¿el self-nulling es un estado ABSORBENTE?
# ---------------------------------------------------------------------------
def poison():
    """
    Que ninguna escena lo dispare no prueba que no exista. Aca se lo dispara A
    MANO: durante 1.5 s se le miente a la estadistica (m_s -> 0, m_n -> 1), o
    sea se le mete la voz entera adentro de Phi_NN. Eso es exactamente el estado
    al que el lazo podria caer solo.

    Lo que importa es lo de DESPUES: si el estado es absorbente, la masa de la
    mascara no vuelve nunca; si hay mecanismo de recuperacion, vuelve, y el
    numero interesante es cuanto tarda. La constante de tiempo del nucleo
    (alpha=0.99, hop 128 @ 16 kHz) es ~0.8 s: por debajo de eso no hay nada que
    esperar, muy por encima es que el lazo se realimenta a si mismo.
    """
    try:
        itp1 = tf.lite.Interpreter(model_path=MODEL_1); itp1.allocate_tensors()
        itp2 = tf.lite.Interpreter(model_path=MODEL_2); itp2.allocate_tensors()
    except Exception:
        itp1 = itp2 = None

    # Largo del veneno en frames. El default (190 = 1.5 s) es ~2 constantes de
    # tiempo del nucleo; con `--poison-frames 600` (4.8 s = 6 tau) Phi_NN se
    # satura por completo de voz, que es el peor caso que se puede construir.
    T0 = 400
    n_poison = 190
    if "--poison-frames" in sys.argv:
        n_poison = int(sys.argv[sys.argv.index("--poison-frames") + 1])
    T1 = T0 + n_poison
    variants = {
        "raw (b=0)":   dict(leak=0.0),
        "leak b=0.05": dict(leak=0.05),
        "leak b=0.20": dict(leak=0.20),
        "floor 0.05":  dict(leak=0.05, mask_floor=0.05),
        "dual":        dict(leak=0.0, guard="dual"),
    }
    sinks, procs = {}, {}
    for name, kw in variants.items():
        for tag, poi in (("sano", None), ("veneno", (T0, T1))):
            key = f"{name} | {tag}"
            sinks[key] = {}
            pr = NM_MVDR_OFB(**kw, **CFG)
            pr.poison = poi
            pr.diag_sink = sinks[key]
            procs[key] = pr

    provider = MirdDatasetProvider(root_dir=os.path.abspath(f"{ROOT}/tools/data/rirs/mird"))
    param_grid = {
        'rt60': [0.610], 'target_angle': [0], 'target_dist': [1.0],
        'interf_configs': [[(45, 1.0)]], 'isir_db': [-5],
        'mismatch_gain': [0], 'mismatch_phase': [0],
        'use_wpe': [False], 'wpe_method': ['online'], 'wpe_taps': [7], 'wpe_delay': [2],
        'error_angle_deg': [0.0], 'error_distance_m': [0.0],
    }
    df = run_mird_grid_search(
        grid_params=param_grid, dataset_provider=provider, processors=procs,
        scene_base_config=base_config(),
        output_dir=os.path.join(OUT_DIR, "poison"),
        interpreter_1=itp1, interpreter_2=itp2, save_catalog=False,
        apply_dtln_post=False)

    print(f"\n=== RECUPERACION tras envenenar frames {T0}-{T1} "
          f"({(T1-T0)*128/16000:.1f} s) ===")
    print("   masa de mascara (300-3400 Hz) promediada en ventanas de 0.8 s "
          "despues del veneno, contra la corrida SANA")
    for name in variants:
        d_ok, d_bad = sinks[f"{name} | sano"], sinks[f"{name} | veneno"]
        if not d_ok or not d_bad:
            continue
        f = d_ok['freqs']
        gb = (f >= 300.0) & (f <= 3400.0)
        a = d_ok['m_raw'][gb].mean(axis=0)
        b = d_bad['m_raw'][gb].mean(axis=0)
        win = 100                                    # ~0.8 s
        cells = []
        for i in range(6):
            s0 = T1 + i * win
            cells.append(f"{b[s0:s0+win].mean():.2f}/{a[s0:s0+win].mean():.2f}")
        # frame en que la masa vuelve a la mitad de la sana, de forma sostenida
        rec = None
        for t in range(T1, len(b) - win):
            if b[t:t+win].mean() > 0.5 * a[t:t+win].mean():
                rec = (t - T1) * 128 / 16000
                break
        print(f"  {name:13s} veneno/sano por 0.8 s: {'  '.join(cells)}"
              f"   recupera en {('%.2f s' % rec) if rec is not None else 'NUNCA'}")

    cols = [c for c in df.columns if c.startswith('proc_PESQ')]
    if cols:
        print("\n   PESQ (el veneno esta DENTRO de la ventana de metricas):")
        print(df.groupby('processor')[cols[0]].mean().round(3).to_string())


if __name__ == "__main__":
    if "--poison" in sys.argv:
        poison()
    elif "--diag" in sys.argv:
        diag()
    else:
        main()
