"""
CUANTO NIVEL DE ENTRADA AGUANTA EL SISTEMA (¿HACE FALTA UN AGC?)

Todo el benchmark MIRD corre con la mezcla normalizada por PICO
(`mix_and_normalize`: el maximo sobre todos los micros queda en 0.99). Como
control experimental esta bien -- fija el nivel para que no ensucie los ejes que
si se estudian -- pero implica que el sistema NUNCA fue evaluado fuera de ese
punto de operacion, y el port a tiempo real si va a ver nivel variable
(distancia al hablante, ganancia del ADC, volumen del locutor).

El bloque sensible es el DTLN: es una red CUANTIZADA que come la magnitud SIN
normalizar del bloque, asi que su punto de operacion es absoluto. Medido fuera
de este harness (un archivo de voz limpia, correlacion de la mascara contra la
de nivel nominal): >=0.94 en -6..+6 dB, 0.81 en -20, 0.52 en +20, y la masa de
la rama de ruido (1-m)^8 se va por un factor 10-100 en los extremos. Pero eso
es correlacion de mascara sobre un ARCHIVO de voz limpia; aca se mide PESQ
sobre celdas reales, y adentro del lazo ese factor 10-100 NO aparece (ver EL
MECANISMO).

DISENO
------
Un decorador, `LevelScaled`, que aplica g a la ENTRADA del procesador y 1/g a su
SALIDA. El nucleo de Souden es invariante a escala (la carga es relativa a la
traza y lambda es un cociente), asi que el camino de metricas ve exactamente la
misma senal que sin decorador: lo UNICO que cambia es el punto de operacion de
la red. La ganancia se aplica DESPUES de `mix_and_normalize`, asi que el iSIR y
el SNR de la escena no se tocan (el iSIR es un cociente).

El procesador interno es `OFB_MVDR` (la version cristalizada, la que se porta a
C++), construido como en tests/ofb_auto_benchmark.py.

Un solo eje de iSIR alcanza: nivel e iSIR son independientes por construccion,
asi que cruzar toda la grilla seria trabajo sin informacion.

LAS TRES VERIFICACIONES (asserts, corren siempre)
-------------------------------------------------
 1. gain_db=0 es BIT A BIT identico al procesador sin decorar (g=1.0 exacto en
    IEEE754). Es lo que garantiza que agregar este eje no invalida nada previo.
    Ademas, `LevelScaled` NO muta el buffer de entrada (el harness lo cachea y
    lo reusa para todos los procesadores de la celda).
 2. Las metricas de ENTRADA no cambian con el offset. En este harness son
    `base_*` (pre-WPE) y `wpe_*` (la entrada real del BF): se calculan UNA vez
    por celda, antes del lazo de procesadores, asi que la igualdad es
    ESTRUCTURAL -- el assert verifica que la ganancia no se filtro a la escena.
 3. Invariancia de escala del nucleo con mascara CONGELADA. `OFB_MVDR` no tiene
    gancho para congelar la mascara (la red esta adentro del lazo), asi que se
    verifica sobre `SoudenCore` directamente, alimentado con una STFT de
    magnitud REALISTA y la misma secuencia de mascaras a las dos escalas. Toda
    la diferencia del barrido tiene que venir de la mascara, no del core.

RESULTADO (4 celdas: rt60 0.360/0.610 x iSIR 0/10 dB, interf a 45 deg,
mediana sobre celdas, referencia early, smooth=0.2)
--------------------------------------------------------------------
    offset   dPESQ   dSTOI    dSDR    dSAR  | dPESQ - nominal
    -24 dB   0.800   0.051    1.76  -21.91  |     -0.310
    -18 dB   1.085   0.103    5.42  -18.18  |     -0.025
    -12 dB   1.101   0.103    5.91  -17.91  |     -0.009
     -6 dB   1.112   0.102    5.46  -17.85  |     +0.002
      0 dB   1.110   0.102    6.28  -17.84  |      0.000
     +6 dB   1.101   0.100    5.59  -17.87  |     -0.010
    +12 dB   1.082   0.095    4.98  -18.39  |     -0.028
    +18 dB   1.007   0.089    4.52  -19.42  |     -0.103
    +24 dB   0.891   0.069    3.24  -20.56  |     -0.219

La forma es una meseta PLANA de -18 a +12 (todo adentro de 0.03 del nominal)
con dos caidas en los bordes, y es ASIMETRICA: aguanta mas hacia abajo. Lo
mismo en STOI y en SDR, que se mueven en fase con el PESQ. El SIR NO se puede
leer en esta corrida (la mediana rebota entre 12.5 y 21 dB sin tendencia: es la
cola pesada conocida del SIR de BSS-Eval con solo 4 celdas).

EL MECANISMO (`return_diag=True`, masas promediadas sobre la ventana de
metricas y sobre las celdas, relativas al nominal)
--------------------------------------------------------------------------
    offset   m_scm^8   (1-m_scm)^8   sesgo iSIR   c_pf
    -24 dB     0.62x      0.79x        -0.63 dB   0.59
    -18 dB     0.95x      0.42x        +0.54 dB   0.64
    -12 dB     0.97x      0.54x        +0.93 dB   0.65
     -6 dB     0.96x      0.83x        +0.78 dB   0.65
      0 dB     1.00x      1.00x        +1.19 dB   0.67
     +6 dB     0.98x      1.09x        +1.12 dB   0.67
    +12 dB     0.82x      1.12x        -0.42 dB   0.61
    +18 dB     0.66x      1.17x        -1.27 dB   0.58
    +24 dB     0.58x      1.28x        -1.48 dB   0.58

De los dos mecanismos que se sospechaban, el que manda es el del SCM, y NO es
el que se esperaba. Hacia ARRIBA la mascara se corre hacia abajo (la rama de
senal pierde el 42 % de su masa y la de ruido gana el 28 %): la red deja de ver
voz donde la hay y el SCM de senal se queda flaco. Hacia ABAJO no hay
desbalance sino PERDIDA DE CONTRASTE: en -24 las DOS ramas pierden masa
(0.62x y 0.79x), o sea que la mascara colapsa hacia el medio, Phi_XX y Phi_NN
se parecen y el filtro degrada hacia e_ref/M -- que es exactamente el modo
suave de fallar que el nucleo sin sustraccion garantiza (ver `SoudenCore`).

El otro mecanismo NO aparece: el estimador de iSIR NO se descalibra. El sesgo
contra el iSIR verdadero se queda adentro de +-1.5 dB en TODO el rango de
+-24 dB, contra un sigmoide de 3.5 dB de ancho, y el coeficiente de la agenda
se mueve entre 0.58 y 0.67. Tiene sentido: el estimador es un COCIENTE de dos
potencias de la misma mascara, asi que lo que le pasa a la mascara se le
cancela en primer orden. La agenda del post-filtro no es el eslabon debil.

La verificacion 3 aporta un tercer mecanismo, chico pero REAL y relevante para
el port: la carga diagonal del `solve` tiene un termino ABSOLUTO (`+1e-12`
sobre Phi_NN) que al bajar el nivel equivale a subir `min_loading`. En -24 dB
equivale a min_loading=2.9e-6 (contra el 1e-9 nominal) y mueve los pesos un
0.3 %. Segun la tabla medida en `SoudenCore` eso cuesta ~0.002 dPESQ: dos
ordenes de magnitud por debajo de los 0.31 que se miden aca, asi que la curva
es 100 % de la mascara. Pero el margen se acaba: una decada mas abajo (-44 dB)
ese termino solo ya seria min_loading~3e-4, que cuesta -0.05.

CONCLUSION
----------
ZONA MUERTA (|dPESQ - nominal| <= 0.05): de -18 a +12 dB alrededor del nominal
(pico 0.99), ASIMETRICA. Los bordes reales caen entre -24 y -18 y entre +12 y
+18; la grilla es de 6 dB y no los resuelve mas fino.

Contra la regla de decision fijada ANTES de correr: adentro de +-12 dB la
caida es 0.028 en el peor punto, o sea que se cumple el segundo caso (dentro
de 0.05 en todo +-12 dB, con caida solo en los extremos). NO SE JUSTIFICA UN
CONTROL DE NIVEL AUTOMATICO. Alcanza con una constante de calibracion fija y
documentada que deje el nivel medio de operacion adentro de esa meseta.

Como esa constante se elige una sola vez y la variacion natural (distancia al
hablante, volumen del locutor) se la come entera, lo que hay que especificar es
el PRESUPUESTO: la ventana util mide 30 dB, asi que apuntando a -3 dB del
nominal quedan 15 dB de margen para abajo y 15 para arriba. Que la ventana sea
asimetrica dice ademas de que lado conviene errarle: subestimar la ganancia
cuesta menos que pasarse.

USO
---
    conda activate tesis_beam
    python tests/ofb_level_sweep.py --quick
    python tests/ofb_level_sweep.py
"""

import os
import argparse

import numpy as np
import pandas as pd
import scipy.signal as sig
import tensorflow as tf

from propagation.mird_loader import MirdDatasetProvider
from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search
from evaluation.bf_wrappers import OFB_MVDR
from beamforming.mask.ofb import SoudenCore

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT_DIR = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "ofb_level")
MODEL_1 = f"{PROJECT_ROOT}/src/dnn_denoise/models/model_quant_1.tflite"
MODEL_2 = f"{PROJECT_ROOT}/src/dnn_denoise/models/model_quant_2.tflite"

LEVELS_DB = [-24, -18, -12, -6, 0, 6, 12, 18, 24]
LEVELS_DB_QUICK = [-24, 0, 24]

# Umbrales de la regla de decision, fijados ANTES de ver los resultados.
DEADZONE_TOL = 0.05      # |dPESQ - nominal| que define la zona muerta
DECISION_TOL = 0.10      # caida dentro de +-12 dB que justificaria un AGC
DECISION_RANGE_DB = 12.0


def proc_name(db):
    return f"L{int(db):+03d}"


class LevelScaled:
    """Aplica g a la entrada del procesador y 1/g a su salida.

    Aisla el efecto del nivel SOBRE LA RED: el beamformer es invariante a escala
    (Souden normaliza por la traza) y las metricas ven exactamente la misma senal
    que sin el decorador. Lo unico que cambia es el punto de operacion del DTLN.

    Ademas junta, por llamada, el resumen del diagnostico del procesador interno
    (masa de cada rama del SCM, iSIR estimado). Es gratis: el harness no expone
    un gancho por celda, pero el decorador ya esta en el medio de cada llamada.
    """

    def __init__(self, inner, gain_db, collect_diag=True, eval_start_s=5.0):
        self.inner, self.gain_db = inner, float(gain_db)
        self.collect_diag = collect_diag
        self.eval_start_s = float(eval_start_s)
        self.diag_rows = []

    def process(self, mic_signals, scene_config):
        g = 10.0 ** (self.gain_db / 20.0)
        # VERIFICACION 1b: el harness cachea `mic_signals_ready` y lo reusa para
        # TODOS los procesadores de la celda; si el decorador lo mutara, el
        # barrido se contaminaria a si mismo en cascada.
        guard = mic_signals.copy()
        y, w = self.inner.process(mic_signals * g, scene_config)
        assert np.array_equal(mic_signals, guard), \
            "LevelScaled muto el buffer de entrada (el harness lo reusa entre procesadores)."
        if self.collect_diag:
            self._collect(scene_config)
        return y / g, w

    def _collect(self, scene_config):
        d = getattr(self.inner, "diag", None)
        if not d:
            return
        hop = (scene_config['stft_window'] - scene_config['stft_overlap'])
        t0 = min(int(self.eval_start_s * scene_config['fs'] / hop),
                 d["m_out"].shape[1] - 1)
        m_out, m_ref = d["m_out"][:, t0:], d["m_ref"][:, t0:]
        m_scm = 0.5 * (m_out + m_ref)
        p = 8.0
        self.diag_rows.append({
            "gain_db": self.gain_db,
            "isir_scene": scene_config.get("isir_db", np.nan),
            "isir_est": float(np.mean(d["isir_db"][t0:])),
            "c_pf": float(np.mean(d["c_pf"][t0:])),
            "m_out_s": float(np.mean(m_out ** p)),
            "m_out_n": float(np.mean((1.0 - m_out) ** p)),
            "m_ref_s": float(np.mean(m_ref ** p)),
            "m_ref_n": float(np.mean((1.0 - m_ref) ** p)),
            "m_scm_s": float(np.mean(m_scm ** p)),
            "m_scm_n": float(np.mean((1.0 - m_scm) ** p)),
        })


# ---------------------------------------------------------------------------
# VERIFICACIONES
# ---------------------------------------------------------------------------

def probe_signal(fs=16000, dur=1.5, M=8, snr_db=60.0, seed=0):
    """Senal de sonda de 8 canales con el nivel y el condicionamiento del harness.

    No pretende ser voz. Lo que si reproduce es lo que hace falta para que las
    dos verificaciones numericas midan algo: el NIVEL (pico 0.99, como
    `mix_and_normalize`) y el CONDICIONAMIENTO de las covarianzas (dos fuentes
    con retardos distintos + ruido termico independiente por canal al mismo
    snr_db=60 de la escena base). Con una sola fuente las SCM quedan rango 1 y
    cualquier carga absoluta se amplifica un factor 100 de mas.
    """
    rng = np.random.default_rng(seed)
    n = int(fs * dur)

    def band(lo, hi):
        b, a = sig.butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band")
        return sig.lfilter(b, a, rng.standard_normal(n))

    s1 = band(200, 3500) * (0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * np.arange(n) / fs))
    s2 = band(150, 5000)
    X = np.stack([np.roll(s1, 2 * m) + 0.5 * np.roll(s2, -3 * m) for m in range(M)])
    X = X + rng.standard_normal(X.shape) * np.sqrt(np.mean(X ** 2) * 10 ** (-snr_db / 10))
    return 0.99 * X / np.max(np.abs(X))


def check_zero_gain_identity(smooth):
    """VERIFICACION 1: gain_db=0 es BIT A BIT el procesador sin decorar."""
    x = probe_signal()
    cfg = {'fs': 16000, 'stft_window': 512, 'stft_overlap': 384,
           'ref_mic_idx': 4, 'dtln_model_path': MODEL_1}

    y_ref, w_ref = OFB_MVDR(smooth=smooth).process(x, cfg)
    dec = LevelScaled(OFB_MVDR(smooth=smooth), 0, collect_diag=False)
    y_dec, w_dec = dec.process(x, cfg)

    assert np.array_equal(y_ref, y_dec), (
        "gain_db=0 NO es bit a bit identico al procesador sin decorar "
        f"(max|dy|={np.max(np.abs(y_ref - y_dec)):.3e}): el eje de nivel "
        "invalidaria los resultados previos.")
    assert np.array_equal(w_ref, w_dec), "gain_db=0 cambia los pesos."
    print(f"[ok] VERIF 1: gain_db=0 bit a bit identico "
          f"({y_ref.size} muestras, {w_ref.shape} pesos).")


def check_core_scale_invariance(levels_db, eps_abs=1e-12, eps_tol=1e-5):
    """VERIFICACION 3: el nucleo no aporta nada a la curva de nivel.

    `OFB_MVDR` no tiene gancho para congelar la mascara (la red esta adentro del
    lazo), asi que se prueba `SoudenCore` directo -- que es donde vive la cuenta
    -- con la MISMA secuencia de mascaras a las dos escalas y con frames de STFT
    de magnitud y condicionamiento realistas.

    El nucleo es invariante a escala salvo por UNA constante: la carga diagonal
    de `solve` es `min_loading * tr/M + 1e-12`, y ese `1e-12` es ABSOLUTO. Al
    bajar el nivel la covarianza se achica y la carga pesa cada vez mas, o sea
    que equivale a correr con un `min_loading` mayor:

        eps_equivalente = 1e-12 / (tr(Phi_NN)/M)

    Asi que lo que se verifica no es una tolerancia numerica arbitraria sino que
    ese eps equivalente se quede por debajo de 1e-5 en todo el rango del
    barrido, que es el valor mas alto con costo MEDIDO en `SoudenCore`
    (-0.004 dPESQ) -- dos ordenes por debajo de lo que mide este test.
    """
    x = probe_signal(dur=1.0)
    _, _, Zxx = sig.stft(x, fs=16000, window='boxcar', nperseg=512,
                         noverlap=384, nfft=512)
    X = np.transpose(Zxx, (1, 2, 0))[:, :60, :]          # (K, T, M)
    K, T, M = X.shape
    rng = np.random.default_rng(1)
    masks = rng.uniform(0.05, 0.95, size=(K, T))         # CONGELADAS

    def run(g):
        core = SoudenCore(K, M, M // 2, alpha=0.99, min_loading=1e-9)
        for t in range(T):
            m = masks[:, t]
            core.update(g * X[:, t, :], m ** 8.0, (1.0 - m) ** 8.0)
        Phi_NN = core.Num_NN / (core.Den_NN + 1e-15)
        tr_M = float(np.mean(np.real(np.trace(Phi_NN, axis1=1, axis2=2))) / M)
        return core.solve(), tr_M

    w0, tr0 = run(1.0)
    print(f"[ok] VERIF 3: invariancia de escala del nucleo (mascara congelada, "
          f"tr(Phi_NN)/M = {tr0:.2e} en el nominal)")
    worst_eps = 0.0
    for db in levels_db:
        wg, tr = run(10.0 ** (db / 20.0))
        rel = float(np.max(np.abs(wg - w0)) / np.max(np.abs(w0)))
        eps = eps_abs / tr
        worst_eps = max(worst_eps, eps)
        print(f"        {db:+3d} dB  max|dw|/max|w| = {rel:.2e}   "
              f"carga absoluta equivalente = {eps:.1e}")
    assert worst_eps < eps_tol, (
        f"la carga absoluta del solve equivale a min_loading={worst_eps:.1e} "
        f"(> {eps_tol:.0e}) en el extremo del barrido: parte de la degradacion "
        "vendria del core y no de la mascara.")


def check_input_metrics_invariant(df):
    """VERIFICACION 2: las metricas de ENTRADA no cambian con el offset.

    En este harness `base_*` (pre-WPE) y `wpe_*` (la entrada real del BF) se
    calculan UNA sola vez por celda, antes del lazo de procesadores: la igualdad
    es ESTRUCTURAL y el assert lo que verifica es que la ganancia no se filtro a
    la escena (si se aplicara antes de `mix_and_normalize`, o si mutara el
    buffer cacheado, estas columnas se moverian).
    """
    cols = [c for c in df.columns if c.startswith("base_") or c.startswith("wpe_")]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    assert cols, "no hay columnas de metricas de entrada en el dataframe."
    key = ["rt60", "isir_db", "interf_configs", "target_angle", "target_dist"]
    key = [k for k in key if k in df.columns]
    spread = df.groupby(key)[cols].agg(lambda s: s.max() - s.min())
    worst = float(np.nanmax(np.abs(spread.values)))
    assert worst == 0.0, (
        f"las metricas de ENTRADA cambian con el offset (max spread {worst:.3e}): "
        "la ganancia se esta aplicando en el lugar equivocado.")
    print(f"[ok] VERIF 2: {len(cols)} metricas de entrada identicas entre "
          f"offsets en las {len(spread)} celdas.")


# ---------------------------------------------------------------------------
# BARRIDO
# ---------------------------------------------------------------------------

def build_processors(levels_db, smooth, eval_start_s):
    return {proc_name(db): LevelScaled(OFB_MVDR(smooth=smooth, diag=True), db,
                                       eval_start_s=eval_start_s)
            for db in levels_db}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    ap.add_argument("--quick", action="store_true", help="1 celda, 3 niveles (plomeria)")
    ap.add_argument("--smooth", type=float, default=0.2)
    ap.add_argument("--duration", type=float, default=15)
    ap.add_argument("--levels", type=float, nargs="+", default=None,
                    help="offsets en dB (default: -24..+24 de a 6)")
    ap.add_argument("--catalog", dest="save_catalog", action="store_true",
                    help="guarda el catalogo H5 (caro y no aporta a este test)")
    ap.add_argument("--only-checks", action="store_true",
                    help="corre las tres verificaciones y sale")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    levels = args.levels if args.levels is not None else (
        LEVELS_DB_QUICK if args.quick else LEVELS_DB)
    levels = [int(round(v)) for v in levels]
    if 0 not in levels:
        raise ValueError("el nivel nominal (0 dB) tiene que estar en la grilla: "
                         "es la referencia de todo el reporte.")

    # --- VERIFICACIONES 1 y 3 (antes del barrido: si fallan, no hay que correrlo)
    check_zero_gain_identity(args.smooth)
    check_core_scale_invariance(levels)
    if args.only_checks:
        return

    interpreter_1 = tf.lite.Interpreter(model_path=MODEL_1)
    interpreter_1.allocate_tensors()
    interpreter_2 = tf.lite.Interpreter(model_path=MODEL_2)
    interpreter_2.allocate_tensors()

    provider = MirdDatasetProvider(root_dir=f"{PROJECT_ROOT}/tools/data/rirs/mird")

    # La MISMA escena base que tests/ofb_auto_benchmark.py.
    base_config = {
        'fs': 16000,
        'duration': args.duration,
        't_early': 0.050,
        'array_center': [3.0, 3.0, 1.2],
        'mird_spacing': "3-3-3-8-3-3-3",
        'snr_db': 60.0,
        'source_path': f"{PROJECT_ROOT}/tools/data/signals/p002_emo_adoration_sentences.wav",
        'interf_paths': [f"{PROJECT_ROOT}/tools/data/signals/techno_gated commune.wav"],

        'wpe_taps': 7, 'wpe_delay': 2, 'wpe_alpha': 0.9999,
        'wpe_stft_size': 512, 'wpe_stft_shift': 128,
        'wpe_fixed_bits': None, 'wpe_fixed_round': 'nearest', 'wpe_backend': 'cov',
        'wpe_block_L': 512, 'wpe_block_shift': 2, 'wpe_block_iters': 2,
        'wpe_block_reg': 1e-6, 'wpe_block_solver': 'cholesky', 'wpe_block_mode': 'resolve',

        'stft_window': 512,
        'stft_overlap': 384,
        'eval_references': ['early'],
        'dtln_model_path': MODEL_1,
    }
    eval_start_s = min(5.0, args.duration * 0.3)

    param_grid = {
        'rt60': [0.610] if args.quick else [0.360, 0.610],
        'target_angle': [0],
        'target_dist': [1.0],
        'interf_configs': [[(45, 1.0)]],
        'isir_db': [0] if args.quick else [0, 10],
        'mismatch_gain': [0], 'mismatch_phase': [0],
        'use_wpe': [False], 'wpe_method': ['online'], 'wpe_taps': [7], 'wpe_delay': [2],
        'error_angle_deg': [0.0], 'error_distance_m': [0.0],
    }

    processors_dict = build_processors(levels, args.smooth, eval_start_s)
    print(f"[*] offsets de nivel: {levels} dB")

    df = run_mird_grid_search(
        grid_params=param_grid,
        dataset_provider=provider,
        processors=processors_dict,
        scene_base_config=base_config,
        output_dir=args.out_dir,
        interpreter_1=interpreter_1,
        interpreter_2=interpreter_2,
        apply_dtln_post=False,
        save_catalog=args.save_catalog,
    )

    # --- VERIFICACION 2 (necesita el dataframe) ---------------------------
    check_input_metrics_invariant(df)

    diag = pd.DataFrame([r for p in processors_dict.values() for r in p.diag_rows])
    if not diag.empty:
        diag.to_csv(os.path.join(args.out_dir, "level_sweep_diag.csv"), index=False)

    summarize(df, diag, levels, args.out_dir)


def summarize(df, diag, levels, out_dir):
    cols = [c for c in ["Delta_bf_PESQ_early", "Delta_bf_STOI_early",
                        "Delta_bf_SDR_early", "Delta_bf_SIR_early",
                        "Delta_bf_SAR_early"] if c in df.columns]
    order = [proc_name(db) for db in levels if proc_name(db) in set(df["processor"])]

    print("\n" + "=" * 78)
    print("CURVA DE NIVEL -- MEDIANA sobre las celdas (referencia early)")
    print("=" * 78)
    t = df.groupby("processor")[cols].median().reindex(order)
    pesq = t["Delta_bf_PESQ_early"]
    nominal = pesq.loc[proc_name(0)]
    t = t.copy()
    t.insert(0, "offset_db", [db for db in levels if proc_name(db) in order])
    t["dPESQ_vs_nominal"] = pesq - nominal
    print(t.round(3).to_string())

    print("\n--- Delta PESQ por iSIR (media sobre salas) ---")
    print(df.pivot_table(index="processor", columns="isir_db",
                         values="Delta_bf_PESQ_early").reindex(order).round(3).to_string())

    if not diag.empty:
        print("\n" + "=" * 78)
        print("MECANISMO -- masa de cada rama del SCM y estimador de iSIR")
        print("(promedio sobre la ventana de metricas y sobre las celdas)")
        print("=" * 78)
        d = diag.groupby("gain_db").mean(numeric_only=True)
        d["isir_bias"] = d["isir_est"] - d["isir_scene"]
        show = ["m_out_s", "m_out_n", "m_ref_s", "m_ref_n", "m_scm_s", "m_scm_n",
                "isir_est", "isir_scene", "isir_bias", "c_pf"]
        print(d[show].round(4).to_string(float_format=lambda v: f"{v:9.4f}"))
        print("\n  m_*_s = mean(m^8) (rama de SENAL)   m_*_n = mean((1-m)^8) (rama de RUIDO)")
        print("  m_scm = (m_out + m_ref)/2, que es lo que realmente entra al SCM.")
        ref = d.loc[0.0] if 0.0 in d.index else d.loc[0]
        print("\n  desbalance relativo al nominal (x):")
        for db in d.index:
            print(f"    {db:+6.1f} dB   senal {d.loc[db, 'm_scm_s'] / ref['m_scm_s']:6.2f}x"
                  f"   ruido {d.loc[db, 'm_scm_n'] / ref['m_scm_n']:6.2f}x"
                  f"   sesgo iSIR {d.loc[db, 'isir_bias']:+6.2f} dB")

    # --- CONCLUSION EN dB: la zona muerta ---------------------------------
    print("\n" + "=" * 78)
    print("CONCLUSION")
    print("=" * 78)
    dev = (pesq - nominal).abs()
    lo, hi = 0, 0
    for db in [l for l in levels if l < 0][::-1]:
        if dev.loc[proc_name(db)] <= DEADZONE_TOL:
            lo = db
        else:
            break
    for db in [l for l in levels if l > 0]:
        if dev.loc[proc_name(db)] <= DEADZONE_TOL:
            hi = db
        else:
            break
    print(f"ZONA MUERTA (|dPESQ - nominal| <= {DEADZONE_TOL}): "
          f"[{lo:+d}, {hi:+d}] dB alrededor del nominal (pico 0.99).")
    if abs(lo) != abs(hi):
        print("  -> ASIMETRICA: la zona muerta del control tambien tiene que serlo.")
    if lo == min(levels) or hi == max(levels):
        print("  -> OJO: la zona muerta toca el BORDE de la grilla; el rango real es mas ancho.")

    inside = [l for l in levels if abs(l) <= DECISION_RANGE_DB]
    if len(inside) < 2:
        print(f"\nREGLA DE DECISION: la grilla no tiene niveles distintos de 0 dentro "
              f"de +-{DECISION_RANGE_DB:.0f} dB ({levels}); corre el barrido completo "
              f"(sin --quick) para evaluarla.")
        df.to_csv(os.path.join(out_dir, "level_sweep_metrics.csv"), index=False)
        print(f"\n[ok] {out_dir}")
        return
    worst = float(dev.loc[[proc_name(l) for l in inside]].max())
    worst_db = inside[int(np.argmax(dev.loc[[proc_name(l) for l in inside]].values))]
    print(f"\nREGLA DE DECISION (fijada antes de ver los resultados):")
    print(f"  peor desviacion dentro de +-{DECISION_RANGE_DB:.0f} dB: "
          f"{worst:.3f} en {worst_db:+d} dB")
    if worst > DECISION_TOL:
        print(f"  -> dPESQ cae mas de {DECISION_TOL} dentro de +-{DECISION_RANGE_DB:.0f} dB: "
              f"EL CONTROL DE NIVEL SE JUSTIFICA, con zona muerta [{lo:+d}, {hi:+d}] dB.")
    elif worst <= DEADZONE_TOL:
        print(f"  -> dPESQ se mantiene dentro de {DEADZONE_TOL} en todo "
              f"+-{DECISION_RANGE_DB:.0f} dB: NO hace falta control automatico, "
              f"alcanza una constante de calibracion fija y documentada.")
    else:
        print(f"  -> zona gris ({DEADZONE_TOL} < {worst:.3f} <= {DECISION_TOL}): "
              f"no justifica un lazo automatico, pero la constante de calibracion "
              f"tiene que quedar adentro de [{lo:+d}, {hi:+d}] dB.")

    df.to_csv(os.path.join(out_dir, "level_sweep_metrics.csv"), index=False)
    print(f"\n[ok] {out_dir}")


if __name__ == "__main__":
    main()
