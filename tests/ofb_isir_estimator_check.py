"""
CALIBRACION Y VALIDACION DEL ESTIMADOR CIEGO DE iSIR (agenda de `pf_mask='isir'`)
================================================================================
`NM_MVDR_OFB(pf_mask='isir')` interpola la mascara del post-filtro entre la de
la SALIDA del beamformer (buena con iSIR bajo) y la de ATRAS (el canal de
referencia, buena con iSIR alto) con un sigmoide agendado por el iSIR. Para que
el sistema sea autonomo ese iSIR hay que ESTIMARLO, y el estimador solo puede
mirar lo que ya esta calculado: la mascara del DTLN sobre el canal de
referencia (`a`) y el espectro de ese canal.

Este test NO corre el beamformer: el estimador depende UNICAMENTE de (a, X_ref)
-- no del lazo, no de los pesos, no del post-filtro -- asi que se puede validar
con un DTLN por escena y sin ninguna metrica perceptual de por medio. Eso es lo
que lo hace barato y lo que permite barrer muchas mas escenas que el benchmark.
Los dos modos que el sistema realmente usa se corren con el `ISIRTracker` de
produccion (sin calibrar, que es lo que este test justamente mide); el resto son
ablaciones que viven solo aca.

RESULTADO (la corrida que fijo los defaults de NM_MVDR_OFB_AUTO)
----------------------------------------------------------------
Con interferente NO hablado, los dos modos son monotonos por escena (Spearman
1.00) y 'band' es el que menos se mueve al cambiar de escena: 1.9 dB reales de
dispersion contra 2.6 del historico 'sum', con el sigmoide de 3.5 dB de ancho.
Lo que arregla es concreto: 'sum' depende del CICLO DE TRABAJO del locutor -- a
+20 dB de iSIR da 5.9 dB con un locutor y 14.6 con otro -- y la normalizacion
por masa de 'band' cancela eso (el residuo que queda es la sala, no el
locutor). Con alpha=0.998 la calibracion es hat = 0.496 * iSIR + 3.77 y la
convergencia es < 1 s.

Con interferente HABLADO no funciona ninguno (Spearman ~0, negativo por
escena). No es calibracion: la mascara del DTLN detecta VOZ, no el TARGET.

QUE SE MIDE
-----------
Para cada escena (rt60 x angulo de interferente x locutor x interferente) y cada
iSIR verdadero de la grilla se corre el DTLN sobre el canal de referencia y se
calculan varios estimadores. De cada uno interesa:

  1. MONOTONIA en el iSIR verdadero (Spearman pooled y por escena). El sigmoide
     no necesita un estimador insesgado: necesita uno MONOTONO, porque el sesgo
     se absorbe calibrando el centro.
  2. DISPERSION entre escenas a iSIR fijo (desvio de isir_hat sobre las escenas).
     Esto es lo que decide si un centro FIJO sirve: si a iSIR verdadero fijo el
     estimador se mueve mas que el ancho del sigmoide, la agenda se rompe al
     cambiar de locutor o de sala.
  3. PENDIENTE dB/dB del ajuste lineal isir_hat = g * isir + b, que fija en que
     unidades hay que expresar el ancho del sigmoide.
  4. ERROR DE DECISION: |c_hat - c_oraculo| con c = sigmoide((isir - c0)/w),
     que es la unica magnitud que el sistema realmente consume.
  5. CONVERGENCIA: segundos hasta que |isir_hat(t) - isir_hat(final)| < 1 dB.

LOS ESTIMADORES
---------------
Todos son causales, O(K) por frame, y comparten la misma recursion de un polo
(`alpha` largo a proposito: el iSIR es una propiedad de la ESCENA, no del
frame). Px = |X_ref|^2.

  sum   s = <sum_k a Px>,  n = <sum_k (1-a) Px>,  hat = 10log10(s/n).
        EL ACTUAL (el que esta hoy en output_feedback.py). Su problema conocido
        es el CICLO DE TRABAJO del locutor: en los silencios a->0, asi que `s`
        no acumula y `n` si. El valor converge a ~ (duty * P_target)/P_ruido, o
        sea que se mueve con cuanto habla el locutor y no solo con el iSIR.
  norm  el mismo cociente pero POR BIN OCUPADO: se dividen las potencias por la
        masa de mascara acumulada (s/ma) / (n/mn). El duty cycle se cancela a
        primer orden -- es la correccion barata (dos acumuladores escalares mas).
  band  `norm` restringido a 300-3400 Hz. Afuera de esa banda la mascara del
        DTLN es poco informativa (abajo domina el rumble, arriba casi no hay
        energia) y solo agrega varianza.
  bandsum  ABLACION: la banda SIN la normalizacion por masa (o sea 'sum' en la
        banda). Separa cual de los dos ingredientes de 'band' aporta.
  wien  `norm` con el reparto de potencia de Wiener (a^2 Px y (1-a)^2 Px) en vez
        del lineal. Es el reparto coherente con interpretar `a` como ganancia.
  sat   fraccion de bins con a > 0.9, suavizada. NO esta en dB: es el detector
        de SATURACION que el docstring de output_feedback.py dejo pendiente.
        Se reporta aparte (su correlacion con el iSIR es lo unico comparable).

Uso
---
    conda activate tesis_beam
    python tests/ofb_isir_estimator_check.py
    python tests/ofb_isir_estimator_check.py --isir -10 -5 0 5 10 15 --quick

Salida: tests/dataset_out/ofb_isir_est/
    isir_estimator_cells.csv    una fila por (escena, iSIR, estimador)
    isir_estimator_report.txt   el resumen que se lee
    isir_estimator.png          panel de 4 graficos
"""

import os
import argparse
import itertools
import time

import numpy as np
import pandas as pd
import scipy.signal as sig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from propagation.simulate_acoustics_v1 import SimAcoustic
from propagation.mird_loader import (
    MirdDatasetProvider, generate_mird_linear_array_from_spacing,
)
from beamforming.array.microphone import Microphone
from beamforming.mask.blind_feedback import DTLNStream
from beamforming.mask.output_feedback import ISIRTracker, ISIR_BAND_HZ

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT_DIR = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "ofb_isir_est")
SIGNALS = os.path.join(PROJECT_ROOT, "tools", "data", "signals")
MODEL1 = os.path.join(PROJECT_ROOT, "src", "dnn_denoise", "models", "model_quant_1.tflite")

EST_DB = ("sum", "norm", "band", "bandsum", "wien")   # los que estan en dB
EST_ALL = EST_DB + ("sat",)


# ---------------------------------------------------------------------------
# escena (nodos 1-3 del benchmark MIRD, sin WPE)
# ---------------------------------------------------------------------------
def build_acoustic_scene(cfg, provider, rt60, target_angle, target_dist,
                         interf_configs, source_path, interf_path):
    """
    Convoluciona UNA vez la escena acustica. El iSIR NO entra aca: se aplica
    despues en `mix_and_normalize`, que es barato -- por eso el barrido de iSIR
    reusa la misma convolucion (igual que la cascada del benchmark).
    """
    array_center = np.array(cfg['array_center'])
    mic_coords = generate_mird_linear_array_from_spacing(cfg['mird_spacing']) + array_center

    scene = SimAcoustic(array_geometry=mic_coords, array_mismatch=0.0,
                        duration=cfg['duration'], fs=cfg['fs'])

    _ = provider.load_rir(rt60, cfg['mird_spacing'], target_dist, target_angle)
    abs_pos_target = array_center + provider.export_position('cartesian').squeeze()
    scene.set_source(source_path, gain=1.0, position=abs_pos_target.reshape(1, 3))

    for (i_ang, i_dist) in interf_configs:
        _ = provider.load_rir(rt60, cfg['mird_spacing'], i_dist, i_ang)
        abs_pos_interf = array_center + provider.export_position('cartesian').squeeze()
        scene.set_interference(audio_path=interf_path, gain=1.0,
                               position=abs_pos_interf.reshape(1, 3))

    scene.import_rirs(dataset_provider=provider, target_t60=rt60,
                      array_center=array_center, spacing_cfg=cfg['mird_spacing'])
    scene.convolve_signals(t_early=cfg['t_early'])
    return scene, mic_coords


def mix_cell(scene, cfg, isir_db, ref):
    """
    Mezcla a un iSIR y le pasa la emulacion de hardware (mismatch 0, solo ruido
    termico), igual que el Nodo 3 del benchmark. Devuelve el canal de referencia
    de la mezcla y el iSIR VERDADERO medido en ese canal (que no es exactamente
    el nominal: el nominal se define sobre el mic 0).
    """
    scene_data = scene.mix_and_normalize(iSIR_dB=isir_db)
    mic_sim = Microphone(fs=cfg['fs'])
    mic_sim.set_seed(1234)
    mic_sim.set_custom_errors(std_gain_dB=0.0, std_phase_deg=0.0, snr_dB=cfg['snr_db'])
    degraded = mic_sim.emulate(scene_data["mic_signals"])

    tgt = scene_data["target_early"][ref] + scene_data["target_late"][ref]
    itf = scene_data["interference_early"][ref] + scene_data["interference_late"][ref]
    isir_true_ref = 10.0 * np.log10((np.mean(tgt ** 2) + 1e-20) /
                                    (np.mean(itf ** 2) + 1e-20))
    return degraded[ref], float(isir_true_ref)


# ---------------------------------------------------------------------------
# los estimadores
# ---------------------------------------------------------------------------
def run_estimators(x_ref, cfg, dtln, alpha=0.995):
    """
    Corre el DTLN sobre el canal de referencia (EXACTAMENTE como lo hace el lazo:
    ventana rectangular, magnitud del bloque escalada por nperseg) y devuelve la
    TRAYECTORIA de cada estimador, un valor por frame.

    Devuelve (dict nombre -> (T,) array, mask_mean) con isir_hat en dB salvo
    'sat', que es una fraccion en [0,1].
    """
    nperseg, noverlap = cfg['stft_window'], cfg['stft_overlap']
    freqs, _, Z = sig.stft(x_ref, fs=cfg['fs'], window='boxcar', nperseg=nperseg,
                           noverlap=noverlap, nfft=nperseg)
    T = Z.shape[1]
    band = (freqs >= ISIR_BAND_HZ[0]) & (freqs <= ISIR_BAND_HZ[1])

    traj = {k: np.zeros(T) for k in EST_ALL}
    # Los dos modos que EXISTEN en el sistema salen del `ISIRTracker` de
    # produccion, sin `calib` (se quiere el estimador CRUDO: la calibracion es
    # justamente lo que este test mide). Los otros son ablaciones que viven solo
    # aca, y se calculan con la misma recursion de un polo para que la
    # comparacion sea limpia.
    trk = {"sum": ISIRTracker(mode="sum", alpha=alpha),
           "band": ISIRTracker(mode="band", alpha=alpha, bins=band)}
    st = {k: None for k in EST_DB if k not in trk}
    sat = None
    mass_acc = 0.0

    for t in range(T):
        Xf = Z[:, t]
        a = np.clip(np.asarray(dtln.step(np.abs(nperseg * Xf)), dtype=np.float64), 0.0, 1.0)
        Px = np.abs(Xf) ** 2
        mass_acc += float(np.mean(a))

        for k, tr in trk.items():
            traj[k][t] = tr.update(a, Px)

        upd = {
            # (potencia_senal, masa_senal, potencia_ruido, masa_ruido)
            "norm": (float(np.sum(a * Px)), float(np.sum(a)),
                     float(np.sum((1.0 - a) * Px)), float(np.sum(1.0 - a))),
            # ABLACION: la banda SIN normalizar por masa, para separar cual de
            # los dos ingredientes de 'band' es el que hace el trabajo.
            "bandsum": (float(np.sum(a[band] * Px[band])), 1.0,
                        float(np.sum((1.0 - a[band]) * Px[band])), 1.0),
            "wien": (float(np.sum(a ** 2 * Px)), float(np.sum(a ** 2)),
                     float(np.sum((1.0 - a) ** 2 * Px)), float(np.sum((1.0 - a) ** 2))),
        }
        for k, v in upd.items():
            v = np.asarray(v, dtype=np.float64)
            st[k] = v if st[k] is None else alpha * st[k] + (1.0 - alpha) * v
            s, ms, n, mn = st[k]
            traj[k][t] = 10.0 * np.log10(((s / max(ms, 1e-20)) + 1e-20) /
                                         ((n / max(mn, 1e-20)) + 1e-20))

        f_sat = float(np.mean(a[band] > 0.9))
        sat = f_sat if sat is None else alpha * sat + (1.0 - alpha) * f_sat
        traj["sat"][t] = sat

    return traj, mass_acc / max(T, 1)


def convergence_s(y, hop, fs, tol=1.0):
    """
    Segundos hasta que la trayectoria ENTRA por primera vez a `tol` del valor
    final. Se usa la primera entrada y no la ultima salida a proposito: con una
    fuente no estacionaria el estimador FLUCTUA para siempre alrededor de su
    valor de regimen, asi que "la ultima vez que salio" mide la fluctuacion, no
    la convergencia. La fluctuacion se reporta aparte (hat_std).
    """
    ok = np.where(np.abs(y - y[-1]) <= tol)[0]
    return (ok[0] if len(ok) else len(y) - 1) * hop / fs


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--isir", type=float, nargs="+",
                    default=[-10, -5, 0, 5, 10, 15, 20])
    ap.add_argument("--rt60", type=float, nargs="+", default=[0.360, 0.610])
    ap.add_argument("--interf-angle", type=float, nargs="+", default=[45, 90])
    ap.add_argument("--alpha", type=float, default=0.998,
                    help="suavizado del estimador (pf_isir_alpha). El default es "
                         "el de NM_MVDR_OFB_AUTO: la calibracion (g,b) que sale "
                         "de este test SOLO vale para el alpha con el que se corrio "
                         "(mas suavizado comprime menos y sube la pendiente).")
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--eval-start", type=float, default=5.0,
                    help="segundos de warm-up excluidos del valor de regimen")
    ap.add_argument("--interfs", type=str, nargs="+",
                    default=["techno_gated commune.wav", "p011_emo_anger_sentences.wav"],
                    help="interferentes (nombres dentro de tools/data/signals). El "
                         "que empieza con 'p0' se trata como interferente de VOZ y "
                         "se reporta aparte.")
    ap.add_argument("--center", type=float, default=2.2,
                    help="centro de la agenda EN dB REALES de iSIR (el cruce "
                         "medido entre pf_mask='out' y pf_mask='back')")
    ap.add_argument("--width", type=float, default=3.5,
                    help="ancho del sigmoide en dB REALES")
    ap.add_argument("--from-csv", action="store_true",
                    help="no recalcula nada: rehace reporte y figura desde el CSV")
    ap.add_argument("--quick", action="store_true",
                    help="una sola escena acustica (para probar el script)")
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    csv = os.path.join(args.out_dir, "isir_estimator_cells.csv")
    if args.from_csv:
        df = pd.read_csv(csv)
        report(df, args, os.path.join(args.out_dir, "isir_estimator_report.txt"))
        plot(df, os.path.join(args.out_dir, "isir_estimator.png"),
             center=args.center, width=args.width)
        return

    cfg = {
        'fs': 16000, 'duration': args.duration, 't_early': 0.050,
        'array_center': [3.0, 3.0, 1.2], 'mird_spacing': "3-3-3-8-3-3-3",
        'snr_db': 60.0, 'stft_window': 512, 'stft_overlap': 384,
    }
    hop = cfg['stft_window'] - cfg['stft_overlap']

    # Locutores e interferentes distintos: el sesgo del estimador depende del
    # CICLO DE TRABAJO del locutor y del espectro del interferente, que es
    # justamente lo que hay que ver si mueve el centro del sigmoide.
    sources = [os.path.join(SIGNALS, "p002_emo_adoration_sentences.wav"),
               os.path.join(SIGNALS, "p014_rainbow_04_loud.wav")]
    interfs = [os.path.join(SIGNALS, n) for n in args.interfs]
    rt60s, angles = list(args.rt60), list(args.interf_angle)
    if args.quick:
        sources, interfs, rt60s, angles = sources[:1], interfs[:1], rt60s[-1:], angles[:1]

    provider = MirdDatasetProvider(root_dir=os.path.join(PROJECT_ROOT, "tools", "data", "rirs", "mird"))
    dtln_path = MODEL1

    rows = []
    scenes = list(itertools.product(rt60s, angles, sources, interfs))
    t0 = time.time()
    for si, (rt60, ang, src, itf) in enumerate(scenes):
        print(f"\n[*] escena {si+1}/{len(scenes)}: rt60={rt60} interf={ang}deg "
              f"src={os.path.basename(src)} itf={os.path.basename(itf)}")
        scene, mic_coords = build_acoustic_scene(
            cfg, provider, rt60, target_angle=0, target_dist=1.0,
            interf_configs=[(ang, 1.0)], source_path=src, interf_path=itf)
        M = mic_coords.shape[0]
        ref = M // 2

        for isir in args.isir:
            x_ref, isir_true = mix_cell(scene, cfg, isir, ref)
            # DTLN NUEVO por celda: el estado LSTM no puede cruzar escenas.
            dtln = DTLNStream(dtln_path)
            traj, mask_mean = run_estimators(x_ref, cfg, dtln, alpha=args.alpha)
            f0 = int(args.eval_start * cfg['fs'] / hop)
            for name in EST_ALL:
                y = traj[name]
                rows.append({
                    "rt60": rt60, "interf_angle": ang,
                    "source": os.path.basename(src), "interf": os.path.basename(itf),
                    "isir_db": isir, "isir_true_ref_db": isir_true,
                    "estimator": name,
                    "hat": float(np.mean(y[f0:])),
                    "hat_std": float(np.std(y[f0:])),
                    "t_conv_s": convergence_s(y, hop, cfg['fs'],
                                              tol=(1.0 if name in EST_DB else 0.02)),
                    "mask_mean": mask_mean,
                })
            print(f"    iSIR {isir:+6.1f} (ref {isir_true:+6.2f})  " +
                  "  ".join(f"{n}={np.mean(traj[n][f0:]):+6.2f}" for n in EST_ALL))

    df = pd.DataFrame(rows)
    df.to_csv(csv, index=False)
    print(f"\n[*] {len(df)} filas -> {csv}   ({time.time()-t0:.0f}s)")

    report(df, args, os.path.join(args.out_dir, "isir_estimator_report.txt"))
    plot(df, os.path.join(args.out_dir, "isir_estimator.png"),
         center=args.center, width=args.width)
    print(f"[ok] {args.out_dir}")


# ---------------------------------------------------------------------------
def spearman(x, y):
    xr = pd.Series(x).rank().values
    yr = pd.Series(y).rank().values
    if np.std(xr) == 0 or np.std(yr) == 0:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])


def report(df, args, path):
    lines = []
    def P(s=""):
        lines.append(s)
        print(s)

    # El interferente de VOZ es un caso APARTE, no una escena mas: la mascara
    # del DTLN detecta VOZ, no el TARGET, asi que con un interferente hablado
    # ninguno de estos estimadores puede funcionar (ver la seccion del final).
    # La calibracion se hace sobre el subconjunto NO-voz y el otro se reporta
    # como el limite conocido.
    is_speech = df["interf"].str.startswith(("p0", "MC", "MF", "FA"))
    dnv, dv = df[~is_speech], df[is_speech]

    P("=" * 78)
    P("ESTIMADOR CIEGO DE iSIR -- calibracion contra el iSIR verdadero")
    P("=" * 78)
    n_scn = df.groupby(["rt60", "interf_angle", "source", "interf"]).ngroups
    P(f"escenas: {n_scn}   iSIR: {sorted(df['isir_db'].unique())}   alpha={args.alpha}")
    P(f"interferente NO-voz: {sorted(dnv['interf'].unique())}")
    P(f"interferente VOZ   : {sorted(dv['interf'].unique())}  (se reporta aparte)")
    P()

    P("--- CALIBRACION (solo interferente NO-voz) ---------------------------")
    P(f"{'est':>8} {'rho':>7} {'rho_min':>8} {'g[dB/dB]':>9} {'b[dB]':>7} "
      f"{'sigma':>7} {'sigma/g':>8} {'fluct/g':>8} {'t_conv':>7}")
    P(f"{'':>8} {'':>7} {'':>8} {'':>9} {'':>7} {'(dB)':>7} {'(dB real)':>8} "
      f"{'(dB real)':>8} {'(s,p90)':>7}")
    summary = {}
    for name in EST_ALL:
        d = dnv[dnv["estimator"] == name]
        rho = spearman(d["isir_db"].values, d["hat"].values)
        rho_min = min(spearman(g["isir_db"].values, g["hat"].values)
                      for _, g in d.groupby(["rt60", "interf_angle", "source", "interf"]))
        g_fit, b_fit = np.polyfit(d["isir_db"].values, d["hat"].values, 1)
        sig_scn = float(d.groupby("isir_db")["hat"].std().mean())
        fluct = float(d["hat_std"].mean())
        tconv = float(np.percentile(d["t_conv_s"].values, 90))
        summary[name] = dict(rho=rho, rho_min=rho_min, g=g_fit, b=b_fit,
                             sigma=sig_scn, fluct=fluct, tconv=tconv)
        P(f"{name:>8} {rho:>7.3f} {rho_min:>8.3f} {g_fit:>9.3f} {b_fit:>7.2f} "
          f"{sig_scn:>7.2f} {sig_scn/g_fit:>8.2f} {fluct/g_fit:>8.2f} {tconv:>7.1f}")
    P()
    P("  rho       Spearman(isir_hat, iSIR) sobre todas las celdas NO-voz.")
    P("  rho_min   la peor escena. Si cae, el estimador se rompe en algun caso.")
    P("  g,b       ajuste isir_hat = g*iSIR + b. Es la CALIBRACION: el sistema")
    P("            invierte esta recta para hablar en dB reales.")
    P("  sigma/g   dispersion entre escenas a iSIR fijo, EN dB REALES. Es la")
    P("            cifra de merito: cuanto se corre el cruce de la agenda al")
    P("            cambiar de sala, de locutor o de angulo. Comparar contra el")
    P("            ancho del sigmoide (unos 3.5 dB reales).")
    P("  fluct/g   fluctuacion temporal dentro de una celda, en dB reales (la")
    P("            fuente no es estacionaria: el estimador nunca se queda quieto).")
    P("  t_conv    p90 de la PRIMERA entrada a 1 dB del valor final.")
    P()
    P("  La dispersion crece arriba de +15 dB en todos los estimadores, pero")
    P("  ahi la agenda ya esta saturada (c ~ 1): el error no cambia ninguna")
    P("  decision. Lo que importa es la dispersion CERCA DEL CRUCE.")
    P()

    P("--- ERROR DE DECISION (lo unico que el sistema consume) --------------")
    P("c = sigmoide((isir_est - centro)/ancho) con centro/ancho en dB REALES y")
    P(f"isir_est = (hat - b)/g. c_oraculo usa el iSIR verdadero. centro="
      f"{args.center:g} dB, ancho={args.width:g} dB.")
    P(f"{'est':>8} {'|dc|med':>9} {'|dc|p90':>9} {'lado erroneo':>13}")
    for name in EST_DB:
        d = dnv[dnv["estimator"] == name]
        s = summary[name]
        est = (d["hat"].values - s["b"]) / s["g"]
        c_hat = 1.0 / (1.0 + np.exp(-(est - args.center) / args.width))
        c_or = 1.0 / (1.0 + np.exp(-(d["isir_db"].values - args.center) / args.width))
        dc = np.abs(c_hat - c_or)
        nbad = int(np.sum((c_hat > 0.5) != (c_or > 0.5)))
        P(f"{name:>8} {np.median(dc):>9.3f} {np.percentile(dc,90):>9.3f} "
          f"{nbad:>6}/{len(dc):<6}")
    P()
    P("  'lado erroneo' cuenta las celdas donde el estimador cruza el centro")
    P("  para el lado que no era. Cerca del cruce eso no cuesta casi nada (las")
    P("  dos mascaras empatan ahi); lo que importa es que no pase LEJOS.")
    P()

    if dv.empty:
        P("--- (sin interferente de VOZ en esta corrida) ------------------------")
        P()
        best = min(EST_DB, key=lambda n: summary[n]["sigma"] / summary[n]["g"])
        s = summary[best]
        P(f"--- RECOMENDACION: '{best}'   isir_est = (hat - {s['b']:.2f}) / {s['g']:.3f}")
        P(f"    dispersion {s['sigma']/s['g']:.2f} dB reales entre escenas, "
          f"rho={s['rho']:.3f} (peor escena {s['rho_min']:.3f})")
        P("=" * 78)
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[*] reporte -> {path}")
        return

    P("--- LIMITE CONOCIDO: interferente de VOZ -----------------------------")
    P("La mascara del DTLN detecta VOZ, no el TARGET. Con un interferente")
    P("hablado, la 'senal' del estimador incluye al interferente y el iSIR deja")
    P("de ser observable por esta via. No es un problema de calibracion:")
    P(f"{'est':>8} {'rho NO-voz':>11} {'rho VOZ':>9} {'rho_min VOZ':>12}")
    for name in EST_ALL:
        dd = dv[dv["estimator"] == name]
        rho_v = spearman(dd["isir_db"].values, dd["hat"].values)
        rmin_v = min(spearman(g["isir_db"].values, g["hat"].values)
                     for _, g in dd.groupby(["rt60", "interf_angle", "source", "interf"]))
        P(f"{name:>8} {summary[name]['rho']:>11.3f} {rho_v:>9.3f} {rmin_v:>12.3f}")
    P()
    P("  Consecuencia practica: en ese caso la agenda elige un punto arbitrario")
    P("  de la interpolacion. El dano esta ACOTADO -- los dos extremos son")
    P("  mascaras razonables (0.46 vs 0.51 de Delta PESQ en la peor celda del")
    P("  barrido de pf_mask) -- pero la ganancia de la agenda se pierde.")
    P()

    best = min(EST_DB, key=lambda n: summary[n]["sigma"] / summary[n]["g"])
    s = summary[best]
    P(f"--- RECOMENDACION: '{best}'   isir_est = (hat - {s['b']:.2f}) / {s['g']:.3f}")
    P(f"    dispersion {s['sigma']/s['g']:.2f} dB reales entre escenas, "
      f"rho={s['rho']:.3f} (peor escena {s['rho_min']:.3f})")
    P("=" * 78)

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[*] reporte -> {path}")


def plot(df, path, center=2.2, width=3.5):
    """
    Cuatro paneles. Los tres primeros son SOLO con interferente no-voz (el
    regimen donde el estimador esta definido); el cuarto es el contraejemplo.
    """
    is_speech = df["interf"].str.startswith(("p0", "MC", "MF", "FA"))
    dnv, dv = df[~is_speech], df[is_speech]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = dict(zip(EST_ALL, plt.cm.tab10(np.linspace(0, 1, len(EST_ALL)))))
    fit = {}
    for name in EST_DB:
        d = dnv[dnv["estimator"] == name]
        fit[name] = np.polyfit(d["isir_db"].values, d["hat"].values, 1)

    ax = axes[0, 0]
    for name in EST_DB:
        d = dnv[dnv["estimator"] == name].groupby("isir_db")["hat"]
        ax.errorbar(d.mean().index, d.mean().values, yerr=d.std().values,
                    marker="o", capsize=3, color=colors[name],
                    label=f"{name} (g={fit[name][0]:.2f})")
    lim = np.array([dnv["isir_db"].min(), dnv["isir_db"].max()])
    ax.plot(lim, lim, "k--", lw=1, label="identidad")
    ax.set_xlabel("iSIR verdadero [dB]"); ax.set_ylabel("isir_hat [dB]")
    ax.set_title("calibracion, interferente NO-voz\n(barra = desvio entre escenas)")
    ax.grid(alpha=.3); ax.legend(fontsize=8)

    ax = axes[0, 1]
    for name in EST_DB:
        d = dnv[dnv["estimator"] == name].groupby("isir_db")["hat"].std()
        ax.plot(d.index, d.values / fit[name][0], marker="o",
                color=colors[name], label=name)
    ax.axhline(width, color="k", ls="--", lw=1, label=f"ancho del sigmoide ({width:g} dB)")
    ax.set_xlabel("iSIR verdadero [dB]")
    ax.set_ylabel("dispersion entre escenas [dB REALES]")
    ax.set_title("cuanto se corre el cruce al cambiar de escena\n"
                 "(debajo de la linea = un centro fijo sirve)")
    ax.grid(alpha=.3); ax.legend(fontsize=8)

    # El estimador elegido, escena por escena, con los dos tipos de interferente
    # superpuestos: es el panel que muestra el limite.
    ax = axes[1, 0]
    for d, col, lab in ((dnv, "tab:blue", "interferente NO-voz"),
                        (dv, "tab:red", "interferente VOZ")):
        d = d[d["estimator"] == "band"]
        for _, g in d.groupby(["source", "interf", "rt60", "interf_angle"]):
            g = g.sort_values("isir_db")
            ax.plot(g["isir_db"], g["hat"], color=col, alpha=.5, lw=1.2)
        if len(d):
            ax.plot([], [], color=col, label=lab)
    if len(dnv):
        g_, b_ = fit["band"]
        ax.plot(lim, g_ * lim + b_, "k--", lw=1.5,
                label=f"calibracion: hat = {g_:.3f} iSIR + {b_:.2f}")
    ax.set_xlabel("iSIR verdadero [dB]"); ax.set_ylabel("isir_hat [dB]")
    ax.set_title("'band' escena por escena")
    ax.grid(alpha=.3); ax.legend(fontsize=8)

    # Lo que el sistema consume de verdad: el peso de la mascara de ATRAS.
    ax = axes[1, 1]
    d = dnv[dnv["estimator"] == "band"]
    if len(d):
        g_, b_ = fit["band"]
        est = (d["hat"].values - b_) / g_
        c_hat = 1.0 / (1.0 + np.exp(-(est - center) / width))
        x = np.linspace(lim[0], lim[1], 200)
        ax.plot(x, 1.0 / (1.0 + np.exp(-(x - center) / width)), "k-", lw=2,
                label="agenda con el iSIR verdadero")
        ax.scatter(d["isir_db"].values, c_hat, s=18, color="tab:purple",
                   alpha=.7, label="con el estimador")
    ax.set_xlabel("iSIR verdadero [dB]")
    ax.set_ylabel("c = peso de la mascara de ATRAS")
    ax.set_title(f"la decision (centro {center:g} dB, ancho {width:g} dB)")
    ax.grid(alpha=.3); ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"[*] figura -> {path}")


if __name__ == "__main__":
    main()
