"""
kria_ofb_benchmark.py -- EL OFB SOBRE RIRs MEDIDAS POR EL ARO DE LA KRIA.

Primer benchmark INTRUSIVO sobre la acustica real del aro de 12 capsulas: voz
convolucionada con RIRs MEDIDAS (mic-array-platform/tools/rir_lab, sweeps por
cable) en lugar de MIRD o ISM, con las mismas filas que
tests/ofb_auto_benchmark.py para poner una tabla al lado de la otra.

ES UN WRAPPER DELGADO de src/evaluation/full_benchmark_test_dtln_kria.py (el
banco sobre RIRs medidas): la escena sale de SU plan_array / build_config /
build_grid (fuentes por nombre, ch6 de referencia, descriptores medidos pegados
a cada fila) y corre por el mismo run_mird_grid_search. Lo que agrega es lo que
ese driver no tiene: (1) la preparacion de las RIRs (HPF + Lundeby), (2) el
chequeo de canales y el ch10 fuera, (3) las filas OFB en vez de su par
OFB_AUTO/OFB_AUTO_BAN, (4) TODOS los targets (no uno fijo) con un filtro de
separacion, y (5) un proceso por par, en paralelo.

QUE SE MIDE
-----------
Escena = target en una posicion medida + 1 interferente en OTRA posicion medida
del MISMO arreglo (mismo hw_key). Target: la voz de siempre
(p002_emo_adoration_sentences.wav); interferente: la de siempre
('techno_gated commune.wav', NO voz: con un interferente hablado el estimador
de iSIR de OFB_AUTO no funciona, ver el docstring de NM_MVDR_OFB_AUTO).
iSIR en {-5, 0, 5, 10} dB (medido en el mic 0 de la seleccion, como en todo el
banco), ruido de sensor a 60 dB (snr_db del banco base). Metricas contra la
imagen EARLY del target (directo + 50 ms) en ch6, el mic de referencia del
proyecto (provider.ref_mic_index(), NO M // 2).

Se usan como pares todas las combinaciones ORDENADAS de posiciones separadas
por >= --min-sep grados (default 45): un par con 3 grados de separacion no le
deja nada a un beamformer de 12 cm y solo mide la mascara.

FILAS
-----
    REF          el mic de referencia sin procesar (las metricas ABSOLUTAS de
                 base_*; los Delta son contra esto)
    MVDR         NM_MVDR_OFB sin post-filtro (smooth=None): el lazo cerrado
                 sobre la salida, solo el beamformer
    PF_OUT       pf_mask='out'   (el extremo bueno con iSIR bajo)
    PF_BACK      pf_mask='back'  (el extremo bueno con iSIR alto)
    OFB_AUTO     NM_MVDR_OFB_AUTO: agenda del PF por iSIR estimado a ciegas
    OFB_AUTO_ORC lo mismo con el iSIR VERDADERO (oraculo)

Todas con smooth=0.5 y el mismo core (rect/hann, sharpen 8, alpha 0.99,
block_update 1, leak 0, fuse='mean'), igual que ofb_auto_benchmark.py.

PREPARACION DE LAS RIRs (en este script, el loader NO se toca)
-------------------------------------------------------------
  * Pasaaltos Butterworth de fase cero a 70 Hz (--hpf): abajo de eso el
    parlante no entrega nada y lo unico que hay es la linea de 52 Hz y su
    armonica de ~102 Hz del ruido de fondo.
  * Truncado en el cruce de Lundeby (rir_lab/acoustics.lundeby) con un fade
    coseno de --fade-ms que TERMINA en el cruce. El cruce es UNO por medicion
    (mediana de los cruces por canal): truncar cada canal en su propio punto
    meteria diferencias entre canales que no son acusticas. --no-trunc lo
    desactiva (solo HPF) para medir cuanto pesa.
  * Canales: el usuario declara ch10 MUERTO (capsula danada) y el .npz no lo
    marca (mics_off = []), asi que se saca en memoria (meta['geometry']
    ['active']), no en el loader. Ademas se chequea CADA medicion: energia del
    directo por canal contra la mediana (muerto < -10 dB, basura > +6 dB) y
    residuo del TDOA de cada canal contra la onda plana del DOA medido
    (> 40 us = basura). Si cae algun canal de la media aro intermitente
    (posiciones 8,9,10,11,0,1 = ch6,7,4,5,10,11) la medicion se excluye.
  * Distancia: el ToF de estas mediciones esta FUERA DE RANGO (-1.46 m en la
    1, 9-13 m en las demas, en una habitacion; el propio .npz avisa "K no
    corresponde a este camino"). Se anula en memoria y el loader cae a la
    nominal de 1.5 m. Al OFB no le importa (es ciego); solo ubica la fuente.

COMO LEERLO
-----------
Las tablas son Delta (procesador - REF) promediadas sobre los pares, por iSIR.
Mirar (a) que OFB_AUTO quede arriba de PF_OUT y PF_BACK en la media, (b) que el
oraculo no le saque mas que ruido, (c) que el orden de las filas sea el mismo
que en MIRD/ISM (ofb_auto_benchmark.py). Los numeros ABSOLUTOS no son
comparables con MIRD: la sala, la distancia y el arreglo son otros.

USO
---
    conda activate tesis_beam
    python tests/kria_ofb_benchmark.py --quick          # 1 par, 1 iSIR (plomeria)
    python tests/kria_ofb_benchmark.py                  # el barrido entero
    python tests/kria_ofb_benchmark.py --check-only     # solo el chequeo de RIRs
    python tests/kria_ofb_benchmark.py --no-trunc --procs MVDR OFB_AUTO

RESULTADOS (2026-11-16, habitacion_pre_test/escritorio, 18 pares x 4 iSIR,
M = 11 sin ch10, 12.6 min con 7 procesos)
------------------------------------------------------------------------
Mediciones: las 6 pasan el chequeo (directo de cada canal a +-3.3 dB de la
mediana, TDOA a <= 22 us de la onda plana). Ninguna con media aro muerta.
OJO: ch10 NO esta mudo en estas RIRs (-0.9..-3.3 dB, TDOA a +1..+11 us); se
saca igual por la declaracion de capsula danada (en la 4 tiene la cola ~7 dB
mas ruidosa que el resto). Las fuentes 3/4/5 estan en la misma direccion
(191-197 deg, separadas 2-5 deg) a distancia creciente: entre ellas no hay
pares; los 18 pares salen de 1 y frente_cercana contra todo.

    Delta PESQ       -5      0      5     10  | media
    REF (abs)      1.059  1.089  1.166  1.316 | 1.157
    MVDR (sin PF)  0.458  0.757  0.964  0.961 | 0.785
    PF_OUT         0.651  0.936  1.101  1.060 | 0.937
    PF_BACK        0.592  0.924  1.136  1.119 | 0.943
    OFB_AUTO       0.647  0.942  1.138  1.119 | 0.961
    OFB_AUTO_ORC   0.647  0.940  1.136  1.119 | 0.960

    Delta SI-SDR     -5      0      5     10  | media
    REF (abs)     -5.655 -0.865  3.509  7.029 | 1.005
    MVDR (sin PF) 10.511  7.058  3.188 -0.345 | 5.103
    PF_OUT        10.539  6.802  2.775 -0.840 | 4.819
    PF_BACK        9.911  6.368  2.533 -0.956 | 4.464
    OFB_AUTO      10.496  6.695  2.669 -0.914 | 4.736

    Delta STOI       -5      0      5     10  | media
    REF (abs)      0.576  0.695  0.798  0.875 | 0.736
    MVDR / PF_OUT  0.292  0.209  0.124  0.055 | 0.170 / 0.171
    PF_BACK        0.286  0.202  0.117  0.050 | 0.164
    OFB_AUTO       0.296  0.209  0.121  0.052 | 0.169

  * La agenda funciona igual que en MIRD: OFB_AUTO toma PF_OUT con iSIR bajo
    y PF_BACK con iSIR alto, y gana la media de Delta PESQ (gana a PF_OUT en
    48/72 celdas, a PF_BACK en 43/72 y al MVDR sin PF en 72/72).
  * Estimador ciego vs oraculo: -0.0007 de media (peor celda +0.022), el ciego
    gana en 38/72. Saber el iSIR no compra nada, como en MIRD (+0.0004).
  * Contra MIRD (ofb_auto_benchmark.py, mismas filas, iSIR -5..10): Delta PESQ
    de OFB_AUTO 0.961 aca vs 0.934 en MIRD, mismo orden de filas
    (OFB_AUTO > PF_BACK > PF_OUT). Delta STOI algo mayor (0.169 vs 0.151).
    Delta SI-SDR ~1.3 dB MENOR (4.74 vs 6.00) y NEGATIVO con iSIR 10 (-0.9 vs
    +0.5): con poca interferencia el PF recorta mas voz de la que saca de
    interferente+cola. El MVDR sin PF es el mejor en SI-SDR en todos los iSIR
    y el peor en PESQ: el PF compra PESQ con SI-SDR.
  * Lo que manda entre pares es el TARGET, no la separacion: target 1 y
    frente_cercana (DRR +0.8 / +9 dB) dan Delta PESQ 0.93-1.25; 3/4/5 (DRR
    -1.8 a -4.5 dB, mas lejos) 0.73-0.88. Spearman por par: DRR 0.50,
    separacion 0.30.
  * Truncado de Lundeby: no cambia nada. La meseta de ruido es -61..-71 dB de
    la energia de la RIR despues del HPF (-52..-66 sin el), y truncado - sin
    truncar da Delta PESQ +0.0003 de media (max 0.043 en una celda), SI-SDR
    -0.003 dB, STOI +0.0001 (OFB_AUTO, 72 celdas). Lo que limpia la cola es el
    HPF de 70 Hz (lineas de 52/102 Hz); el truncado queda por higiene.
"""

import argparse
import contextlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

PLATFORM = os.path.expanduser("~/Documents/Tesis/mic-array-platform")
RIR_ROOT = os.path.join(PLATFORM, "data", "rirs")
RIR_LAB = os.path.join(PLATFORM, "tools")
SIGNALS = os.path.join(PROJECT_ROOT, "tools", "data", "signals")
OUT_DIR = os.path.join(PROJECT_ROOT, "tests", "dataset_out", "kria_ofb")

ROOM, ARRAY = "habitacion_pre_test", "escritorio"
FORCED_OFF = (10,)                     # ch10: capsula danada (declarado)
HALF_RING_POS = (8, 9, 10, 11, 0, 1)   # posiciones con contacto intermitente
ORDER = ["MVDR", "PF_OUT", "PF_BACK", "OFB_AUTO", "OFB_AUTO_ORC"]
C_SOUND = 343.0


# ============================================================ RIRs ==========
def check_channels(ir, pos, az_deg, el_deg, fs, ref=6):
    """Chequeo por canal de UNA medicion cruda (N, 12).

    Devuelve {canal: dict(dir_db, tdoa_err_us, rho, estado)}. El directo se
    ubica en el pico de energia sumada; la TDOA se mide contra ch6 en 300-5000
    Hz con interpolacion parabolica y se compara contra la onda plana del DOA
    que rir_lab guardo en la medicion.
    """
    from scipy import signal as sg
    pk = int(np.argmax(np.sum(ir ** 2, axis=1)))
    e_dir = np.array([10 * np.log10(np.sum(ir[pk - 16:pk + 48, m] ** 2) + 1e-30)
                      for m in range(ir.shape[1])])
    med = np.median(e_dir)
    sos = sg.butter(4, [300, 5000], btype="band", fs=fs, output="sos")
    w = sg.sosfiltfilt(sos, ir, axis=0)[pk - 20:pk + 100, :]
    az, el = np.radians(az_deg), np.radians(el_deg)
    u = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    tau = -(pos @ u) / C_SOUND
    nf = 4096
    R = np.conj(np.fft.rfft(w[:, ref], nf))
    out = {}
    for m in range(ir.shape[1]):
        c = np.fft.irfft(np.fft.rfft(w[:, m], nf) * R, nf)
        cc = np.concatenate([c[-40:], c[:41]])
        i = int(np.argmax(cc))
        y0, y1, y2 = cc[max(i - 1, 0)], cc[i], cc[min(i + 1, len(cc) - 1)]
        den = y0 - 2 * y1 + y2
        d = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        lag = (i - 40 + d) / fs
        rho = cc[i] / np.sqrt(np.sum(w[:, m] ** 2) * np.sum(w[:, ref] ** 2) + 1e-300)
        err = (lag - (tau[m] - tau[ref])) * 1e6
        rel = e_dir[m] - med
        st = "ok"
        if rel < -10:
            st = "MUERTO"
        elif rel > 6 or abs(err) > 40:
            st = "BASURA"
        out[m] = dict(dir_db=float(rel), tdoa_err_us=float(err), rho=float(rho),
                      estado=st)
    return out


def prep_rir(h_all, fs, hpf_hz=70.0, trunc=True, fade_ms=10.0):
    """HPF de fase cero + truncado comun en el cruce de Lundeby.

    h_all (N, M) con SOLO los canales buenos. Devuelve (h, info)."""
    from scipy import signal as sg
    if RIR_LAB not in sys.path:
        sys.path.insert(0, RIR_LAB)
    from rir_lab.acoustics import lundeby
    h = np.asarray(h_all, dtype=float).copy()
    if hpf_hz:
        sos = sg.butter(2, hpf_hz, btype="high", fs=fs, output="sos")
        h = sg.sosfiltfilt(sos, h, axis=0)
    crosses, enr = [], []
    for m in range(h.shape[1]):
        cr, noise, _, _, found = lundeby(h[:, m], fs)
        if found:
            crosses.append(cr)
        enr.append(10 * np.log10(noise / (np.max(h[:, m] ** 2) + 1e-300) + 1e-300))
    cut = int(np.median(crosses)) if crosses else h.shape[0]
    info = {"cross_ms": cut / fs * 1e3, "cross_spread_ms":
            (np.ptp(crosses) / fs * 1e3 if crosses else float("nan")),
            "enr_db": float(np.median(enr)), "N": h.shape[0]}
    e0 = np.sum(h ** 2)
    if trunc and cut < h.shape[0]:
        nf = max(1, int(fade_ms * 1e-3 * fs))
        a = max(0, cut - nf)
        win = np.ones(h.shape[0])
        win[a:cut] = 0.5 * (1 + np.cos(np.pi * np.arange(cut - a) / max(cut - a, 1)))
        win[cut:] = 0.0
        h = h * win[:, None]
        h = h[:cut]
    info["removed_db"] = float(10 * np.log10(max(e0 - np.sum(h ** 2), 1e-300) / e0))
    return h, info


def make_provider(trunc=True, hpf_hz=70.0, fade_ms=10.0, keep_ch10=False, verbose=False):
    """KriaRirProvider de la sala, con canales/distancia arreglados EN MEMORIA
    y las RIRs preparadas (HPF + Lundeby) al pedirlas."""
    from propagation.kria_rir_loader import KriaRirProvider

    class PreparedKriaProvider(KriaRirProvider):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._cache = {}
            self.prep_info = {}

        def load_rir(self, room, array, distance=None, angle=None, source=None):
            h = super().load_rir(room, array, distance, angle, source)
            key = self.current.path
            if key not in self._cache:
                hh, info = prep_rir(h, self.current.fs, hpf_hz=hpf_hz,
                                    trunc=trunc, fade_ms=fade_ms)
                self._cache[key] = hh
                self.prep_info[self.current.source] = info
            return self._cache[key]

    prov = PreparedKriaProvider(root=os.path.join(RIR_ROOT, ROOM), verbose=verbose)
    for m in prov.measurements:
        g = m.meta.setdefault("geometry", {})
        off = () if keep_ch10 else FORCED_OFF
        g["active"] = [c for c in (g.get("active") or range(12)) if c not in off]
        doa = m.meta.get("doa") or {}
        if doa.get("dist_m") is not None:
            doa["dist_m_tof_invalido"] = doa["dist_m"]
            doa["dist_m"] = None
    return prov


def audit(prov, keep_ch10=False, verbose=True):
    """Chequeo de canales de cada medicion de la sala/aro. Devuelve
    (fuentes_validas, tabla de texto)."""
    ok, lines = [], []
    order = None
    for s in prov.sources(ROOM, ARRAY):
        m = prov.get(ROOM, ARRAY, s)
        z = np.load(m.path, allow_pickle=False)
        ir, pos = np.asarray(z["ir"], float), np.asarray(z["positions"], float)
        order = m.meta["geometry"]["order"]
        az, el, _ = m.direction()
        chk = check_channels(ir, pos, az, el, m.fs, ref=m.ref_mic)
        bad = {c: v["estado"] for c, v in chk.items()
               if v["estado"] != "ok" and (keep_ch10 or c not in FORCED_OFF)}
        half = [c for c in range(12) if order[c] in HALF_RING_POS]
        half_bad = [c for c in bad if c in half]
        excl = len(half_bad) > 0 or len(bad) > 1
        d = m.descriptors()
        q = m.quality
        lines.append(
            f"  {s:<15} az {az:6.1f} el {el:5.1f}  T {d['T_wide']:.2f} s  "
            f"DRR {d['DRR_wide']:+5.1f} dB  reps {q['reps_used']}  "
            f"dir[dB rel] min {min(v['dir_db'] for v in chk.values()):+5.1f} "
            f"(ch{min(chk, key=lambda c: chk[c]['dir_db'])})  "
            f"|TDOA err| max {max(abs(v['tdoa_err_us']) for v in chk.values()):4.0f} us  "
            f"ch10: {chk[10]['dir_db']:+.1f} dB / {chk[10]['tdoa_err_us']:+.0f} us  "
            + (f"-> EXCLUIDA ({bad})" if excl else ("malos: %s" % bad if bad else "ok")))
        if not excl:
            ok.append(s)
    txt = "\n".join(lines)
    if verbose:
        print(f"[audit] media aro intermitente = posiciones {HALF_RING_POS} "
              f"= ch{[c for c in range(12) if order[c] in HALF_RING_POS]}; "
              f"ch10 fuera por declaracion: {not keep_ch10}")
        print(txt)
    return ok, txt


def plan_pairs(prov, sources, min_sep):
    """Pares ORDENADOS (target, interferente, separacion) con separacion >=
    min_sep. Para cada target valido usa plan_array() del banco Kria (K=1,
    todas las combinaciones) y se queda con los interferentes validos."""
    from evaluation.full_benchmark_test_dtln_kria import plan_array, _sep
    az = {s: prov.get(ROOM, ARRAY, s).direction()[0] for s in sources}
    pairs = []
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        for t in sources:
            pl = plan_array(prov, ROOM, ARRAY, t, [1], None, 0)
            for (i,) in (pl["combos"] if pl else []):
                if i in sources and _sep(az[t], az[i]) >= min_sep:
                    pairs.append((t, i, _sep(az[t], az[i])))
    return pairs, az


# ======================================================= PROCESADORES ======
def build_processors(smooth, names=None):
    from evaluation.bf_wrappers import NM_MVDR_OFB, NM_MVDR_OFB_AUTO
    common = dict(win_type='rect', synth='hann', sharpen_exp=8.0, alpha=0.99,
                  block_update=1, leak=0.0, fuse='mean')
    p = {
        "MVDR": lambda: NM_MVDR_OFB(smooth=None, **common),
        "PF_OUT": lambda: NM_MVDR_OFB(pf_mask='out', smooth=smooth, **common),
        "PF_BACK": lambda: NM_MVDR_OFB(pf_mask='back', smooth=smooth, **common),
        "OFB_AUTO": lambda: NM_MVDR_OFB_AUTO(smooth=smooth),
        "OFB_AUTO_ORC": lambda: NM_MVDR_OFB_AUTO(smooth=smooth, isir_db='scene'),
    }
    names = names or ORDER
    return {n: p[n]() for n in names}


# ============================================================ CORRIDA ======
def run_pair(job):
    """Un par (target, interferente) con todos sus iSIR, en un proceso aparte
    (la salida del motor va a un log).

    NO arma su propia escena: reusa plan_array / build_config / build_grid de
    src/evaluation/full_benchmark_test_dtln_kria.py (el banco sobre RIRs
    medidas), con el plan recortado a UN interferente. Lo unico propio es el
    provider preparado (make_provider) y los procesadores OFB.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    a = job["args"]
    tag = f"{job['target']}__vs__{job['interf']}" + ("" if a["trunc"] else "__notrunc")
    log = os.path.join(a["out_dir"], "logs", tag + ".log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    t0 = time.time()
    with open(log, "w") as fh, contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh):
        import pandas as pd
        from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search
        from evaluation.full_benchmark_test_dtln_kria import (
            plan_array, build_config, build_grid, DEFAULT_INTERF_AUDIO, SIGNALS as K_SIGNALS)
        prov = make_provider(trunc=a["trunc"], hpf_hz=a["hpf"], fade_ms=a["fade_ms"],
                             keep_ch10=a["keep_ch10"])
        plan = plan_array(prov, ROOM, ARRAY, job["target"], [1], None, 0)
        plan["combos"] = [(job["interf"],)]
        prov.require_same_array([plan["ms"][job["target"]], plan["ms"][job["interf"]]])
        # Un interferente -> una senal: la misma de ofb_auto_benchmark.py (techno).
        base = build_config(prov, ROOM, ARRAY, job["target"],
                            [os.path.join(K_SIGNALS, DEFAULT_INTERF_AUDIO[0])])
        assert prov.current.active[base['ref_mic_mode']] == 6
        base.update({'out_prefix': f"kria_{tag}", 'snr_db': a["snr"],
                     'duration': a["duration"]})
        grid, info = build_grid(ROOM, plan, a["isir"])
        df = run_mird_grid_search(
            grid_params=grid, dataset_provider=prov,
            processors=build_processors(a["smooth"], a["procs"]),
            scene_base_config=base, output_dir=os.path.join(a["out_dir"], "per_pair"),
            interpreter_1=None, interpreter_2=None, apply_dtln_post=False,
            save_catalog=False, show_progress=False)
        meta = pd.DataFrame([info[k] for k in df["interf_configs"]], index=df.index)
        df = pd.concat([df, meta], axis=1)
        M = len(prov.current.active)
        print("M =", M, "ref idx", base['ref_mic_mode'], "prep", prov.prep_info)
    df["target"] = job["target"]
    df["interf"] = job["interf"]
    df["sep_deg"] = job["sep"]
    df["trunc"] = a["trunc"]
    df["M_used"] = M
    df["tgt_cross_ms"] = prov.prep_info.get(job["target"], {}).get("cross_ms", np.nan)
    print(f"[ok] {tag:40s} {time.time() - t0:6.1f} s", flush=True)
    return df


# =========================================================== RESUMEN =======
def summarize(df, out_dir):
    import pandas as pd
    pd.set_option("display.width", 200)
    metrics = [("PESQ", "PESQ"), ("SI-SDR", "SI-SDR"), ("STOI", "STOI")]
    for trunc in sorted(df["trunc"].unique(), reverse=True):
        d = df[df["trunc"] == trunc]
        order = [p for p in ORDER if p in d["processor"].unique()]
        print("\n" + "=" * 78)
        print(f"RIRs {'TRUNCADAS (Lundeby)' if trunc else 'SIN TRUNCAR (solo HPF)'}: "
              f"{d.groupby(['target', 'interf']).ngroups} pares, "
              f"{d['isir_db'].nunique()} iSIR, M = {sorted(d['M_used'].unique())}")
        print("=" * 78)
        one = d[d["processor"] == order[0]]
        for lab, col in metrics:
            base =one.groupby("isir_db")[f"base_{col}_early"].mean()
            p = d.pivot_table(index="processor", columns="isir_db",
                              values=f"Delta_bf_{col}_early", aggfunc="mean").reindex(order)
            p["media"] = p.mean(axis=1)
            base_row = pd.DataFrame([list(base.values) + [base.mean()]],
                                    columns=p.columns, index=["REF (abs)"])
            print(f"\n--- {lab}: REF absoluto y Delta por iSIR (media sobre pares) ---")
            print(pd.concat([base_row, p]).round(3).to_string())
        if "OFB_AUTO" in order:
            key = ["target", "interf", "isir_db"]
            auto = d[d["processor"] == "OFB_AUTO"].set_index(key)
            print("\n--- celdas ganadas por OFB_AUTO (Delta PESQ / SI-SDR / STOI) ---")
            for p_ in order:
                if p_ == "OFB_AUTO":
                    continue
                cur = d[d["processor"] == p_].set_index(key).reindex(auto.index)
                w = [int((auto[f"Delta_bf_{c}_early"] > cur[f"Delta_bf_{c}_early"]).sum())
                     for _, c in metrics]
                print(f"  vs {p_:13s} n={len(auto):3d}  PESQ:{w[0]}  SI-SDR:{w[1]}  STOI:{w[2]}")
        if {"OFB_AUTO", "OFB_AUTO_ORC"} <= set(order):
            key = ["target", "interf", "isir_db"]
            a = d[d["processor"] == "OFB_AUTO"].set_index(key)["Delta_bf_PESQ_early"]
            o = d[d["processor"] == "OFB_AUTO_ORC"].set_index(key)["Delta_bf_PESQ_early"]
            dd = o - a.reindex(o.index)
            print(f"\n--- COSTO DEL ESTIMADOR (oraculo - ciego, Delta PESQ) ---")
            print(f"    media {dd.mean():+.4f}  mediana {dd.median():+.4f}  "
                  f"peor celda {dd.max():+.4f}  ciego gana en {int((dd < 0).sum())}/{len(dd)}")
        print("\n--- Delta PESQ de OFB_AUTO por par (media sobre iSIR) ---")
        if "OFB_AUTO" in order:
            t = d[d["processor"] == "OFB_AUTO"].groupby(["target", "interf"]).agg(
                sep=("sep_deg", "first"), base_PESQ=("base_PESQ_early", "mean"),
                dPESQ=("Delta_bf_PESQ_early", "mean"), dSISDR=("Delta_bf_SI-SDR_early", "mean"),
                dSTOI=("Delta_bf_STOI_early", "mean"))
            print(t.round(3).to_string())
    if df["trunc"].nunique() == 2:
        key = ["target", "interf", "isir_db", "processor"]
        a = df[df["trunc"]].set_index(key)
        b = df[~df["trunc"]].set_index(key).reindex(a.index)
        ok = b["Delta_bf_PESQ_early"].notna()
        print("\n--- TRUNCADO vs SIN TRUNCAR (mismas celdas; truncado - sin truncar) ---")
        for lab, c in metrics:
            for col in (f"base_{c}_early", f"Delta_bf_{c}_early"):
                x = (a[col] - b[col])[ok]
                print(f"  {col:24s} media {x.mean():+.4f}  max|.| {x.abs().max():.4f}  (n={ok.sum()})")
    print(f"\n[ok] {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--quick", action="store_true", help="1 par, iSIR 0 (plomeria)")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--isir", type=float, nargs="+", default=[-5, 0, 5, 10])
    ap.add_argument("--min-sep", type=float, default=45.0)
    ap.add_argument("--smooth", type=float, default=0.5)
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--snr", type=float, default=60.0, help="ruido de sensor (dB)")
    ap.add_argument("--hpf", type=float, default=70.0)
    ap.add_argument("--fade-ms", type=float, default=10.0)
    ap.add_argument("--no-trunc", action="store_true", help="solo HPF, sin Lundeby")
    ap.add_argument("--both-trunc", action="store_true",
                    help="correr tambien sin truncar (procesadores de --ablation-procs)")
    ap.add_argument("--ablation-procs", nargs="+", default=["OFB_AUTO"])
    ap.add_argument("--keep-ch10", action="store_true")
    ap.add_argument("--procs", nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    prov = make_provider(trunc=not args.no_trunc, hpf_hz=args.hpf, fade_ms=args.fade_ms,
                         keep_ch10=args.keep_ch10, verbose=True)
    srcs, _ = audit(prov, keep_ch10=args.keep_ch10)
    prov.require_same_array([prov.get(ROOM, ARRAY, s) for s in srcs])
    print("\n[prep] RIRs (HPF %.0f Hz, %s):" % (args.hpf, "sin truncar" if args.no_trunc
                                                 else "Lundeby + fade %.0f ms" % args.fade_ms))
    for s in srcs:
        prov.load_rir(ROOM, ARRAY, source=s)
        i = prov.prep_info[s]
        print(f"  {s:<15} M={len(prov.current.active)} cruce {i['cross_ms']:5.0f} ms "
              f"(dispersion entre canales {i['cross_spread_ms']:4.0f} ms)  "
              f"ruido/pico {i['enr_db']:6.1f} dB  energia quitada {i['removed_db']:6.1f} dB "
              f"(de {i['N'] / 16:.0f} ms)")
    pairs, az = plan_pairs(prov, srcs, args.min_sep)
    print(f"\n[plan] {len(pairs)} pares ordenados con separacion >= {args.min_sep:g} deg:")
    for t, i, s in pairs:
        print(f"  target {t:<15} ({az[t]:6.1f})  interf {i:<15} ({az[i]:6.1f})  sep {s:5.1f}")
    if args.check_only:
        return
    if args.quick:
        pairs, args.isir = pairs[:1], [0]

    base_args = dict(out_dir=args.out_dir, hpf=args.hpf, fade_ms=args.fade_ms,
                     keep_ch10=args.keep_ch10, duration=args.duration, snr=args.snr,
                     smooth=args.smooth, isir=args.isir, procs=args.procs)
    jobs = [dict(target=t, interf=i, sep=s, args=dict(base_args, trunc=not args.no_trunc))
            for t, i, s in pairs]
    if args.both_trunc and not args.no_trunc:
        jobs += [dict(target=t, interf=i, sep=s,
                      args=dict(base_args, trunc=False, procs=args.ablation_procs))
                 for t, i, s in pairs]
    print(f"\n[run] {len(jobs)} trabajos en {args.workers} procesos "
          f"(logs en {os.path.join(args.out_dir, 'logs')})")
    t0 = time.time()
    import multiprocessing as mp
    import pandas as pd
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=mp.get_context("spawn")) as ex:
        dfs = list(ex.map(run_pair, jobs))
    df = pd.concat(dfs, ignore_index=True)
    print(f"[run] {len(df)} filas en {(time.time() - t0) / 60:.1f} min")
    stem = os.path.join(args.out_dir, "kria_ofb_benchmark" + ("_quick" if args.quick else ""))
    df.to_csv(stem + ".csv", index=False)
    summarize(df, args.out_dir)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
    main()
