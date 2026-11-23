#!/usr/bin/env python3
"""full_benchmark_test_dtln_kria.py -- el banco, sobre RIRs MEDIDAS con el aro.

MISMO MOTOR, OTRO DATASET. No duplica nada: importa run_mird_grid_search() de
full_benchmark_test_dtln_mird.py y le pasa un KriaRirProvider.

COMO SE ELIGEN LAS ESCENAS (distinto del MIRD, a proposito)

  El dataset medido es una JERARQUIA, no una grilla:

      sala  ->  posicion de aro (A, B, ...)  ->  fuente (1, 2, 3, ...)

  y las fuentes se piden POR NOMBRE, nunca por (angulo, distancia). Se le dice
  "procesame toda la sala": para cada posicion de aro el OBJETIVO es siempre la
  misma fuente (default: la 1) y los INTERFERENTES son combinaciones de las
  demas, de a K (--n-interf 1 2 ...). Con S fuentes hay C(S-1, K) combinaciones
  por posicion de aro: se corren todas, o --max-combos N de ellas elegidas con
  semilla fija y BALANCEADAS (cada fuente aparece la misma cantidad de veces
  como interferente, +-1), para no meter sesgo espacial.

  LA REVERBERACION NO ES UN EJE. No se elige un RT60 ni se hace snap: se corre
  todo, y cada fila del resultado lleva los descriptores MEDIDOS de su objetivo
  (tgt_T_wide, tgt_DRR_wide, tgt_CDR_1000, ...) y la separacion angular con los
  interferentes (sep_min_deg). El orden se hace despues, en el analisis.

  K INTERFERENTES NECESITAN K SENALES DISTINTAS. Con una sola, serian la misma
  senal llegando de K direcciones (fuentes coherentes), que no es lo que se
  quiere medir. Se pasan con --interf-audio; si faltan, se corta.

QUE NO CAMBIA: los procesadores, las metricas, el cacheo, el DTLN. 'rt60' sigue
siendo el nombre de la SALA y 'mird_spacing' el de la posicion del ARO (son los
ejes que el motor sabe barrer y cachear). El microfono de referencia es ch6.

SALIDA (en --out; mismos nombres y formato que la corrida MIRD, con prefijo
kria_<sala>, mas lo que el MIRD no tiene):

  kria_<sala>_benchmark_catalog.h5      UN catalogo por sala, igual al del MIRD
      (<proc>/<metrica>/{best,worst}_case/{metadata,geometry,audio,metrics,
      spatial_<proc>}: lo que lee dashboard.py) con best/worst sobre TODA la
      sala, no por aro. Cada caso suma:
        scene/  attrs: ubicacion de cada fuente (dist/az/el, xyz, de donde
                salio cada dato), descriptores acusticos medidos (T, EDT, DRR,
                CDR por banda, MSC), calidad de la medicion, aro, sala, audios.
        rirs/   las RIRs que entraron a la escena: target, interf_1, ... (N, M)
      y la raiz lleva attrs['run_info'] (json de la corrida).
  kria_<sala>_benchmark_metrics.{parquet,csv}   las metricas del MIRD (mismas
      columnas) + todas las columnas de escena (las mismas claves que scene/).
  kria_<sala>_sources.{parquet,csv}   una fila por medicion usada (fuente x
      aro): posicion, descriptores, calidad, hardware. Para cruzar.
  kria_<sala>_run.json                argumentos, configuracion, procesadores,
      commit: con que se genero todo lo anterior.
  kria_<sala>_parts/                  las tablas parciales por aro que escribe
      el motor (checkpoint si la corrida se corta).

  El catalogo H5 se REGENERA en cada corrida (como el parquet): si no, quedan
  casos de procesadores de una corrida vieja mezclados con los de esta.

USO
    conda run -n tesis_beam python src/evaluation/full_benchmark_test_dtln_kria.py --list
    conda run -n tesis_beam python src/evaluation/full_benchmark_test_dtln_kria.py \\
        --room habitacion_baja --n-interf 1 2 --isir -5 0 5 --dry-run
"""
import argparse
import datetime
import itertools
import json
import os
import random
import subprocess
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
for _p in (SRC_DIR, SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from propagation.kria_rir_loader import KriaRirProvider      # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
SIGNALS = os.path.join(PROJECT_ROOT, "tools", "data", "signals")
ARRAY_CENTER = [3.0, 3.0, 1.2]
# Una senal por interferente simultaneo, en este orden.
DEFAULT_INTERF_AUDIO = ["techno_gated commune.wav", "all minor ab oz.wav",
                        "flute_music.wav", "hairdryer_02_RHODE_NT5.wav",
                        "vacuum_02_RHODE_NT1.wav", "blender_03_RHODE_NT1.wav"]
# Senales para --signal-sweep: espectros DISTINTOS a proposito (de 80 % de la
# potencia bajo 300 Hz a casi nada), para que el resultado no dependa de como
# el iSIR de banda ancha reparte la energia de UNA senal. Sin voz: con un
# interferente hablado el estimador de iSIR del OFB no funciona (documentado).
SWEEP_AUDIO = ["techno_gated commune.wav",                       # musica, graves
               "audioset_realrec_airconditioner_2TE3LoA2OUQ.wav",  # maquina, graves
               "ruido_rosa_16k.wav",                             # estacionario, ancho
               "flute_music.wav",                                # musica tonal, medios
               "hairdryer_02_RHODE_NT5.wav",                     # maquina, agudos
               "vacuum_02_RHODE_NT1.wav",                        # maquina, agudos
               "blender_03_RHODE_NT1.wav"]                       # maquina, agudos

_AUDIO_DESC = {}


def audio_spectrum(path, fs=16000, max_s=60.0):
    """Descriptores espectrales de una SENAL (cacheados): % de potencia bajo
    300 Hz, % en 300-4000 Hz y centroide, a la fs del banco. Es la otra mitad
    de la interaccion senal x funcion de transferencia."""
    if path in _AUDIO_DESC:
        return dict(_AUDIO_DESC[path])
    import scipy.signal as sg
    from scipy.io import wavfile
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fs0, x = wavfile.read(path)
    x = x.astype(float)
    x = x.mean(axis=1) if x.ndim > 1 else x
    x = x[: int(max_s * fs0)]
    g = np.gcd(int(fs0), int(fs))
    if fs0 != fs:
        x = sg.resample_poly(x, fs // g, fs0 // g)
    f, S = sg.welch(x, fs=fs, nperseg=4096)
    tot = S.sum() + 1e-30
    out = {"audio_low_pct": float(100 * S[f < 300].sum() / tot),
           "audio_speech_pct": float(100 * S[(f >= 300) & (f <= 4000)].sum() / tot),
           "audio_centroid_hz": float(np.sum(f * S) / tot)}
    _AUDIO_DESC[path] = out
    return dict(out)


def _sep(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def pick_combos(others, k, max_combos=None, seed=0):
    """Combinaciones de `k` interferentes entre `others`.

    Todas si entran en `max_combos`; si no, una seleccion BALANCEADA: se van
    eligiendo las que usan las fuentes menos usadas hasta el momento, con los
    empates resueltos por un orden aleatorio de semilla fija. Asi ninguna
    posicion queda sobre-representada como interferente.
    """
    allc = list(itertools.combinations(others, k))
    if max_combos is None or len(allc) <= max_combos:
        return allc
    rng = random.Random(f"{seed}:{k}:{','.join(map(str, others))}")
    rng.shuffle(allc)
    used = {s: 0 for s in others}
    out = []
    for _ in range(max_combos):
        best = min(allc, key=lambda c: (sum(used[s] for s in c),
                                        max(used[s] for s in c)))
        allc.remove(best)
        out.append(best)
        for s in best:
            used[s] += 1
    return out


def plan_array(prov, room, array, target, n_interf, max_combos, seed,
               min_sep=0.0, exclude=()):
    """Las escenas de UNA posicion de aro: (objetivo, [combos de interferentes]).

    Devuelve None (y lo dice) si esa posicion no tiene la fuente objetivo.
    """
    srcs = [s for s in prov.sources(room, array) if s not in set(exclude or ())]
    if target not in srcs:
        print(f"[kria] aro '{array}': no tiene la fuente objetivo '{target}' "
              f"(hay {srcs}); se saltea.")
        return None
    ms = {s: prov.get(room, array, s) for s in srcs}
    sin_dir = [s for s in srcs if ms[s].direction()[0] is None]
    if target in sin_dir:
        print(f"[kria] aro '{array}': el objetivo no tiene direccion; se saltea.")
        return None
    others = [s for s in srcs if s != target and s not in sin_dir]
    if sin_dir:
        print(f"[kria] aro '{array}': sin direccion, no se usan: {sin_dir}")
    # Interferentes (casi) en la MISMA direccion que el objetivo: suelen ser
    # re-mediciones del mismo punto. Ahi no hay separacion espacial que medir.
    taz = ms[target].direction()[0]
    cerca = [s for s in others if _sep(taz, ms[s].direction()[0]) < max(min_sep, 1e-9)]
    if cerca:
        print(f"[kria] aro '{array}': a menos de {min_sep:g} deg del objetivo, "
              f"no se usan: {cerca}")
        others = [s for s in others if s not in cerca]
    casi = [s for s in others if _sep(taz, ms[s].direction()[0]) < 10.0]
    if casi:
        print(f"[kria] aro '{array}': OJO, a menos de 10 deg del objetivo: {casi} "
              f"(--min-sep para sacarlos)")
    prov.require_same_array([ms[target]] + [ms[s] for s in others])
    combos = []
    for k in n_interf:
        if k == 0:
            combos.append(())
        elif k > len(others):
            print(f"[kria] aro '{array}': se pidieron {k} interferentes y hay "
                  f"{len(others)} fuentes ademas del objetivo; ese K se saltea.")
        else:
            combos += pick_combos(others, k, max_combos, seed)
    return {"array": array, "target": target, "ms": ms, "combos": combos}


def array_center_for(prov, room, array):
    """(centro del aro en la sala, origen). Si la posicion del aro en la sala
    se midio (x, y, z en la base), se usa: el dashboard dibuja la sala real.
    Si no, ARRAY_CENTER (virtual). No cambia ninguna metrica: las RIRs son
    las medidas; solo ubica la escena en el catalogo."""
    a = prov._for(room, array).array_summary()
    xyz = (a["x_room_m"], a["y_room_m"], a["z_room_m"])
    if all(v is not None for v in xyz):
        return [float(v) for v in xyz], "medido"
    return list(ARRAY_CENTER), "virtual"


def build_config(prov, room, array, target, interf_audio,
                 dims_default=(5.0, 4.0, 2.6), duration=15):
    """base_config con lo que el dataset medido impone."""
    prov.load_rir(room, array, source=target)
    dims = prov.room_dims(room) or dims_default   # solo catalogo H5/dashboard
    center, _ = array_center_for(prov, room, array)
    geom = prov.array_geometry(room, array)
    ref = prov.ref_mic_index(room, array)
    act = prov.active_channels(room, array)
    print(f"[kria] aro '{array}': M = {len(act)} canales {act}, "
          f"referencia indice {ref} (ch{act[ref]}), "
          f"radio {np.linalg.norm(geom[0, :2]) * 1e3:.1f} mm")
    return {
        'fs': prov.get_current_fs(),
        'duration': duration,
        't_early': 0.050,
        'array_center': center,
        'mird_spacing': array,          # <- el eje "posicion de aro"
        'out_prefix': f'kria_{room}_{array}',
        'room_dims': list(dims),
        'ref_mic_mode': ref,            # <- ch6, no M // 2
        'snr_db': 60.0,
        'source_path': os.path.join(
            SIGNALS, "p002_emo_adoration_sentences.wav"),
        'interf_paths': list(interf_audio),
        'wpe_taps': 7, 'wpe_delay': 3, 'wpe_alpha': 0.9999,
        'wpe_stft_size': 512, 'wpe_stft_shift': 128,
        'wpe_fixed_bits': None, 'wpe_fixed_round': 'nearest',
        'wpe_backend': 'cov',
        'wpe_block_L': 512, 'wpe_block_shift': 2, 'wpe_block_iters': 2,
        'wpe_block_reg': 1e-6, 'wpe_block_solver': 'cholesky',
        'wpe_block_mode': 'resolve',
        'stft_window': 512, 'stft_overlap': 384,
        'eval_references': ['early'],
        'dtln_model_path': os.path.join(
            SRC_DIR, "dnn_denoise", "models", "model_quant_1.tflite"),
        # el catalogo H5 guarda las RIRs que entraron a cada caso extremo
        'catalog_save_rirs': True,
        # SIR de entrada por banda (superposicion voz/interferencia) en ch6
        'input_band_sir': True,
    }


def _prefixed(prefix, d):
    return {f"{prefix}{k}": v for k, v in d.items()}


def build_grid(room, plan, isir, prov=None, interf_audio=None, source_path=None,
               variants=None):
    """La grilla de una posicion de aro + la tabla de metadatos por escena.

    Cada interferente va como (az, dist, indice_de_audio, NOMBRE): el nombre
    es lo que usa el motor para pedir la RIR; az y dist solo ubican la fuente.

    `info[str(interf_cfg)]` es un dict PLANO con todo lo que describe la escena
    y que el motor no sabe: se pega a cada fila (scene_info_fn) y al grupo
    'scene' del catalogo H5. Claves:
      room, array, scene_id, interf_srcs, target_az_deg, target_dir_origin,
      interf_az_deg, sep_min_deg, sep_mean_deg,
      tgt_*        la medicion del objetivo: posicion (dist/az/el, xyz relativo
                   al centro del aro, dir_origin, dist_origin, tof_*), calidad
                   (rep_corr_min, n_warnings, warnings, eps, ruido de fondo),
                   excitacion y descriptores (T_wide, DRR_wide, T_<banda>,
                   EDT_<banda>, DRR_<banda>, CDR_<banda>, MSC_rms, ...)
      int<j>_*     lo mismo para el interferente j (1..K), + sep_deg y audio
      int_*_mean   promedios sobre los interferentes (T_wide, T_mid, DRR_wide)
      arr_*        la posicion de aro: M, canales, ref, radio, orden, hw_rev,
                   centro usado en la escena y de donde salio
      room_*       lo que la base SQLite sabe de la sala (dims, volumen)
    `prov`, `interf_audio`, `source_path` son opcionales: sin ellos faltan
    arr_/room_ y los nombres de los audios (compatibilidad con quien ya la usa).

    `variants` (barrido de SENALES): lista de desplazamientos v. Cada
    combinacion se repite una vez por v, y el interferente j recibe la senal
    interf_audio[(v + j) % n] -- con K <= n las K senales son siempre
    distintas. None = una sola variante con la senal j para el interferente j
    (el comportamiento de siempre, mismo scene_id). Con variants, el scene_id
    termina en "|<senales>" y la fila lleva signal_variant.
    """
    ms, target = plan["ms"], plan["target"]
    td, taz, _, _ = ms[target].position()
    tsum = ms[target].summary()
    tsum.pop("src", None)                        # ya esta como 'target_src'
    common = {}
    if prov is not None:
        common.update(_prefixed("arr_", ms[target].array_summary()))
        center, origin = array_center_for(prov, room, plan["array"])
        common.update({"arr_center_x_m": center[0], "arr_center_y_m": center[1],
                       "arr_center_z_m": center[2], "arr_center_origin": origin})
        common.update(_prefixed("room_", prov.room_summary(room)))
    if source_path is not None:
        common["tgt_audio"] = os.path.basename(source_path)
        if os.path.exists(source_path):
            common.update(_prefixed("tgt_", audio_spectrum(source_path)))

    cfgs, info = [], {}
    sweep = variants is not None
    n_aud = len(interf_audio) if interf_audio else 1
    for combo, v in ((c, v) for c in plan["combos"] for v in (variants or [0])):
        cfg = []
        for j, s in enumerate(combo):
            d, az, _, _ = ms[s].position()
            cfg.append((round(az, 3), round(d, 3), (v + j) % n_aud if sweep else j, s))
        cfgs.append(cfg)
        azs = [c[0] for c in cfg]
        seps = [_sep(taz, a) for a in azs]
        sid = f"{room}/{plan['array']}/{target}|{','.join(combo)}"
        if sweep:
            sid += "|" + "+".join(os.path.splitext(os.path.basename(interf_audio[c[2]]))[0]
                                  for c in cfg)
        row = {"room": room, "array": plan["array"],
               "scene_id": sid,
               "interf_srcs": ",".join(combo),
               "target_az_deg": round(taz, 2),
               "target_dir_origin": ms[target].direction()[2],
               "interf_az_deg": ",".join(f"{a:.1f}" for a in azs),
               "sep_min_deg": min(seps) if seps else float("nan"),
               "sep_mean_deg": float(np.mean(seps)) if seps else float("nan")}
        row.update(common)
        row.update(_prefixed("tgt_", tsum))
        for j, s in enumerate(combo, start=1):
            isum = ms[s].summary()
            isum["sep_deg"] = seps[j - 1]
            if interf_audio is not None:
                ap_ = interf_audio[cfg[j - 1][2]]
                isum["audio"] = os.path.basename(ap_)
                if os.path.exists(ap_):
                    isum.update(audio_spectrum(ap_))
            row.update(_prefixed(f"int{j}_", isum))
        for k in ("T_wide", "T_mid", "DRR_wide"):
            vals = [ms[s].descriptors().get(k) for s in combo]
            vals = [v for v in vals if v is not None]
            row[f"int_{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
        if sweep:
            row["signal_variant"] = v
        info[str(cfg)] = row
    grid = {
        'rt60': [room],                 # <- el eje "sala"
        'target_src': [target],
        'target_angle': [round(taz, 3)],
        'target_dist': [round(td, 3)],
        'interf_configs': cfgs,
        'isir_db': list(isir),
        'mismatch_gain': [0], 'mismatch_phase': [0],
        'use_wpe': [False], 'wpe_method': ['online'],
        'wpe_taps': [7], 'wpe_delay': [2],
        'error_angle_deg': [0.0], 'error_distance_m': [0.0],
    }
    return grid, info


def sources_table(prov, room, plans):
    """Una fila por medicion usada (fuente x posicion de aro)."""
    rows = []
    for pl in plans:
        used = {pl["target"]} | {s for c in pl["combos"] for s in c}
        for s in sorted(used):
            m = pl["ms"][s]
            r = {"room": room, "array": pl["array"],
                 "role": "target" if s == pl["target"] else "interf",
                 "path": os.path.relpath(m.path, prov.root)}
            r.update(m.summary())
            r.update(_prefixed("arr_", m.array_summary()))
            rows.append(r)
    return rows


def _git_rev(path):
    try:
        out = subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "-C", path, "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5)
        rev = out.stdout.strip() or None
        return rev and (rev + ("-dirty" if dirty.stdout.strip() else ""))
    except Exception:                                        # noqa: BLE001
        return None


def _jsonable(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, np.ndarray):
        return f"<ndarray {v.shape}>"
    return repr(v)


def print_plan(room, plans, isir, n_proc, n_variants=1):
    tot = 0
    for pl in plans:
        ms, t = pl["ms"], pl["target"]
        print(f"\n  aro {pl['array']}: objetivo '{t}' "
              f"(az {ms[t].position()[1]:.1f}, T {ms[t].descriptors().get('T_wide') or float('nan'):.2f} s, "
              f"DRR {ms[t].descriptors().get('DRR_wide') or float('nan'):+.1f} dB), "
              f"{len(pl['combos'])} combinaciones")
        for c in pl["combos"]:
            azs = [ms[s].position()[1] for s in c]
            sep = min((_sep(ms[t].position()[1], a) for a in azs),
                      default=float("nan"))
            print(f"      interferentes {list(c)}  az {[round(a, 1) for a in azs]}"
                  f"  separacion minima {sep:.1f} deg")
        tot += len(pl["combos"]) * len(isir) * n_variants
    n_arr = len({pl["array"] for pl in plans})
    print(f"\n[kria] sala '{room}': {n_arr} posiciones de aro x {len(plans) // max(n_arr, 1)} objetivos, {tot} "
          f"escenas x {n_proc} procesadores = {tot * n_proc} filas"
          + (f" ({n_variants} variantes de senal por combinacion)" if n_variants > 1 else ""))


# ============================================================ EJECUCION ====
class DtlnMono:
    """DTLN (Westhausen & Meyer) monocanal sobre el mic de referencia que el
    motor inyecta en scene_config['ref_mic_idx'] (el mismo de las metricas).
    quant=True usa model_quant_{1,2}.tflite (pesos int8, rango dinamico);
    quant=False usa model_{1,2}.tflite (float32). Es el mismo lazo tiempo real
    (bloque 512 / salto 128) que el nodo 4.5 del motor (DTLN mono)."""

    def __init__(self, quant=True):
        self.quant = quant
        self._itp = None

    def _interpreters(self):
        if self._itp is None:
            import tensorflow as tf
            models = os.path.join(SRC_DIR, "dnn_denoise", "models")
            tag = "model_quant_" if self.quant else "model_"
            itp = []
            for k in (1, 2):
                it = tf.lite.Interpreter(model_path=os.path.join(models, f"{tag}{k}.tflite"))
                it.allocate_tensors()
                itp.append(it)
            self._itp = tuple(itp)
        return self._itp

    def process(self, mic_signals, scene_config):
        from dnn_denoise.dtln_lite import apply_dtln_post_tflite_realtime
        ref = int(scene_config.get('ref_mic_idx', len(mic_signals) // 2))
        i1, i2 = self._interpreters()
        y = apply_dtln_post_tflite_realtime(i1, i2, np.asarray(mic_signals)[ref])
        return np.asarray(y, dtype=np.float64), None


def make_processors(nopf=False, only=None):
    """Los procesadores del banco. nopf=True agrega las variantes SIN
    post-filtro (smooth=None): el lazo del OFB alimenta la red con la salida
    ANTES del post, asi que los pesos espaciales son IDENTICOS a los de la
    version con post y la diferencia aisla la mascara."""
    from evaluation.bf_wrappers import NM_MVDR_OFB_AUTO
    procs = {
        "NM_MVDR_OFB_AUTO": NM_MVDR_OFB_AUTO(smooth=0.2, sharpen_exp=8.0),
        "NM_MVDR_OFB_AUTO_BAN": NM_MVDR_OFB_AUTO(smooth=0.2, sharpen_exp=8.0,
                                                 ban=True),
    }
    if nopf:
        procs["NM_MVDR_OFB_AUTO_noPF"] = NM_MVDR_OFB_AUTO(smooth=None, sharpen_exp=8.0)
        procs["NM_MVDR_OFB_AUTO_BAN_noPF"] = NM_MVDR_OFB_AUTO(smooth=None, sharpen_exp=8.0,
                                                              ban=True)
    # Referencias monocanal: DTLN sobre el mic de referencia, como "procesador"
    # para que salgan con las MISMAS escenas y metricas que los beamformers
    # (columnas proc_*/Delta_tot_*). Solo se instancian si se piden por --procs:
    # cargan TF-Lite de forma perezosa.
    for name, quant in (("DTLN_mono_q", True), ("DTLN_mono_f", False)):
        if only and name in only:
            procs[name] = DtlnMono(quant=quant)
    if only:
        falta = [k for k in only if k not in procs]
        if falta:
            raise SystemExit(f"[kria] procesadores desconocidos {falta}; hay {list(procs)}")
        procs = {k: procs[k] for k in only}
    return procs


def load_dtln():
    """(itp1, itp2) o (None, None). El motor corre la etapa neuronal si
    `interpreter_1 is not None`."""
    try:
        import tensorflow as tf
        models = os.path.join(SRC_DIR, "dnn_denoise", "models")
        itp1 = tf.lite.Interpreter(
            model_path=os.path.join(models, "model_quant_1.tflite"))
        itp1.allocate_tensors()
        itp2 = tf.lite.Interpreter(
            model_path=os.path.join(models, "model_quant_2.tflite"))
        itp2.allocate_tensors()
        print("[kria] interpretes DTLN TF-Lite listos.")
        return itp1, itp2
    except Exception as e:                                   # noqa: BLE001
        print(f"[kria] sin DTLN ({e}). Las columnas *_dtln* no van a salir.")
        return None, None


def run_plan(prov, room, pl, audio, a, itp1, itp2, parts_dir, catalog,
             leaderboard, tag=None, show_progress=True, variants=None):
    """Corre UNA posicion de aro (o un pedazo de ella: pl['combos'] puede ser
    un subconjunto). Devuelve (df, base_config)."""
    from evaluation.full_benchmark_test_dtln_mird import run_mird_grid_search
    base = build_config(prov, room, pl["array"], pl["target"], audio,
                        duration=a.duration)
    if tag is not None:
        base['out_prefix'] = f"{base['out_prefix']}_{tag}"
    grid, info = build_grid(room, pl, a.isir, prov=prov, interf_audio=audio,
                            source_path=base['source_path'], variants=variants)
    df = run_mird_grid_search(
        grid, prov, make_processors(getattr(a, "nopf", False), getattr(a, "procs", None)), base,
        output_dir=parts_dir,
        interpreter_1=itp1, interpreter_2=itp2,
        save_catalog=not a.no_catalog, nonintrusive=a.nonintrusive,
        show_progress=show_progress,
        scene_info_fn=lambda exp, _i=info: _i[str(exp['interf_configs'])],
        leaderboard=leaderboard, catalog_path=catalog)
    return df, base


# --- paralelo: un proceso por worker, cada uno con SU provider y SU DTLN ----
_W = {}


def _worker_init(root, use_dtln, log_dir):
    _W["prov"] = KriaRirProvider(**({} if root is None else {"root": root}),
                                 verbose=False)
    _W["log_dir"] = log_dir
    _W["itp"] = load_dtln() if use_dtln else (None, None)


def _worker_run(job):
    """Un job = una combinacion de interferentes de una posicion de aro, con
    todos sus iSIR (asi el cacheo por nodos del motor sigue sirviendo).
    Escribe su propio catalogo; el padre los une al final."""
    import contextlib
    import time
    idx, room, array, target, combo, audio, a, var = job
    log = os.path.join(_W["log_dir"], f"job{idx:03d}.log")
    t0 = time.time()
    with open(log, "w") as fh, contextlib.redirect_stdout(fh), \
            contextlib.redirect_stderr(fh):
        prov = _W["prov"]
        pl = plan_array(prov, room, array, target, [len(combo)], None, a.seed,
                        a.min_sep, getattr(a, "exclude_src", None))
        pl["combos"] = [tuple(combo)]
        cat = (None if a.no_catalog else
               os.path.join(_W["log_dir"], f"catalog_job{idx:03d}.h5"))
        if cat and os.path.exists(cat):
            os.remove(cat)
        itp1, itp2 = _W["itp"]
        df, base = run_plan(prov, room, pl, audio, a, itp1, itp2,
                            _W["log_dir"], cat, {}, tag=f"job{idx:03d}",
                            show_progress=False,
                            variants=None if var is None else [var])
    return idx, df, base, cat, time.time() - t0


TRACKED = ["Delta_tot_PESQ", "Delta_tot_STOI", "Delta_tot_SDR", "Delta_tot_SIR",
           "Delta_tot_SAR", "Delta_tot_SINR", "Delta_tot_CD"]


def merge_catalogs(paths, dest):
    """Une catalogos parciales en uno: para cada (procesador, metrica) se queda
    con el best/worst de TODOS, con el mismo criterio que el leaderboard del
    motor (mayor es mejor salvo CD; ante empate gana el primero en `paths`)."""
    import h5py
    best = {}                       # (proc, metric, case) -> (val, path)
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        with h5py.File(p, "r") as f:
            for proc in f:
                for metric in f[proc]:
                    for case in f[proc][metric]:
                        v = f[proc][metric][case]["metrics"].attrs.get(f"{metric}_early")
                        if v is None or isinstance(v, (str, bytes)):
                            continue
                        v = float(v)
                        low_good = metric == "Delta_tot_CD"
                        want_max = (case == "best_case") != low_good
                        k = (proc, metric, case)
                        if (k not in best or (v > best[k][0] if want_max
                                              else v < best[k][0])):
                            best[k] = (v, p)
    with h5py.File(dest, "w") as out:
        for (proc, metric, case), (_, p) in sorted(best.items()):
            with h5py.File(p, "r") as f:
                parent = out.require_group(f"{proc}/{metric}")
                f.copy(f[f"{proc}/{metric}/{case}"], parent, name=case)
    return len(best)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=None, help="raiz de data/rirs")
    ap.add_argument("--list", action="store_true",
                    help="mostrar las mediciones y salir")
    ap.add_argument("--room", help="la sala: se procesa ENTERA")
    ap.add_argument("--array", nargs="*", default=None,
                    help="posiciones de aro (default: todas las de la sala)")
    ap.add_argument("--target", default="1",
                    help="la fuente objetivo, la misma en todas las posiciones "
                         "de aro (default: 1). 'all' = CADA fuente hace de "
                         "objetivo contra las demas (pares ordenados con K=1)")
    ap.add_argument("--exclude-src", nargs="*", default=[],
                    help="fuentes que no se usan ni como objetivo ni como "
                         "interferente (p.ej. una toma repetida)")
    ap.add_argument("--procs", nargs="*", default=None,
                    help="subconjunto de procesadores por nombre (default: todos)")
    ap.add_argument("--n-interf", type=int, nargs="+", default=[1],
                    help="cuantos interferentes simultaneos; varios valores = "
                         "varias tandas (p.ej. 1 2 3)")
    ap.add_argument("--max-combos", type=int, default=None,
                    help="tope de combinaciones por K y por posicion de aro "
                         "(default: todas). Se eligen balanceadas")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-sep", type=float, default=0.0,
                    help="descartar interferentes a menos de estos grados del "
                         "objetivo (default 0: no se descarta ninguno)")
    ap.add_argument("--interf-audio", nargs="+", default=None,
                    help="senales de interferencia, una por interferente "
                         "simultaneo")
    ap.add_argument("--isir", type=float, nargs="+", default=[0.0])
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "results"))
    ap.add_argument("--dry-run", action="store_true",
                    help="mostrar que se correria y salir")
    ap.add_argument("--no-dtln", action="store_true",
                    help="correr sin la etapa neuronal")
    ap.add_argument("--duration", type=float, default=15,
                    help="segundos de senal por escena (default 15; las "
                         "metricas arrancan en min(5, 0.3*duracion))")
    ap.add_argument("--nonintrusive", action="store_true",
                    help="agregar DNSMOS / SQUIM (ver evaluation/nonintrusive.py)")
    ap.add_argument("--no-catalog", action="store_true",
                    help="no escribir el catalogo H5 (solo las tablas)")
    ap.add_argument("--nopf", action="store_true",
                    help="agregar las variantes SIN post-filtro (ablacion: "
                         "mismos pesos espaciales, sin la mascara a la salida)")
    ap.add_argument("--signal-sweep", action="store_true",
                    help="repetir cada combinacion con cada senal de "
                         "interferencia (rotando: el interferente j recibe la "
                         "senal v+j). Sin --interf-audio usa SWEEP_AUDIO")
    ap.add_argument("--workers", type=int, default=1,
                    help="procesos en paralelo (default 1). Cada uno ~2.2 GB "
                         "de RAM con DTLN; reparte las combinaciones de "
                         "interferentes. Mismo resultado que en serie.")
    a = ap.parse_args(argv)

    prov = KriaRirProvider(**({} if a.root is None else {"root": a.root}))
    if a.list or not a.room:
        print(prov.describe())
        print("\nPara correr hace falta --room (y opcionalmente --array, "
              "--target, --n-interf).")
        return 0
    if a.room not in prov.rooms():
        raise SystemExit(f"[kria] no existe la sala '{a.room}'. Hay: {prov.rooms()}")

    arrays = a.array or prov.arrays(a.room)
    excl = set(a.exclude_src or ())
    plans = [pl for pl in (plan_array(prov, a.room, arr, t, a.n_interf,
                                      a.max_combos, a.seed, a.min_sep, excl)
                           for arr in arrays
                           for t in ([x for x in prov.sources(a.room, arr) if x not in excl]
                                     if a.target == "all" else [a.target]))
             if pl and pl["combos"]]
    if not plans:
        raise SystemExit("[kria] no quedo ninguna escena para correr.")
    audio = [p if os.path.isabs(p) else os.path.join(SIGNALS, p)
             for p in (a.interf_audio or
                       (SWEEP_AUDIO if a.signal_sweep else DEFAULT_INTERF_AUDIO))]
    variants = list(range(len(audio))) if a.signal_sweep else None
    kmax = max(len(c) for pl in plans for c in pl["combos"])
    faltan = [p for p in (audio if a.signal_sweep else audio[:kmax])
              if not os.path.exists(p)]
    if kmax > len(audio) or faltan:
        raise SystemExit(
            f"[kria] hacen falta {kmax} senales de interferencia DISTINTAS y "
            f"hay {len(audio)}" + (f"; no existen: {faltan}" if faltan else "") +
            ". Pasarlas con --interf-audio.")
    print_plan(a.room, plans, a.isir, len(a.procs) if a.procs else (4 if a.nopf else 2),
               len(variants) if variants else 1)
    if a.signal_sweep:
        print("[kria] senales: " + ", ".join(
            f"{os.path.basename(p)} ({audio_spectrum(p)['audio_low_pct']:.0f} % < 300 Hz)"
            for p in audio))
    if a.dry_run:
        return 0

    import pandas as pd

    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, f"kria_{a.room}")
    parts_dir = stem + "_parts"
    os.makedirs(parts_dir, exist_ok=True)
    catalog = stem + "_benchmark_catalog.h5"
    if not a.no_catalog and os.path.exists(catalog):
        print(f"[kria] se regenera {catalog} (el de la corrida anterior se reemplaza).")
        os.remove(catalog)

    dfs, bases = [], {}
    use_dtln = not a.no_dtln
    if a.workers <= 1:
        itp1, itp2 = load_dtln() if use_dtln else (None, None)
        use_dtln = itp1 is not None
        # UN leaderboard para toda la sala: el motor se llama una vez por aro,
        # pero el best/worst del catalogo es sobre TODAS las escenas.
        leaderboard = {}
        for pl in plans:
            df, base = run_plan(prov, a.room, pl, audio, a, itp1, itp2,
                                parts_dir, None if a.no_catalog else catalog,
                                leaderboard, variants=variants,
                                tag=f"t{pl['target']}" if a.target == "all" else None)
            dfs.append(df)
            bases[f"{pl['array']}/{pl['target']}"] = base
    else:
        import multiprocessing as mp
        import time
        # Cada proceso usa pocos hilos: el paralelismo esta en los procesos.
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS"):
            os.environ[v] = str(max(1, (os.cpu_count() or 2) // a.workers))
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        jobs = [(i, a.room, pl["array"], pl["target"], list(c), audio, a, v)
                for i, (pl, c, v) in enumerate(
                    (pl, c, v) for pl in plans for c in pl["combos"]
                    for v in (variants or [None]))]
        # las de mas interferentes primero: son las mas largas
        order = sorted(jobs, key=lambda j: -len(j[4]))
        print(f"[kria] {len(jobs)} jobs en {a.workers} procesos "
              f"(logs en {parts_dir}/job*.log)")
        res, t0 = {}, time.time()
        ctx = mp.get_context("spawn")
        with ctx.Pool(a.workers, initializer=_worker_init,
                      initargs=(a.root, use_dtln, parts_dir)) as pool:
            for k, (idx, df, base, cat, dt) in enumerate(
                    pool.imap_unordered(_worker_run, order), 1):
                res[idx] = (df, base, cat)
                sig = ("" if jobs[idx][7] is None else
                       f" senal v{jobs[idx][7]}")
                print(f"[kria] {k}/{len(jobs)} job{idx:03d} {base['mird_spacing']} "
                      f"{jobs[idx][4]}{sig} ({dt:.0f} s, total {time.time() - t0:.0f} s)",
                      flush=True)
        dfs = [res[i][0] for i in sorted(res)]
        bases = {f"{jobs[i][2]}/{jobs[i][3]}": res[i][1] for i in sorted(res)}
        use_dtln = any(c.startswith("dtln_alone_") for c in dfs[0].columns)
        if not a.no_catalog:
            cats = [res[i][2] for i in sorted(res)]
            n = merge_catalogs(cats, catalog)
            print(f"[kria] catalogo unido: {n} casos de {len(cats)} parciales")
            for c in cats:
                if c and os.path.exists(c):
                    os.remove(c)

    full = pd.concat(dfs, ignore_index=True)
    for col in full.columns:            # mismo criterio que el motor
        if full[col].apply(lambda x: isinstance(x, (list, tuple))).any():
            full[col] = full[col].astype(str)
    full.to_csv(stem + "_benchmark_metrics.csv", index=False)
    full.to_parquet(stem + "_benchmark_metrics.parquet", index=False)

    src = pd.DataFrame(sources_table(prov, a.room, plans))
    src.to_csv(stem + "_sources.csv", index=False)
    src.to_parquet(stem + "_sources.parquet", index=False)

    run_info = {
        "script": os.path.relpath(os.path.abspath(__file__), PROJECT_ROOT),
        "argv": list(sys.argv[1:] if argv is None else argv),
        "args": vars(a),
        "when": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_beamformer": _git_rev(PROJECT_ROOT),
        "git_dataset": _git_rev(prov.root),
        "dataset_root": prov.root,
        "room": a.room,
        "arrays": [pl["array"] for pl in plans],
        "target": a.target,
        "interf_audio": [os.path.basename(p) for p in audio],
        "dtln": bool(use_dtln),
        "processors": {k: _jsonable(getattr(v, "__dict__", repr(v)))
                       for k, v in make_processors(a.nopf, a.procs).items()},
        "base_config": {arr: {k: _jsonable(v) for k, v in b.items()
                              if k not in ("mic_coords", "source_pos",
                                           "interferences_pos", "VAD",
                                           "oracle_target", "oracle_noise")}
                        for arr, b in bases.items()},
        "n_rows": int(len(full)),
        "n_scenes": int(full["scene_id"].nunique()) if "scene_id" in full else None,
        "outputs": {"catalog": None if a.no_catalog else os.path.basename(catalog),
                    "metrics": os.path.basename(stem) + "_benchmark_metrics.parquet",
                    "sources": os.path.basename(stem) + "_sources.parquet"},
    }
    with open(stem + "_run.json", "w") as fh:
        json.dump(run_info, fh, indent=2, ensure_ascii=False)
    if not a.no_catalog and os.path.exists(catalog):
        import h5py
        with h5py.File(catalog, "a") as f:
            f.attrs["run_info"] = json.dumps(run_info, ensure_ascii=False)
            f.attrs["dataset"] = "kria"
            f.attrs["room"] = a.room

    print(f"\n[kria] {len(full)} filas x {full.shape[1]} columnas -> "
          f"{stem}_benchmark_metrics.parquet")
    print(f"[kria] {len(src)} mediciones -> {stem}_sources.parquet")
    if not a.no_catalog:
        print(f"[kria] catalogo -> {catalog}")
    cols = [c for c in ("array", "target_src", "interf_srcs", "sep_min_deg",
                        "tgt_T_wide", "tgt_DRR_wide", "processor", "isir_db",
                        "Delta_tot_PESQ_early", "Delta_tot_SIR_early",
                        "Delta_tot_STOI_early") if c in full.columns]
    if cols:
        print(full[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
