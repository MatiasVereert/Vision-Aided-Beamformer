#!/usr/bin/env python3
"""full_benchmark_test_dtln_scenes.py -- el banco sobre ESCENAS REALES grabadas.

Tercer gemelo de full_benchmark_test_dtln_mird.py / _kria.py. Las dos primeras
arman mezclas con RIRs y miden con referencia limpia; esta toma grabaciones
REALES de mic-array-platform/tools/scene_lab (una persona hablandole al aro con
musica o ruido de fondo, el OFB corriendo EN LA KRIA) y, como no hay senal
limpia, mide con metricas NO INTRUSIVAS (DNSMOS P.835 + TorchAudio-SQUIM,
evaluation/nonintrusive.py) y con NIVELES por tramo.

QUE SE COMPARA, POR TOMA

    base          el mic de referencia crudo (ch5: el del OFB con M = 11)
    OFB_kria      la salida que mando la Kria (ch12): el sistema en tiempo real
    DTLN_mono_f   la linea de base: DTLN mono FLOAT (model_1 + model_2) sobre el
                  mismo mic de referencia, con la MISMA GANANCIA FIJA que la
                  Kria le aplica al OFB (+35 dB, sin normalizar; se deshace a
                  la salida): lo que haria un DTLN corriendo en el dispositivo.
    DTLN_mono_pk  control: el mismo DTLN normalizado por el PICO DEL ARCHIVO,
                  como DtlnMono del banco de la Kria. No es causal (un AGC que
                  conoce el futuro); con --no-dtln-pk no se corre.
    --procs X     ademas, cualquier procesador del banco corrido OFFLINE sobre
                  los mismos 11 crudos CON LA GANANCIA DE LA KRIA (entrada x g,
                  salida / g, como OfbStream). P.ej. NM_MVDR_OFB_AUTO: la
                  version Python del OFB; control de paridad con OFB_kria.

NIVEL DE ENTRADA. El OFB (lazo int8) rinde parejo con la entrada entre -20 y
+6 dB del pico normalizado de los bancos y colapsa por debajo de ~-30 dB
(tests/ofb_level_sweep.py en simulacion; confirmado en tomas reales
2026-11-20). La Kria NO normaliza: aplica +35 dB fijos, asi que el punto de
operacion depende de lo fuerte que hable el locutor y de la distancia. Cada
fila lleva in_level_mics_db / in_level_ref_db (0 = el punto de los bancos) e
in_level_ok; fuera de [-20, +6] dB la toma se avisa.

DONDE SE MIDE. En la ventana 'Ce' que calculo scene_lab al grabar: el tramo C
(voz + interferencia) sin sus primeros 5 s -- el mismo descarte que el banco --
y sin un margen de 0.5 s en el borde final. Las metricas no intrusivas se
normalizan a -26 dBFS antes de puntuar (nonintrusive.prepare_nonintrusive), asi
que la ganancia de cada procesador no entra.
Con --c-skip se recalcula la ventana con otro descarte.

NIVELES (mismas ventanas, en dBFS sin ganancia; el ch12 y los crudos estan en
la misma escala, asi que las diferencias son atenuaciones reales):
    *_L_{A,B,Ce,D}_dbfs   nivel por tramo
    *_nr_B_db / nr_D_db   lo que baja la interferencia SOLA respecto del mic
                          de referencia, antes (B) y despues (D) de la voz
    *_snr_est_db          (P_Ce - P_N) / P_N, con N = B + D (estimado: supone
                          la interferencia de C con la potencia de B y D)

SALIDA (en --out, mismos nombres y convenciones que los otros dos):

  scenes_<sala>_benchmark_metrics.{parquet,csv}  una fila por (toma, procesador)
      con las columnas del banco para no intrusivas:
          base_<K>, proc_<K>, Delta_tot_<K> = proc - base
          dtln_alone_<K>, Delta_dtln_alone_<K> = dtln - base
          Delta_vs_dtln_<K> = proc - dtln          (la comparacion de la tesis)
      con K en NONINTRUSIVE_KEYS, mas los niveles (base_/proc_/dtln_alone_ +
      L_*, nr_*, snr_est_db), la escena de la toma (take_*, interf, speaker,
      target_src, ...) y los descriptores medidos por rir_lab de la posicion del
      locutor (tgt_*) y del parlante (int_*) si existen.
  scenes_<sala>_takes.{parquet,csv}   una fila por toma (escena, plan,
      duraciones reales, niveles, captura, lo que reporto la Kria).
  scenes_<sala>_run.json              argumentos, commit, procesadores.
  scenes_<sala>_audio/                con --save-audio: los wav de cada toma.

USO
    conda run -n tesis_beam python src/evaluation/full_benchmark_test_dtln_scenes.py --list
    conda run -n tesis_beam python src/evaluation/full_benchmark_test_dtln_scenes.py \\
        --room Evaluacion_cuarto
    ... --room Evaluacion_cuarto --procs NM_MVDR_OFB_AUTO --save-audio
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
for _p in (SRC_DIR, SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from propagation.kria_scene_loader import KriaSceneProvider, DEFAULT_ROOT  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
DTLN_LATENCY = 384        # bloque 512 / salto 128: la salida sale 24 ms tarde
BASE, OFB, DTLN, DTLN_PK = "base", "OFB_kria", "DTLN_mono_f", "DTLN_mono_pk"
LEVEL_OK_DB = (-20.0, 6.0)   # meseta del OFB respecto del pico normalizado


# ----------------------------------------------------------------- niveles
def _pow(x, w):
    if w is None:
        return None
    a, b = int(w[0]), int(w[1])
    if b - a <= 0:
        return None
    y = np.asarray(x[a:b], dtype=np.float64)
    return float(np.mean(y * y))


def _db(p):
    return None if p is None or p <= 0 else float(10 * np.log10(p))


def level_metrics(y, ref, win):
    """Niveles de `y` por tramo y lo que baja respecto de `ref`."""
    out = {}
    pw = {k: _pow(y, win.get(k)) for k in ("A", "B", "Ce", "D")}
    pr = {k: _pow(ref, win.get(k)) for k in ("A", "B", "Ce", "D")}
    for k, v in pw.items():
        out[f"L_{k}_dbfs"] = _db(v)
    for k in ("B", "D", "Ce"):
        a, b = _db(pr[k]), _db(pw[k])
        out[f"nr_{k}_db"] = None if a is None or b is None else a - b
    ns = [(pw[k], win.get(k)) for k in ("B", "D") if pw[k] is not None]
    if ns and pw["Ce"] is not None:
        tot = sum(w[1] - w[0] for _, w in ns)
        pn = sum(p * (w[1] - w[0]) for p, w in ns) / tot
        out["snr_est_db"] = float(10 * np.log10(max(pw["Ce"] - pn, pn * 1e-6) / pn))
    else:
        out["snr_est_db"] = None
    return out


def eval_windows(t, c_skip=None, guard=0.5):
    """Las ventanas que guardo scene_lab, o recalculadas con otro descarte."""
    win = dict(t.windows)
    if c_skip is not None:
        c = t.segments.get("C")
        if c and c[0] is not None and c[1] is not None:
            a = int(c[0] + c_skip * t.fs)
            b = int(c[1] - guard * t.fs)
            win["Ce"] = [a, b] if b - a > int(0.25 * t.fs) else None
    return win


# ------------------------------------------------------------ procesadores
def offline_processors(names):
    if not names:
        return {}
    from evaluation.full_benchmark_test_dtln_kria import make_processors
    return make_processors(only=names)


def run_dtln(dtln, t, normalize):
    """DTLN mono sobre la referencia. normalize=False: con la ganancia fija de
    la Kria (entrada x g, salida / g); True: normalizado por el pico del
    archivo (DtlnMono del banco)."""
    from dnn_denoise.dtln_lite import apply_dtln_post_tflite_realtime
    g = 1.0 if normalize else 10 ** (t.kria_gain_db / 20)
    i1, i2 = dtln._interpreters()
    y = apply_dtln_post_tflite_realtime(i1, i2, t.ref() * g, normalize=normalize)
    y = np.asarray(y, dtype=np.float64) / g
    return np.concatenate([y[DTLN_LATENCY:], np.zeros(DTLN_LATENCY)])


def run_offline(proc, t):
    """Un procesador del banco sobre los crudos, con la MISMA ganancia fija
    que la Kria (entrada x g, salida / g, como OfbStream). Sin ella la entrada
    queda ~34 dB debajo del punto de operacion y el OFB colapsa."""
    # mismas claves que build_config() del banco de la Kria
    cfg = {"fs": t.fs, "ref_mic_idx": t.ref_index, "ref_mic_mode": t.ref_index,
           "mic_coords": t.mic_coords(), "stft_window": 512, "stft_overlap": 384,
           "dtln_model_path": os.path.join(SRC_DIR, "dnn_denoise", "models",
                                           "model_quant_1.tflite")}
    g = 10 ** (t.kria_gain_db / 20)
    x = t.mics()
    y, _ = proc.process(x * g, cfg)
    y = np.asarray(y, dtype=np.float64).squeeze() / g
    n = x.shape[1]
    return y[:n] if len(y) >= n else np.concatenate([y, np.zeros(n - len(y))])


# ------------------------------------------------------------ descriptores
def rir_descriptors(rir_prov, t):
    """tgt_* / int_*: lo que rir_lab midio en las posiciones del locutor y del
    parlante de esta toma (mismos nombres de sala y aro)."""
    out = {}
    if rir_prov is None:
        return out
    for pre, src in (("tgt_", t.scene.get("target_src")),
                     ("int_", t.scene.get("interf_src"))):
        if not src:
            continue
        try:
            m = rir_prov.get(t.room, t.array, str(src))
        except Exception:                                    # noqa: BLE001
            out[pre + "found"] = False
            continue
        out[pre + "found"] = True
        out.update({pre + k: v for k, v in m.descriptors().items()})
        try:
            dist, az, el, origin = m.position()      # marco interno del aro
            out.update({pre + "dist_m": dist, pre + "az_deg": az,
                        pre + "el_deg": el, pre + "dir_origin": origin,
                        pre + "dist_origin": m.dist_origin})
        except Exception:                                    # noqa: BLE001
            pass
    return out


def _git_rev(path):
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except Exception:                                        # noqa: BLE001
        return None


def _save_wav(path, y, fs):
    import soundfile as sf
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pk = float(np.max(np.abs(y))) if len(y) else 0.0
    sf.write(path, (y / pk * 0.9 if pk > 0 else y).astype(np.float32), fs)


# -------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=DEFAULT_ROOT, help="raiz de data/escenas")
    ap.add_argument("--rir-root", default=None,
                    help="raiz de data/rirs para los descriptores (default: "
                         "la de kria_rir_loader)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--room", help="la sala (default: todas, una tabla por sala)")
    ap.add_argument("--array", default=None)
    ap.add_argument("--procs", nargs="*", default=[],
                    help="procesadores del banco a correr OFFLINE sobre los crudos")
    ap.add_argument("--no-dtln", action="store_true",
                    help="sin la linea de base DTLN mono")
    ap.add_argument("--no-dtln-pk", action="store_true",
                    help="sin el control DTLN normalizado por pico")
    ap.add_argument("--no-ofb", action="store_true",
                    help="sin la fila de la salida de la Kria")
    ap.add_argument("--include-bad", action="store_true",
                    help="incluir tomas MARCADAS (huecos, CRC, sanidad)")
    ap.add_argument("--c-skip", type=float, default=None,
                    help="recalcular la ventana de evaluacion descartando estos "
                         "segundos del inicio de C (default: la de la grabacion, 5 s)")
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "results"))
    ap.add_argument("--save-audio", action="store_true")
    a = ap.parse_args(argv)

    prov = KriaSceneProvider(a.root)
    if a.list:
        print(prov.describe())
        return 0
    rooms = [a.room] if a.room else prov.rooms()
    if a.room and a.room not in prov.rooms():
        raise SystemExit(f"[escenas] no existe la sala '{a.room}'. Hay: {prov.rooms()}")

    import pandas as pd
    from evaluation.nonintrusive import (score_nonintrusive, nonintrusive_status,
                                         NONINTRUSIVE_KEYS)
    st = nonintrusive_status()
    print(f"[escenas] DNSMOS {'ok' if st['dnsmos'] else 'NO DISPONIBLE'} · "
          f"SQUIM {'ok' if st['squim'] else 'NO DISPONIBLE'}")

    rir_prov = None
    try:
        from propagation.kria_rir_loader import KriaRirProvider
        rir_prov = KriaRirProvider(**({"root": a.rir_root} if a.rir_root else {}),
                                   verbose=False)
    except Exception as e:                                   # noqa: BLE001
        print(f"[escenas] sin descriptores de rir_lab ({e})")

    dtln = None
    if not a.no_dtln:
        from evaluation.full_benchmark_test_dtln_kria import DtlnMono
        dtln = DtlnMono(quant=False)
    procs = offline_processors(a.procs)
    os.makedirs(a.out, exist_ok=True)

    for room in rooms:
        takes = list(prov.iter_takes(room, a.array, include_bad=a.include_bad))
        if not takes:
            print(f"[escenas] {room}: ninguna toma utilizable")
            continue
        print(f"[escenas] {room}: {len(takes)} tomas")
        rows, trows = [], []
        stem = os.path.join(a.out, f"scenes_{room}")
        for i, t in enumerate(takes, 1):
            t0 = time.time()
            win = eval_windows(t, a.c_skip)
            ce = win.get("Ce")
            fs = t.fs
            ref = t.ref()
            sigs = {}
            if not a.no_ofb:
                sigs[OFB] = t.ofb()
            for name, p in procs.items():
                try:
                    sigs[name] = run_offline(p, t)
                except Exception as e:                       # noqa: BLE001
                    print(f"  [!!] {t.key} {name}: {type(e).__name__}: {e}")
            y_dtln = run_dtln(dtln, t, normalize=False) if dtln is not None else None
            if y_dtln is not None:
                sigs[DTLN] = y_dtln
                if not a.no_dtln_pk:
                    sigs[DTLN_PK] = run_dtln(dtln, t, normalize=True)
            lev = t.input_levels()
            lo, hi = LEVEL_OK_DB
            lev_ok = all(v is not None and lo <= v <= hi for v in lev.values())
            if not lev_ok:
                print(f"  [!!] {t.key}: entrada fuera de la meseta del OFB "
                      f"(mics {lev['mics']:+.1f} dB, ref {lev['ref']:+.1f} dB "
                      f"respecto del pico normalizado; meseta [{lo:g}, {hi:+g}])")

            def ni(y):
                if ce is None:
                    return {}
                return score_nonintrusive(y[ce[0]:ce[1]], fs)

            base_ni, base_lv = ni(ref), level_metrics(ref, ref, win)
            dt_ni = ni(y_dtln) if y_dtln is not None else {}
            dt_lv = level_metrics(y_dtln, ref, win) if y_dtln is not None else {}
            tsum = t.summary()
            tsum.update({"eval_s": None if ce is None else (ce[1] - ce[0]) / fs,
                         "kria_gain_db": t.kria_gain_db,
                         "in_level_mics_db": lev["mics"],
                         "in_level_ref_db": lev["ref"], "in_level_ok": lev_ok})
            desc = rir_descriptors(rir_prov, t)
            tsum.update(desc)
            trows.append(tsum)
            for name, y in sigs.items():
                pr_ni = dt_ni if name == DTLN else ni(y)
                pr_lv = dt_lv if name == DTLN else level_metrics(y, ref, win)
                r = {"scene_id": t.key, "room": t.room, "array": t.array,
                     "take": t.take, "processor": name, "is_realtime": name == OFB,
                     "eval_s": tsum["eval_s"], "ref_mic": t.ref_mic,
                     "M": len(t.active), "kria_gain_db": t.kria_gain_db,
                     "in_level_mics_db": lev["mics"],
                     "in_level_ref_db": lev["ref"], "in_level_ok": lev_ok}
                for k in ("interf", "interf_kind", "interf_sha1",
                          "interf_gain_db", "speaker",
                          "target_src", "interf_src", "ok", "c_ended_by",
                          "hop_mean_ms", "hops_late", "take_snr_in_db",
                          "take_inr_db", "take_nr_B_db", "take_nr_D_db"):
                    r[k] = tsum.get(k)
                for k in NONINTRUSIVE_KEYS:
                    b = base_ni.get(k, np.nan)
                    p = pr_ni.get(k, np.nan)
                    d = dt_ni.get(k, np.nan)
                    r[f"base_{k}"] = b
                    r[f"proc_{k}"] = p
                    r[f"Delta_tot_{k}"] = p - b
                    r[f"dtln_alone_{k}"] = d
                    r[f"Delta_dtln_alone_{k}"] = d - b
                    r[f"Delta_vs_dtln_{k}"] = p - d
                for pre, lvd in (("base_", base_lv), ("proc_", pr_lv),
                                 ("dtln_alone_", dt_lv)):
                    for k, v in lvd.items():
                        r[pre + k] = v
                r.update(desc)
                rows.append(r)
            if a.save_audio:
                d = os.path.join(stem + "_audio", t.array, t.take)
                _save_wav(os.path.join(d, "base_ref.wav"), ref, fs)
                for name, y in sigs.items():
                    _save_wav(os.path.join(d, f"{name}.wav"), y, fs)
            t.unload()
            ofb_row = next((r for r in rows[::-1] if r["scene_id"] == t.key
                            and r["processor"] == OFB), None)
            msg = ""
            if ofb_row is not None:
                msg = (f" OVRL base {ofb_row['base_DNSMOS_OVRL']:.2f} OFB "
                       f"{ofb_row['proc_DNSMOS_OVRL']:.2f} DTLN "
                       f"{ofb_row['dtln_alone_DNSMOS_OVRL']:.2f}")
            print(f"[escenas] {i}/{len(takes)} {t.key} ({t.meta['interference']['name']})"
                  f"{msg} ({time.time() - t0:.0f} s)", flush=True)

        full = pd.DataFrame(rows)
        tk = pd.DataFrame(trows)
        for df in (full, tk):
            for col in df.columns:
                if df[col].apply(lambda x: isinstance(x, (list, tuple, dict))).any():
                    df[col] = df[col].astype(str)
        full.to_csv(stem + "_benchmark_metrics.csv", index=False)
        full.to_parquet(stem + "_benchmark_metrics.parquet", index=False)
        tk.to_csv(stem + "_takes.csv", index=False)
        tk.to_parquet(stem + "_takes.parquet", index=False)
        run_info = {
            "script": os.path.relpath(os.path.abspath(__file__), PROJECT_ROOT),
            "argv": list(sys.argv[1:] if argv is None else argv),
            "args": vars(a),
            "when": datetime.datetime.now().isoformat(timespec="seconds"),
            "git_beamformer": _git_rev(PROJECT_ROOT),
            "git_dataset": _git_rev(prov.root),
            "dataset_root": prov.root, "room": room,
            "takes": [t.key for t in takes],
            "processors": ([OFB] if not a.no_ofb else []) + list(procs) +
                          ([DTLN] if dtln is not None else []) +
                          ([DTLN_PK] if dtln is not None and not a.no_dtln_pk else []),
            "baseline": "DTLN float (model_1 + model_2) sobre el mic de "
                        "referencia con la ganancia fija de la Kria, sin "
                        "normalizar; DTLN_mono_pk = control normalizado por pico",
            "offline_gain": "procesadores --procs con la ganancia de la Kria "
                            "(entrada x g, salida / g)",
            "nonintrusive": st,
            "n_rows": int(len(full)),
            "outputs": {"metrics": os.path.basename(stem) + "_benchmark_metrics.parquet",
                        "takes": os.path.basename(stem) + "_takes.parquet"},
        }
        with open(stem + "_run.json", "w") as fh:
            json.dump(run_info, fh, indent=2, ensure_ascii=False, default=str)
        print(f"[escenas] {len(full)} filas -> {stem}_benchmark_metrics.parquet")
        cols = [c for c in ("array", "take", "interf", "processor", "eval_s",
                            "take_snr_in_db", "proc_DNSMOS_OVRL",
                            "Delta_tot_DNSMOS_OVRL", "Delta_vs_dtln_DNSMOS_OVRL",
                            "proc_nr_D_db") if c in full.columns]
        if cols:
            print(full[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
