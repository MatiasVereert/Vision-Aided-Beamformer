#!/usr/bin/env python3
"""kria_scene_loader.py -- ESCENAS DE USO grabadas con el OFB en tiempo real.

Hermano de kria_rir_loader.py, para el otro dataset de la prueba final. Aca no
hay RIRs ni mezclas: cada toma es una grabacion REAL de mic-array-platform/
tools/scene_lab -- una persona hablandole al aro con musica/ruido de fondo --
con los 12 canales crudos y la salida del OFB que corrio EN LA KRIA (canal 12,
ya alineada con los crudos por el server).

    data/escenas/<sala>/<posicion de aro>/<toma>__<fecha>.npz

Cada toma sigue la secuencia A (silencio) / B (interferencia) / C (voz +
interferencia) / D (interferencia, sin voz) y trae los limites EN FRAMES
(meta['segments']) y las ventanas de medicion ya recortadas
(meta['levels']['windows']; 'Ce' = la voz que se evalua, sin los primeros 5 s).

LA INTERFERENCIA es un archivo reproducido por el parlante
(meta['interference']['kind'] = "file", con sha1 y ganancia) o una fuente
EXTERNA prendida a mano -- licuadora, secador -- ("external": sin archivo ni
ganancia; hay un tramo extra 'on' = la transicion entre A y B, que no se mide).

AUTOSUFICIENTE, como kria_rir_loader: lee todo del .npz (audio, tramos,
geometria, escena, log de la Kria); no importa nada de mic-array-platform.

LOS NOMBRES DE SALA Y ARO SON LOS DE data/rirs: con target_src / interf_src
(las fuentes de rir_lab donde estaban el locutor y el parlante) cada toma se
cruza con los descriptores medidos (T30, DRR, CDR) de esas posiciones.

ESCALA. raw / 2^31: la misma que usa todo el banco (soundfile, ofb_offline).
El ch12 viene en la MISMA escala que los crudos (el server deshace su ganancia
de entrada de 35 dB antes de mandar).

UNA TOMA REPETIDA (mismo numero) deja dos archivos: get() devuelve la ULTIMA
sin problemas de captura. Las tomas con ok = False (huecos, CRC, sanidad) no
se devuelven salvo include_bad=True.
"""
import json
import os

import numpy as np

DEFAULT_ROOT = os.path.expanduser(
    "~/Documents/Tesis/mic-array-platform/data/escenas")
FULL_SCALE = 2.0 ** 31


class SceneTake:
    def __init__(self, path, meta):
        self.path = path
        self.meta = meta
        self._raw = None

    # ------------------------------------------------------------ escena
    @property
    def scene(self):
        return self.meta["scene"]

    @property
    def room(self):
        return self.scene["room"]

    @property
    def array(self):
        return self.scene["array"]

    @property
    def take(self):
        return str(self.scene["take"])

    @property
    def key(self):
        return f"{self.room}/{self.array}/{self.take}"

    @property
    def fs(self):
        return int(self.meta["fs"])

    @property
    def ok(self):
        return bool(self.meta.get("ok", True))

    @property
    def out_ch(self):
        return int(self.meta.get("out_ch", 12))

    @property
    def mics_off(self):
        return [int(c) for c in self.meta.get("mics_off", [])]

    @property
    def active(self):
        """Canales crudos que entran (los 12 menos los anulados), en orden."""
        return [c for c in range(self.out_ch) if c not in set(self.mics_off)]

    @property
    def ref_mic(self):
        """Canal crudo de referencia: el que uso el OFB (lo dice la Kria)."""
        return int(self.meta.get("ref_mic", 5))

    @property
    def ref_index(self):
        """Indice del mic de referencia DENTRO de `active` (lo que se le pasa
        a los procesadores como scene_config['ref_mic_idx'])."""
        return self.active.index(self.ref_mic)

    @property
    def segments(self):
        return self.meta["segments"]

    @property
    def windows(self):
        return (self.meta.get("levels") or {}).get("windows") or {}

    # ------------------------------------------------------------- audio
    def _load(self):
        if self._raw is None:
            with np.load(self.path, allow_pickle=False) as z:
                self._raw = z["raw"]
        return self._raw

    def mics(self, active_only=True):
        """(M, N) float64 en x / 2^31."""
        x = self._load()
        sel = self.active if active_only else list(range(self.out_ch))
        return np.ascontiguousarray(x[:, sel].T, dtype=np.float64) / FULL_SCALE

    def ref(self):
        return self._load()[:, self.ref_mic].astype(np.float64) / FULL_SCALE

    def ofb(self):
        """La salida del OFB que corrio en la Kria (N,)."""
        return self._load()[:, self.out_ch].astype(np.float64) / FULL_SCALE

    @property
    def kria_gain_db(self):
        """Ganancia de entrada FIJA que el server aplico antes del OFB (y
        deshizo antes de mandar el ch12): la del log de la Kria, o la de la
        configuracion con que se lanzo, o 35 dB."""
        for d in (self.meta.get("kria") or {}, self.meta.get("kria_cfg") or {}):
            if d.get("gain_db") is not None:
                return float(d["gain_db"])
        return 35.0

    def input_levels(self, norm_peak=0.99):
        """Cuantos dB queda la entrada CON la ganancia de la Kria respecto de
        la normalizacion por pico de los bancos: 'mics' = pico global de los
        crudos activos contra 0.99 (mix_and_normalize, lo que ve el OFB) y
        'ref' = pico del mic de referencia contra 1.0 (lo que ve el DTLN).
        0 = el punto de operacion de los bancos."""
        g = self.kria_gain_db
        pm = float(np.max(np.abs(self.mics())))
        pr = float(np.max(np.abs(self.ref())))
        lv = lambda p, n: None if p <= 0 else float(20 * np.log10(p / n) + g)
        return {"mics": lv(pm, norm_peak), "ref": lv(pr, 1.0)}

    def unload(self):
        self._raw = None

    # --------------------------------------------------------- geometria
    def mic_coords(self, active_only=True):
        """(M, 3) en metros, marco INTERNO del aro (como array_geometry() de
        kria_rir_loader), centrado en el aro."""
        pos = np.asarray(self.meta["geometry"]["positions"], dtype=np.float64)
        return pos[self.active] if active_only else pos

    # ------------------------------------------------------ lo que se midio
    def levels(self):
        return self.meta.get("levels") or {}

    def kria(self):
        return self.meta.get("kria") or {}

    def summary(self):
        """Una fila por toma: escena, plan, niveles, captura y Kria."""
        sc, it = self.scene, self.meta.get("interference") or {}
        lv = self.levels()
        d = lv.get("derived") or {}
        k = self.kria()
        ks = k.get("summary") or {}
        st = self.meta.get("stream") or {}
        out = {"room": self.room, "array": self.array, "take": self.take,
               "path": self.path, "when": self.meta.get("when"), "ok": self.ok,
               "speaker": sc.get("speaker"), "target_src": sc.get("target_src"),
               "interf_src": sc.get("interf_src"), "text_words": sc.get("text_words"),
               "notes": sc.get("notes"),
               "interf": it.get("name"), "interf_kind": it.get("kind", "file"),
               "interf_notes": it.get("notes"), "interf_sha1": it.get("sha1"),
               "interf_gain_db": it.get("gain_db"),
               "ref_mic": self.ref_mic, "M": len(self.active),
               "mics_off": json.dumps(self.mics_off),
               "c_ended_by": (self.meta.get("events") or {}).get("c_ended_by"),
               "b0_source": (self.meta.get("segments_src") or {}).get("b0"),
               "gaps": len(self.meta.get("gaps") or []),
               "crc_errors": st.get("crc_errors"), "blocks": st.get("blocks"),
               "kria_M": k.get("M"), "kria_ref": k.get("ref_mic"),
               "kria_model": k.get("model"), "kria_gain_db": k.get("gain_db"),
               "kria_P": k.get("P"),
               "hop_mean_ms": ks.get("hop_mean_ms"), "hop_max_ms": ks.get("hop_max_ms"),
               "hops_late": ks.get("hops_late"), "lag_max_pct": ks.get("lag_max_pct"),
               "send_blocked_max_ms": ks.get("send_blocked_max_ms"),
               "kria_lost_frames": ks.get("lost_frames"),
               "warnings": json.dumps(self.meta.get("warnings") or [],
                                      ensure_ascii=False)}
        for s, v in (self.meta.get("dur_s") or {}).items():
            out[f"dur_{s}_s"] = v
        for s, v in (self.meta.get("plan") or {}).items():
            out[f"plan_{s}_s"] = v
        for s, v in (lv.get("ref") or {}).items():
            out[f"ref_L_{s}_dbfs"] = v
        for s, v in (lv.get("out") or {}).items():
            out[f"ofb_L_{s}_dbfs"] = v
        out.update({f"take_{k2}": v for k2, v in d.items()})
        return out

    def __repr__(self):
        return f"SceneTake({self.key}, ok={self.ok}, {os.path.basename(self.path)})"


class KriaSceneProvider:
    """Indice de data/escenas: sala -> aro -> toma -> [archivos]."""

    def __init__(self, root=DEFAULT_ROOT, verbose=True):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.registry = {}
        self.broken = []
        self._index(verbose)

    def _index(self, verbose):
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"no existe el dataset de escenas {self.root}")
        n = 0
        for dirpath, dirs, files in os.walk(self.root):
            dirs.sort()
            if os.path.basename(dirpath).startswith("_"):
                continue
            for fn in sorted(files):
                if not fn.endswith(".npz"):
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    with np.load(p, allow_pickle=False) as z:
                        meta = json.loads(str(z["meta"]))
                    if meta.get("kind") != "scene_take":
                        continue
                    t = SceneTake(p, meta)
                    self.registry.setdefault(t.room, {}).setdefault(
                        t.array, {}).setdefault(t.take, []).append(t)
                    n += 1
                except Exception as e:                       # noqa: BLE001
                    self.broken.append((p, f"{type(e).__name__}: {e}"))
        if verbose:
            print(f"[escenas] {n} tomas en {self.root}" +
                  (f" ({len(self.broken)} ilegibles)" if self.broken else ""))

    def rooms(self):
        return sorted(self.registry)

    def arrays(self, room):
        return sorted(self.registry.get(room, {}))

    def takes(self, room, array):
        return sorted(self.registry.get(room, {}).get(array, {}),
                      key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else 0, s))

    def get(self, room, array, take, include_bad=False):
        """La ULTIMA grabacion de esa toma (las ok primero)."""
        lst = self.registry.get(room, {}).get(array, {}).get(str(take), [])
        good = [t for t in lst if t.ok]
        pool = good or (lst if include_bad else [])
        if not pool:
            return None
        return max(pool, key=lambda t: t.meta.get("when") or "")

    def iter_takes(self, room=None, array=None, include_bad=False):
        for r in ([room] if room else self.rooms()):
            for a in ([array] if array else self.arrays(r)):
                for k in self.takes(r, a):
                    t = self.get(r, a, k, include_bad=include_bad)
                    if t is not None:
                        yield t

    def describe(self):
        lines = [f"Escenas en {self.root}:"]
        for r in self.rooms():
            lines.append(f"  {r}")
            for a in self.arrays(r):
                tt = []
                for k in self.takes(r, a):
                    lst = self.registry[r][a][k]
                    t = self.get(r, a, k, include_bad=True)
                    flag = "" if t.ok else " (MARCADA)"
                    rep = f" x{len(lst)}" if len(lst) > 1 else ""
                    tt.append(f"{k}:{t.meta.get('interference', {}).get('name')}"
                              f"{rep}{flag}")
                lines.append(f"    {a}: " + (", ".join(tt) or "-"))
        return "\n".join(lines)
