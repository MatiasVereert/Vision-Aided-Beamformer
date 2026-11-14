#!/usr/bin/env python3
"""kria_rir_loader.py -- RIRs MEDIDAS con el arreglo de la Kria, para el banco.

Hermano de mird_loader.py: misma funcion, otro origen. En vez de la base de
Bar-Ilan, carga las respuestas que mide mic-array-platform/tools/rir_lab con el
aro de 12 capsulas sobre la Kria KV260, en salas reales.

QUE SE ESTA CARGANDO. No es "la acustica de la sala": es la respuesta de la
CADENA parlante -> sala -> arreglo, con la coloracion del parlante y de los
microfonos adentro, ninguno calibrado. Eso es exactamente lo que se quiere para
evaluar el beamformer -- es la senal que de verdad le va a llegar -- y las
comparaciones entre condiciones valen porque la cadena es la misma en todas.

AUTOSUFICIENTE A PROPOSITO. Cada .npz lleva su geometria y sus metadatos
adentro, asi que esto NO importa nada de mic-array-platform para funcionar: se
puede copiar el arbol de data/rirs/ a Colab y anda. Solo el camino opcional de
estimar una posicion que no se midio necesita rir_lab, y si no esta, se dice y
se sigue.

TRES COSAS QUE HAY QUE TENER EN LA CABEZA AL USARLO

 1. EL MARCO DE LOS ANGULOS. Las posiciones que devuelve array_geometry() y los
    azimuts que devuelve export_position() estan en el marco INTERNO del aro:
    el canal m en el angulo 30 * ORDER[m], creciendo en sentido antihorario
    visto desde arriba, que es la convencion matematica estandar y la misma que
    usa el resto del banco. NO es el marco en que se leen los informes de
    rir_lab (cero en la entrada del cable, sentido horario): ese es para
    humanos. Mezclarlos espeja el resultado sin avisar.

 2. EL MICROFONO DE REFERENCIA NO ES M // 2. El proyecto usa ch6 en todo (el
    OFB, las comparaciones de DNSMOS, la auditoria del DTLN). Con los 11
    canales de hoy, M // 2 cae en ch5. Hay que pasarle al motor
        scene_base_config['ref_mic_mode'] = provider.ref_mic_index()
    porque si no, las metricas se calculan contra otro canal y la diferencia de
    camino acustico entre los dos micros se contabiliza como distorsion del
    procesador (medido sobre MIRD: ~3 dB de SI-SDR).

 3. MEDICIONES DE ARREGLOS DISTINTOS NO SE MEZCLAN. Cada posicion de arreglo
    guarda su estado de hardware (hw_rev: que capsulas estaban anuladas, que
    orden de canales). El ch1 se recupero el 2026-11-14 y el ch10 sigue mudo,
    asi que hay mediciones de M = 10 y de M = 11 en el mismo dataset.
    require_same_array() lo chequea y levanta un error en vez de promediar
    peras con manzanas.

COMO SE ENGANCHA AL MOTOR DEL BANCO (las claves exactas, del NODE 1 de
full_benchmark_test_dtln_mird.py):

    from propagation.kria_rir_loader import KriaRirProvider
    prov = KriaRirProvider()                       # indexa data/rirs/

    prov.load_rir(room, array, source="f1")        # fija la escena actual
    base_config.update({
        "fs":            prov.get_current_fs(),        # 16000, sin resamplear
        "mic_coords":    prov.array_geometry() + array_center,
        "ref_mic_mode":  prov.ref_mic_index(),         # ch6, NO M // 2
        "mird_spacing":  array,                        # el nombre de la pos.
    })
    # y en el barrido, 'rt60' pasa a ser el nombre de la SALA:
    grid_params = {"rt60": [room], "target_dist": [...], "target_angle": [...]}

    acoustic_scene.import_rirs(dataset_provider=prov, target_t60=room,
                               array_center=array_center, spacing_cfg=array)

Los nombres 'rt60' y 'mird_spacing' quedan como estan a proposito: son los ejes
que el motor ya sabe barrer y cachear, y renombrarlos obligaria a tocar el
motor entero para no ganar nada. Con este provider significan sala y posicion
de arreglo.
"""
import glob
import json
import os
import sqlite3

import numpy as np

DEFAULT_ROOT = os.path.expanduser(
    "~/Documents/Tesis/mic-array-platform/data/rirs")


def _f(v):
    return None if v is None else float(v)


class KriaMeasurement:
    """Una medicion: la RIR, su geometria y de donde vino el sonido."""

    __slots__ = ("path", "meta", "_ir", "_pos")

    def __init__(self, path, meta):
        self.path = path
        self.meta = meta
        self._ir = None
        self._pos = None

    # ---- identidad de la escena -----------------------------------------
    @property
    def scene(self):
        return self.meta.get("scene") or {}

    @property
    def room(self):
        return self.scene.get("room")

    @property
    def array(self):
        return self.scene.get("array")

    @property
    def source(self):
        return self.scene.get("source")

    @property
    def fs(self):
        return int(self.meta["fs"])

    @property
    def active(self):
        return list((self.meta.get("geometry") or {}).get("active") or [])

    @property
    def ref_mic(self):
        return (self.meta.get("geometry") or {}).get("ref_mic")

    @property
    def hw_key(self):
        """Identidad del ARREGLO con el que se midio. Dos mediciones con hw_key
        distinto no son del mismo arreglo y no se pueden comparar."""
        g = self.meta.get("geometry") or {}
        return (tuple(g.get("order") or ()), tuple(g.get("active") or ()),
                _f(g.get("radius_m")))

    # ---- datos -----------------------------------------------------------
    def _load(self):
        if self._ir is None:
            z = np.load(self.path, allow_pickle=False)
            self._ir = np.asarray(z["ir"], dtype=float)
            self._pos = np.asarray(z["positions"], dtype=float)
        return self._ir, self._pos

    def rir(self, active_only=True):
        """(N, M) float64. Con active_only, solo los canales buenos, EN ORDEN,
        de modo que la columna j corresponde a array_geometry()[j]."""
        ir, _ = self._load()
        return ir[:, self.active] if active_only else ir

    def geometry(self):
        """(M, 3) en metros, centrada en el aro, marco INTERNO (ver encabezado)."""
        _, pos = self._load()
        return pos[self.active, :]

    def position(self):
        """(dist_m, az_deg, el_deg, de_donde_salio) en el marco interno.

        `de_donde_salio` es 'rel'/'xyz' si se midio con cinta o de un plano, y
        'estimated' si la saco locate() de la propia RIR. Que quede explicito
        no es un detalle: una posicion estimada alimentando un beamformer
        geometrico es defendible, pero el lector tiene que saberlo.
        """
        sc = self.scene
        if sc.get("dist_m") is not None and sc.get("az_deg") is not None:
            return (_f(sc["dist_m"]), _f(sc["az_deg"]), _f(sc.get("el_deg") or 0.0),
                    sc.get("pos_source") or "rel")
        est = self.estimate_position()
        if est is None:
            return (None, None, None, None)
        return est

    def estimate_position(self, rir_lab_path=None):
        """Estima la direccion de la propia RIR. Necesita rir_lab."""
        import sys
        p = rir_lab_path or os.path.expanduser(
            "~/Documents/Tesis/mic-array-platform/tools")
        if p not in sys.path:
            sys.path.insert(0, p)
        try:
            from rir_lab import locate as loc
        except Exception as e:                               # noqa: BLE001
            print(f"[KriaRir] no puedo estimar la posicion de "
                  f"{os.path.basename(self.path)}: falta rir_lab ({e}). "
                  f"Medila con cinta o corre `measure.py locate`.")
            return None
        ir, _ = self._load()
        acs = self.meta.get("acoustics") or {}
        e = loc.locate(ir, self.fs, active=self.active,
                       rot_deg=_f(self.meta.get("rot_deg")) or 0.0,
                       drr_db=acs.get("DRR_wide", np.nan),
                       rc_m=acs.get("critical_distance_m", np.nan))
        # La distancia NO sale de la geometria: se usa la cota inferior, que es
        # lo unico defendible (ver locate.range_lower_bound).
        return (float(e["r_min_m"]), float(e["az_deg"]), float(e["el_deg"]),
                "estimated")

    @property
    def quality(self):
        rc = self.meta.get("rep_corr") or []
        return {"reps_used": self.meta.get("reps_used"),
                "rep_corr_min": (min(rc) if rc else None),
                "eps_ppm": (self.meta.get("eps") or 0.0) * 1e6,
                "eps_resid_us": self.meta.get("eps_resid_us"),
                "gaps": len(self.meta.get("gaps") or []),
                "warnings": list(self.meta.get("warnings") or [])}

    def __repr__(self):
        d, a, _, src = self.position()
        return (f"<KriaMeasurement {self.room}/{self.array}/{self.source} "
                f"M={len(self.active)} "
                f"dist={'?' if d is None else format(d, '.2f')} "
                f"az={'?' if a is None else format(a, '.1f')} ({src})>")


class KriaRirProvider:
    """Indexa las mediciones y las entrega, con la API de MirdDatasetProvider.

    El indice se arma recorriendo los .npz (no depende de la base SQLite: la
    base es un indice para consultar, los archivos son la fuente de verdad).
    """

    def __init__(self, root=DEFAULT_ROOT, verbose=True):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.measurements = []
        self.registry = {}           # [room][array][source] = KriaMeasurement
        self.current = None
        self._index(verbose)

    # -------------------------------------------------------------- indice
    def _index(self, verbose=True):
        n_bad = 0
        for path in sorted(glob.glob(os.path.join(self.root, "**", "*.npz"),
                                     recursive=True)):
            try:
                z = np.load(path, allow_pickle=False)
                meta = json.loads(str(z["meta"]))
            except Exception:                                # noqa: BLE001
                n_bad += 1
                continue
            m = KriaMeasurement(path, meta)
            if not (m.room and m.array and m.source):
                n_bad += 1
                continue
            self.measurements.append(m)
            self.registry.setdefault(m.room, {}).setdefault(
                m.array, {}).setdefault(m.source, []).append(m)
        if verbose:
            print(f"[KriaRir] {len(self.measurements)} mediciones en "
                  f"{self.root}" + (f" ({n_bad} sin escena, salteadas)"
                                    if n_bad else ""))

    # ---------------------------------------------------- enumerar escenas
    def rooms(self):
        return sorted(self.registry)

    def arrays(self, room):
        return sorted(self.registry.get(room, {}))

    def sources(self, room, array):
        return sorted(self.registry.get(room, {}).get(array, {}))

    def get(self, room, array, source, prefer_clean=True):
        """La medicion de esa escena. Con varias tomas, la MEJOR: sin avisos y
        con la correlacion entre repeticiones mas alta. Repetir una medicion es
        normal (el camino de audio tiene hipos); elegir a ciegas, no."""
        lst = self.registry.get(room, {}).get(array, {}).get(source, [])
        if not lst:
            raise ValueError(
                f"no hay medicion para {room}/{array}/{source}. Hay: "
                f"{self.sources(room, array) or '(ninguna)'}")
        if not prefer_clean or len(lst) == 1:
            return lst[-1]
        def key(m):
            q = m.quality
            return (len(q["warnings"]) == 0, q["rep_corr_min"] or 0.0,
                    m.meta.get("when") or "")
        return max(lst, key=key)

    # --------------------------------------------- API estilo MIRD loader
    def load_rir(self, room, array, distance=None, angle=None, source=None):
        """(N, M) de la escena pedida. Compatible en posicion con
        MirdDatasetProvider.load_rir(t60, spacing, distance, angle): aca
        `room` ocupa el lugar de t60 y `array` el de spacing.

        Se puede pedir por nombre de fuente (`source=`) o por (distancia,
        azimut), que es como barre el motor del banco. En el segundo caso se
        busca la medicion mas cercana y se avisa cuanto se tuvo que mover --
        esta base es DISCRETA y chica, no una grilla densa como MIRD.
        """
        if source is not None:
            m = self.get(room, array, source)
        else:
            m = self._nearest(room, array, distance, angle)
        self.current = m
        return m.rir()

    def _nearest(self, room, array, distance, angle, tol_deg=10.0, tol_m=0.5):
        cands = []
        for s in self.sources(room, array):
            mm = self.get(room, array, s)
            d, a, _, _ = mm.position()
            if a is None:
                continue
            da = abs((a - float(angle) + 180.0) % 360.0 - 180.0)
            dd = (abs(d - float(distance))
                  if (d is not None and distance is not None) else 0.0)
            cands.append((da + 20.0 * dd, da, dd, s, mm))
        if not cands:
            raise ValueError(
                f"ninguna medicion de {room}/{array} tiene posicion conocida; "
                f"medila con cinta o deja que se estime (necesita rir_lab)")
        cands.sort()
        _, da, dd, s, mm = cands[0]
        if da > tol_deg or dd > tol_m:
            print(f"[KriaRir] OJO: la escena mas cercana a "
                  f"(dist={distance}, az={angle}) es '{s}' y esta a "
                  f"{da:.1f} deg / {dd:.2f} m. Esta base es discreta: no "
                  f"interpola.")
        return mm

    def get_current_fs(self):
        return int(self.current.fs) if self.current else 16000

    def export_position(self, mode="cartesian"):
        """Posicion de la fuente de la ultima carga, RELATIVA al centro del aro.

        Misma forma (1, 3) que MirdDatasetProvider, para que
        SimAcoustic/import_rirs no noten la diferencia.
        """
        if self.current is None:
            raise RuntimeError("primero hay que llamar a load_rir()")
        d, az, el, src = self.current.position()
        if d is None:
            raise RuntimeError(
                f"{self.current} no tiene posicion: medila o estimala")
        a, e = np.radians(az), np.radians(el or 0.0)
        cart = np.array([[d * np.cos(e) * np.cos(a),
                          d * np.cos(e) * np.sin(a),
                          d * np.sin(e)]])
        if mode == "cartesian":
            return cart
        if mode == "spherical":
            return np.array([d, a, np.pi / 2.0 - e])
        raise ValueError("mode tiene que ser 'cartesian' o 'spherical'")

    def resolve_grid(self, rel_vec):
        """Para import_rirs: de un vector relativo al centro, los argumentos de
        load_rir. Reemplaza el redondeo a la grilla de 15 grados del MIRD, que
        no tiene sentido acá."""
        rel = np.asarray(rel_vec, dtype=float).squeeze()
        r = float(np.linalg.norm(rel[:2]))
        az = float(np.degrees(np.arctan2(rel[1], rel[0])) % 360.0)
        return r, az

    # ----------------------------------------------------------- geometria
    def array_geometry(self, room=None, array=None):
        """(M, 3) centrada en el aro, marco interno. Es lo que hay que pasarle
        a SimAcoustic como array_geometry (mas el array_center)."""
        m = self.current or self._any(room, array)
        return m.geometry()

    def active_channels(self, room=None, array=None):
        return (self.current or self._any(room, array)).active

    def ref_mic_index(self, room=None, array=None):
        """Indice de ch6 DENTRO de la seleccion: es lo que va en
        scene_base_config['ref_mic_mode']. Ver la nota 2 del encabezado."""
        m = self.current or self._any(room, array)
        act = m.active
        ref = m.ref_mic
        if ref is None or ref not in act:
            raise RuntimeError(f"{m} no declara un microfono de referencia "
                               f"valido (ref={ref}, activos={act})")
        return act.index(int(ref))

    def _any(self, room=None, array=None):
        for m in self.measurements:
            if (room is None or m.room == room) and (array is None
                                                     or m.array == array):
                return m
        raise RuntimeError("el dataset esta vacio")

    def require_same_array(self, measurements):
        """Levanta error si las mediciones no son del mismo arreglo fisico."""
        keys = {m.hw_key for m in measurements}
        if len(keys) > 1:
            raise RuntimeError(
                "estas mediciones son de arreglos DISTINTOS (cambio el orden "
                "de canales, el radio o que capsulas entran). No se pueden "
                "comparar ni promediar: " +
                "; ".join(f"{m.room}/{m.array}/{m.source} M={len(m.active)}"
                          for m in measurements))
        return True

    def room_dims(self, room):
        """(x, y, z) de la sala en metros, de la base SQLite si esta.

        Solo entra en el H5 del catalogo y en el dashboard, no en las metricas.
        Si la sala no tiene dimensiones cargadas devuelve None y quien llama
        pone su default: es mejor que inventar un volumen, porque de ahi sale
        la distancia critica.
        """
        db = os.path.join(self.root, "rirs.sqlite")
        if not os.path.exists(db):
            return None
        try:
            cx = sqlite3.connect(db)
            cx.row_factory = sqlite3.Row
            r = cx.execute("SELECT dim_x, dim_y, dim_z FROM room WHERE slug=?",
                           (room,)).fetchone()
            cx.close()
        except Exception:                                    # noqa: BLE001
            return None
        if not r or r["dim_x"] is None:
            return None
        return (float(r["dim_x"]), float(r["dim_y"]), float(r["dim_z"]))

    def scene_table(self, room, array):
        """[(fuente, dist, az, el, origen)] de las fuentes medidas, ordenadas
        por azimut. Es de aca que sale la grilla del barrido: esta base es
        DISCRETA, asi que los ejes no se eligen, se leen."""
        out = []
        for s_ in self.sources(room, array):
            m = self.get(room, array, s_)
            d, az, el, src = m.position()
            if d is None or az is None:
                continue
            out.append((s_, float(d), float(az), float(el or 0.0), src))
        return sorted(out, key=lambda t: t[2])

    # ------------------------------------------------------------- resumen
    def describe(self):
        head = (f"  {'escena':<26} {'M':>2} {'fs':>5} {'dist':>6} {'az':>7} "
                f"{'el':>6} {'origen':<10} {'reps':>5} {'corr':>7}")
        out = [head, "  " + "-" * (len(head) - 2)]
        for room in self.rooms():
            for arr in self.arrays(room):
                for s_ in self.sources(room, arr):
                    m = self.get(room, arr, s_)
                    d, az, el, src = m.position()
                    q = m.quality
                    g = lambda v, n=2: "     ?" if v is None else f"{v:6.{n}f}"
                    out.append(
                        f"  {room + '/' + arr + '/' + s_:<26} "
                        f"{len(m.active):2d} {m.fs:5d} {g(d)} {g(az, 1):>7} "
                        f"{g(el, 1):>6} {str(src or '-'):<10} "
                        f"{q['reps_used'] or 0:5d} "
                        f"{q['rep_corr_min'] or float('nan'):7.4f}"
                        f"{'  !! con avisos' if q['warnings'] else ''}")
        return "\n".join(out)


def generate_kria_uca(radius_m=0.1235 / 2, order=None, rot_deg=0.0, n=12,
                      active=None):
    """Coordenadas teoricas del aro de la Kria, para cuando no se carga una
    medicion (p.ej. para simular la MISMA geometria en el ISM y comparar).

    El canal m va en el angulo rot_deg + 360/n * order[m], en z = 0. `order` es
    REWIRED_ORDER; si no se pasa, se usa el cableado vigente al 2026-11.
    OJO: para evaluar RIRs MEDIDAS no hace falta -- la geometria viaja dentro
    del .npz y se saca con provider.array_geometry(), que es la que de verdad
    corresponde a esos datos.
    """
    if order is None:
        order = (7, 6, 3, 2, 10, 11, 8, 9, 5, 4, 0, 1)
    ang = np.deg2rad(float(rot_deg) + 360.0 / n * np.asarray(order))
    pos = np.stack([radius_m * np.cos(ang), radius_m * np.sin(ang),
                    np.zeros(n)], axis=1)
    return pos if active is None else pos[list(active), :]


if __name__ == "__main__":
    # Prueba de integracion: una RIR MEDIDA entrando al motor del banco por el
    # mismo camino que las del MIRD. Misma idea que el __main__ de
    # mird_loader.py. No necesita la Kria ni el parlante: lee los .npz.
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from propagation.simulate_acoustics_v1 import SimAcoustic

    SRC = os.path.expanduser("~/Documents/Tesis/Vision-Aided-Beamformer/tools/"
                             "data/signals/p002_emo_adoration_sentences.wav")
    CENTER = np.array([3.0, 3.0, 1.2])

    prov = KriaRirProvider()
    print(prov.describe())
    room = prov.rooms()[-1]
    arr = prov.arrays(room)[0]
    src = prov.sources(room, arr)[-1]
    print(f"\n[1] escena: {room}/{arr}/{src}")

    rir = prov.load_rir(room, arr, source=src)
    m = prov.current
    geom = prov.array_geometry()
    d, az, el, origen = m.position()
    print(f"    RIR {rir.shape}  fs {prov.get_current_fs()}  "
          f"geometria {geom.shape}")
    print(f"    fuente a {d:.2f} m, az {az:.1f} deg, el {el:.1f} deg ({origen})")
    print(f"    ref_mic_mode para el motor: {prov.ref_mic_index()} "
          f"(ch{m.ref_mic}); M//2 daria ch{m.active[len(m.active) // 2]}")

    # --- el motor, igual que el NODE 1 del benchmark ----------------------
    mic_coords = geom + CENTER
    scene = SimAcoustic(array_geometry=mic_coords, array_mismatch=0.0,
                        duration=4.0, fs=prov.get_current_fs())
    abs_pos = (CENTER + prov.export_position("cartesian").squeeze()).reshape(1, 3)
    print(f"\n[2] fuente absoluta {np.round(abs_pos.squeeze(), 3)} "
          f"(centro {CENTER})")
    if not os.path.exists(SRC):
        print(f"    (sin {os.path.basename(SRC)}: no se puede convolucionar)")
        _sys.exit(0)
    scene.set_source(SRC, gain=1.0, position=abs_pos)
    scene.import_rirs(dataset_provider=prov, target_t60=room,
                      array_center=CENTER, spacing_cfg=arr)
    scene.convolve_signals(t_early=0.050)
    x = scene.target_early_unscaled + scene.target_late_unscaled
    print(f"\n[3] senales de microfono {x.shape}, "
          f"finitas={np.all(np.isfinite(x))}, "
          f"RMS por canal {np.round(20 * np.log10(np.maximum(np.sqrt(np.mean(x ** 2, axis=1)), 1e-30)), 1)}")

    # --- el chequeo que de verdad importa: el AZIMUT sobrevive el viaje ---
    # Si la geometria, el orden de canales o el signo de la fase se mezclaron
    # en algun punto, la direccion de llegada de la senal convolucionada no va
    # a coincidir con la posicion que se le puso a la fuente.
    h = np.stack([scene.rirs[k][0] for k in range(scene.M)], axis=1)
    p0 = int(np.argmax(np.sum(h ** 2, axis=1)))
    a0, b0 = max(0, p0 - 16), min(h.shape[0], p0 + 32)
    w = np.hanning(b0 - a0)[:, None]
    seg = (h[a0:b0, :] * w)
    nf = 1 << int(np.ceil(np.log2(4 * (b0 - a0))))
    ref = prov.ref_mic_index()
    tau = np.zeros(scene.M)
    R = np.fft.rfft(seg[:, ref], nf)
    for k in range(scene.M):
        c = np.fft.irfft(np.fft.rfft(seg[:, k], nf) * np.conj(R), nf)
        lag = np.argmax(np.concatenate([c[-8:], c[:9]])) - 8
        tau[k] = lag / scene.fs
    A = np.column_stack([np.ones(scene.M), -geom[:, 0], -geom[:, 1]])
    sol, *_ = np.linalg.lstsq(A, tau, rcond=None)
    az_back = np.degrees(np.arctan2(sol[2], sol[1])) % 360.0
    err = (az_back - az + 180) % 360 - 180
    print(f"\n[4] azimut puesto {az:.1f} deg -> recuperado de las RIRs "
          f"importadas {az_back:.1f} deg  (error {err:+.1f})")
    print("    " + ("OK: la geometria y el orden de canales sobreviven el viaje"
                    if abs(err) < 5 else
                    "!! NO COINCIDE: revisar geometria/orden/signo de fase"))

    # --- [5] REGRESION DEL CAMINO MIRD -----------------------------------
    # Para que este loader funcione hubo que generalizar resolve_grid_mapping()
    # en SimAcoustic.import_rirs, que es codigo COMPARTIDO con las corridas del
    # MIRD que ya estaban validadas. El cambio es aditivo (sin resolve_grid en
    # el provider, el comportamiento es el de antes), pero "es aditivo" es una
    # afirmacion, no una prueba: esto lo comprueba cargando una RIR del MIRD
    # por el motor y comparandola contra la que da el loader a mano.
    MIRD = os.path.expanduser("~/Documents/Tesis/Vision-Aided-Beamformer/"
                              "tools/data/rirs/mird")
    if not os.path.isdir(MIRD):
        print("\n[5] (sin la base MIRD: no se corre la regresion)")
    else:
        from propagation.mird_loader import (
            MirdDatasetProvider, generate_mird_linear_array_from_spacing)
        from scipy import signal as _sg
        SPACING = "3-3-3-8-3-3-3"
        mp = MirdDatasetProvider(root_dir=MIRD)
        t60 = mp.get_available_t60s()[1]
        g8 = generate_mird_linear_array_from_spacing(SPACING)
        s8 = SimAcoustic(array_geometry=g8 + CENTER, array_mismatch=0.0,
                         duration=3.0, fs=16000)
        rel = np.array([2.0 * np.cos(np.radians(30)),
                        2.0 * np.sin(np.radians(30)), 0.0])
        s8.set_source(SRC, gain=1.0, position=(CENTER + rel).reshape(1, 3))
        s8.import_rirs(dataset_provider=mp, target_t60=t60,
                       array_center=CENTER, spacing_cfg=SPACING)
        imp = np.stack([s8.rirs[k][0] for k in range(s8.M)], axis=1)
        raw = np.atleast_2d(mp.load_rir(t60, SPACING, 2.0, 30))
        gg = np.gcd(16000, mp.get_current_fs())
        man = np.stack([_sg.resample_poly(raw[:, k], 16000 // gg,
                                          mp.get_current_fs() // gg)
                        for k in range(s8.M)], axis=1)
        nn = min(len(imp), len(man))
        dd = float(np.max(np.abs(imp[:nn] - man[:nn])))
        print(f"\n[5] regresion MIRD (t60={t60} ms, M={s8.M}): "
              f"max|diferencia| = {dd:.3e}")
        print("    " + ("OK: el camino MIRD quedo bit a bit igual"
                        if dd == 0.0 else
                        "!! CAMBIO: la generalizacion rompio el camino MIRD"))
