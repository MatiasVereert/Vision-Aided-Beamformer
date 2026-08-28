"""
bench_ui.py
===========
Consola unificada para los benchmarks (`full_benchmark_test_dtln.py`,
`full_benchmark_test_dtln_mird.py`, `full_benchmark_real.py`).

Que resuelve
------------
Los benchmarks encadenan muchas etapas pesadas (RIRs, WPE, DTLN, beamformers,
metricas) y cada libreria interna imprime su propio progreso por frame
("Processing frame 812 of 1873", "-> Computing DTLN mask ...", avisos de ventana
STFT, etc.). El resultado es un scroll ilegible donde no se ve ni cuanto falta ni
que se esta corriendo.

`BenchmarkUI` reemplaza eso por un panel FIJO de 3 lineas que se re-escriben en
el lugar (no generan renglones nuevos):

    Benchmark    45%|#########       | 9/20 exp [02:11<02:40, 14.7s/exp]
    exp 10/20 · rt60=0.61 isir=-5 wpe=True    60%|######    | 6/10
    -> NM-MVDR_alpha_0.99: beamforming

  * linea 1: progreso TOTAL del benchmark (con ETA).
  * linea 2: progreso de la prueba EN CURSO; se RESETEA en cada experimento y su
    descripcion es la config compacta de esa prueba.
  * linea 3: el proceso puntual que se esta ejecutando ahora mismo.

Ademas expone `quiet()`, un contexto que silencia el stdout de los procesadores y
de la evaluacion de metricas. Es puramente cosmetico: NO toca argumentos, orden
de llamadas ni valores devueltos. Si la etapa silenciada lanza una excepcion, el
stdout capturado se vuelca antes de propagarla, asi no se pierde el diagnostico.

Notas de implementacion
-----------------------
  * Las barras viven en STDERR y `log()/warn()` tambien, de modo que el stdout
    (donde caen los prints internos, las tablas finales y cualquier redireccion
    a archivo del usuario) queda limpio y no interfiere con el repintado.
  * Sin TTY (nohup, `> log.txt`, CI) las barras se desactivan solas y se emite
    una linea por experimento: el log queda legible en vez de miles de \\r.
  * En notebook la linea de estado se pliega dentro de la barra de etapa
    (los widgets de tqdm.notebook no renderizan un bar_format solo-descripcion).
"""

import atexit
import contextlib
import io
import os
import shutil
import sys

try:  # tqdm.auto = widget en notebook, texto en terminal
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback defensivo
    from tqdm import tqdm


def _in_notebook():
    return "ipykernel" in sys.modules


# Ejes que mas identifican una escena; se muestran primero en la etiqueta.
_LABEL_PRIORITY = (
    "rt60", "topology", "M", "diameter", "target_angle", "target_dist",
    "interf_scenario", "interf_configs", "source_dist", "isir_db",
    "use_wpe", "wpe_method", "wpe_taps", "wpe_delay",
    "mismatch_gain", "mismatch_phase", "error_angle_deg", "error_distance_m",
)


def varying_keys(experiments):
    """
    Claves cuyo valor CAMBIA a lo largo del barrido, es decir los ejes reales de
    la grilla. Sirve para que la etiqueta de cada prueba muestre solo lo que la
    distingue de las demas en vez de repetir 15 constantes. Devuelve None (=
    mostrar todo) si hay un solo experimento o si nada varia.
    """
    if not experiments or len(experiments) < 2:
        return None
    keys = {k for k in experiments[0]
            if len({str(e.get(k)) for e in experiments}) > 1}
    return keys or None


def compact_config(exp, max_len=64, skip=(), keep=None):
    """
    Resume el dict de un experimento en una etiqueta corta de una linea.

    Las rutas se reducen al nombre de archivo (sin extension) y las listas a su
    repr recortado, para que la config entre en la descripcion de la barra sin
    empujar el resto de la linea fuera de la pantalla. `keep` (p.ej. la salida de
    `varying_keys`) limita la etiqueta a esas claves; los ejes mas identificatorios
    van primero.
    """
    items = sorted(
        exp.items(),
        key=lambda kv: (_LABEL_PRIORITY.index(kv[0])
                        if kv[0] in _LABEL_PRIORITY else len(_LABEL_PRIORITY)),
    )
    parts = []
    for k, v in items:
        if k in skip or (keep is not None and k not in keep):
            continue
        if isinstance(v, str) and (os.sep in v or "/" in v):
            v = os.path.splitext(os.path.basename(v))[0][:18]
        elif isinstance(v, float):
            v = f"{v:g}"
        elif isinstance(v, (list, tuple)):
            v = str(v).replace(" ", "")
        parts.append(f"{k}={v}")
    label = " ".join(parts)
    if len(label) > max_len:
        label = label[: max_len - 1] + "…"
    return label


class BenchmarkUI:
    """
    Panel de progreso de 3 lineas + silenciador de prints internos.

    Uso tipico::

        with BenchmarkUI(len(experiments), desc="MIRD Benchmark") as ui:
            for i, exp in enumerate(experiments):
                ui.begin_experiment(i, compact_config(exp), steps=n_steps)
                ui.stage("[NODE 1] RIRs")
                with ui.quiet():
                    ...
                ui.end_experiment()

    Parametros
    ----------
    total : int          cantidad de experimentos/fases del benchmark completo.
    desc : str           titulo de la barra total.
    unit : str           unidad de la barra total ('exp', 'fase', ...).
    quiet : bool         si False, `quiet()` es un no-op (util para depurar).
    enabled : bool       si False, toda la UI es un no-op silencioso.
    """

    def __init__(self, total, desc="Benchmark", unit="exp", quiet=True, enabled=True):
        self.total = int(total)
        self.desc = desc
        self._quiet_enabled = bool(quiet)
        self._stream = sys.stderr
        self._notebook = _in_notebook()
        self._tty = self._notebook or bool(getattr(self._stream, "isatty", lambda: False)())
        self._enabled = bool(enabled) and self.total > 0
        # Sin TTY las barras solo ensucian: se apagan y se loguea 1 linea por prueba.
        self._bars = self._enabled and self._tty

        self._exp_label = ""
        self._exp_idx = 0
        self._stage_text = ""

        self.bar_total = None
        self.bar_stage = None
        self.line_status = None

        # El panel vive en stderr; si quedo texto pendiente en el buffer de stdout
        # (setup, avisos de carga) se vacia primero para que no aparezca mezclado
        # con las barras o, al redirigir a archivo, fuera de orden.
        try:
            sys.stdout.flush()
        except Exception:
            pass

        if self._bars:
            self.bar_total = tqdm(
                total=self.total, desc=desc, unit=unit, position=0, leave=True,
                dynamic_ncols=True, file=self._stream,
                bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {unit} [{elapsed}<{remaining}, {rate_fmt}]",
            )
            self.bar_stage = tqdm(
                total=1, desc="", position=1, leave=False,
                dynamic_ncols=True, file=self._stream,
                bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
            )
            if not self._notebook:
                self.line_status = tqdm(
                    total=0, position=2, leave=False, dynamic_ncols=True,
                    file=self._stream, bar_format="{desc}",
                )
            # Si el benchmark muere por excepcion, el panel se cierra igual y la
            # terminal no queda con el cursor pisado (close() es idempotente).
            atexit.register(self.close)

    # ------------------------------------------------------------------ log
    def log(self, msg):
        """Mensaje PERSISTENTE por encima del panel (no lo rompe)."""
        if self._bars:
            tqdm.write(str(msg), file=self._stream)
        elif self._enabled:
            print(str(msg), file=self._stream, flush=True)

    def warn(self, msg):
        self.log(str(msg))

    # ------------------------------------------------------- ciclo de prueba
    def begin_experiment(self, index, label="", steps=1):
        """
        Arranca una prueba: RESETEA la barra de etapa y fija su descripcion.

        index : indice 0-based del experimento.
        label : config compacta (ver `compact_config`).
        steps : cantidad estimada de etapas de esta prueba (100% de la barra).
        """
        self._exp_idx = int(index)
        self._exp_label = str(label)
        self._stage_text = ""
        steps = max(1, int(steps))
        if self.bar_stage is not None:
            self.bar_stage.reset(total=steps)
            self.bar_stage.set_description_str(self._exp_prefix(), refresh=True)
        self._set_status("")
        if self._enabled and not self._bars:
            print(f"[{self._exp_idx + 1}/{self.total}] {self._exp_label}",
                  file=self._stream, flush=True)

    def _exp_prefix(self):
        head = f"exp {self._exp_idx + 1}/{self.total}"
        if not self._exp_label:
            return head
        # Recorte al ancho real de la terminal para no partir la linea en dos.
        width = shutil.get_terminal_size((100, 24)).columns
        budget = max(20, width - 42 - len(head))
        label = self._exp_label
        if len(label) > budget:
            label = label[: budget - 1] + "…"
        return f"{head} · {label}"

    def stage(self, text, advance=True):
        """
        Anuncia la etapa en curso (linea 3) y avanza la barra de la prueba.

        `advance=False` para re-etiquetar sin consumir un paso (p.ej. sub-etapas
        de una misma unidad de trabajo).
        """
        self._stage_text = str(text)
        if self.bar_stage is not None and advance:
            # Si la prueba tuvo mas etapas de las estimadas, se estira el total
            # en vez de mostrar un 120% imposible.
            if self.bar_stage.n + 1 > self.bar_stage.total:
                self.bar_stage.total = self.bar_stage.n + 1
            self.bar_stage.update(1)
        self._set_status(self._stage_text)

    def _set_status(self, text):
        if self.line_status is not None:
            self.line_status.set_description_str(f" -> {text}" if text else "", refresh=True)
        elif self.bar_stage is not None:
            # Notebook: no hay linea propia, la etapa va pegada a la descripcion.
            suffix = f" · {text}" if text else ""
            self.bar_stage.set_description_str(self._exp_prefix() + suffix, refresh=True)

    def end_experiment(self):
        """Cierra la prueba: completa su barra y avanza la barra total."""
        if self.bar_stage is not None and self.bar_stage.total:
            remaining = self.bar_stage.total - self.bar_stage.n
            if remaining > 0:
                self.bar_stage.update(remaining)
        if self.bar_total is not None:
            self.bar_total.update(1)

    # -------------------------------------------------------------- silencio
    @contextlib.contextmanager
    def quiet(self):
        """
        Silencia el stdout de la etapa (prints por frame de DTLN/Souden, avisos
        de ventana STFT, DNSMOS, etc.). Si hay excepcion, vuelca lo capturado.

        Solo redirige STDOUT: las barras y los logs viven en stderr y siguen
        visibles. No altera ninguna llamada ni valor de retorno.
        """
        if not self._quiet_enabled:
            yield
            return
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                yield
        except BaseException:
            captured = buf.getvalue().strip()
            if captured:
                self.log("[stdout capturado antes del error]\n" + captured[-4000:])
            raise

    # ----------------------------------------------------------------- cierre
    def close(self):
        try:
            atexit.unregister(self.close)
        except Exception:
            pass
        for bar in (self.line_status, self.bar_stage, self.bar_total):
            if bar is not None:
                try:
                    bar.close()
                except Exception:
                    pass
        self.line_status = self.bar_stage = self.bar_total = None
        if self._bars:
            # Deja el cursor en limpio debajo del panel.
            self._stream.write("\n")
            self._stream.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


@contextlib.contextmanager
def quiet_stdout():
    """`BenchmarkUI.quiet()` suelto, para scripts que no arman panel."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            yield
    except BaseException:
        captured = buf.getvalue().strip()
        if captured:
            print("[stdout capturado antes del error]\n" + captured[-4000:],
                  file=sys.stderr, flush=True)
        raise
