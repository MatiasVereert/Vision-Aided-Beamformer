"""
nonintrusive.py
===============
Metricas de calidad SIN referencia limpia, para evaluar el realce de voz sobre
grabaciones reales (donde no existe un target anecoico de ground truth).

Dos familias:

  1. DNSMOS P.835 (Microsoft) -> SIG / BAK / OVRL / P808_MOS
     Predictor MOS no-intrusivo. SIG = calidad de la voz, BAK = intrusividad del
     ruido de fondo, OVRL = global. Es JUSTO el eje transparencia-vs-supresion:
     el beamformer distortionless tiende a SIG alto / BAK mas bajo; el DNN
     monocanal agresivo a BAK alto / SIG mas bajo.
     Requiere: onnxruntime + librosa + los 2 modelos ONNX (sig_bak_ovr.onnx,
     model_v8.onnx) del repo Microsoft DNS-Challenge (carpeta DNSMOS).
     Implementacion fiel al dnsmos_local.py oficial (misma preproc, mismos
     polyfit de calibracion P.835).

  2. TorchAudio-SQUIM (objective) -> STOI / PESQ / SI-SDR ESTIMADOS sin referencia.
     Da los numeros "clasicos" sobre datos reales sin necesitar audio limpio.
     Requiere: torchaudio (matcheando la version de torch instalada).

Todo con carga perezosa y degradacion elegante: si falta una dep o un modelo,
la metrica devuelve NaN y se imprime UNA sola vez que instalar/descargar.
"""

import os
import numpy as np

SR = 16000
_INPUT_LENGTH = 9.01  # segundos por segmento (fijo del modelo DNSMOS)

# Carpeta donde el usuario coloca los ONNX de DNSMOS (ver instrucciones al pie).
DNSMOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dnsmos_models")
DNSMOS_PRIMARY = os.path.join(DNSMOS_DIR, "sig_bak_ovr.onnx")
DNSMOS_P808 = os.path.join(DNSMOS_DIR, "model_v8.onnx")

# Flags para imprimir el aviso de "falta instalar" una sola vez
_warned = {"dnsmos": False, "squim": False}
# Familias cuya carga ya fallo: no se reintenta en cada senal (p.ej. una descarga
# de pesos de SQUIM sin red se repetiria en cada llamada).
_unavailable = set()


def _warn_once(key, msg):
    if not _warned[key]:
        print(msg)
        _warned[key] = True


def _to_16k_mono(x, fs):
    """Asegura mono float32 @ 16 kHz (resample simple si hiciera falta)."""
    x = np.asarray(x, dtype=np.float64).squeeze()
    if x.ndim > 1:
        x = x.mean(axis=0)
    if fs != SR:
        # resample lineal por FFT (scipy) para no depender de librosa aca
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(fs), SR)
        x = resample_poly(x, SR // g, int(fs) // g)
    return x.astype(np.float32)


# =============================================================================
# DNSMOS P.835  (port fiel de Microsoft DNS-Challenge / dnsmos_local.py)
# =============================================================================
class _DNSMOS:
    _instance = None

    def __init__(self):
        import onnxruntime as ort  # lazy
        self.ort = ort
        self.sess_primary = ort.InferenceSession(DNSMOS_PRIMARY, providers=["CPUExecutionProvider"])
        self.sess_p808 = ort.InferenceSession(DNSMOS_P808, providers=["CPUExecutionProvider"])

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = _DNSMOS()
        return cls._instance

    @staticmethod
    def _melspec(audio, n_mels=120, frame_size=320, hop_length=160, sr=SR):
        import librosa  # lazy
        mel = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop_length, n_mels=n_mels
        )
        mel = (librosa.power_to_db(mel, ref=np.max) + 40) / 40
        return mel.T

    @staticmethod
    def _polyfit(sig, bak, ovr):
        # Coeficientes de calibracion P.835 (no-personalizado) del repo oficial
        p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
        p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
        p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
        return p_sig(sig), p_bak(bak), p_ovr(ovr)

    def __call__(self, audio):
        fs = SR
        len_samples = int(_INPUT_LENGTH * fs)
        audio = np.asarray(audio, dtype=np.float32)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)  # repetir hasta cubrir 9.01 s

        hop = fs
        num_hops = int(np.floor(len(audio) / fs) - _INPUT_LENGTH) + 1
        sigs, baks, ovrs, p808s = [], [], [], []
        for idx in range(num_hops):
            seg = audio[int(idx * hop): int((idx + _INPUT_LENGTH) * hop)]
            if len(seg) < len_samples:
                continue
            feats = seg.astype(np.float32)[np.newaxis, :]
            p808_feats = self._melspec(seg[:-160]).astype(np.float32)[np.newaxis, :, :]
            p808 = self.sess_p808.run(None, {"input_1": p808_feats})[0][0][0]
            sig_raw, bak_raw, ovr_raw = self.sess_primary.run(None, {"input_1": feats})[0][0]
            s, b, o = self._polyfit(sig_raw, bak_raw, ovr_raw)
            sigs.append(s); baks.append(b); ovrs.append(o); p808s.append(p808)

        if not sigs:
            return {}
        return {
            "DNSMOS_SIG": float(np.mean(sigs)),
            "DNSMOS_BAK": float(np.mean(baks)),
            "DNSMOS_OVRL": float(np.mean(ovrs)),
            "DNSMOS_P808": float(np.mean(p808s)),
        }


def _dnsmos_model():
    """Instancia (cacheada) de DNSMOS, o None si faltan modelos/deps (avisa una vez)."""
    if not (os.path.isfile(DNSMOS_PRIMARY) and os.path.isfile(DNSMOS_P808)):
        _warn_once("dnsmos",
                   f"[metricas] DNSMOS deshabilitado: faltan modelos ONNX en {DNSMOS_DIR}\n"
                   f"           (sig_bak_ovr.onnx y model_v8.onnx del repo Microsoft DNS-Challenge/DNSMOS).")
        return None
    if "dnsmos" in _unavailable:
        return None
    try:
        import librosa  # noqa: F401  (la usa _melspec; se chequea aca para no fallar a mitad)
        return _DNSMOS.get()
    except ImportError:
        _warn_once("dnsmos", "[metricas] DNSMOS deshabilitado: falta `onnxruntime` y/o `librosa` (pip install onnxruntime librosa).")
    except Exception as e:
        _warn_once("dnsmos", f"[metricas] DNSMOS fallo: {e}")
    _unavailable.add("dnsmos")
    return None


def compute_dnsmos(x, fs):
    """Devuelve dict con DNSMOS_SIG/BAK/OVRL/P808 (o {} si no esta disponible)."""
    model = _dnsmos_model()
    if model is None:
        return {}
    try:
        return model(_to_16k_mono(x, fs))
    except Exception as e:
        _warn_once("dnsmos", f"[metricas] DNSMOS fallo: {e}")
        return {}


# =============================================================================
# TorchAudio-SQUIM (objective): STOI / PESQ / SI-SDR estimados sin referencia
# =============================================================================
class _SQUIM:
    _model = None

    @classmethod
    def get(cls):
        if cls._model is None:
            from torchaudio.pipelines import SQUIM_OBJECTIVE  # lazy
            cls._model = SQUIM_OBJECTIVE.get_model()
            cls._model.eval()
        return cls._model


def _squim_model():
    """Modelo SQUIM (cacheado), o None si falta torchaudio/los pesos (avisa una vez)."""
    if "squim" in _unavailable:
        return None
    try:
        import torch  # noqa: F401
        return _SQUIM.get()
    except ImportError:
        _warn_once("squim", "[metricas] SQUIM deshabilitado: falta `torchaudio` (instalar la version que matchee tu torch).")
    except Exception as e:
        _warn_once("squim", f"[metricas] SQUIM no disponible: {e}")
    _unavailable.add("squim")
    return None


def compute_squim(x, fs, chunk_s=10.0):
    """
    Devuelve dict con SQUIM_STOI/PESQ/SISDR (estimados, sin referencia).
    Procesa en ventanas de chunk_s y promedia para robustez/memoria.
    """
    model = _squim_model()
    if model is None:
        return {}

    try:
        import torch
        wav = _to_16k_mono(x, fs)
        n = int(chunk_s * SR)
        stois, pesqs, sisdrs = [], [], []
        with torch.no_grad():
            for start in range(0, max(1, len(wav)), n):
                seg = wav[start:start + n]
                if len(seg) < SR:  # < 1 s, ignorar colas cortas
                    continue
                t = torch.from_numpy(seg).float().unsqueeze(0)
                stoi_h, pesq_h, sisdr_h = model(t)
                stois.append(float(stoi_h)); pesqs.append(float(pesq_h)); sisdrs.append(float(sisdr_h))
        if not stois:
            return {}
        return {
            "SQUIM_STOI": float(np.mean(stois)),
            "SQUIM_PESQ": float(np.mean(pesqs)),
            "SQUIM_SISDR": float(np.mean(sisdrs)),
        }
    except Exception as e:
        _warn_once("squim", f"[metricas] SQUIM fallo: {e}")
        return {}


# =============================================================================
# API unificada
# =============================================================================
def compute_nonintrusive(x, fs):
    """Junta DNSMOS + SQUIM en un solo dict (claves ausentes si no disponibles)."""
    out = {}
    out.update(compute_dnsmos(x, fs))
    out.update(compute_squim(x, fs))
    return out


def nonintrusive_status():
    """
    Carga (una sola vez) los modelos y devuelve {'dnsmos': bool, 'squim': bool}.
    Conviene llamarla al arrancar una corrida y FUERA de cualquier bloque que
    silencie stdout: los avisos de "falta instalar X" salen una sola vez, asi
    que si se disparan dentro de un ui.quiet() se pierden y las columnas quedan
    en NaN sin explicacion. De paso saca la carga de los modelos de la 1a celda.
    """
    return {"dnsmos": _dnsmos_model() is not None, "squim": _squim_model() is not None}


# =============================================================================
# Preproceso comun: ventana de evaluacion + nivel
# =============================================================================
# Nivel al que se lleva cada senal antes de puntuar. DNSMOS depende del nivel y
# las salidas del simulador estan a escala fisica (cada procesador con su propia
# ganancia): sin normalizar, la comparacion mezclaria ganancia con calidad.
TARGET_DBFS = -26.0


def prepare_nonintrusive(x, fs, start_s=0.0, target_dbfs=TARGET_DBFS):
    """
    Recorta los primeros `start_s` segundos (la misma ventana que ignoran las
    metricas intrusivas: transitorios de convergencia) y normaliza a
    `target_dbfs` RMS. Devuelve float64 1-D.
    """
    x = np.asarray(x, dtype=np.float64).squeeze()
    x = x[int(start_s * fs):]
    rms = np.sqrt(np.mean(x ** 2))
    return x * (10 ** (target_dbfs / 20) / max(rms, 1e-12))


def score_nonintrusive(x, fs, start_s=0.0, target_dbfs=TARGET_DBFS):
    """prepare_nonintrusive + compute_nonintrusive (claves ausentes si no disponibles)."""
    return compute_nonintrusive(prepare_nonintrusive(x, fs, start_s, target_dbfs), fs)


# Claves en orden de reporte (para armar tablas/CSV de forma estable)
NONINTRUSIVE_KEYS = [
    "DNSMOS_SIG", "DNSMOS_BAK", "DNSMOS_OVRL", "DNSMOS_P808",
    "SQUIM_STOI", "SQUIM_PESQ", "SQUIM_SISDR",
]
