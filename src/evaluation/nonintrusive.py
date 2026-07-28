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


def compute_dnsmos(x, fs):
    """Devuelve dict con DNSMOS_SIG/BAK/OVRL/P808 (o {} si no esta disponible)."""
    if not (os.path.isfile(DNSMOS_PRIMARY) and os.path.isfile(DNSMOS_P808)):
        _warn_once("dnsmos",
                   f"[metricas] DNSMOS deshabilitado: faltan modelos ONNX en {DNSMOS_DIR}\n"
                   f"           (sig_bak_ovr.onnx y model_v8.onnx del repo Microsoft DNS-Challenge/DNSMOS).")
        return {}
    try:
        return _DNSMOS.get()(_to_16k_mono(x, fs))
    except ImportError:
        _warn_once("dnsmos", "[metricas] DNSMOS deshabilitado: falta `onnxruntime` y/o `librosa` (pip install onnxruntime librosa).")
        return {}
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


def compute_squim(x, fs, chunk_s=10.0):
    """
    Devuelve dict con SQUIM_STOI/PESQ/SISDR (estimados, sin referencia).
    Procesa en ventanas de chunk_s y promedia para robustez/memoria.
    """
    try:
        import torch
        model = _SQUIM.get()
    except ImportError:
        _warn_once("squim", "[metricas] SQUIM deshabilitado: falta `torchaudio` (instalar la version que matchee tu torch).")
        return {}
    except Exception as e:
        _warn_once("squim", f"[metricas] SQUIM no disponible: {e}")
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


# Claves en orden de reporte (para armar tablas/CSV de forma estable)
NONINTRUSIVE_KEYS = [
    "DNSMOS_SIG", "DNSMOS_BAK", "DNSMOS_OVRL", "DNSMOS_P808",
    "SQUIM_STOI", "SQUIM_PESQ", "SQUIM_SISDR",
]
