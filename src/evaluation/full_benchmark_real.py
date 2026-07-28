"""
full_benchmark_real.py
=======================
Version ADAPTADA del benchmark (full_benchmark_test_dtln.py) para procesar
senales REALES grabadas con la interfaz de 8 microfonos PDM sobre la Kria KV260.

DIFERENCIAS CLAVE respecto al benchmark sintetico
-------------------------------------------------
El benchmark original genera toda la fisica por simulacion (ISM/SimAcoustic):
crea las RIRs, convoluciona voz limpia + interferentes, y produce referencias
ground-truth (target_anechoic/early/late, interference_early/late, VAD oraculo)
contra las que calcula PESQ/STOI/SDR/SIR/SAR.

Con una grabacion REAL "solo la mezcla" (target + ruido, sin close-talk ni
grabaciones separadas) NO existen esas referencias limpias. Por lo tanto:

  * NO se calculan metricas referenciadas (PESQ/STOI/SDR/SIR) -- no hay ground
    truth. La evaluacion es CUALITATIVA + metricas SIN referencia.
  * Se corre el beamformer CIEGO DTLN-MVDR (basado en mascara neuronal), que NO
    necesita geometria del array ni VAD ni posicion de la fuente.
  * Se guarda el audio procesado (WAV) + el mic de referencia crudo + DTLN mono
    (comparacion) y diagnosticos sin referencia (RMS, SNR segmental estimado en
    silencios via VAD por energia, y espectrogramas PNG).

Los beamformers GEOMETRICOS (DS, MVDR, MPDR, ...) quedan cableados pero
DESACTIVADOS por defecto: requieren mic_coords + source_pos reales. Cuando midas
la geometria (ver build_placeholder_geometry) podes activarlos.

CONTRATO DE ENTRADA (handoff bitacora hardware sec 19.7)
-------------------------------------------------------
  * WAV de 8 canales, PCM int32 entrelazado ch0..ch7, fs=16000.0 EXACTA.
  * muestra i = canal i%8; el periodo arranca en ch0 (alineado muestra a muestra).
  * Ganancia 35 dB LINEAL e IDENTICA en los 8 canales (matcheada).
  * Mapeo logico ch0..ch7 -> posicion fisica: A DEFINIR con el tono por canal
    (bitacora sec 19.6 c'); ver build_placeholder_geometry().

USO
---
    conda activate tesis_beam
    python src/evaluation/full_benchmark_real.py <grabacion_8ch.wav> [output_dir]

o editar DEFAULT_INPUT_WAV / OUTPUT_DIR abajo y correr sin argumentos.
"""

import os
import sys
import time
import contextlib

import numpy as np
import soundfile as sf
import scipy.signal as sig

# --- Asegurar que 'src/' este en el path para los imports del paquete ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import tensorflow as tf
from dnn_denoise.dtln_lite import apply_dtln_post_tflite_realtime
from evaluation.nonintrusive import compute_nonintrusive, NONINTRUSIVE_KEYS
from beamforming.mask.souden_mvdr import MVDR_Souden_recursive_mask
from beamforming.mask.dtln_masks import get_dtln_masks_sharpen
# Variante BAN duplicada con factor de olvido alpha (el original BAN no lo tiene)
from beamforming.mask.souden_mvdr import MVDR_Souden_recursive_mask_BAN_alpha


class DTLN_Souden_MVDR_Processor:
    """
    Beamformer ciego NUEVO: mascara DTLN (variante con sharpening parametrizable,
    get_dtln_masks_sharpen) + algoritmo de MVDR de Souden (MVDR_Souden_recursive_mask).

    No edita ningun algoritmo de procesamiento: solo llama a las funciones
    existentes/duplicadas. Sirve para barrer el exponente de sharpening de la
    mascara sobre la formulacion de Souden.

    Parametros:
      sharpen_exp : exponente de agudizado de la mascara (4.0 = igual al original).
      alpha       : factor de olvido de las covarianzas en Souden.
      min_loading : carga diagonal minima del algoritmo Souden.
    """
    def __init__(self, nperseg=512, noverlap=384, sharpen_exp=4.0, alpha=0.99, min_loading=1e-6):
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.sharpen_exp = sharpen_exp
        self.alpha = alpha
        self.min_loading = min_loading

    def process(self, mic_signals, scene_config):
        fs = scene_config['fs']
        model_path = scene_config.get('dtln_model_path', r'dnn_denoise\models\model_quant_1.tflite')

        nperseg_dyn = scene_config.get('stft_window', self.nperseg)
        noverlap_dyn = scene_config.get('stft_overlap', self.noverlap)
        nfft_dyn = nperseg_dyn
        hop_length_dyn = nperseg_dyn - noverlap_dyn

        M_tot = mic_signals.shape[0]
        ref_mic_idx = M_tot // 2

        # Mascara con la VARIANTE (sharpening parametrizable)
        mask_s, mask_n = get_dtln_masks_sharpen(
            mic_signals, ref_mic_idx, model_path,
            block_len=nperseg_dyn, block_shift=hop_length_dyn,
            sharpen_exp=self.sharpen_exp
        )

        freqs, times, Zxx = sig.stft(
            mic_signals, fs=fs, window='hamming',
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )
        X_stft = np.transpose(Zxx, (1, 2, 0))

        min_frames = min(X_stft.shape[1], mask_s.shape[1])
        X_stft = X_stft[:, :min_frames, :]
        mask_s = mask_s[:, :min_frames]
        mask_n = mask_n[:, :min_frames]

        # Algoritmo de Souden (sin editar): MVDR_Souden_recursive_mask
        Y_stft, weights = MVDR_Souden_recursive_mask(
            X_stft, mask_s, mask_n,
            min_loading=self.min_loading, save_weights=True, alpha=self.alpha
        )

        _, y_time = sig.istft(
            Y_stft, fs=fs, window='hamming',
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )
        y_time = y_time[:mic_signals.shape[1]]
        return y_time, weights


class DTLN_Souden_BAN_MVDR_Processor(DTLN_Souden_MVDR_Processor):
    """
    Igual que DTLN_Souden_MVDR_Processor pero con el algoritmo de Souden + BAN
    (Blind Analytic Normalization, MVDR_Souden_recursive_mask_BAN), que aplica un
    post-filtro/normalizacion pensado para atacar el residual de ruido.
    Reusa la MISMA mascara variante (sharpening parametrizable). No edita ningun
    algoritmo: usa la variante DUPLICADA MVDR_Souden_recursive_mask_BAN_alpha, que
    agrega el factor de olvido `alpha` (el BAN original acumula sin olvido = alpha=1.0).
    """
    def process(self, mic_signals, scene_config):
        fs = scene_config['fs']
        model_path = scene_config.get('dtln_model_path', r'dnn_denoise\models\model_quant_1.tflite')

        nperseg_dyn = scene_config.get('stft_window', self.nperseg)
        noverlap_dyn = scene_config.get('stft_overlap', self.noverlap)
        nfft_dyn = nperseg_dyn
        hop_length_dyn = nperseg_dyn - noverlap_dyn

        M_tot = mic_signals.shape[0]
        ref_mic_idx = M_tot // 2

        mask_s, mask_n = get_dtln_masks_sharpen(
            mic_signals, ref_mic_idx, model_path,
            block_len=nperseg_dyn, block_shift=hop_length_dyn,
            sharpen_exp=self.sharpen_exp
        )

        freqs, times, Zxx = sig.stft(
            mic_signals, fs=fs, window='hamming',
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )
        X_stft = np.transpose(Zxx, (1, 2, 0))

        min_frames = min(X_stft.shape[1], mask_s.shape[1])
        X_stft = X_stft[:, :min_frames, :]
        mask_s = mask_s[:, :min_frames]
        mask_n = mask_n[:, :min_frames]

        # Souden + BAN con factor de olvido alpha (variante duplicada)
        Y_stft, weights = MVDR_Souden_recursive_mask_BAN_alpha(
            X_stft, mask_s, mask_n,
            min_loading=self.min_loading, save_weights=True, alpha=self.alpha
        )

        _, y_time = sig.istft(
            Y_stft, fs=fs, window='hamming',
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )
        y_time = y_time[:mic_signals.shape[1]]
        return y_time, weights

# Geometricos (opcionales, requieren geometria real). Import perezoso mas abajo.

@contextlib.contextmanager
def _quiet():
    """Silencia el stdout verboso de las librerias DTLN (impresion por frame)."""
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            yield


MODELS_DIR = os.path.join(SRC_DIR, "dnn_denoise", "models")
DTLN_MODEL_1 = os.path.join(MODELS_DIR, "model_quant_1.tflite")  # modelo de mascara (usado por DTLN-MVDR)
DTLN_MODEL_2 = os.path.join(MODELS_DIR, "model_quant_2.tflite")  # 2do modelo (solo DTLN mono)


# =============================================================================
# CARGA Y PRE-PROCESO DE LA GRABACION REAL
# =============================================================================
def load_multichannel_wav(path, expected_fs=16000, expected_channels=8):
    """
    Carga un WAV multicanal grabado con pdm_record.
    Devuelve (mic_signals, fs) con mic_signals de forma (M, N) en float [-1, 1].

    pdm_record escribe int32 entrelazado; soundfile lo normaliza a float [-1,1).
    """
    data, fs = sf.read(path, dtype="float64", always_2d=True)  # (N, M)
    mic_signals = data.T.copy()  # (M, N)
    M, N = mic_signals.shape

    if fs != expected_fs:
        print(f"[!] AVISO: fs del WAV = {fs} Hz, se esperaba {expected_fs} Hz. "
              f"Los modelos DTLN asumen 16 kHz -- resultados pueden degradarse.")
    if M != expected_channels:
        print(f"[!] AVISO: el WAV tiene {M} canales, se esperaban {expected_channels}.")

    dur = N / fs
    print(f"[*] Grabacion cargada: {M} canales, {N} muestras ({dur:.2f} s) @ {fs} Hz")
    return mic_signals, fs


def per_channel_rms_normalize(mic_signals, enable=False):
    """
    Normalizacion OPCIONAL por canal (compensa tolerancia de sensibilidad
    +/-3 dB entre mics). Por defecto DESACTIVADA: el hardware ya aplica 35 dB
    lineal e identico en los 8 canales, y para el mask-based DTLN-MVDR la
    mascara sale del mic de referencia. Se deja como toggle para experimentar.
    """
    if not enable:
        return mic_signals
    ref_rms = np.sqrt(np.mean(mic_signals[0] ** 2)) + 1e-12
    out = np.empty_like(mic_signals)
    for m in range(mic_signals.shape[0]):
        rms = np.sqrt(np.mean(mic_signals[m] ** 2)) + 1e-12
        out[m] = mic_signals[m] * (ref_rms / rms)
    print("[*] Normalizacion RMS por canal aplicada (toggle ON).")
    return out


# =============================================================================
# GEOMETRIA PLACEHOLDER (para habilitar beamformers geometricos mas adelante)
# =============================================================================
def build_placeholder_geometry(M=8, fs=16000):
    """
    PLACEHOLDER de geometria -- REEMPLAZAR con las posiciones MEDIDAS del array.

    Layout fisico real (bitacora sec 19.7): 8 mics en 4 lineas DOUT, 2 mics por
    linea (~5 cm el par de una misma linea), sobre protoboard. El mapeo logico
    ch0..ch7 -> posicion fisica se determina con el tono por canal (sec 19.6 c').

    Aca se asume, SOLO como placeholder, un arreglo lineal uniforme sobre X de
    ~30 cm de apertura, centrado en el origen (z=0), con la fuente a broadside
    (perpendicular al eje) a 0.6 m. NO uses esto para conclusiones geometricas:
    medí y sobrescribí mic_coords y source_pos.
    """
    span = 0.30  # m, apertura total (placeholder)
    x = np.linspace(-span / 2, span / 2, M)
    mic_coords = np.zeros((M, 3))
    mic_coords[:, 0] = x
    source_pos = np.array([[0.0, 0.6, 0.0]])  # broadside, 0.6 m (placeholder)
    return mic_coords, source_pos


# =============================================================================
# DIAGNOSTICOS SIN REFERENCIA
# =============================================================================
def energy_vad(x, fs, frame_ms=25.0, hop_ms=10.0, thr_db_below_peak=-25.0):
    """
    VAD binario simple por energia (envolvente de Hilbert), estilo el VAD oraculo
    del simulador pero calculado sobre la MEZCLA (no hay target limpio).
    Se usa SOLO para diagnostico (SNR segmental estimado), NO para el beamformer.
    Devuelve mascara booleana por muestra (True = voz/activo).
    """
    env = np.abs(sig.hilbert(x)) + 1e-12
    # Suavizado con ventana de ~frame_ms
    win = max(1, int(fs * frame_ms / 1000.0))
    kernel = np.ones(win) / win
    env_s = np.convolve(env, kernel, mode="same")
    thr = (10 ** (thr_db_below_peak / 20.0)) * np.max(env_s)
    return env_s > thr


def rms_db(x):
    return 20.0 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)


def segmental_snr_estimate(x, vad):
    """
    SNR segmental ESTIMADO sin referencia: potencia en tramos activos (voz+ruido)
    vs. potencia en tramos inactivos (solo ruido). Es una COTA/estimacion, no una
    metrica ground-truth. Sirve para comparar entrada vs salida de forma relativa.
    """
    if vad.sum() == 0 or (~vad).sum() == 0:
        return np.nan
    p_active = np.mean(x[vad] ** 2) + 1e-12
    p_noise = np.mean(x[~vad] ** 2) + 1e-12
    return 10.0 * np.log10(p_active / p_noise)


def save_spectrograms(signals_dict, fs, out_png):
    """Guarda un panel de espectrogramas (una fila por senal)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[!] No se pudo importar matplotlib para espectrogramas: {e}")
        return

    n = len(signals_dict)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.0 * n), squeeze=False)
    for ax, (name, x) in zip(axes[:, 0], signals_dict.items()):
        f, t, Sxx = sig.spectrogram(x, fs=fs, nperseg=512, noverlap=384)
        ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading="gouraud")
        ax.set_title(name)
        ax.set_ylabel("Hz")
    axes[-1, 0].set_xlabel("Tiempo (s)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"[*] Espectrogramas guardados: {out_png}")


def save_wav_normalized(path, x, fs, peak=0.95):
    """Guarda WAV PCM_16 con normalizacion a pico (para escuchar sin clipping)."""
    m = np.max(np.abs(x)) + 1e-12
    y = (x / m) * peak
    sf.write(path, y.astype(np.float32), fs, subtype="PCM_16")


# =============================================================================
# ORQUESTADOR
# =============================================================================
def run_real_benchmark(input_wav, output_dir, base_config,
                       interpreter_1=None, interpreter_2=None,
                       geometric_processors=None, extra_processors=None):
    os.makedirs(output_dir, exist_ok=True)
    fs_cfg = base_config["fs"]

    # 1. Cargar y (opcional) normalizar la grabacion
    mic_signals, fs = load_multichannel_wav(input_wav, expected_fs=fs_cfg)
    base_config["fs"] = fs  # usar la fs real del archivo
    mic_signals = per_channel_rms_normalize(mic_signals, enable=base_config.get("per_channel_norm", False))
    M = mic_signals.shape[0]

    ref_mic_idx = M // 2  # mic de referencia = canal central
    ref_mic = mic_signals[ref_mic_idx]

    # Geometria placeholder (solo la usan los beamformers geometricos, si se activan)
    mic_coords, source_pos = build_placeholder_geometry(M=M, fs=fs)
    proc_config = base_config.copy()
    proc_config["mic_coords"] = mic_coords
    proc_config["source_pos"] = source_pos
    # VAD de diagnostico (NO se usa para DTLN-MVDR ciego; si para geometricos que lo pidan)
    vad_diag = energy_vad(ref_mic, fs)
    proc_config["VAD"] = vad_diag

    outputs = {}  # nombre -> senal 1D procesada

    # -------------------------------------------------------------------------
    # Referencia: mic crudo (para A/B) y DTLN mono (denoise de un solo canal)
    # -------------------------------------------------------------------------
    outputs["ref_mic_raw"] = ref_mic
    use_dtln = interpreter_1 is not None and interpreter_2 is not None
    if use_dtln:
        print("[*] DTLN mono sobre mic de referencia (comparacion single-channel)...")
        t0 = time.time()
        with _quiet():
            outputs["dtln_mono"] = apply_dtln_post_tflite_realtime(
                interpreter_1=interpreter_1, interpreter_2=interpreter_2, audio_mono=ref_mic
            )
        print(f"    ({time.time() - t0:.1f} s)")

    # Beamformer nuevo: mascara variante (sharpening parametrizable) + Souden
    sharpen_exp = base_config.get("souden_sharpen_exp", 4.0)
    alpha = base_config.get("souden_alpha", 0.99)
    print(f"[*] DTLN-Souden-MVDR (sharpen_exp={sharpen_exp}, alpha={alpha})...")
    t0 = time.time()
    with _quiet():
        y_souden, _w = DTLN_Souden_MVDR_Processor(sharpen_exp=sharpen_exp, alpha=alpha).process(mic_signals, proc_config)
    print(f"    ({time.time() - t0:.1f} s)")
    outputs["dtln_souden_mvdr"] = y_souden

    # Souden + BAN (post-filtro Blind Analytic Normalization) con factor de olvido alpha
    print(f"[*] DTLN-Souden-BAN-MVDR (sharpen_exp={sharpen_exp}, alpha={alpha})...")
    t0 = time.time()
    with _quiet():
        y_ban, _w = DTLN_Souden_BAN_MVDR_Processor(sharpen_exp=sharpen_exp, alpha=alpha).process(mic_signals, proc_config)
    print(f"    ({time.time() - t0:.1f} s)")
    outputs["dtln_souden_ban_mvdr"] = y_ban

    # Cascada BAN -> DTLN: BAN es la etapa espacial mas transparente (SIG alto),
    # el DTLN mono hace la supresion del residual.
    if use_dtln:
        print("[*] DTLN post BAN (cascada BAN -> DTLN)...")
        with _quiet():
            outputs["dtln_souden_ban_then_dtln"] = apply_dtln_post_tflite_realtime(
                interpreter_1=interpreter_1, interpreter_2=interpreter_2, audio_mono=y_ban
            )

    # -------------------------------------------------------------------------
    # BEAMFORMERS GEOMETRICOS (opcionales; requieren geometria MEDIDA)
    # -------------------------------------------------------------------------
    if geometric_processors:
        print("[!] AVISO: corriendo beamformers geometricos con GEOMETRIA PLACEHOLDER. "
              "Medí mic_coords/source_pos antes de sacar conclusiones espaciales.")
        for name, proc in geometric_processors.items():
            print(f"[*] {name} (geometrico)...")
            t0 = time.time()
            with _quiet():
                y, _w = proc.process(mic_signals, proc_config)
            print(f"    ({time.time() - t0:.1f} s)")
            outputs[name] = y

    # -------------------------------------------------------------------------
    # PROCESADORES EXTRA (mask-based / oracle; NO geometricos). Se corren con el
    # mismo proc_config (que puede traer 'oracle_target'/'oracle_noise' para los
    # oracle). No usan mic_coords/source_pos, asi que no aplica el aviso de arriba.
    # -------------------------------------------------------------------------
    if extra_processors:
        for name, proc in extra_processors.items():
            print(f"[*] {name} (extra)...")
            t0 = time.time()
            with _quiet():
                y, _w = proc.process(mic_signals, proc_config)
            print(f"    ({time.time() - t0:.1f} s)")
            outputs[name] = y

    # -------------------------------------------------------------------------
    # SALIDAS: WAVs + diagnosticos sin referencia + espectrogramas
    # -------------------------------------------------------------------------
    print("\n=== Guardando salidas ===")
    for name, x in outputs.items():
        wav_path = os.path.join(output_dir, f"{name}.wav")
        save_wav_normalized(wav_path, np.asarray(x, dtype=np.float64), fs)
        print(f"[*] {wav_path}")

    # Diagnosticos sin referencia (relativos; NO son ground-truth)
    print("\n=== Diagnosticos SIN referencia ===")
    print("[*] Calculando metricas no-intrusivas (DNSMOS / SQUIM)... "
          "(la 1a vez puede tardar por carga de modelos)")
    diag_rows = []
    for name, x in outputs.items():
        x = np.asarray(x, dtype=np.float64)
        # VAD por senal para el segSNR (sobre la propia salida, para comparar)
        vad_x = energy_vad(x, fs)
        seg = segmental_snr_estimate(x, vad_x)
        r = rms_db(x)
        pk = 20 * np.log10(np.max(np.abs(x)) + 1e-12)
        row = {"senal": name, "rms_dbfs": r, "peak_dbfs": pk, "segSNR_est_db": seg}
        # Metricas no-intrusivas (DNSMOS P.835 + SQUIM). NaN si no estan disponibles.
        ni = compute_nonintrusive(x, fs)
        for k in NONINTRUSIVE_KEYS:
            row[k] = ni.get(k, np.nan)
        diag_rows.append(row)

    # Tabla: basicas + no-intrusivas mas relevantes
    print(f"\n{'senal':<24} {'RMS':>7} {'segSNR':>7} "
          f"{'SIG':>6} {'BAK':>6} {'OVRL':>6} {'sqSTOI':>7} {'sqPESQ':>7} {'sqSISDR':>8}")
    for rr in diag_rows:
        def g(k):
            v = rr.get(k, np.nan)
            return "  nan" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.2f}"
        print(f"{rr['senal']:<24} {rr['rms_dbfs']:>7.1f} {rr['segSNR_est_db']:>7.2f} "
              f"{g('DNSMOS_SIG'):>6} {g('DNSMOS_BAK'):>6} {g('DNSMOS_OVRL'):>6} "
              f"{g('SQUIM_STOI'):>7} {g('SQUIM_PESQ'):>7} {g('SQUIM_SISDR'):>8}")

    # CSV resumen (todas las columnas)
    try:
        import csv
        csv_path = os.path.join(output_dir, "diagnostics_real.csv")
        fields = ["senal", "rms_dbfs", "peak_dbfs", "segSNR_est_db"] + NONINTRUSIVE_KEYS
        with open(csv_path, "w", newline="") as fcsv:
            w = csv.DictWriter(fcsv, fieldnames=fields)
            w.writeheader()
            w.writerows(diag_rows)
        print(f"\n[*] Resumen CSV: {csv_path}")
    except Exception as e:
        print(f"[!] No se pudo escribir el CSV: {e}")

    # Espectrogramas comparativos (mic crudo vs salidas principales)
    spec_signals = {"ref_mic_raw (entrada)": np.asarray(outputs["ref_mic_raw"], dtype=np.float64)}
    if "dtln_mono" in outputs:
        spec_signals["dtln_mono"] = np.asarray(outputs["dtln_mono"], dtype=np.float64)
    save_spectrograms(spec_signals, fs, os.path.join(output_dir, "spectrograms_real.png"))

    print("\n=== LISTO ===")
    print("Recorda: sin referencia limpia estas metricas son RELATIVAS/cualitativas. "
          "Escucha los WAV y compara los espectrogramas para evaluar la mejora.")
    return outputs


# =============================================================================
# MAIN
# =============================================================================
# Editar estos defaults o pasar por linea de comandos:
DEFAULT_INPUT_WAV = os.path.join(PROJECT_ROOT, "recordings", "8_mic.wav")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "tests", "real_benchmark_out")

if __name__ == "__main__":
    input_wav = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT_WAV
    output_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_DIR

    if not os.path.isfile(input_wav):
        print(f"[!] No existe el WAV de entrada: {input_wav}")
        print("    Pasa la ruta como argumento: python full_benchmark_real.py <grabacion_8ch.wav> [output_dir]")
        sys.exit(1)

    # DTLN interpreters (para DTLN mono y cascada). Si falla, sigue sin ellos.
    try:
        interpreter_1 = tf.lite.Interpreter(model_path=DTLN_MODEL_1)
        interpreter_1.allocate_tensors()
        interpreter_2 = tf.lite.Interpreter(model_path=DTLN_MODEL_2)
        interpreter_2.allocate_tensors()
        print("[*] Interpretes DTLN TF-Lite cargados.")
    except Exception as e:
        print(f"[!] No se pudieron cargar los modelos DTLN mono. Sigo sin ellos. Detalle: {e}")
        interpreter_1, interpreter_2 = None, None

    base_config = {
        "fs": 16000,
        # STFT alineado con DTLN (block_len=512, block_shift=128 -> overlap=384)
        "stft_window": 512,
        "stft_overlap": 384,
        "dtln_model_path": DTLN_MODEL_1,   # modelo de mascara para DTLN-MVDR
        "per_channel_norm": False,         # toggle normalizacion RMS por canal
        "souden_sharpen_exp": 4.0,         # exponente sharpening mascara del DTLN-Souden-MVDR (4.0 = original)
        "souden_alpha": 0.99,              # factor de olvido de covarianzas (Souden y BAN). 1.0 = acumulativo (sin olvido)
    }

    # Beamformers GEOMETRICOS opcionales. Vacio por defecto (geometria placeholder).
    # Para activarlos cuando midas la geometria, descomenta:
    # from evaluation.bf_wrappers import DS_Processor, MVDR_Recursive_Processor
    # geometric_processors = {"DS": DS_Processor(), "MVDR": MVDR_Recursive_Processor()}
    geometric_processors = {}

    run_real_benchmark(
        input_wav=input_wav,
        output_dir=output_dir,
        base_config=base_config,
        interpreter_1=interpreter_1,
        interpreter_2=interpreter_2,
        geometric_processors=geometric_processors,
    )
