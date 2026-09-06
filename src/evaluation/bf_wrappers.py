import numpy as np
import scipy.signal as sig
# Assuming compute_rtf_steering_vector is imported in your module
from beamforming.signal_model import compute_rtf_steering_vector
from beamforming.MVDR.base import MVDR_recursive
from beamforming.MVDR.oracle_scm import MVDR_geo_oracle_scm
from beamforming.kmvdr.kmvdr_base import KMVDR_recursive
from beamforming.MWF.SP_SDW_MWF_base import sdw_mwf
from beamforming.MPDRxWPE.mpdr import MPDR_recursive
from beamforming.MVDR.RTF_estimation import RTF_MVDR_recursive
from beamforming.MVDR.SPP import SPP_MVDR_recursive
from beamforming.MVDR.SPP_mono import SPP_mono_MVDR_recursive

from beamforming.mask.dtln_masks import (get_dtln_masks, get_dtln_masks_sharpen,
                                        get_dtln_masks_soft, align_mask_frames)
from beamforming.mask.oracle_masks import get_oracle_masks
from beamforming.mask.souden_mvdr import (
    MVDR_Souden_recursive_mask,
    MVDR_Souden_recursive_mask_BAN,
    MVDR_Souden_recursive_mask_slow,
    MVDR_Souden_recursive_mask_specsub_base,
    MVDR_Souden_recursive_mask_BAN_specsub_base,
    MVDR_Souden_recursive_oracle,
    MVDR_Souden_recursive_mask_BAN_alpha,
    MVDR_Souden_recursive_mask_subtract,
    MVDR_Souden_recursive_mask_calibrated,
)
from beamforming.mask.scm_calibration import diffuse_coherence, masks_from_raw
from beamforming.mask.ds_mask import (
    fixed_bf_signal, array_gain, backproject_mask, stretch_sharpen,
    blind_bf_signal, estimate_rtf_recursive,
)
from beamforming.mask.blind_feedback import (
    blind_feedback_stft, DTLNStream,
)
from beamforming.MWF.wiener_postfilter import (
    MVDR_Souden_mask_specsub_MWF, estimate_isir_db, schedule_aggressiveness,
)

# ---------------------------------------------------------------------------
# VENTANA DE LA STFT (analisis/sintesis) -- PARAMETRIZABLE
# ---------------------------------------------------------------------------
# Historicamente TODOS los wrappers usaban window='hamming' hardcodeado. Eso es
# incompatible con acoplar el beamformer y el DTLN en UNA sola STFT: el DTLN
# (tanto get_dtln_masks como apply_dtln_post_tflite_realtime) enmarca con ventana
# RECTANGULAR (in_buffer -> np.fft.rfft, sin ventana) y hace overlap-add por suma
# simple con block_len=512 / block_shift=128.
#
# `resolve_stft_window` deja elegir la ventana sin tocar el default: si no se
# especifica nada devuelve 'hamming' -> resultados IDENTICOS a los previos.
#
# Prioridad: argumento de la instancia (win_type=...) > scene_config['stft_win_type']
# > 'hamming'.
#
# Alias aceptados:
#   'hamming'                       -> hamming (default historico)
#   'hann' / 'hanning'              -> hann
#   'rect' / 'rectangular' / 'boxcar' / 'none' -> boxcar (la del DTLN)
#   'sqrt_hann'                     -> raiz de hann (WOLA simetrico analisis/sintesis)
_WIN_ALIASES = {
    'hamming': 'hamming',
    'hann': 'hann', 'hanning': 'hann',
    'rect': 'boxcar', 'rectangular': 'boxcar', 'boxcar': 'boxcar', 'none': 'boxcar',
    'blackman': 'blackman',
}


def resolve_stft_window(scene_config, override=None, nperseg=512):
    """Devuelve el argumento `window` para scipy.signal.stft/istft.

    scene_config : dict de escena (se lee la clave opcional 'stft_win_type').
    override     : valor de la instancia del wrapper (gana sobre la escena).
    nperseg      : necesario solo para las ventanas construidas a mano.
    """
    w = override if override is not None else scene_config.get('stft_win_type', 'hamming')
    if not isinstance(w, str):
        return w  # ya es un array de ventana
    key = w.strip().lower()
    if key == 'sqrt_hann':
        return np.sqrt(sig.get_window('hann', int(nperseg), fftbins=True))
    return _WIN_ALIASES.get(key, key)


def ola_taper(Y_stft, nperseg, hop, synth, n_out):
    """
    Overlap-add con ventana de SINTESIS distinta de la de analisis.

    Pensado para el camino ANALISIS RECTANGULAR + SINTESIS CON TAPER: el analisis
    rectangular es el que hace falta para tener UNA sola STFT en todo el sistema
    (el DTLN enmarca asi, ver dtln_masks.py) y deja la mascara en el mismo dominio
    que las SCM; el taper de sintesis suprime las discontinuidades de borde de
    frame que aparecen cuando el filtro cambia frame a frame, que es lo unico que
    el all-rect perdia contra hamming (medido: la brecha de PESQ se cierra en 8/8
    escenas MIRD, ver tests/window_mismatch/).

    Cuesta `nperseg` multiplicaciones por frame de salida y NINGUNA transformada
    extra respecto de la iSTFT.

    Y_stft : (K, T) espectro de salida, tal como lo devuelve scipy.signal.stft con
             ventana RECTANGULAR (o sea escalado por 1/nperseg: se deshace aca).
    synth  : nombre de ventana de scipy. Se normaliza por su suma OLA, asi que sin
             modificar Y_stft la reconstruccion es exacta (verificado, 1e-15).
    """
    nperseg, hop = int(nperseg), int(hop)
    w = sig.get_window(synth, nperseg, fftbins=True)
    acc = np.zeros(nperseg)
    for m in range(0, nperseg, hop):
        acc += np.roll(w, m)
    w = w / acc.mean()                      # COLA: sum_m w(n - m*hop) = 1
    frames = np.fft.irfft(Y_stft * nperseg, n=nperseg, axis=0)
    T = Y_stft.shape[1]
    y = np.zeros((T - 1) * hop + nperseg)
    for t in range(T):
        y[t * hop:t * hop + nperseg] += frames[:, t] * w
    # el recorte compensa el boundary='zeros' de scipy.signal.stft
    return y[nperseg // 2:nperseg // 2 + n_out]


class DS:
    """
    Standard Delay-and-Sum beamformer for baseline comparison.
    Produces a completely static spatial filter.
    """
    def __init__(self, nperseg=512, noverlap=384):
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nperseg

    def process(self, mic_signals: np.ndarray, scene_config: dict) -> tuple:
        # Extract configuration parameters
        fs = scene_config['fs']
        source_pos = scene_config['source_pos'].reshape(1, 3)
        mic_coords = scene_config['mic_coords']

        # Dynamic STFT configuration
        nperseg_dyn = scene_config.get('stft_window', self.nperseg)
        noverlap_dyn = scene_config.get('stft_overlap', self.noverlap)
        nfft_dyn = nperseg_dyn

        # Compute STFT
        # Input shape: (M, N_samples). Output shape X_stft: (M, K, T)
        freqs, times, X_stft = sig.stft(
            mic_signals, fs=fs, window='hamming',
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )

        # Transpose to (K, T, M) for spatial frequency-domain processing
        X_stft_ds = np.transpose(X_stft, (1, 2, 0))
        K, T, M = X_stft_ds.shape

        # Compute exact steering vector.
        # ref_mic_idx normaliza la RTF: el DS entrega la voz tal como llega a ESE
        # canal. Se toma el mismo microfono de referencia que usan los mask-based
        # (inyectado por el benchmark en 'ref_mic_idx'); default historico 0.
        sv = compute_rtf_steering_vector(
            freqs, source_pos, mic_coords,
            ref_mic_idx=int(scene_config.get('ref_mic_idx', 0)),
            mode="near_field", squeeze=True
        )

        # Calculate static Delay-and-Sum weights
        # Normalize the steering vector by the number of microphones
        w_ds = sv / M

        # Expand weights across the time axis (static over time)
        # Shape: (K, 1, M) broadcasted to (K, T, M)
        weights = np.broadcast_to(w_ds[:, np.newaxis, :], (K, T, M))

        # Vectorized application of the beamformer
        # Performs the complex conjugate dot product: y = w^H * X
        # 'km,ktm->kt' means: sum over M (m), keeping K (k) and T (t)
        X_hat_stft = np.einsum('km,ktm->kt', np.conj(w_ds), X_stft_ds)

        # Compute ISTFT to return to the time domain
        _, y_time = sig.istft(
            X_hat_stft, fs=fs, window='hamming',
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )

        # Return both the processed 1D signal and the weights matrix
        return y_time, weights




class DTLN_MB_MVDR:
    """
    Wrapper for the DTLN mask-based MVDR beamformer.
    Integrates offline neural mask estimation with recursive spatial filtering.
    """
    def __init__(self, nperseg=512, noverlap=384):
        # STFT configuration aligned with DTLN (block_len=512, block_shift=128)
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nperseg
        self.hop_length = nperseg - noverlap

    def process(self, mic_signals: np.ndarray, scene_config: dict) -> tuple:
        # 1. Extract physical and operational configurations
        fs = scene_config['fs']

        # Dynamically extract DTLN model path from the benchmark config
        # Defaults to a standard path if not provided in the dictionary
        model_path = scene_config.get('dtln_model_path', r'dnn_denoise\models\model_quant_1.tflite')

        # Dynamic STFT configuration
        nperseg_dyn = scene_config.get('stft_window', self.nperseg)
        noverlap_dyn = scene_config.get('stft_overlap', self.noverlap)
        nfft_dyn = nperseg_dyn
        hop_length_dyn = nperseg_dyn - noverlap_dyn

        # Warning to use the same windows as the DTLN model was trained on
        if nperseg_dyn != 512 or hop_length_dyn != 128:
            print(f"[Warning]: Window length ({nperseg_dyn}) and hop length ({hop_length_dyn}) should ideally match DTLN training (512/128).")

        # Define Reference Microphone Index as middle index
        M_tot = mic_signals.shape[0]
        ref_mic_idx = M_tot // 2

        # 2. Extract masks using block_shift (hop_length)
        mask_s, mask_n = get_dtln_masks(
            mic_signals,
            ref_mic_idx,
            model_path,
            block_len=nperseg_dyn,
            block_shift=hop_length_dyn
        )
        # El bloque i del DTLN cubre las mismas muestras que el frame i-1 de
        # la STFT; sin esto a cada frame se le aplica la mascara del anterior.
        mask_s, mask_n = align_mask_frames((mask_s, mask_n),
                                           getattr(self, 'mask_shift', None))

        # 3. Compute STFT
        # Input shape: (M, N_samples). Output shape Zxx: (M, K, T)
        freqs, times, Zxx = sig.stft(
            mic_signals, fs=fs, window='hamming',
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )

        # Transpose to (K, T, M) for spatial frequency-domain processing
        X_stft = np.transpose(Zxx, (1, 2, 0))

        # 4. Ensure time dimensions match between STFT and neural masks
        min_frames = min(X_stft.shape[1], mask_s.shape[1])
        X_stft = X_stft[:, :min_frames, :]
        mask_s = mask_s[:, :min_frames]
        mask_n = mask_n[:, :min_frames]

        # 5. Execute the core mathematical function passing the STFT matrix
        Y_stft, weights = MVDR_recursive_exp_mask_based(
            X_stft,
            mask_s,
            mask_n,
            save_weights=True
        )

        # 6. Compute ISTFT to return to the time domain
        _, y_time = sig.istft(
            Y_stft, fs=fs, window='hamming',
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )

        # 7. Ensure the output length exactly matches the original input signal
        original_length = mic_signals.shape[1]
        y_time = y_time[:original_length]

        return y_time, weights



class DTLN_MB_MVDR_SOUDEN_BAN_alphaless:
    """
    Wrapper for the DTLN mask-based MVDR beamformer.
    Integrates offline neural mask estimation with recursive spatial filtering.
    """
    def __init__(self, nperseg=512, noverlap=384, min_loading=1e-6, win_type=None, alpha = 0.99):
        # STFT configuration aligned with DTLN (block_len=512, block_shift=128)
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nperseg
        self.hop_length = nperseg - noverlap
        self.min_loading = min_loading
        # Ventana de la STFT del beamformer. None -> 'hamming' (historico).
        self.win_type = win_type
        self.alpha = alpha

    def process(self, mic_signals: np.ndarray, scene_config: dict) -> tuple:
        # 1. Extract physical and operational configurations
        fs = scene_config['fs']

        # Dynamically extract DTLN model path from the benchmark config
        # Defaults to a standard path if not provided in the dictionary
        model_path = scene_config.get('dtln_model_path', r'dnn_denoise\models\model_quant_1.tflite')

        # Dynamic STFT configuration
        nperseg_dyn = scene_config.get('stft_window', self.nperseg)
        noverlap_dyn = scene_config.get('stft_overlap', self.noverlap)
        nfft_dyn = nperseg_dyn
        hop_length_dyn = nperseg_dyn - noverlap_dyn

        # Warning to use the same windows as the DTLN model was trained on
        if nperseg_dyn != 512 or hop_length_dyn != 128:
            print(f"[Warning]: Window length ({nperseg_dyn}) and hop length ({hop_length_dyn}) should ideally match DTLN training (512/128).")

        # Define Reference Microphone Index as middle index
        # Microfono de REFERENCIA: fija el canal que reconstruye el filtro (o sea el
        # DOMINIO de la salida) y por lo tanto contra cual hay que medir. El benchmark
        # lo inyecta en 'ref_mic_idx' para que TODOS los procesadores y las metricas
        # usen el mismo canal; sin esa clave se conserva el default historico.
        M_tot = mic_signals.shape[0]
        ref_mic_idx = int(scene_config.get('ref_mic_idx', M_tot // 2))

        # 2. Extract masks using block_shift (hop_length)
        mask_s, mask_n = get_dtln_masks(
            mic_signals,
            ref_mic_idx,
            model_path,
            block_len=nperseg_dyn,
            block_shift=hop_length_dyn
        )
        # El bloque i del DTLN cubre las mismas muestras que el frame i-1 de
        # la STFT; sin esto a cada frame se le aplica la mascara del anterior.
        mask_s, mask_n = align_mask_frames((mask_s, mask_n),
                                           getattr(self, 'mask_shift', None))

        # 3. Compute STFT (ventana configurable: ver resolve_stft_window)
        win_spec = resolve_stft_window(scene_config, self.win_type, nperseg_dyn)
        # Input shape: (M, N_samples). Output shape Zxx: (M, K, T)
        freqs, times, Zxx = sig.stft(
            mic_signals, fs=fs, window=win_spec,
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )

        # Transpose to (K, T, M) for spatial frequency-domain processing
        X_stft = np.transpose(Zxx, (1, 2, 0))

        # 4. Ensure time dimensions match between STFT and neural masks
        min_frames = min(X_stft.shape[1], mask_s.shape[1])
        X_stft = X_stft[:, :min_frames, :]
        mask_s = mask_s[:, :min_frames]
        mask_n = mask_n[:, :min_frames]

        # 5. Execute the core mathematical function passing the STFT matrix
        Y_stft, weights = MVDR_Souden_recursive_mask_BAN(
            X_stft,
            mask_s,
            mask_n,
            min_loading= self.min_loading,
            save_weights=True,
            ref_mic_idx=ref_mic_idx,

        )

        # 6. Compute ISTFT to return to the time domain
        _, y_time = sig.istft(
            Y_stft, fs=fs, window=win_spec,
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )

        # 7. Ensure the output length exactly matches the original input signal
        original_length = mic_signals.shape[1]
        y_time = y_time[:original_length]

        return y_time, weights

class DTLN_MB_MVDR_SOUDEN_BAN:
    """
    Wrapper for the DTLN mask-based MVDR beamformer.
    Integrates offline neural mask estimation with recursive spatial filtering.
    """
    def __init__(self, nperseg=512, noverlap=384, min_loading=1e-6, win_type=None, alpha = 0.99):
        # STFT configuration aligned with DTLN (block_len=512, block_shift=128)
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nperseg
        self.hop_length = nperseg - noverlap
        self.min_loading = min_loading
        # Ventana de la STFT del beamformer. None -> 'hamming' (historico).
        self.win_type = win_type
        self.alpha = alpha

    def process(self, mic_signals: np.ndarray, scene_config: dict) -> tuple:
        # 1. Extract physical and operational configurations
        fs = scene_config['fs']

        # Dynamically extract DTLN model path from the benchmark config
        # Defaults to a standard path if not provided in the dictionary
        model_path = scene_config.get('dtln_model_path', r'dnn_denoise\models\model_quant_1.tflite')

        # Dynamic STFT configuration
        nperseg_dyn = scene_config.get('stft_window', self.nperseg)
        noverlap_dyn = scene_config.get('stft_overlap', self.noverlap)
        nfft_dyn = nperseg_dyn
        hop_length_dyn = nperseg_dyn - noverlap_dyn

        # Warning to use the same windows as the DTLN model was trained on
        if nperseg_dyn != 512 or hop_length_dyn != 128:
            print(f"[Warning]: Window length ({nperseg_dyn}) and hop length ({hop_length_dyn}) should ideally match DTLN training (512/128).")

        # Define Reference Microphone Index as middle index
        # Microfono de REFERENCIA: fija el canal que reconstruye el filtro (o sea el
        # DOMINIO de la salida) y por lo tanto contra cual hay que medir. El benchmark
        # lo inyecta en 'ref_mic_idx' para que TODOS los procesadores y las metricas
        # usen el mismo canal; sin esa clave se conserva el default historico.
        M_tot = mic_signals.shape[0]
        ref_mic_idx = int(scene_config.get('ref_mic_idx', M_tot // 2))

        # 2. Extract masks using block_shift (hop_length)
        mask_s, mask_n = get_dtln_masks(
            mic_signals,
            ref_mic_idx,
            model_path,
            block_len=nperseg_dyn,
            block_shift=hop_length_dyn
        )
        # El bloque i del DTLN cubre las mismas muestras que el frame i-1 de
        # la STFT; sin esto a cada frame se le aplica la mascara del anterior.
        mask_s, mask_n = align_mask_frames((mask_s, mask_n),
                                           getattr(self, 'mask_shift', None))

        # 3. Compute STFT (ventana configurable: ver resolve_stft_window)
        win_spec = resolve_stft_window(scene_config, self.win_type, nperseg_dyn)
        # Input shape: (M, N_samples). Output shape Zxx: (M, K, T)
        freqs, times, Zxx = sig.stft(
            mic_signals, fs=fs, window=win_spec,
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )

        # Transpose to (K, T, M) for spatial frequency-domain processing
        X_stft = np.transpose(Zxx, (1, 2, 0))

        # 4. Ensure time dimensions match between STFT and neural masks
        min_frames = min(X_stft.shape[1], mask_s.shape[1])
        X_stft = X_stft[:, :min_frames, :]
        mask_s = mask_s[:, :min_frames]
        mask_n = mask_n[:, :min_frames]

        # 5. Execute the core mathematical function passing the STFT matrix
        Y_stft, weights = MVDR_Souden_recursive_mask_BAN_alpha(
            X_stft,
            mask_s,
            mask_n,
            min_loading= self.min_loading,
            save_weights=True,
            ref_mic_idx=ref_mic_idx,
            alpha = self.alpha,
        )

        # 6. Compute ISTFT to return to the time domain
        _, y_time = sig.istft(
            Y_stft, fs=fs, window=win_spec,
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )

        # 7. Ensure the output length exactly matches the original input signal
        original_length = mic_signals.shape[1]
        y_time = y_time[:original_length]

        return y_time, weights


class DTLN_MB_MVDR_SOUDEN_SLOW:
    """
    Wrapper for the DTLN mask-based MVDR beamformer.
    Integrates offline neural mask estimation with recursive spatial filtering.
    """
    def __init__(self, nperseg=512, noverlap=384, min_loading=1e-6, alpha=0.99):
        # STFT configuration aligned with DTLN (block_len=512, block_shift=128)
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nperseg
        self.hop_length = nperseg - noverlap
        self.min_loading = min_loading
        self.alpha = alpha

    def process(self, mic_signals: np.ndarray, scene_config: dict) -> tuple:
        # 1. Extract physical and operational configurations
        fs = scene_config['fs']

        # Dynamically extract DTLN model path from the benchmark config
        # Defaults to a standard path if not provided in the dictionary
        model_path = scene_config.get('dtln_model_path', r'dnn_denoise\models\model_quant_1.tflite')

        # Dynamic STFT configuration
        nperseg_dyn = scene_config.get('stft_window', self.nperseg)
        noverlap_dyn = scene_config.get('stft_overlap', self.noverlap)
        nfft_dyn = nperseg_dyn
        hop_length_dyn = nperseg_dyn - noverlap_dyn

        # Warning to use the same windows as the DTLN model was trained on
        if nperseg_dyn != 512 or hop_length_dyn != 128:
            print(f"[Warning]: Window length ({nperseg_dyn}) and hop length ({hop_length_dyn}) should ideally match DTLN training (512/128).")

        # Define Reference Microphone Index as middle index
        # Microfono de REFERENCIA: fija el canal que reconstruye el filtro (o sea el
        # DOMINIO de la salida) y por lo tanto contra cual hay que medir. El benchmark
        # lo inyecta en 'ref_mic_idx' para que TODOS los procesadores y las metricas
        # usen el mismo canal; sin esa clave se conserva el default historico.
        M_tot = mic_signals.shape[0]
        ref_mic_idx = int(scene_config.get('ref_mic_idx', M_tot // 2))

        # 2. Extract masks using block_shift (hop_length)
        mask_s, mask_n = get_dtln_masks(
            mic_signals,
            ref_mic_idx,
            model_path,
            block_len=nperseg_dyn,
            block_shift=hop_length_dyn
        )
        # El bloque i del DTLN cubre las mismas muestras que el frame i-1 de
        # la STFT; sin esto a cada frame se le aplica la mascara del anterior.
        mask_s, mask_n = align_mask_frames((mask_s, mask_n),
                                           getattr(self, 'mask_shift', None))

        # 3. Compute STFT
        # Input shape: (M, N_samples). Output shape Zxx: (M, K, T)
        freqs, times, Zxx = sig.stft(
            mic_signals, fs=fs, window='hamming',
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )

        # Transpose to (K, T, M) for spatial frequency-domain processing
        X_stft = np.transpose(Zxx, (1, 2, 0))

        # 4. Ensure time dimensions match between STFT and neural masks
        min_frames = min(X_stft.shape[1], mask_s.shape[1])
        X_stft = X_stft[:, :min_frames, :]
        mask_s = mask_s[:, :min_frames]
        mask_n = mask_n[:, :min_frames]

        # 5. Execute the core mathematical function passing the STFT matrix
        Y_stft, weights = MVDR_Souden_recursive_mask_slow(
            X_stft,
            mask_s,
            mask_n,
            min_loading= self.min_loading,
            save_weights=True,
            alpha= self.alpha,
            ref_mic_idx=ref_mic_idx
        )

        # 6. Compute ISTFT to return to the time domain
        _, y_time = sig.istft(
            Y_stft, fs=fs, window='hamming',
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )

        # 7. Ensure the output length exactly matches the original input signal
        original_length = mic_signals.shape[1]
        y_time = y_time[:original_length]

        return y_time, weights


class NM_MVDR:
    """
    Wrapper for the DTLN mask-based MVDR beamformer.
    Integrates offline neural mask estimation with recursive spatial filtering.

    El exponente de realce de la mascara (`sharpen_exp`) es ahora un argumento de
    la clase: controla cuan abrupta es la transicion voz/ruido de la mascara DTLN,
    analogo al del Oracle. sharpen_exp=4.0 (default) reproduce EXACTAMENTE el
    comportamiento original (get_dtln_masks tenia el `** 4` fijo). Se puede
    overridear por escena con scene_config['dtln_sharpen_exp'].
    """
    def __init__(self, nperseg=512, noverlap=384, min_loading=1e-6, alpha=0.99, sharpen_exp=4.0,
                 win_type=None, alpha_lf=None, alpha_fsplit_hz=300.0, mask_shift=None):
        # Alineacion mascara <-> STFT. None -> dtln_masks.DTLN_MASK_SHIFT (=1,
        # CORREGIDO). mask_shift=0 reproduce el comportamiento historico, en el
        # que a cada frame se le aplicaba la mascara del frame anterior.
        self.mask_shift = mask_shift
        # STFT configuration aligned with DTLN (block_len=512, block_shift=128)
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nperseg
        self.hop_length = nperseg - noverlap
        self.min_loading = min_loading
        self.alpha = alpha
        self.sharpen_exp = sharpen_exp
        # Ventana de la STFT del beamformer. None -> 'hamming' (comportamiento
        # historico). 'rect'/'boxcar' = la misma que usa el DTLN internamente.
        self.win_type = win_type
        # Factor de olvido dependiente de la frecuencia: alpha_lf debajo de
        # alpha_fsplit_hz, alpha arriba. None -> alpha unico (comportamiento
        # historico, bit a bit identico).
        self.alpha_lf = alpha_lf
        self.alpha_fsplit_hz = alpha_fsplit_hz

    def process(self, mic_signals: np.ndarray, scene_config: dict) -> tuple:
        # 1. Extract physical and operational configurations
        fs = scene_config['fs']

        # Dynamically extract DTLN model path from the benchmark config
        # Defaults to a standard path if not provided in the dictionary
        model_path = scene_config.get('dtln_model_path', r'dnn_denoise\models\model_quant_1.tflite')

        # Dynamic STFT configuration
        nperseg_dyn = scene_config.get('stft_window', self.nperseg)
        noverlap_dyn = scene_config.get('stft_overlap', self.noverlap)
        nfft_dyn = nperseg_dyn
        hop_length_dyn = nperseg_dyn - noverlap_dyn

        # Warning to use the same windows as the DTLN model was trained on
        if nperseg_dyn != 512 or hop_length_dyn != 128:
            print(f"[Warning]: Window length ({nperseg_dyn}) and hop length ({hop_length_dyn}) should ideally match DTLN training (512/128).")

        # --- MICROFONO DE REFERENCIA ---
        # El filtro de Souden proyecta la salida sobre UN canal de referencia: ese
        # microfono define el "punto de escucha" (la voz se estima tal como llega a
        # el). Si el benchmark inyecta 'ref_mic_idx' (p.ej. el mic mas cercano al
        # centro geometrico del arreglo, geometry.select_reference_mic), se usa ese;
        # si no, se cae al default historico M//2.
        M_tot = mic_signals.shape[0]
        ref_mic_idx = int(scene_config.get('ref_mic_idx', M_tot // 2))

        # Exponente de realce de la mascara (controlable por instancia o por escena)
        sharpen_exp = scene_config.get('dtln_sharpen_exp', self.sharpen_exp)

        # 2. Extract masks using block_shift (hop_length)
        # get_dtln_masks_sharpen con sharpen_exp=4.0 == get_dtln_masks original.
        mask_s, mask_n = get_dtln_masks_sharpen(
            mic_signals,
            ref_mic_idx,
            model_path,
            block_len=nperseg_dyn,
            block_shift=hop_length_dyn,
            sharpen_exp=sharpen_exp
        )
        # El bloque i del DTLN cubre las mismas muestras que el frame i-1 de la
        # STFT: sin esto, a cada frame se le aplica la mascara del anterior.
        mask_s, mask_n = align_mask_frames((mask_s, mask_n), self.mask_shift)

        # 3. Compute STFT (ventana configurable: ver resolve_stft_window)
        win_spec = resolve_stft_window(scene_config, self.win_type, nperseg_dyn)
        # Input shape: (M, N_samples). Output shape Zxx: (M, K, T)
        freqs, times, Zxx = sig.stft(
            mic_signals, fs=fs, window=win_spec,
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )

        # Transpose to (K, T, M) for spatial frequency-domain processing
        X_stft = np.transpose(Zxx, (1, 2, 0))

        # 4. Ensure time dimensions match between STFT and neural masks
        min_frames = min(X_stft.shape[1], mask_s.shape[1])
        X_stft = X_stft[:, :min_frames, :]
        mask_s = mask_s[:, :min_frames]
        mask_n = mask_n[:, :min_frames]

        # alpha por bin, si se pidio un alpha distinto para la banda grave.
        alpha_arg = self.alpha
        if self.alpha_lf is not None:
            K_bins = nfft_dyn // 2 + 1
            f_bins = np.arange(K_bins) * fs / nfft_dyn
            alpha_arg = np.where(f_bins < self.alpha_fsplit_hz,
                                 self.alpha_lf, self.alpha)

        # 5. Execute the core mathematical function passing the STFT matrix
        Y_stft, weights = MVDR_Souden_recursive_mask(
            X_stft,
            mask_s,
            mask_n,
            min_loading= self.min_loading,
            save_weights=True,
            alpha=alpha_arg,
            ref_mic_idx=ref_mic_idx
        )

        # 6. Compute ISTFT to return to the time domain
        _, y_time = sig.istft(
            Y_stft, fs=fs, window=win_spec,
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )

        # 7. Ensure the output length exactly matches the original input signal
        original_length = mic_signals.shape[1]
        y_time = y_time[:original_length]

        return y_time, weights


class NM_MVDR_DSM:
    """
    NM_MVDR con la MASCARA ESTIMADA SOBRE UN BEAMFORMER FIJO ("DS-mask").

    Identico a NM_MVDR en TODO el resto de la cadena (mismo core de Souden,
    mismo alpha, mismo ref_mic, mismo framing, mismo stretch + sharpen). La
    UNICA diferencia es que el DTLN no ve el canal de referencia crudo sino la
    salida de un filtro fijo apuntado al target:

        m_raw = DTLN( w_fix^H x )   en vez de   m_raw = DTLN( x_ref )

    `w_fix` sale de la GEOMETRIA y del DOA (que en este sistema lo da la vision),
    no de la senal, asi que no puede realimentar errores de la propia mascara.
    Le entrega al DTLN una entrada con mejor SNR justo donde el arreglo tiene
    ganancia (arriba de ~1 kHz), y eso se traduce en una Phi_NN con menos fuga de
    voz. Medido contra las SCM oracle sobre MIRD (tests/ds_mask_scm_run.py, 4
    escenas): -1.0 dB de perdida con bf_mode="ds" y -1.7 dB con "sd", en 4/4
    escenas, sobre un techo de 3.6 dB (el que marca la mascara ideal).

    NO LLEVA PROYECCION HACIA ATRAS -- y eso esta medido, no asumido. La mascara
    entra al core como PESO de un promedio de outer products, no como ganancia
    sobre una senal: un factor por bin se cancela exacto, y la unica correccion
    con efecto (el corrimiento del punto de operacion, `beta`) EMPEORA el
    resultado de forma monotona en las 4 escenas. Se deja `beta` expuesto para
    poder reproducir esa ablacion, con default 0.

    Parametros propios:
        bf_mode : "ds" delay-and-sum | "sd" superdirectivo cargado (mejor, pero
            paga WNG: a 500 Hz el WNG del "sd" con loading=1e-2 es -4 dB, o sea
            que amplifica el ruido propio de los microfonos; verificar contra el
            SNR real del hardware antes de adoptarlo).
        sd_loading : carga del modo "sd" (1.0 -> DS exacto).
        beta : fuerza de la proyeccion hacia atras (0 = ninguna; ver arriba).
        mask_shift : corrimiento de la mascara en frames. El bloque i del DTLN
            cubre las mismas muestras que el frame i-1 de scipy.stft, pero el
            pipeline los aparea en el MISMO indice; mask_shift=1 lo compensa.
            Default 0 = comportamiento historico (el efecto es ortogonal a este
            wrapper: vale para toda la familia mask-based).

    BAN (ban=True)
    --------------
    Blind Analytical Normalization sobre los pesos del core:

        g_BAN = sqrt( (w^H Phi_NN^2 w) / M ) / (w^H Phi_NN w) ,   w <- g_BAN w

    Un reescalado REAL por (k,t) que lleva la salida al nivel de ruido de UN
    microfono. No cambia la direccion del filtro -- o sea que la SINR
    instantanea no se mueve -- pero si la respuesta en frecuencia efectiva de la
    salida, y con ella PESQ/STOI. Es ortogonal al post-filtro: BAN normaliza la
    ESCALA del beamformer, el PF aplica una GANANCIA por mascara encima; se
    pueden combinar y ese es el punto de tenerlo como flag.

    ban=False (default) deja el camino bit a bit identico al anterior. Con
    core="base" el BAN entra por MVDR_Souden_recursive_mask_BAN_alpha (el core
    base + BAN, mismo alpha y misma carga); con core="subtract", por el flag
    homonimo del core, que lo aplica antes del gate.
    """
    def __init__(self, nperseg=512, noverlap=384, min_loading=1e-6, alpha=0.99,
                 sharpen_exp=4.0, win_type=None, bf_mode="ds", sd_loading=1e-2,
                 beta=0.0, field="spherical", mask_shift=None, core="base",
                 mu=0.0, lambda_floor=1e-3, psd_project=True, ban=False):
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nperseg
        self.hop_length = nperseg - noverlap
        self.min_loading = min_loading
        self.alpha = alpha
        self.sharpen_exp = sharpen_exp
        self.win_type = win_type
        self.bf_mode = bf_mode
        self.sd_loading = sd_loading
        self.beta = beta
        self.field = field
        self.mask_shift = mask_shift
        # core = "base" (MVDR_Souden_recursive_mask, el original) o "subtract"
        # (Phi_SS = Phi_XX - Phi_NN). Los dos efectos son ADITIVOS: medido contra
        # las SCM oracle, la sustraccion aporta ~0.57 dB y el front-end fijo
        # ~1.03 dB, y juntos dan 1.60 dB (ver tests/ds_mask_scm_run.py, columnas
        # L_nu0 / L_nu1). Por eso el default recomendado para el benchmark es
        # core="subtract".
        self.core = core
        self.mu = mu
        self.lambda_floor = lambda_floor
        self.psd_project = psd_project
        # BAN: normalizacion analitica ciega de la escala (ver el docstring).
        self.ban = ban

    def process(self, mic_signals: np.ndarray, scene_config: dict) -> tuple:
        fs = scene_config['fs']
        model_path = scene_config.get('dtln_model_path',
                                      'dnn_denoise/models/model_quant_1.tflite')
        nperseg_dyn = scene_config.get('stft_window', self.nperseg)
        noverlap_dyn = scene_config.get('stft_overlap', self.noverlap)
        nfft_dyn = nperseg_dyn
        hop_length_dyn = nperseg_dyn - noverlap_dyn

        M_tot = mic_signals.shape[0]
        ref_mic_idx = int(scene_config.get('ref_mic_idx', M_tot // 2))
        sharpen_exp = scene_config.get('dtln_sharpen_exp', self.sharpen_exp)
        # La ventana se resuelve ACA (y no en el paso 4) porque el front-end fijo
        # tiene que usar la MISMA que el beamformer: fixed_bf_signal toma
        # window='hamming' por default, asi que sin pasarsela una corrida con
        # win_type='rect' dejaba el front-end en hamming. Con el default
        # (hamming) el camino es identico al anterior.
        win_spec = resolve_stft_window(scene_config, self.win_type, nperseg_dyn)

        # 1. FILTRO FIJO -> senal mono con mejor SNR para el DTLN. La
        # normalizacion distortionless respecto de ref_mic deja esa senal en el
        # MISMO dominio que el canal de referencia (la voz sale como llega ahi),
        # que es lo que hace que la mascara siga siendo la mascara del sistema.
        y_fix, w_fix, f_fix = fixed_bf_signal(
            mic_signals, scene_config['mic_coords'],
            np.asarray(scene_config['source_pos']).reshape(1, 3), fs,
            ref_mic_idx=ref_mic_idx, nperseg=nperseg_dyn, noverlap=noverlap_dyn,
            window=win_spec,
            mode=self.bf_mode, loading=self.sd_loading, field=self.field)

        # 2. Mascara sobre esa senal (mismo DTLN, mismo framing)
        m_raw, _ = get_dtln_masks_soft(y_fix[None, :], 0, model_path,
                                       block_len=nperseg_dyn,
                                       block_shift=hop_length_dyn)
        if self.beta != 0.0:
            ag, _ = array_gain(w_fix, scene_config['mic_coords'], f_fix,
                               field=self.field)
            m_raw = backproject_mask(m_raw, np.clip(ag, 1.0, float(M_tot)),
                                     beta=self.beta)
        # 3. Post-proceso IDENTICO al del sistema actual (stretch global + **n)
        mask_s, mask_n = stretch_sharpen(m_raw, sharpen_exp=sharpen_exp)
        mask_s, mask_n = align_mask_frames((mask_s, mask_n), self.mask_shift)

        # 4. STFT del beamformer y core de Souden: sin cambios
        freqs, times, Zxx = sig.stft(mic_signals, fs=fs, window=win_spec,
                                     nperseg=nperseg_dyn, noverlap=noverlap_dyn,
                                     nfft=nfft_dyn)
        X_stft = np.transpose(Zxx, (1, 2, 0))
        min_frames = min(X_stft.shape[1], mask_s.shape[1])
        X_stft = X_stft[:, :min_frames, :]
        mask_s, mask_n = mask_s[:, :min_frames], mask_n[:, :min_frames]

        if self.core == "subtract":
            Y_stft, weights = MVDR_Souden_recursive_mask_subtract(
                X_stft, mask_s, mask_n, min_loading=self.min_loading,
                save_weights=True, alpha=self.alpha, mu=self.mu,
                lambda_floor=self.lambda_floor, psd_project=self.psd_project,
                ref_mic_idx=ref_mic_idx, ban=self.ban)
        elif self.core == "base":
            # El core base + BAN ya existe como funcion propia (misma recursion,
            # mismo alpha y misma carga que MVDR_Souden_recursive_mask).
            core_fn = (MVDR_Souden_recursive_mask_BAN_alpha if self.ban
                       else MVDR_Souden_recursive_mask)
            Y_stft, weights = core_fn(
                X_stft, mask_s, mask_n, min_loading=self.min_loading,
                save_weights=True, alpha=self.alpha, ref_mic_idx=ref_mic_idx)
        else:
            raise ValueError(f"core desconocido: {self.core!r} ('base' | 'subtract')")

        _, y_time = sig.istft(Y_stft, fs=fs, window=win_spec, nperseg=nperseg_dyn,
                              noverlap=noverlap_dyn, nfft=nfft_dyn)
        return y_time[:mic_signals.shape[1]], weights


class NM_MVDR_SUB:
    """
    NM_MVDR con SUSTRACCION DE COVARIANZA (Phi_SS = Phi_XX - Phi_NN).

    Identico a NM_MVDR en TODO el resto de la cadena (mascara DTLN sharpen,
    framing STFT, alineacion de frames, ISTFT); la unica diferencia es el core:
    MVDR_Souden_recursive_mask_subtract en vez de MVDR_Souden_recursive_mask.

    Corrige el colapso de escala de la normalizacion de Souden a baja frecuencia:
    el core estandar normaliza por lambda = lambda_S + M, que esta acotado por
    abajo por M, asi que en los bins donde la mascara no encuentra voz el filtro
    degenera a w = u/M (la banda sale -20*log10(M) dB con ganancia de arreglo
    nula). Ver el docstring del core y tests/lowfreq_diagnostic_run.py.

    mu : trade-off PMWF. 0 = MVDR distortionless (la correccion pura), 1 = MWF,
         M = mismo denominador que el core actual. Barrer entre 0 y 1.
    """
    def __init__(self, nperseg=512, noverlap=384, min_loading=1e-9, alpha=0.99,
                 sharpen_exp=4.0, win_type=None, mu=0.0, lambda_floor=1e-3,
                 psd_project=True, rank1=False, gate_thresh=None, gate_sharp=2.0,
                 gate_fmax_hz=None, smooth=None, alpha_lf=None, alpha_fsplit_hz=300.0,
                 mask_shift=None):
        # Alineacion mascara <-> STFT (ver dtln_masks.align_mask_frames).
        # None -> DTLN_MASK_SHIFT (=1, corregido); 0 = comportamiento historico.
        self.mask_shift = mask_shift
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nperseg
        self.hop_length = nperseg - noverlap
        self.min_loading = min_loading
        self.alpha = alpha
        self.sharpen_exp = sharpen_exp
        self.win_type = win_type
        self.mu = mu
        self.lambda_floor = lambda_floor
        self.psd_project = psd_project
        self.rank1 = rank1
        # Gate por confianza (lambda_S/M). None -> desactivado.
        self.gate_thresh = gate_thresh
        self.gate_sharp = gate_sharp
        # Tope de frecuencia OPCIONAL para el gate (red de seguridad; el gate por
        # lambda es agnostico a la frecuencia por diseno). None -> sin tope.
        self.gate_fmax_hz = gate_fmax_hz
        # POST-FILTRO de sustraccion espectral sobre la SALIDA, identico al de
        # NM_MVDR_PF: G = smooth + (1-smooth)*mask_orig. None -> sin post-filtro.
        # Es la etapa que debe hacerse cargo de la banda que el gate devuelve en
        # passthrough (abajo de ~130 Hz no hay nada espacial que extraer).
        self.smooth = smooth
        # Factor de olvido DEPENDIENTE DE LA FRECUENCIA. alpha_lf se aplica debajo
        # de alpha_fsplit_hz, alpha arriba. Motivacion: el campo a baja frecuencia
        # decorrelaciona mas lento (la coherencia se mantiene sobre lambda/2), asi
        # que promediar mas tiempo ahi casi no cuesta tracking. None -> alpha unico.
        self.alpha_lf = alpha_lf
        self.alpha_fsplit_hz = alpha_fsplit_hz

    def process(self, mic_signals: np.ndarray, scene_config: dict) -> tuple:
        fs = scene_config['fs']
        model_path = scene_config.get('dtln_model_path',
                                      r'dnn_denoise\models\model_quant_1.tflite')

        nperseg_dyn = scene_config.get('stft_window', self.nperseg)
        noverlap_dyn = scene_config.get('stft_overlap', self.noverlap)
        nfft_dyn = nperseg_dyn
        hop_length_dyn = nperseg_dyn - noverlap_dyn

        M_tot = mic_signals.shape[0]
        ref_mic_idx = int(scene_config.get('ref_mic_idx', M_tot // 2))
        sharpen_exp = scene_config.get('dtln_sharpen_exp', self.sharpen_exp)

        mask_s, mask_n = get_dtln_masks_sharpen(
            mic_signals, ref_mic_idx, model_path,
            block_len=nperseg_dyn, block_shift=hop_length_dyn,
            sharpen_exp=sharpen_exp
        )
        # El bloque i del DTLN cubre las mismas muestras que el frame i-1 de la
        # STFT: sin esto, a cada frame se le aplica la mascara del anterior.
        mask_s, mask_n = align_mask_frames((mask_s, mask_n), self.mask_shift)

        win_spec = resolve_stft_window(scene_config, self.win_type, nperseg_dyn)
        freqs, times, Zxx = sig.stft(
            mic_signals, fs=fs, window=win_spec,
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )
        X_stft = np.transpose(Zxx, (1, 2, 0))       # (K, T, M)

        min_frames = min(X_stft.shape[1], mask_s.shape[1])
        X_stft = X_stft[:, :min_frames, :]
        mask_s = mask_s[:, :min_frames]
        mask_n = mask_n[:, :min_frames]

        # alpha por bin, si se pidio un alpha distinto para graves.
        alpha_arg = self.alpha
        if self.alpha_lf is not None:
            K_bins = nfft_dyn // 2 + 1
            f_bins = np.arange(K_bins) * fs / nfft_dyn
            alpha_arg = np.where(f_bins < self.alpha_fsplit_hz,
                                 self.alpha_lf, self.alpha)

        # Tope del gate en indice de bin (df = fs / nfft).
        gate_kmax = None
        if self.gate_fmax_hz is not None:
            gate_kmax = int(np.floor(self.gate_fmax_hz * nfft_dyn / fs))

        Y_stft, weights = MVDR_Souden_recursive_mask_subtract(
            X_stft, mask_s, mask_n,
            min_loading=self.min_loading,
            save_weights=True,
            alpha=alpha_arg,
            mu=self.mu,
            lambda_floor=self.lambda_floor,
            psd_project=self.psd_project,
            rank1=self.rank1,
            ref_mic_idx=ref_mic_idx,
            gate_thresh=self.gate_thresh,
            gate_sharp=self.gate_sharp,
            gate_kmax=gate_kmax,
        )

        # Post-filtro espectral (misma ganancia que NM_MVDR_PF). mask_s_soft es la
        # mascara ORIGINAL del DTLN, sin el realce que usa el beamformer.
        if self.smooth is not None:
            mask_s_soft = np.clip(mask_s ** (1.0 / sharpen_exp), 0.0, 1.0)
            Tm = min(Y_stft.shape[1], mask_s_soft.shape[1])
            G = self.smooth + (1.0 - self.smooth) * mask_s_soft[:, :Tm]
            Y_stft = Y_stft.copy()
            Y_stft[:, :Tm] *= G

        _, y_time = sig.istft(
            Y_stft, fs=fs, window=win_spec,
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )
        return y_time[:mic_signals.shape[1]], weights


class NM_MVDR_CAL:
    """
    NM_MVDR con la transformacion mascara -> covarianza CALIBRADA por banda.

    Identico a NM_MVDR_SUB en TODA la cadena (mascara DTLN sharpen, framing STFT,
    alineacion de frames, ISTFT, post-filtro opcional); la unica diferencia es el
    core: MVDR_Souden_recursive_mask_calibrated, que generaliza la sustraccion con
    dos parametros POR BIN ajustados contra las SCM oracle por el banco de
    `beamforming/mask/scm_calibration.py`:

        nu_k    escala de la sustraccion, Phi_SS = Phi_XX - nu_k Phi_NN.
                nu = 0 -> core base (NM_MVDR); nu = 1 -> NM_MVDR_SUB.
        gamma_k shrinkage de Phi_NN hacia la coherencia de campo difuso, que se
                deriva de las POSICIONES de los microfonos (scene_config['mic_coords']).

    De donde salen los parametros
    -----------------------------
      calib_path : ruta al .npz que escribe tests/scm_calibration_run.py
                   (claves: freqs, nu_k, gamma_k). Se INTERPOLAN en frecuencia,
                   asi que el .npz puede haberse ajustado con otra STFT.
      nu / gamma : alternativamente, escalares o arrays (K,) explicitos. Sirven
                   para ablaciones: nu=1, gamma=0 reproduce NM_MVDR_SUB exacto.
    Si se pasan ambos, mandan `nu` / `gamma` explicitos.

    field : modelo de campo difuso, "spherical" (sinc, isotropico 3D) o
        "cylindrical" (J0, isotropico en el plano). Tiene que coincidir con el
        que se uso al calibrar.
    """
    # La calibracion de este wrapper (.npz) se AJUSTO con el desfasaje de
    # 1 frame entre la mascara del DTLN y la STFT todavia puesto, asi que
    # se lo deja pinneado hasta rehacer el ajuste. Ver
    # dtln_masks.align_mask_frames y tests/ds_mask_benchmark.py.
    mask_shift = 0

    def __init__(self, nperseg=512, noverlap=384, min_loading=1e-9, alpha=0.99,
                 sharpen_exp=4.0, win_type=None, mu=0.0, lambda_floor=1e-3,
                 psd_project=True, calib_path=None, nu=None, gamma=None,
                 field="spherical", smooth=None, alpha_lf=None,
                 alpha_fsplit_hz=300.0):
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nperseg
        self.hop_length = nperseg - noverlap
        self.min_loading = min_loading
        self.alpha = alpha
        self.sharpen_exp = sharpen_exp
        self.win_type = win_type
        self.mu = mu
        self.lambda_floor = lambda_floor
        self.psd_project = psd_project
        self.field = field
        self.smooth = smooth
        self.alpha_lf = alpha_lf
        self.alpha_fsplit_hz = alpha_fsplit_hz

        self.nu, self.gamma = nu, gamma
        self._calib_freqs = None
        if calib_path is not None and (nu is None or gamma is None):
            z = np.load(calib_path, allow_pickle=True)
            self._calib_freqs = np.asarray(z["freqs"], dtype=float)
            if self.nu is None:
                self.nu = np.asarray(z["nu_k"], dtype=float)
            if self.gamma is None:
                self.gamma = np.asarray(z["gamma_k"], dtype=float)
        if self.nu is None:
            raise ValueError("NM_MVDR_CAL necesita calib_path o nu/gamma explicitos.")
        if self.gamma is None:
            self.gamma = 0.0

    def _resolve_params(self, freqs):
        """Lleva nu_k / gamma_k a la grilla de frecuencias de ESTA corrida."""
        def _fit(p):
            p = np.asarray(p, dtype=float)
            if p.ndim == 0:
                return float(p)
            if p.shape == freqs.shape:
                return p
            if self._calib_freqs is None or p.shape != self._calib_freqs.shape:
                raise ValueError(
                    f"parametro de calibracion de shape {p.shape} incompatible con "
                    f"{freqs.shape} bins y sin eje de frecuencias para interpolar.")
            # Interpolacion lineal en frecuencia: el ajuste es por BANDA (constante
            # a tramos), asi que interpolar solo suaviza los escalones.
            return np.interp(freqs, self._calib_freqs, p)
        return _fit(self.nu), _fit(self.gamma)

    def process(self, mic_signals: np.ndarray, scene_config: dict) -> tuple:
        fs = scene_config['fs']
        model_path = scene_config.get('dtln_model_path',
                                      r'dnn_denoise\models\model_quant_1.tflite')

        nperseg_dyn = scene_config.get('stft_window', self.nperseg)
        noverlap_dyn = scene_config.get('stft_overlap', self.noverlap)
        nfft_dyn = nperseg_dyn
        hop_length_dyn = nperseg_dyn - noverlap_dyn

        M_tot = mic_signals.shape[0]
        ref_mic_idx = int(scene_config.get('ref_mic_idx', M_tot // 2))
        sharpen_exp = scene_config.get('dtln_sharpen_exp', self.sharpen_exp)

        mask_s, mask_n = get_dtln_masks_sharpen(
            mic_signals, ref_mic_idx, model_path,
            block_len=nperseg_dyn, block_shift=hop_length_dyn,
            sharpen_exp=sharpen_exp
        )
        # El bloque i del DTLN cubre las mismas muestras que el frame i-1 de
        # la STFT; sin esto a cada frame se le aplica la mascara del anterior.
        mask_s, mask_n = align_mask_frames((mask_s, mask_n),
                                           getattr(self, 'mask_shift', None))

        win_spec = resolve_stft_window(scene_config, self.win_type, nperseg_dyn)
        freqs, times, Zxx = sig.stft(
            mic_signals, fs=fs, window=win_spec,
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )
        X_stft = np.transpose(Zxx, (1, 2, 0))       # (K, T, M)

        min_frames = min(X_stft.shape[1], mask_s.shape[1])
        X_stft = X_stft[:, :min_frames, :]
        mask_s = mask_s[:, :min_frames]
        mask_n = mask_n[:, :min_frames]

        nu_k, gamma_k = self._resolve_params(freqs)

        # El target del shrinkage sale de la GEOMETRIA. Solo se construye si hace
        # falta, para no exigir 'mic_coords' en corridas con gamma = 0.
        Gamma = None
        if np.any(np.asarray(gamma_k) > 0.0):
            if 'mic_coords' not in scene_config:
                raise KeyError("NM_MVDR_CAL con gamma > 0 necesita "
                               "scene_config['mic_coords'] (posiciones de los micros).")
            Gamma = diffuse_coherence(scene_config['mic_coords'], freqs,
                                      field=self.field)

        alpha_arg = self.alpha
        if self.alpha_lf is not None:
            alpha_arg = np.where(freqs < self.alpha_fsplit_hz, self.alpha_lf, self.alpha)

        Y_stft, weights = MVDR_Souden_recursive_mask_calibrated(
            X_stft, mask_s, mask_n,
            nu=nu_k, gamma=gamma_k, Gamma_diff=Gamma,
            min_loading=self.min_loading,
            save_weights=True,
            alpha=alpha_arg,
            mu=self.mu,
            lambda_floor=self.lambda_floor,
            psd_project=self.psd_project,
            ref_mic_idx=ref_mic_idx,
        )

        # Post-filtro espectral opcional (misma ganancia que NM_MVDR_PF).
        if self.smooth is not None:
            mask_s_soft = np.clip(mask_s ** (1.0 / sharpen_exp), 0.0, 1.0)
            Tm = min(Y_stft.shape[1], mask_s_soft.shape[1])
            G = self.smooth + (1.0 - self.smooth) * mask_s_soft[:, :Tm]
            Y_stft = Y_stft.copy()
            Y_stft[:, :Tm] *= G

        _, y_time = sig.istft(
            Y_stft, fs=fs, window=win_spec,
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )
        return y_time[:mic_signals.shape[1]], weights


class NM_MVDR_MCAL:
    """
    NM_MVDR con la MASCARA calibrada (etapa 2 del banco) -- el reemplazo del
    `stretch` global + `** 4` de `get_dtln_masks_sharpen`.

    QUE CAMBIA RESPECTO DE NM_MVDR_SUB
    ----------------------------------
    Solo COMO se construyen mask_s y mask_n a partir de la salida del DTLN. El
    beamformer, el framing, el ref_mic y alpha son identicos.

    Camino ACTUAL (el que se reemplaza):
        m      = (m_raw - min(m_raw)) / (max(m_raw) - min(m_raw))   <- GLOBAL
        mask_s = m ** 4 ;  mask_n = (1 - m) ** 4
    El stretch usa el min/max de TODO el archivo y de TODAS las frecuencias: no
    es causal (mira el futuro), depende del archivo entero y acopla los bins.
    No es implementable en el sistema online que se lleva a HLS.

    Camino NUEVO (causal, sin estado global, ajustado contra las SCM oracle):
        mask_s = sigma(a_s * logit(m_raw) + b_s)
        mask_n = sigma(a_n * logit(1 - m_raw) + b_n)
    con (a, b) POR BIN. Lo que salio del ajuste (tests/scm_mask_calibration_run.py):
    la rama de VOZ quiere la identidad (a_s ~ 1, b_s ~ 0: la mascara CRUDA tal
    cual), y la de RUIDO un a_n ~ 2 con b_n ~ -8, o sea un ODDS-RATIO RECORTADO:
    crece como ((1-m)/m)^{a_n} en el grueso de las celdas y SATURA en 1 para las
    de ruido mas confiable. El recorte no es un detalle -- el odds-ratio puro
    (sin sigmoide) rinde 0.46 dB PEOR, porque un punado de celdas pasa a dominar
    Phi_NN. b_n fija donde recorta.

    De donde salen los parametros
    -----------------------------
      calib_path : .npz de tests/scm_mask_calibration_run.py (a_s, b_s, a_n, b_n,
                   freqs). Se interpolan en frecuencia.
      const_a_n  : ignora la tabla y usa (a_s=1, b_s=0, a_n=const_a_n, b_n=b_n_const).
                   Es la version SIN tabla por banda -- mas simple de defender y
                   de implementar en hardware. Ver el barrido en el informe.

    nu/gamma se pasan al core calibrado (default nu=1, gamma=0 = el punto de
    NM_MVDR_SUB, que es donde se ajusto la mascara).
    """
    # La calibracion de este wrapper (.npz) se AJUSTO con el desfasaje de
    # 1 frame entre la mascara del DTLN y la STFT todavia puesto, asi que
    # se lo deja pinneado hasta rehacer el ajuste. Ver
    # dtln_masks.align_mask_frames y tests/ds_mask_benchmark.py.
    mask_shift = 0

    def __init__(self, nperseg=512, noverlap=384, min_loading=1e-9, alpha=0.99,
                 win_type=None, mu=0.0, lambda_floor=1e-3, psd_project=True,
                 calib_path=None, const_a_n=None, b_n_const=-8.0,
                 nu=1.0, gamma=0.0, field="spherical", smooth=None):
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nperseg
        self.hop_length = nperseg - noverlap
        self.min_loading = min_loading
        self.alpha = alpha
        self.win_type = win_type
        self.mu = mu
        self.lambda_floor = lambda_floor
        self.psd_project = psd_project
        self.nu, self.gamma = nu, gamma
        self.field = field
        self.smooth = smooth
        self.const_a_n = const_a_n
        self.b_n_const = b_n_const

        self._cf = None
        self._theta = None
        if const_a_n is None:
            if calib_path is None:
                raise ValueError("NM_MVDR_MCAL necesita calib_path o const_a_n.")
            z = np.load(calib_path, allow_pickle=True)
            self._cf = np.asarray(z["freqs"], dtype=float)
            self._theta = tuple(np.asarray(z[k], dtype=float)
                                for k in ("a_s", "b_s", "a_n", "b_n"))

    def _theta_for(self, freqs):
        if self.const_a_n is not None:
            return (1.0, 0.0, float(self.const_a_n), float(self.b_n_const))
        out = []
        for p in self._theta:
            out.append(p if p.shape == freqs.shape else np.interp(freqs, self._cf, p))
        return tuple(out)

    def process(self, mic_signals: np.ndarray, scene_config: dict) -> tuple:
        fs = scene_config['fs']
        model_path = scene_config.get('dtln_model_path',
                                      r'dnn_denoise\models\model_quant_1.tflite')
        nperseg_dyn = scene_config.get('stft_window', self.nperseg)
        noverlap_dyn = scene_config.get('stft_overlap', self.noverlap)
        nfft_dyn = nperseg_dyn
        hop_length_dyn = nperseg_dyn - noverlap_dyn

        M_tot = mic_signals.shape[0]
        ref_mic_idx = int(scene_config.get('ref_mic_idx', M_tot // 2))

        # mascara CRUDA del DTLN: SIN stretch y SIN sharpen. Es el punto de
        # partida del warp (get_dtln_masks_soft devuelve la salida de la red).
        m_raw, _ = get_dtln_masks_soft(mic_signals, ref_mic_idx, model_path,
                                       block_len=nperseg_dyn,
                                       block_shift=hop_length_dyn)

        win_spec = resolve_stft_window(scene_config, self.win_type, nperseg_dyn)
        freqs, times, Zxx = sig.stft(
            mic_signals, fs=fs, window=win_spec,
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn)
        X_stft = np.transpose(Zxx, (1, 2, 0))

        min_frames = min(X_stft.shape[1], m_raw.shape[1])
        X_stft = X_stft[:, :min_frames, :]
        m_raw = m_raw[:, :min_frames]

        mask_s, mask_n = masks_from_raw(m_raw, *self._theta_for(freqs))
        # El bloque i del DTLN cubre las mismas muestras que el frame i-1 de
        # la STFT; sin esto a cada frame se le aplica la mascara del anterior.
        mask_s, mask_n = align_mask_frames((mask_s, mask_n),
                                           getattr(self, 'mask_shift', None))

        Gamma = None
        if np.any(np.asarray(self.gamma) > 0.0):
            if 'mic_coords' not in scene_config:
                raise KeyError("NM_MVDR_MCAL con gamma > 0 necesita "
                               "scene_config['mic_coords'].")
            Gamma = diffuse_coherence(scene_config['mic_coords'], freqs,
                                      field=self.field)

        Y_stft, weights = MVDR_Souden_recursive_mask_calibrated(
            X_stft, mask_s, mask_n, nu=self.nu, gamma=self.gamma,
            Gamma_diff=Gamma, min_loading=self.min_loading, save_weights=True,
            alpha=self.alpha, mu=self.mu, lambda_floor=self.lambda_floor,
            psd_project=self.psd_project, ref_mic_idx=ref_mic_idx)

        if self.smooth is not None:
            Tm = min(Y_stft.shape[1], m_raw.shape[1])
            G = self.smooth + (1.0 - self.smooth) * np.clip(m_raw[:, :Tm], 0.0, 1.0)
            Y_stft = Y_stft.copy()
            Y_stft[:, :Tm] *= G

        _, y_time = sig.istft(Y_stft, fs=fs, window=win_spec, nperseg=nperseg_dyn,
                              noverlap=noverlap_dyn, nfft=nfft_dyn)
        return y_time[:mic_signals.shape[1]], weights


class ORACLE_MB_MVDR_SOUDEN:
    """
    Espejo de NM_MVDR pero usando MASCARAS ORACLE (ideales)
    calculadas a partir de las señales limpias de referencia en lugar del modelo
    DTLN. Sirve como cota superior / referencia agnostica al modelo neuronal para
    comparar Oracle vs DTLN dentro del mismo pipeline mask-based (Souden MVDR).

    El resto de la cadena (framing STFT, alineacion de frames, algoritmo core,
    ISTFT) es identica a la version DTLN, de modo que la unica diferencia sea el
    ORIGEN de la mascara.

    Requiere en scene_config las señales limpias de referencia:
      - 'oracle_target': (M, N) target limpio (p.ej. target_early + target_late)
      - 'oracle_noise' : (M, N) ruido + interferencia limpio
    Opcional:
      - 'oracle_sharpen_exp': exponente de realce (default 1.0 = mascara suave).
    """
    def __init__(self, nperseg=512, noverlap=384, min_loading=1e-6, alpha=0.99, sharpen_exp=1.0):
        # STFT configuration aligned with DTLN (block_len=512, block_shift=128)
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nperseg
        self.hop_length = nperseg - noverlap
        self.min_loading = min_loading
        self.alpha = alpha
        self.sharpen_exp = sharpen_exp

    def process(self, mic_signals: np.ndarray, scene_config: dict) -> tuple:
        # 1. Extract physical and operational configurations
        fs = scene_config['fs']

        # Dynamic STFT configuration
        nperseg_dyn = scene_config.get('stft_window', self.nperseg)
        noverlap_dyn = scene_config.get('stft_overlap', self.noverlap)
        nfft_dyn = nperseg_dyn
        hop_length_dyn = nperseg_dyn - noverlap_dyn

        # Warning to keep the framing aligned with the DTLN path (fair comparison)
        if nperseg_dyn != 512 or hop_length_dyn != 128:
            print(f"[Warning]: Window length ({nperseg_dyn}) and hop length ({hop_length_dyn}) should ideally match the DTLN path (512/128) for a fair Oracle-vs-DTLN comparison.")

        # --- MICROFONO DE REFERENCIA ---
        # El filtro de Souden proyecta la salida sobre UN canal de referencia: ese
        # microfono define el "punto de escucha" (la voz se estima tal como llega a
        # el). Si el benchmark inyecta 'ref_mic_idx' (p.ej. el mic mas cercano al
        # centro geometrico del arreglo, geometry.select_reference_mic), se usa ese;
        # si no, se cae al default historico M//2.
        M_tot = mic_signals.shape[0]
        ref_mic_idx = int(scene_config.get('ref_mic_idx', M_tot // 2))

        # 2. Retrieve the clean reference signals and build the ORACLE masks
        # These are the ground-truth (pre-hardware/pre-WPE) components; the mask
        # encodes perfect per-T-F SNR knowledge and is applied to the real observation.
        speech_ref = scene_config['oracle_target']
        noise_ref = scene_config['oracle_noise']
        sharpen_exp = scene_config.get('oracle_sharpen_exp', self.sharpen_exp)

        mask_s, mask_n = get_oracle_masks(
            speech_ref,
            noise_ref,
            ref_mic=ref_mic_idx,
            block_len=nperseg_dyn,
            block_shift=hop_length_dyn,
            sharpen_exp=sharpen_exp
        )
        # El bloque i del DTLN cubre las mismas muestras que el frame i-1 de
        # la STFT; sin esto a cada frame se le aplica la mascara del anterior.
        mask_s, mask_n = align_mask_frames((mask_s, mask_n),
                                           getattr(self, 'mask_shift', None))

        # 3. Compute STFT
        # Input shape: (M, N_samples). Output shape Zxx: (M, K, T)
        freqs, times, Zxx = sig.stft(
            mic_signals, fs=fs, window='hamming',
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )

        # Transpose to (K, T, M) for spatial frequency-domain processing
        X_stft = np.transpose(Zxx, (1, 2, 0))

        # 4. Ensure time dimensions match between STFT and oracle masks
        min_frames = min(X_stft.shape[1], mask_s.shape[1])
        X_stft = X_stft[:, :min_frames, :]
        mask_s = mask_s[:, :min_frames]
        mask_n = mask_n[:, :min_frames]

        # 5. Execute the same core mathematical function as the DTLN Souden path
        Y_stft, weights = MVDR_Souden_recursive_mask(
            X_stft,
            mask_s,
            mask_n,
            min_loading=self.min_loading,
            save_weights=True,
            alpha=self.alpha,
            ref_mic_idx=ref_mic_idx
        )

        # 6. Compute ISTFT to return to the time domain
        _, y_time = sig.istft(
            Y_stft, fs=fs, window='hamming',
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )

        # 7. Ensure the output length exactly matches the original input signal
        original_length = mic_signals.shape[1]
        y_time = y_time[:original_length]

        return y_time, weights


class SOUDEN_ORACLE_SCM:
    """
    Souden MVDR con covarianzas ORACLE estimadas DIRECTAMENTE de las señales
    limpias multicanal (target y ruido de referencia), SIN mascara. Es la cota
    superior mas limpia: mientras el oracle-mask todavia estima las SCM desde la
    mezcla ruidosa enmascarada, aca Phi_SS y Phi_NN salen de la estadistica real
    de la señal y el ruido. Los pesos se aplican a la mezcla observada.

    Requiere en scene_config las señales limpias de referencia:
      - 'oracle_target': (M, N) target limpio (p.ej. target_early + target_late)
      - 'oracle_noise' : (M, N) ruido + interferencia limpio
    """
    def __init__(self, nperseg=512, noverlap=384, min_loading=1e-6, alpha=0.99):
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nperseg
        self.hop_length = nperseg - noverlap
        self.min_loading = min_loading
        self.alpha = alpha

    def process(self, mic_signals: np.ndarray, scene_config: dict) -> tuple:
        # 1. Configuraciones
        fs = scene_config['fs']

        nperseg_dyn = scene_config.get('stft_window', self.nperseg)
        noverlap_dyn = scene_config.get('stft_overlap', self.noverlap)
        nfft_dyn = nperseg_dyn

        # 2. Señales limpias de referencia (mismas que usa el oracle-mask)
        speech_ref = np.asarray(scene_config['oracle_target'])
        noise_ref = np.asarray(scene_config['oracle_noise'])

        # 3. STFT de las tres señales: mezcla observada (para filtrar) + limpias (para SCM)
        def _stft(sig_in):
            _, _, Z = sig.stft(
                sig_in, fs=fs, window='hamming',
                nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
            )
            return np.transpose(Z, (1, 2, 0))  # (K, T, M)

        X_stft = _stft(mic_signals)
        S_stft = _stft(speech_ref)
        N_stft = _stft(noise_ref)

        # Microfono de referencia (punto de escucha del filtro de Souden). El
        # benchmark puede inyectar 'ref_mic_idx' (mic mas cercano al centro
        # geometrico del arreglo); default historico M//2.
        ref_mic_idx = int(scene_config.get('ref_mic_idx', mic_signals.shape[0] // 2))

        # 4. Alinear frames al minimo comun
        min_frames = min(X_stft.shape[1], S_stft.shape[1], N_stft.shape[1])
        X_stft = X_stft[:, :min_frames, :]
        S_stft = S_stft[:, :min_frames, :]
        N_stft = N_stft[:, :min_frames, :]

        # 5. Core: covarianzas oracle directas + Souden MVDR
        Y_stft, weights = MVDR_Souden_recursive_oracle(
            X_stft, S_stft, N_stft,
            min_loading=self.min_loading,
            save_weights=True,
            alpha=self.alpha,
            ref_mic_idx=ref_mic_idx
        )

        # 6. ISTFT
        _, y_time = sig.istft(
            Y_stft, fs=fs, window='hamming',
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )

        # 7. Ajustar largo exacto
        original_length = mic_signals.shape[1]
        y_time = y_time[:original_length]

        return y_time, weights


class MVDR_GEO_ORACLE_SCM:
    """
    MVDR clasico GEOMETRICO (steering vector desde la posicion asumida de la fuente)
    pero con la covarianza de ruido ORACLE, estimada directamente de la señal de
    ruido+interferencia limpia multicanal (sin VAD, sin mascara):

        w = Phi_NN^-1 d / (d^H Phi_NN^-1 d)

    Es el eslabon que faltaba entre las dos familias:
      - MVDR_Recursive     : d geometrico + Phi_NN de la mezcla via VAD.
      - ESTE               : d geometrico + Phi_NN oracle  -> aisla el error del
                             MODELO GEOMETRICO (RTF de campo cercano en sala real)
                             con informacion de ruido perfecta.
      - SOUDEN_ORACLE_SCM  : sin d, Phi_SS/Phi_NN oracle -> techo sin geometria.

    Requiere en scene_config:
      - 'oracle_noise' : (M, N) ruido + interferencia limpio (mismo dominio que la
                         señal filtrada; lo arma el benchmark en Node 3/4).
      - 'source_pos', 'mic_coords' : geometria (el benchmark ya inyecta aca el
                         error de DOA/distancia si la grilla lo barre).

    Carga diagonal con la MISMA forma que el resto del repo:
    load = max(rel_loading * tr(Phi_NN)/M, min_loading), con rel_loading como escala
    relativa y min_loading como piso absoluto. El default rel_loading=1e-2 es donde
    este beamformer rinde mejor (ver el docstring del core); pasar 1e-6 para igualar
    el valor que usan los mask-based.
    `ref_mic_idx=None` -> M//2, la referencia de la familia Souden.
    """
    def __init__(self, nperseg=512, noverlap=384, rel_loading=1e-2, min_loading=1e-9,
                 alpha=0.99, ref_mic_idx=None):
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nperseg
        self.hop_length = nperseg - noverlap
        self.rel_loading = rel_loading
        self.min_loading = min_loading
        self.alpha = alpha
        self.ref_mic_idx = ref_mic_idx

    def process(self, mic_signals: np.ndarray, scene_config: dict) -> tuple:
        # 1. Configuraciones
        fs = scene_config['fs']
        source_pos = np.asarray(scene_config['source_pos']).reshape(1, 3)
        mic_coords = scene_config['mic_coords']

        nperseg_dyn = scene_config.get('stft_window', self.nperseg)
        noverlap_dyn = scene_config.get('stft_overlap', self.noverlap)
        nfft_dyn = nperseg_dyn

        # 2. Referencia oracle de ruido (la misma que consumen los otros oracles).
        #    El MVDR clasico solo necesita Phi_NN: la informacion de la voz entra
        #    por el steering vector geometrico, no por una SCM.
        noise_ref = np.asarray(scene_config['oracle_noise'])

        # 3. STFT de la mezcla observada (se filtra) y del ruido limpio (da Phi_NN)
        def _stft(sig_in):
            _, _, Z = sig.stft(
                sig_in, fs=fs, window='hamming',
                nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
            )
            return np.transpose(Z, (1, 2, 0))  # (K, T, M)

        X_stft = _stft(mic_signals)
        N_stft = _stft(noise_ref)

        # 4. Alinear frames al minimo comun
        min_frames = min(X_stft.shape[1], N_stft.shape[1])
        X_stft = X_stft[:, :min_frames, :]
        N_stft = N_stft[:, :min_frames, :]

        # 5. Core: MVDR clasico con d geometrico + Phi_NN oracle
        Y_stft, weights = MVDR_geo_oracle_scm(
            X_stft, N_stft, fs=fs,
            array_geometry=mic_coords,
            source_pos=source_pos,
            rel_loading=self.rel_loading,
            min_loading=self.min_loading,
            alpha=self.alpha,
            ref_mic_idx=(self.ref_mic_idx if self.ref_mic_idx is not None
                         else scene_config.get('ref_mic_idx')),
            save_weights=True
        )

        # 6. ISTFT
        _, y_time = sig.istft(
            Y_stft, fs=fs, window='hamming',
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )

        # 7. Ajustar largo exacto
        original_length = mic_signals.shape[1]
        y_time = y_time[:original_length]

        return y_time, weights


class MVDR_Recursive:
    """
    Wrapper for the isolated MVDR_recursive algorithm.
    Handles STFT/ISTFT transformations, parameter extraction from the
    pipeline config, and ensures the expected output format.
    """
    def __init__(self, nperseg=512, noverlap=256, min_loading=1e-6, rel_loading=1e-2,
                 alpha=0.99):
        self.nperseg = nperseg
        self.noverlap = noverlap
        # Carga diagonal: load = max(rel_loading * tr(R_nn)/M, min_loading). La
        # escala RELATIVA es rel_loading (misma forma que la familia Souden, donde
        # ese rol lo cumple su parametro min_loading); aca min_loading es solo el
        # piso ABSOLUTO del arranque. Defaults = comportamiento historico.
        self.min_loading = min_loading
        self.rel_loading = rel_loading
        self.alpha = alpha

    def process(self, mic_signals: np.ndarray, scene_config: dict) -> tuple:
        """
        Executes the spatial filtering process by acting as a bridge
        between the orchestration pipeline and the core mathematical function.
        """
        # 1. Extract configurations
        fs = scene_config['fs']
        source_pos = scene_config['source_pos'].reshape(1, 3)
        mic_coords = scene_config['mic_coords']
        vad = scene_config['VAD']

        # Microfono de REFERENCIA: fija el canal que reconstruye el filtro (o sea el
        # DOMINIO de la salida) y por lo tanto contra cual hay que medir. El benchmark
        # lo inyecta en 'ref_mic_idx' para que TODOS los procesadores y las metricas
        # usen el mismo canal; sin esa clave se conserva el default historico.
        ref_mic_idx = int(scene_config.get('ref_mic_idx', 0))

        # Dynamic STFT configuration
        nperseg_dyn = scene_config.get('stft_window', self.nperseg)
        noverlap_dyn = scene_config.get('stft_overlap', self.noverlap)
        nfft_dyn = nperseg_dyn
        hop_length_dyn = nperseg_dyn - noverlap_dyn

        # 2. Forward STFT
        # Input shape: (M, N_samples). Output shape Zxx: (M, K, T)
        freqs, times, Zxx = sig.stft(
            mic_signals, fs=fs, window='hamming',
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )

        # Transpose to shape (K, T, M) for spatial processing
        X_stft = np.transpose(Zxx, (1, 2, 0))

        # 3. Pad VAD to avoid index out of bounds during the last STFT frames
        vad_padded = np.pad(vad, (0, nperseg_dyn + hop_length_dyn), mode='constant')

        # 4. Call the isolated core function
        # CRITICAL: We pass save_weights=True to get the matrix for the benchmark H5 files
        Y_stft, weights_rec = MVDR_recursive(
            X_stft=X_stft,
            vad=vad_padded,
            fs=fs,
            array_geometry=mic_coords,
            source_pos=source_pos,
            length_fft=nperseg_dyn,
            hop_length_fft=hop_length_dyn,
            rel_loading=self.rel_loading,
            min_loading=self.min_loading,
            alpha=self.alpha,
            save_weights=True,
            ref_mic_idx=ref_mic_idx
        )

        # 5. Inverse STFT to return to time domain
        _, y_time = sig.istft(
            Y_stft, fs=fs, window='hamming',
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )

        # Ensure the output length exactly matches the original input signal
        original_length = mic_signals.shape[1]
        y_time = y_time[:original_length]

        return y_time, weights_rec

class RTF_MVDR_Recursive:
    """
    Wrapper for the isolated MVDR_recursive algorithm.
    Handles STFT/ISTFT transformations, parameter extraction from the
    pipeline config, and ensures the expected output format.
    """
    def __init__(self, nperseg=512, noverlap=256, min_loading=1e-6):
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.min_loading = min_loading

    def process(self, mic_signals: np.ndarray, scene_config: dict) -> tuple:
        """
        Executes the spatial filtering process by acting as a bridge
        between the orchestration pipeline and the core mathematical function.
        """
        # 1. Extract configurations
        fs = scene_config['fs']
        source_pos = scene_config['source_pos'].reshape(1, 3)
        mic_coords = scene_config['mic_coords']
        vad = scene_config['VAD']

        # Dynamic STFT configuration
        nperseg_dyn = scene_config.get('stft_window', self.nperseg)
        noverlap_dyn = scene_config.get('stft_overlap', self.noverlap)
        nfft_dyn = nperseg_dyn
        hop_length_dyn = nperseg_dyn - noverlap_dyn

        # 2. Forward STFT
        # Input shape: (M, N_samples). Output shape Zxx: (M, K, T)
        freqs, times, Zxx = sig.stft(
            mic_signals, fs=fs, window='hamming',
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )

        # Transpose to shape (K, T, M) for spatial processing
        X_stft = np.transpose(Zxx, (1, 2, 0))

        # 3. Pad VAD to avoid index out of bounds during the last STFT frames
        vad_padded = np.pad(vad, (0, nperseg_dyn + hop_length_dyn), mode='constant')

        # 4. Call the isolated core function
        # CRITICAL: We pass save_weights=True to get the matrix for the benchmark H5 files
        Y_stft, weights_rec = RTF_MVDR_recursive(
            X_stft=X_stft,
            vad=vad_padded,
            fs=fs,
            array_geometry=mic_coords,
            source_pos=source_pos,
            length_fft=nperseg_dyn,
            hop_length_fft=hop_length_dyn,
            save_weights=True
        )

        # 5. Inverse STFT to return to time domain
        _, y_time = sig.istft(
            Y_stft, fs=fs, window='hamming',
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )

        # Ensure the output length exactly matches the original input signal
        original_length = mic_signals.shape[1]
        y_time = y_time[:original_length]

        return y_time, weights_rec


class SPP_MVDR_Recursive:
    """
    Wrapper for the isolated MVDR_recursive algorithm.
    Handles STFT/ISTFT transformations, parameter extraction from the
    pipeline config, and ensures the expected output format.
    """
    def __init__(self, nperseg=512, noverlap=256, min_loading=1e-6):
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.min_loading = min_loading

    def process(self, mic_signals: np.ndarray, scene_config: dict) -> tuple:
        """
        Executes the spatial filtering process by acting as a bridge
        between the orchestration pipeline and the core mathematical function.
        """
        # 1. Extract configurations
        fs = scene_config['fs']
        source_pos = scene_config['source_pos'].reshape(1, 3)
        mic_coords = scene_config['mic_coords']
        vad = scene_config['VAD']

        # Dynamic STFT configuration
        nperseg_dyn = scene_config.get('stft_window', self.nperseg)
        noverlap_dyn = scene_config.get('stft_overlap', self.noverlap)
        nfft_dyn = nperseg_dyn
        hop_length_dyn = nperseg_dyn - noverlap_dyn

        # 2. Forward STFT
        # Input shape: (M, N_samples). Output shape Zxx: (M, K, T)
        freqs, times, Zxx = sig.stft(
            mic_signals, fs=fs, window='hamming',
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )

        # Transpose to shape (K, T, M) for spatial processing
        X_stft = np.transpose(Zxx, (1, 2, 0))

        # 3. Pad VAD to avoid index out of bounds during the last STFT frames
        vad_padded = np.pad(vad, (0, nperseg_dyn + hop_length_dyn), mode='constant')

        # 4. Call the isolated core function
        # CRITICAL: We pass save_weights=True to get the matrix for the benchmark H5 files
        Y_stft, weights_rec = SPP_MVDR_recursive(
            X_stft=X_stft,
            fs=fs,
            array_geometry=mic_coords,
            source_pos=source_pos,
            save_weights=True
        )

        # 5. Inverse STFT to return to time domain
        _, y_time = sig.istft(
            Y_stft, fs=fs, window='hamming',
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )

        # Ensure the output length exactly matches the original input signal
        original_length = mic_signals.shape[1]
        y_time = y_time[:original_length]

        return y_time, weights_rec


class SPP_mono_MVDR_Recursive:
    """
    Wrapper for the isolated MVDR_recursive algorithm.
    Handles STFT/ISTFT transformations, parameter extraction from the
    pipeline config, and ensures the expected output format.
    """
    def __init__(self, nperseg=512, noverlap=256, min_loading=1e-6):
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.min_loading = min_loading

    def process(self, mic_signals: np.ndarray, scene_config: dict) -> tuple:
        """
        Executes the spatial filtering process by acting as a bridge
        between the orchestration pipeline and the core mathematical function.
        """
        # 1. Extract configurations
        fs = scene_config['fs']
        source_pos = scene_config['source_pos'].reshape(1, 3)
        mic_coords = scene_config['mic_coords']
        vad = scene_config['VAD']

        # Dynamic STFT configuration
        nperseg_dyn = scene_config.get('stft_window', self.nperseg)
        noverlap_dyn = scene_config.get('stft_overlap', self.noverlap)
        nfft_dyn = nperseg_dyn
        hop_length_dyn = nperseg_dyn - noverlap_dyn

        # 2. Forward STFT
        # Input shape: (M, N_samples). Output shape Zxx: (M, K, T)
        freqs, times, Zxx = sig.stft(
            mic_signals, fs=fs, window='hamming',
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )

        # Transpose to shape (K, T, M) for spatial processing
        X_stft = np.transpose(Zxx, (1, 2, 0))

        # 3. Pad VAD to avoid index out of bounds during the last STFT frames
        vad_padded = np.pad(vad, (0, nperseg_dyn + hop_length_dyn), mode='constant')

        # 4. Call the isolated core function
        # CRITICAL: We pass save_weights=True to get the matrix for the benchmark H5 files
        Y_stft, weights_rec = SPP_mono_MVDR_recursive(
            X_stft=X_stft,
            fs=fs,
            array_geometry=mic_coords,
            source_pos=source_pos,
            save_weights=True
        )

        # 5. Inverse STFT to return to time domain
        _, y_time = sig.istft(
            Y_stft, fs=fs, window='hamming',
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )

        # Ensure the output length exactly matches the original input signal
        original_length = mic_signals.shape[1]
        y_time = y_time[:original_length]

        return y_time, weights_rec


class KMVDR_Recursive:
    """
    Wrapper for the Kronecker MVDR beamformer.
    Automatically factorizes the number of microphones M into M1 and M2
    to optimize the Kronecker decomposition, and handles STFT/ISTFT.
    """
    def __init__(self, nperseg=512, noverlap=256, target_P=2, alpha=0.95, ALS_iterations=2, beta=1e-3, min_loading=1e-6):
        self.nperseg = nperseg
        self.noverlap = noverlap

        # KMVDR specific parameters
        self.target_P = target_P
        self.alpha = alpha
        self.ALS_iterations = ALS_iterations
        self.beta = beta
        self.min_loading = min_loading

    def _get_optimal_factors(self, M: int) -> tuple:
        """
        Finds the closest integer factors M1 and M2 such that M = M1 * M2.
        This maximizes the degrees of freedom for the Kronecker sub-filters.
        """
        # Start looking from the square root of M downwards
        for i in range(int(np.sqrt(M)), 0, -1):
            if M % i == 0:
                M2 = i
                M1 = M // i
                # By convention, ensure M1 >= M2
                return max(M1, M2), min(M1, M2)
        return M, 1

    def process(self, mic_signals: np.ndarray, scene_config: dict) -> tuple:
        """
        Executes the spatial filtering process.
        """
        # 1. Extract physical and operational configuration
        fs = scene_config['fs']
        source_pos = scene_config['source_pos'].reshape(1, 3)
        mic_coords = scene_config['mic_coords']
        vad = scene_config['VAD']

        M = mic_signals.shape[0]

        # Dynamic STFT configuration
        nperseg_dyn = scene_config.get('stft_window', self.nperseg)
        noverlap_dyn = scene_config.get('stft_overlap', self.noverlap)
        nfft_dyn = nperseg_dyn
        hop_length_dyn = nperseg_dyn - noverlap_dyn

        # 2. Intelligent factorization for Kronecker Sub-arrays
        M1, M2 = self._get_optimal_factors(M)

        # Determine strict mathematical bound for P: P <= min(M1, M2)
        P = min(self.target_P, M1, M2)

        # 3. Forward STFT
        # Input shape: (M, N_samples). Output shape Zxx: (M, K, T)
        freqs, times, Zxx = sig.stft(
            mic_signals, fs=fs, window='hamming',
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )

        # Transpose to shape (K, T, M) for the core function
        X_stft = np.transpose(Zxx, (1, 2, 0))

        # Pad VAD to avoid out-of-bounds indexing in the final STFT frames
        vad_padded = np.pad(vad, (0, nperseg_dyn + hop_length_dyn), mode='constant')

        # 4. Execute the core mathematical function
        Y_stft, weights_rec = KMVDR_recursive(
            X_stft=X_stft,
            vad=vad_padded,
            fs=fs,
            array_geometry=mic_coords,
            source_pos=source_pos,
            M1=M1,
            M2=M2,
            P=P,
            alpha=self.alpha,
            ALS_iterations=self.ALS_iterations,
            beta=self.beta,
            min_loading=self.min_loading,
            length_fft=nperseg_dyn,
            hop_length_fft=hop_length_dyn,
            save_weights=True
        )

        # 5. Inverse STFT to return to the time domain
        _, y_time = sig.istft(
            Y_stft, fs=fs, window='hamming',
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )

        # Ensure exact length match
        original_length = mic_signals.shape[1]
        y_time = y_time[:original_length]

        return y_time, weights_rec


class SDW_MWF:
    """
    Wrapper for the Speech Distortion Weighted Multichannel Wiener Filter (SP-SDW-MWF).
    Acts as a direct pass-through for the time-domain block processing,
    explicitly discarding intermediate signals to optimize memory usage.
    """
    def __init__(self, constrained=True):
        self.constrained = constrained

    def process(self, mic_signals: np.ndarray, scene_config: dict) -> tuple:
        """
        Executes the SP-SDW-MWF algorithm.
        """
        # 1. Extract physical and operational configurations
        fs = scene_config['fs']
        source_pos = scene_config['source_pos'].reshape(1, 3)
        mic_coords = scene_config['mic_coords']
        vad = scene_config['VAD']

        original_length = mic_signals.shape[1]

        # 2. Call the core mathematical function
        # We explicitly request ouput_weights=True.
        # We use '_' to discard the fixed branch, blocking matrix output, and noise estimate.
        # This prevents the wrapper from holding references to large unused arrays.
        _, _, _, z_out, weights_rec = sdw_mwf(
            u=mic_signals,
            vad=vad,
            mic_coords=mic_coords,
            source_pos=source_pos,
            fs=fs,
            constrained=self.constrained,
            ouput_weights=True
        )

        # 3. Ensure the output length exactly matches the original input signal
        # Block processing (Overlap-Save) might pad the output to a multiple of L.
        if len(z_out) > original_length:
            y_time = z_out[:original_length]
        elif len(z_out) < original_length:
            y_time = np.pad(z_out, (0, original_length - len(z_out)), mode='constant')
        else:
            y_time = z_out

        # Return strictly the 1D processed audio and the 3D weights tensor
        return y_time, weights_rec


class MPDR_Recursive:
    """
    Wrapper for the Recursive Minimum Power Distortionless Response (MPDR) beamformer.
    Handles STFT/ISTFT transformations and parameter extraction for the benchmark pipeline.
    Unlike MVDR, this processor does not require a VAD mask.
    """
    def __init__(self, nperseg=512, noverlap=256, beta=1e-3, min_loading=1e-6):
        # STFT configuration
        self.nperseg = nperseg
        self.noverlap = noverlap

        # Algorithm hyper-parameters
        self.beta = beta
        self.min_loading = min_loading

    def process(self, mic_signals: np.ndarray, scene_config: dict) -> tuple:
        """
        Executes the MPDR spatial filtering bridge.
        """
        # 1. Extract physical and operational configurations from scene_config
        fs = scene_config['fs']
        source_pos = scene_config['source_pos'].reshape(1, 3)
        mic_coords = scene_config['mic_coords']

        # Microfono de REFERENCIA: fija el canal que reconstruye el filtro (o sea el
        # DOMINIO de la salida) y por lo tanto contra cual hay que medir. El benchmark
        # lo inyecta en 'ref_mic_idx' para que TODOS los procesadores y las metricas
        # usen el mismo canal; sin esa clave se conserva el default historico.
        ref_mic_idx = int(scene_config.get('ref_mic_idx', 0))

        # Dynamic STFT configuration
        nperseg_dyn = scene_config.get('stft_window', self.nperseg)
        noverlap_dyn = scene_config.get('stft_overlap', self.noverlap)
        nfft_dyn = nperseg_dyn

        # 2. Forward STFT
        # Input shape: (M, N_samples). Output shape Zxx: (M, K, T)
        freqs, times, Zxx = sig.stft(
            mic_signals, fs=fs, window='hamming',
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )

        # Transpose to shape (K, T, M) to match the core function expectations
        X_stft = np.transpose(Zxx, (1, 2, 0))

        # 3. Call the core mathematical function
        # We pass save_weights=True to ensure the tracking matrix is captured for H5 storage
        Y_stft, weights_rec = MPDR_recursive(
            X_stft=X_stft,
            fs=fs,
            array_geometry=mic_coords,
            source_pos=source_pos,
            beta=self.beta,
            min_loading=self.min_loading,
            save_weights=True,
            ref_mic_idx=ref_mic_idx
        )

        # 4. Inverse STFT to return to the time domain
        _, y_time = sig.istft(
            Y_stft, fs=fs, window='hamming',
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )

        # 5. Ensure exact length match with input signal
        original_length = mic_signals.shape[1]
        y_time = y_time[:original_length]

        return y_time, weights_rec


class NM_MVDR_PF:
    """
    NM_MVDR (neural-mask Souden MVDR) + POST-FILTRO (PF) DE SUSTRACCION ESPECTRAL.
    Toma el CORE BASE (MVDR_Souden_recursive_mask) -- el mismo beamformer del ganador
    NM_MVDR -- y le agrega una ganancia espectral suave por bin sobre la salida.

    Motivacion (verificada empiricamente en el barrido de iSNR): la carga diagonal
    RELATIVA CON PISO ABSOLUTO del core base (max(min_loading*tr/M, 1e-9), inv())
    preserva mejor PESQ/SIR que una carga relativa pura y pesada. Sobre ESE
    beamformer se aplica el post-filtro, que sube fuerte PESQ/SIR a costa de algo de
    STOI/SAR (distorsion inherente del gate espectral monocanal).

    Cadena: mascara DTLN sharpened (para el beamformer) -> core base ->
    G = smooth + (1-smooth)*mask_orig, con mask_orig = mask_sharpen ** (1/sharpen_exp).

    Por defecto min_loading=1e-6 (el del core base/ganador). smooth: 1.0 = sin filtro
    (== NM_MVDR exacto); 0.33 (default) = extraccion suave, piso ~-9.6 dB; 0.5 = mas
    balanceado (recupera STOI/SAR resignando poco PESQ).
    """
    def __init__(self, nperseg=512, noverlap=384, min_loading=1e-6, alpha=0.99,
                 sharpen_exp=4.0, smooth=0.33):
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nperseg
        self.hop_length = nperseg - noverlap
        self.min_loading = min_loading
        self.alpha = alpha
        self.sharpen_exp = sharpen_exp
        self.smooth = smooth

    def process(self, mic_signals: np.ndarray, scene_config: dict) -> tuple:
        # 1. Configuraciones
        fs = scene_config['fs']
        model_path = scene_config.get('dtln_model_path', r'dnn_denoise\models\model_quant_1.tflite')

        nperseg_dyn = scene_config.get('stft_window', self.nperseg)
        noverlap_dyn = scene_config.get('stft_overlap', self.noverlap)
        nfft_dyn = nperseg_dyn
        hop_length_dyn = nperseg_dyn - noverlap_dyn
        if nperseg_dyn != 512 or hop_length_dyn != 128:
            print(f"[Warning]: Window length ({nperseg_dyn}) and hop length ({hop_length_dyn}) should ideally match DTLN training (512/128).")

        # sharpen_exp / smooth desde config si estan, si no los del __init__
        sharpen_exp = scene_config.get('dtln_sharpen_exp', self.sharpen_exp)

        # Microfono de REFERENCIA: fija el canal que reconstruye el filtro (o sea el
        # DOMINIO de la salida) y por lo tanto contra cual hay que medir. El benchmark
        # lo inyecta en 'ref_mic_idx' para que TODOS los procesadores y las metricas
        # usen el mismo canal; sin esa clave se conserva el default historico.
        M_tot = mic_signals.shape[0]
        ref_mic_idx = int(scene_config.get('ref_mic_idx', M_tot // 2))

        # 2. Mascara DTLN sharpened (para el beamformer)
        mask_s, mask_n = get_dtln_masks_sharpen(
            mic_signals, ref_mic_idx, model_path,
            block_len=nperseg_dyn, block_shift=hop_length_dyn, sharpen_exp=sharpen_exp
        )
        # El bloque i del DTLN cubre las mismas muestras que el frame i-1 de
        # la STFT; sin esto a cada frame se le aplica la mascara del anterior.
        mask_s, mask_n = align_mask_frames((mask_s, mask_n),
                                           getattr(self, 'mask_shift', None))

        # 3. STFT
        freqs, times, Zxx = sig.stft(
            mic_signals, fs=fs, window='hamming',
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )
        X_stft = np.transpose(Zxx, (1, 2, 0))

        # 4. Alinear frames STFT/mascaras
        min_frames = min(X_stft.shape[1], mask_s.shape[1])
        X_stft = X_stft[:, :min_frames, :]
        mask_s = mask_s[:, :min_frames]
        mask_n = mask_n[:, :min_frames]

        # 5. Mascara ORIGINAL (sin realce) para el post-filtro de sustraccion
        mask_s_soft = np.clip(mask_s ** (1.0 / sharpen_exp), 0.0, 1.0)

        # 6. Core BASE (el del ganador) + sustraccion espectral
        Y_stft, weights = MVDR_Souden_recursive_mask_specsub_base(
            X_stft, mask_s, mask_n, mask_s_soft,
            min_loading=self.min_loading, alpha=self.alpha,
            smooth=self.smooth, save_weights=True, ref_mic_idx=ref_mic_idx
        )

        # 7. ISTFT
        _, y_time = sig.istft(
            Y_stft, fs=fs, window='hamming',
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )

        # 8. Ajustar largo exacto
        original_length = mic_signals.shape[1]
        y_time = y_time[:original_length]

        return y_time, weights


class NM_MVDR_BAN_PF:
    """
    NM_MVDR + BLIND ANALYTICAL NORMALIZATION (BAN) + POST-FILTRO (PF) DE SUSTRACCION
    ESPECTRAL, todo sobre el CORE BASE.

    Analogo a NM_MVDR_PF pero con una etapa de BAN entre el beamformer y el
    post-filtro. Usa MVDR_Souden_recursive_mask_BAN_specsub_base, que aplica la BAN
    con factor de olvido alpha sobre la carga diagonal RELATIVA CON PISO ABSOLUTO
    (max(min_loading*tr/M, 1e-9), inv()) -- el mismo estilo de loading del ganador,
    NO el core _fixed que usa MVDR_Souden_recursive_mask_BAN_specsub.

    Cadena: mascara DTLN sharpened -> BAN (core base) -> G = smooth + (1-smooth)*mask_orig.
    La BAN fija la escala de salida referida al ruido (buena fidelidad de forma de
    onda a SNR alto); el specsub actua DESPUES sobre la magnitud por bin, asi que las
    dos etapas NO se cancelan. smooth: 1.0 = solo BAN base (sin filtro); 0.33 default.
    """
    def __init__(self, nperseg=512, noverlap=384, min_loading=1e-6, alpha=0.99,
                 sharpen_exp=4.0, smooth=0.33):
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nperseg
        self.hop_length = nperseg - noverlap
        self.min_loading = min_loading
        self.alpha = alpha
        self.sharpen_exp = sharpen_exp
        self.smooth = smooth

    def process(self, mic_signals: np.ndarray, scene_config: dict) -> tuple:
        # 1. Configuraciones
        fs = scene_config['fs']
        model_path = scene_config.get('dtln_model_path', r'dnn_denoise\models\model_quant_1.tflite')

        nperseg_dyn = scene_config.get('stft_window', self.nperseg)
        noverlap_dyn = scene_config.get('stft_overlap', self.noverlap)
        nfft_dyn = nperseg_dyn
        hop_length_dyn = nperseg_dyn - noverlap_dyn
        if nperseg_dyn != 512 or hop_length_dyn != 128:
            print(f"[Warning]: Window length ({nperseg_dyn}) and hop length ({hop_length_dyn}) should ideally match DTLN training (512/128).")

        sharpen_exp = scene_config.get('dtln_sharpen_exp', self.sharpen_exp)

        # Microfono de REFERENCIA: fija el canal que reconstruye el filtro (o sea el
        # DOMINIO de la salida) y por lo tanto contra cual hay que medir. El benchmark
        # lo inyecta en 'ref_mic_idx' para que TODOS los procesadores y las metricas
        # usen el mismo canal; sin esa clave se conserva el default historico.
        M_tot = mic_signals.shape[0]
        ref_mic_idx = int(scene_config.get('ref_mic_idx', M_tot // 2))

        # 2. Mascara DTLN sharpened (para el beamformer)
        mask_s, mask_n = get_dtln_masks_sharpen(
            mic_signals, ref_mic_idx, model_path,
            block_len=nperseg_dyn, block_shift=hop_length_dyn, sharpen_exp=sharpen_exp
        )
        # El bloque i del DTLN cubre las mismas muestras que el frame i-1 de
        # la STFT; sin esto a cada frame se le aplica la mascara del anterior.
        mask_s, mask_n = align_mask_frames((mask_s, mask_n),
                                           getattr(self, 'mask_shift', None))

        # 3. STFT
        freqs, times, Zxx = sig.stft(
            mic_signals, fs=fs, window='hamming',
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )
        X_stft = np.transpose(Zxx, (1, 2, 0))

        # 4. Alinear frames STFT/mascaras
        min_frames = min(X_stft.shape[1], mask_s.shape[1])
        X_stft = X_stft[:, :min_frames, :]
        mask_s = mask_s[:, :min_frames]
        mask_n = mask_n[:, :min_frames]

        # 5. Mascara ORIGINAL (sin realce) para el post-filtro de sustraccion
        mask_s_soft = np.clip(mask_s ** (1.0 / sharpen_exp), 0.0, 1.0)

        # 6. Core BASE + BAN + sustraccion espectral
        Y_stft, weights = MVDR_Souden_recursive_mask_BAN_specsub_base(
            X_stft, mask_s, mask_n, mask_s_soft,
            min_loading=self.min_loading, alpha=self.alpha,
            smooth=self.smooth, save_weights=True, ref_mic_idx=ref_mic_idx
        )

        # 7. ISTFT
        _, y_time = sig.istft(
            Y_stft, fs=fs, window='hamming',
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )

        # 8. Ajustar largo exacto
        original_length = mic_signals.shape[1]
        y_time = y_time[:original_length]

        return y_time, weights


class NM_MVDR_PF_MWF:
    """
    NM_MVDR + POST-FILTRO (PF) DE SUSTRACCION ESPECTRAL + WIENER MONOCANAL RELAJADO
    a la salida, o sea el MWF cerrado por descomposicion (W_MWF = W_MVDR * G_wiener).

    Identico a NM_MVDR_PF salvo la etapa final: donde aquel corta con el gate ciego
    G_pf = smooth + (1-smooth)*mask_soft, este agrega una ganancia de Wiener
    decision-directed estimada sobre la SNR RESIDUAL de la salida ya conformada. La
    PSD de ruido se estima DESPUES del PF y se guia con la mascara de ruido del DTLN
    (update congelado en frames de habla), asi que las dos etapas no se pisan.

    Perillas de relajacion (ver beamforming/MWF/wiener_postfilter.py):
      w_gmin_db : piso de ganancia. 0 dB -> Wiener identidad -> reproduce NM_MVDR_PF
                  bit a bit. -6 dB es el punto util; -12 dB destruye PESQ.
      w_osf     : subestimacion del ruido (<1 = mas relajado). 0.3 default.
      w_beta    : olvido del decision-directed (0.98, evita ruido musical).
    """
    def __init__(self, nperseg=512, noverlap=384, min_loading=1e-6, alpha=0.99,
                 sharpen_exp=4.0, smooth=0.33,
                 w_beta=0.98, w_gmin_db=-6.0, w_alpha_n=0.9, w_osf=0.3,
                 w_xi_min_db=-20.0, w_gmin_mask=False, w_smooth_f=0, w_smooth_t=0.0):
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nperseg
        self.hop_length = nperseg - noverlap
        self.min_loading = min_loading
        self.alpha = alpha
        self.sharpen_exp = sharpen_exp
        self.smooth = smooth
        self.w_beta = w_beta
        self.w_gmin_db = w_gmin_db
        self.w_alpha_n = w_alpha_n
        self.w_osf = w_osf
        self.w_xi_min_db = w_xi_min_db
        self.w_gmin_mask = w_gmin_mask
        self.w_smooth_f = w_smooth_f
        self.w_smooth_t = w_smooth_t

    def process(self, mic_signals: np.ndarray, scene_config: dict) -> tuple:
        # 1. Configuraciones
        fs = scene_config['fs']
        model_path = scene_config.get('dtln_model_path', r'dnn_denoise\models\model_quant_1.tflite')

        nperseg_dyn = scene_config.get('stft_window', self.nperseg)
        noverlap_dyn = scene_config.get('stft_overlap', self.noverlap)
        nfft_dyn = nperseg_dyn
        hop_length_dyn = nperseg_dyn - noverlap_dyn
        if nperseg_dyn != 512 or hop_length_dyn != 128:
            print(f"[Warning]: Window length ({nperseg_dyn}) and hop length ({hop_length_dyn}) should ideally match DTLN training (512/128).")

        sharpen_exp = scene_config.get('dtln_sharpen_exp', self.sharpen_exp)

        # Microfono de REFERENCIA: mismo criterio que el resto de los wrappers (el
        # benchmark lo inyecta en 'ref_mic_idx' para alinear metricas y salida).
        M_tot = mic_signals.shape[0]
        ref_mic_idx = int(scene_config.get('ref_mic_idx', M_tot // 2))

        # 2. Mascara DTLN sharpened (para el beamformer)
        mask_s, mask_n = get_dtln_masks_sharpen(
            mic_signals, ref_mic_idx, model_path,
            block_len=nperseg_dyn, block_shift=hop_length_dyn, sharpen_exp=sharpen_exp
        )
        # El bloque i del DTLN cubre las mismas muestras que el frame i-1 de
        # la STFT; sin esto a cada frame se le aplica la mascara del anterior.
        mask_s, mask_n = align_mask_frames((mask_s, mask_n),
                                           getattr(self, 'mask_shift', None))

        # 3. STFT
        freqs, times, Zxx = sig.stft(
            mic_signals, fs=fs, window='hamming',
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )
        X_stft = np.transpose(Zxx, (1, 2, 0))

        # 4. Alinear frames STFT/mascaras
        min_frames = min(X_stft.shape[1], mask_s.shape[1])
        X_stft = X_stft[:, :min_frames, :]
        mask_s = mask_s[:, :min_frames]
        mask_n = mask_n[:, :min_frames]

        # 5. Mascara ORIGINAL (sin realce) para el post-filtro de sustraccion
        mask_s_soft = np.clip(mask_s ** (1.0 / sharpen_exp), 0.0, 1.0)

        # 6. Core BASE + specsub + Wiener DD (MWF)
        Y_stft, weights = MVDR_Souden_mask_specsub_MWF(
            X_stft, mask_s, mask_n, mask_s_soft,
            min_loading=self.min_loading, alpha=self.alpha, smooth=self.smooth,
            w_beta=self.w_beta, w_gmin_db=self.w_gmin_db, w_alpha_n=self.w_alpha_n,
            w_osf=self.w_osf, w_xi_min_db=self.w_xi_min_db,
            w_gmin_mask=self.w_gmin_mask, w_smooth_f=self.w_smooth_f,
            w_smooth_t=self.w_smooth_t,
            save_weights=True, ref_mic_idx=ref_mic_idx
        )

        # 7. ISTFT
        _, y_time = sig.istft(
            Y_stft, fs=fs, window='hamming',
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )

        # 8. Ajustar largo exacto
        return y_time[:mic_signals.shape[1]], weights


class NM_MVDR_PF_MWF_ADAPT:
    """
    Post-filtro MWF con AGRESIVIDAD ADAPTATIVA, programada por el iSIR estimado a
    ciegas en el microfono de referencia.

    Se apoya en lo medido en el barrido de protecciones de STOI: el post-filtro es
    barato en inteligibilidad a iSIR alto (la mascara acierta y el gate solo saca
    ruido) y caro a iSIR bajo (la mascara falla y el gate pega sobre habla). El
    schedule por lo tanto va AGRESIVO ARRIBA y CONSERVADOR ABAJO -- al reves de la
    intuicion habitual de "mas post-filtro cuando hay mas interferencia".

    Por escena: estima el iSIR con la mascara del DTLN (sin ninguna referencia),
    interpola linealmente (smooth, g_min) entre el extremo conservador y el agresivo,
    y corre el core MWF con esa configuracion.

    lo_db / hi_db estan en unidades del ESTIMADOR (que es sesgado respecto del iSIR
    verdadero); se calibran con tests/pf_mwf_adaptive_calib.py.

    log_path : si se pasa, appendea una linea CSV por escena con el iSIR estimado y
               la configuracion elegida. Sirve para calibrar y para auditar despues
               que hizo el schedule en cada celda del barrido.
    """
    def __init__(self, nperseg=512, noverlap=384, min_loading=1e-6, alpha=0.99,
                 sharpen_exp=4.0,
                 lo_db=5.0, hi_db=12.0,
                 smooth_lo=0.60, smooth_hi=0.40,
                 gmin_lo_db=-3.0, gmin_hi_db=-9.0,
                 w_beta=0.98, w_alpha_n=0.9, w_osf=0.3, w_xi_min_db=-20.0,
                 log_path=None):
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nperseg
        self.hop_length = nperseg - noverlap
        self.min_loading = min_loading
        self.alpha = alpha
        self.sharpen_exp = sharpen_exp
        self.lo_db = lo_db
        self.hi_db = hi_db
        self.smooth_lo = smooth_lo
        self.smooth_hi = smooth_hi
        self.gmin_lo_db = gmin_lo_db
        self.gmin_hi_db = gmin_hi_db
        self.w_beta = w_beta
        self.w_alpha_n = w_alpha_n
        self.w_osf = w_osf
        self.w_xi_min_db = w_xi_min_db
        self.log_path = log_path
        self.log = []          # (isir_est, smooth, g_min_db) por escena

    def process(self, mic_signals: np.ndarray, scene_config: dict) -> tuple:
        fs = scene_config['fs']
        model_path = scene_config.get('dtln_model_path', r'dnn_denoise\models\model_quant_1.tflite')

        nperseg_dyn = scene_config.get('stft_window', self.nperseg)
        noverlap_dyn = scene_config.get('stft_overlap', self.noverlap)
        nfft_dyn = nperseg_dyn
        hop_length_dyn = nperseg_dyn - noverlap_dyn

        sharpen_exp = scene_config.get('dtln_sharpen_exp', self.sharpen_exp)

        M_tot = mic_signals.shape[0]
        ref_mic_idx = int(scene_config.get('ref_mic_idx', M_tot // 2))

        mask_s, mask_n = get_dtln_masks_sharpen(
            mic_signals, ref_mic_idx, model_path,
            block_len=nperseg_dyn, block_shift=hop_length_dyn, sharpen_exp=sharpen_exp
        )
        # El bloque i del DTLN cubre las mismas muestras que el frame i-1 de
        # la STFT; sin esto a cada frame se le aplica la mascara del anterior.
        mask_s, mask_n = align_mask_frames((mask_s, mask_n),
                                           getattr(self, 'mask_shift', None))

        freqs, times, Zxx = sig.stft(
            mic_signals, fs=fs, window='hamming',
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )
        X_stft = np.transpose(Zxx, (1, 2, 0))

        min_frames = min(X_stft.shape[1], mask_s.shape[1])
        X_stft = X_stft[:, :min_frames, :]
        mask_s = mask_s[:, :min_frames]
        mask_n = mask_n[:, :min_frames]

        mask_s_soft = np.clip(mask_s ** (1.0 / sharpen_exp), 0.0, 1.0)

        # --- punto de operacion estimado y configuracion elegida -----------------
        isir_est = estimate_isir_db(X_stft[:, :, ref_mic_idx], mask_s_soft)
        smooth, g_min_db = schedule_aggressiveness(
            isir_est, self.lo_db, self.hi_db,
            self.smooth_lo, self.smooth_hi, self.gmin_lo_db, self.gmin_hi_db)
        self.log.append((isir_est, smooth, g_min_db))
        if self.log_path:
            with open(self.log_path, "a") as fh:
                fh.write(f"{isir_est:.4f},{smooth:.4f},{g_min_db:.4f}\n")

        Y_stft, weights = MVDR_Souden_mask_specsub_MWF(
            X_stft, mask_s, mask_n, mask_s_soft,
            min_loading=self.min_loading, alpha=self.alpha, smooth=smooth,
            w_beta=self.w_beta, w_gmin_db=g_min_db, w_alpha_n=self.w_alpha_n,
            w_osf=self.w_osf, w_xi_min_db=self.w_xi_min_db,
            save_weights=True, ref_mic_idx=ref_mic_idx
        )

        _, y_time = sig.istft(
            Y_stft, fs=fs, window='hamming',
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )
        return y_time[:mic_signals.shape[1]], weights


class NM_MVDR_DSM_BLIND:
    """
    NM_MVDR_DSM CIEGO: el front-end que ve el DTLN se apunta con la RTF ESTIMADA
    de la propia SCM de senal, no con la geometria + el DOA.

    QUE PROBLEMA RESUELVE
    ---------------------
    NM_MVDR_DSM gana (+0.20 PESQ / +2.0 SDR / +2.5 SIR en 16/16 celdas MIRD)
    porque le da al DTLN una entrada con mejor SNR, pero para armar w_fix necesita
    scene_config['source_pos']: el estimador de mascara deja de ser CIEGO. Aca el
    apuntamiento se REALIMENTA de lo que la propia cadena ya estima:

        mascara(1) = DTLN(x_ref)                          <- el sistema de hoy
        Phi_SS     = Phi_XX - Phi_NN                      <- la misma sustraccion
        d          = RTF(Phi_SS)                          <- ciega: sin DOA
        y_fix      = w(d)^H x                             <- w^H d = 1
        mascara(2) = DTLN(y_fix)                          <- la que usa el BF
        Y          = core_Souden(X, mascara(2))

    No entra ni la posicion de la fuente ni la geometria del arreglo: solo la
    senal. `n_iter` > 1 repite el lazo (la mascara(2) vuelve a estimar la RTF).

    LOS DOS SEGUROS DEL LAZO (los parametros nuevos)
    ------------------------------------------------
    Realimentar es justamente lo que el front-end geometrico evitaba, asi que la
    matriz de correlacion que se usa para ESTIMAR la RTF -- que NO es la que
    invierte el beamformer -- lleva sus propios controles:

      rtf_loading : carga diagonal RELATIVA AL NIVEL DE RUIDO sobre esa matriz,
          Psi = R_est + rtf_loading * (tr(Phi_NN)/M) * I. Fija el punto de falla:
          donde la senal estimada es debil frente al ruido, d -> e_ref y el
          front-end degrada CONTINUAMENTE al canal de referencia crudo, o sea al
          sistema actual. Un bin sin informacion no puede inventar apuntamiento.
      rtf_alpha : factor de olvido de esa recursion, INDEPENDIENTE del alpha del
          beamformer. La RTF es una propiedad del cuarto, no de la actividad de
          voz: cambia mucho mas lento, asi que conviene promediarla mas tiempo
          (0.999 vs 0.99). Es lo que baja la varianza del estimador, que es la
          fuente real de inestabilidad del lazo.

    rtf_mode : "cs" (columna de Phi_SS), "evd" (autovector principal) o "cw"
        (autovector principal blanqueado por Phi_NN). Ver `ds_mask`.
    w_mode : "ds" (w = d/(d^H d), el analogo directo del DS geometrico) o "mvdr"
        (w = Phi_NN^-1 d / (d^H Phi_NN^-1 d): mas SNR para el DTLN, pero
        realimenta tambien la SCM de ruido).
    core : "subtract" (default, el ganador) o "base".

    POST-FILTRO DE SUSTRACCION ESPECTRAL (smooth)
    ---------------------------------------------
    Misma ganancia que NM_MVDR_PF, aplicada sobre la salida del beamformer:

        G(k,t) = smooth + (1 - smooth) * m_soft(k,t) ,   Y = Y_bf * G

    con m_soft la mascara ORIGINAL del DTLN (sin el realce que usa el
    beamformer: m_soft = mask_sharpen ** (1/sharpen_exp)). smooth=None (default)
    lo desactiva; 1.0 = sin filtro; 0.33 = extraccion suave, piso ~-9.6 dB.

    LA DIFERENCIA CON NM_MVDR_PF ESTA EN QUE MASCARA ES. En NM_MVDR_PF el gate
    lo dicta el DTLN mirando el canal de referencia CRUDO. Aca, con
    pf_mask_src="fix" (default), lo dicta el DTLN mirando la salida del
    front-end apuntado con la RTF ESTIMADA: el mismo lazo que mejora Phi_NN
    mejora tambien el gate monocanal, que es la etapa donde una mascara mala se
    escucha directo (ruido musical, voz recortada). pf_mask_src="ref" usa la
    mascara de la PRIMERA pasada -- o sea exactamente el post-filtro de
    NM_MVDR_PF -- y sirve como ablacion para separar las dos cosas.
    """
    def __init__(self, nperseg=512, noverlap=384, min_loading=1e-9, alpha=0.99,
                 sharpen_exp=4.0, win_type=None, core="subtract", mu=0.0,
                 lambda_floor=1e-3, psd_project=True, mask_shift=None,
                 rtf_alpha=0.999, rtf_loading=1e-2, rtf_mode="cs", w_mode="ds",
                 bf_loading=1e-6, n_iter=1, smooth=None, pf_mask_src="fix",
                 ban=False, mask_warp=None, synth=None, causal=False,
                 conf_gate=None, conf_band=(300.0, 3400.0), conf_smooth=0.9,
                 conf_alpha=None, sd_eps=1e-2, sd_field="spherical"):
        # Ventana de SINTESIS (None = la misma del analisis: comportamiento
        # historico, bit a bit). Ver `ola_taper`.
        self.synth = synth
        # CADENA CAUSAL. El camino historico tiene dos etapas que miran toda la
        # senal y por lo tanto no son implementables online:
        #   * la normalizacion por el PICO GLOBAL del canal antes de enmarcar
        #     (dentro de get_dtln_masks_*), en las DOS pasadas;
        #   * el STRETCH min-max global de la mascara, tambien en las dos.
        # causal=True las reemplaza: escala fija 1.0 (el hardware entrega
        # unidades de fondo de escala; medido: corr 0.999 / MAE 0.009 contra el
        # pico global) y post-proceso PUNTUAL -- la potencia si mask_warp es
        # None, o el warp calibrado si se pasa. Default False = historico.
        self.causal = causal
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nperseg
        self.hop_length = nperseg - noverlap
        self.min_loading = min_loading
        self.alpha = alpha
        self.sharpen_exp = sharpen_exp
        self.win_type = win_type
        self.core = core
        self.mu = mu
        self.lambda_floor = lambda_floor
        self.psd_project = psd_project
        self.mask_shift = mask_shift
        # --- ARRANQUE EN FRIO del estimador de RTF (todo apagado = historico) --
        # Ver `estimate_rtf_recursive`. Atacan el modo de falla medido en
        # tests/window_mismatch/dsm_blind_feedback_diag.py: si el archivo empieza
        # con ruido NO estacionario y la voz entra tarde, la sustraccion no se
        # cancela, d se va de e_ref y la masa contaminada que se acumula tarda
        # 1/(1-rtf_alpha) = 8 s de voz en olvidarse (con 8 s de prefijo no se
        # recupera dentro del archivo: la ganancia del front-end cae a 0 dB).
        self.conf_gate = conf_gate            # gate duro de la rama de senal
        self.conf_band = conf_band
        # --- SUPERDIRECTIVIDAD del front-end (w_mode="sd") ---------------------
        # Semi-ciego: usa la coherencia difusa TEORICA, que sale solo de la
        # geometria del arreglo. NO usa scene_config['source_pos'], asi que el
        # sistema sigue sin saber donde esta la fuente. sd_eps es la carga
        # diagonal: 1 = DS exacto, ->0 = superdirectivo sin restriccion.
        self.sd_eps = sd_eps
        self.sd_field = sd_field
        self.conf_smooth = conf_smooth
        self.conf_alpha = conf_alpha
        # --- lazo de realimentacion ---
        self.rtf_alpha = rtf_alpha
        self.rtf_loading = rtf_loading
        self.rtf_mode = rtf_mode
        self.w_mode = w_mode
        self.bf_loading = bf_loading
        self.n_iter = int(n_iter)
        # --- post-filtro de sustraccion espectral (None = desactivado) ---
        self.smooth = smooth
        self.pf_mask_src = pf_mask_src
        # BAN: ortogonal al post-filtro, se pueden prender juntos.
        self.ban = ban
        # --- POST-PROCESO DE LA MASCARA DE LA SEGUNDA PASADA ------------------
        # None (default) -> `stretch_sharpen`: el camino de PRODUCCION (stretch
        #   min-max GLOBAL sobre (k,t) + potencia). Se conserva como default para
        #   que el wrapper siga reproduciendo el sistema medido.
        # (a_s, b_s, a_n, b_n) -> `masks_from_raw`: el warp CALIBRADO, causal y
        #   sin estado global (ver beamforming/mask/scm_calibration.py). Ademas
        #   de rendir mejor, saca del camino la unica etapa NO CAUSAL que queda:
        #   el stretch necesita el min/max de todo el archivo, asi que hoy este
        #   wrapper no es implementable online tal cual.
        # OJO: el (a_s, b_s, a_n, b_n) ajustado en scm_mask_calibration_run.py se
        # calibro sobre mascaras del canal de referencia CRUDO. Aca la mascara
        # sale del front-end ciego, con mejor SNR y otra distribucion -> hay que
        # RE-BARRER a_n (ver tests/dsm_blind_an_sweep.py) en vez de transplantar.
        self.mask_warp = mask_warp

    def process(self, mic_signals: np.ndarray, scene_config: dict) -> tuple:
        fs = scene_config['fs']
        model_path = scene_config.get('dtln_model_path',
                                      'dnn_denoise/models/model_quant_1.tflite')
        nperseg_dyn = scene_config.get('stft_window', self.nperseg)
        noverlap_dyn = scene_config.get('stft_overlap', self.noverlap)
        nfft_dyn = nperseg_dyn
        hop_length_dyn = nperseg_dyn - noverlap_dyn

        M_tot = mic_signals.shape[0]
        ref_mic_idx = int(scene_config.get('ref_mic_idx', M_tot // 2))
        sharpen_exp = scene_config.get('dtln_sharpen_exp', self.sharpen_exp)
        win_spec = resolve_stft_window(scene_config, self.win_type, nperseg_dyn)

        # 1. PRIMERA PASADA: la mascara del sistema actual, sobre el canal crudo.
        # Solo sirve para estimar la RTF ciega; no llega al beamformer final.
        _causal = getattr(self, 'causal', False)
        mask_s, mask_n = get_dtln_masks_sharpen(
            mic_signals, ref_mic_idx, model_path,
            block_len=nperseg_dyn, block_shift=hop_length_dyn,
            sharpen_exp=sharpen_exp,
            **({'peak_norm': 1.0, 'stretch': False} if _causal else {}))
        mask_s, mask_n = align_mask_frames((mask_s, mask_n), self.mask_shift)
        mask_s_ref = mask_s  # la del canal crudo: ablacion del post-filtro

        # 2. LAZO: mascara -> Phi_SS -> RTF ciega -> front-end -> mascara nueva.
        for _ in range(max(self.n_iter, 0)):
            y_fix = blind_bf_signal(
                mic_signals, mask_s, mask_n, fs, ref_mic_idx=ref_mic_idx,
                nperseg=nperseg_dyn, noverlap=noverlap_dyn, window=win_spec,
                rtf_alpha=self.rtf_alpha, rtf_loading=self.rtf_loading,
                rtf_mode=self.rtf_mode, w_mode=self.w_mode,
                bf_loading=self.bf_loading,
                conf_gate=getattr(self, 'conf_gate', None),
                mic_coords=(scene_config.get('mic_coords')
                            if self.w_mode == "sd" else None),
                sd_eps=getattr(self, 'sd_eps', 1e-2),
                sd_field=getattr(self, 'sd_field', "spherical"),
                conf_band=getattr(self, 'conf_band', (300.0, 3400.0)),
                conf_smooth=getattr(self, 'conf_smooth', 0.9),
                conf_alpha=getattr(self, 'conf_alpha', None))
            m_raw, _ = get_dtln_masks_soft(y_fix[None, :], 0, model_path,
                                           block_len=nperseg_dyn,
                                           block_shift=hop_length_dyn,
                                           **({'peak_norm': 1.0} if _causal else {}))
            if self.mask_warp is None and _causal:
                # Potencia PUNTUAL, sin el stretch global: mismo realce que
                # produccion pero causal.
                m = np.clip(np.asarray(m_raw, dtype=np.float64), 0.0, 1.0)
                mask_s, mask_n = m ** sharpen_exp, (1.0 - m) ** sharpen_exp
            elif self.mask_warp is None:
                # Post-proceso de PRODUCCION (stretch global + sharpen).
                mask_s, mask_n = stretch_sharpen(m_raw, sharpen_exp=sharpen_exp)
            else:
                # Warp calibrado: causal, sin estado global, ramas desacopladas.
                mask_s, mask_n = masks_from_raw(m_raw, *self.mask_warp)
            mask_s, mask_n = align_mask_frames((mask_s, mask_n), self.mask_shift)
            m_raw_last = m_raw

        # 3. Beamformer final: sin cambios respecto de NM_MVDR / NM_MVDR_SUB.
        freqs, times, Zxx = sig.stft(mic_signals, fs=fs, window=win_spec,
                                     nperseg=nperseg_dyn, noverlap=noverlap_dyn,
                                     nfft=nfft_dyn)
        X_stft = np.transpose(Zxx, (1, 2, 0))
        min_frames = min(X_stft.shape[1], mask_s.shape[1])
        X_stft = X_stft[:, :min_frames, :]
        mask_s, mask_n = mask_s[:, :min_frames], mask_n[:, :min_frames]

        if self.core == "subtract":
            Y_stft, weights = MVDR_Souden_recursive_mask_subtract(
                X_stft, mask_s, mask_n, min_loading=self.min_loading,
                save_weights=True, alpha=self.alpha, mu=self.mu,
                lambda_floor=self.lambda_floor, psd_project=self.psd_project,
                ref_mic_idx=ref_mic_idx, ban=self.ban)
        elif self.core == "base":
            # El core base + BAN ya existe como funcion propia (misma recursion,
            # mismo alpha y misma carga que MVDR_Souden_recursive_mask).
            core_fn = (MVDR_Souden_recursive_mask_BAN_alpha if self.ban
                       else MVDR_Souden_recursive_mask)
            Y_stft, weights = core_fn(
                X_stft, mask_s, mask_n, min_loading=self.min_loading,
                save_weights=True, alpha=self.alpha, ref_mic_idx=ref_mic_idx)
        else:
            raise ValueError(f"core desconocido: {self.core!r} ('base' | 'subtract')")

        # 4. POST-FILTRO: misma ganancia espectral que NM_MVDR_PF, pero con la
        # mascara del front-end ciego (ver el docstring).
        if self.smooth is not None:
            if self.pf_mask_src == "fix":
                m_pf = mask_s
            elif self.pf_mask_src == "ref":
                m_pf = mask_s_ref
            else:
                raise ValueError(f"pf_mask_src desconocido: {self.pf_mask_src!r} "
                                 "('fix' | 'ref')")
            # m_soft = la mascara SUAVE (sin realce). Con el post-proceso de
            # produccion se recupera invirtiendo la potencia; con el warp
            # calibrado esa inversion no aplica -> se usa la mascara CRUDA del
            # DTLN, que es la misma cantidad y ya esta a mano.
            if self.mask_warp is not None and self.pf_mask_src == "fix":
                m_soft = np.clip(m_raw_last, 0.0, 1.0)
            else:
                m_soft = np.clip(m_pf ** (1.0 / sharpen_exp), 0.0, 1.0)
            Tm = min(Y_stft.shape[1], m_soft.shape[1])
            G = self.smooth + (1.0 - self.smooth) * m_soft[:, :Tm]
            Y_stft = Y_stft.copy()
            Y_stft[:, :Tm] *= G

        # synth=None -> iSTFT normal. synth='hann' con win_type='rect' habilita
        # ANALISIS RECTANGULAR + SINTESIS CON TAPER (ver `ola_taper`): una sola
        # FFT por canal en toda la cadena, sin perder calidad de reconstruccion.
        n_out = mic_signals.shape[1]
        if getattr(self, 'synth', None) is None:
            _, y_time = sig.istft(Y_stft, fs=fs, window=win_spec, nperseg=nperseg_dyn,
                                  noverlap=noverlap_dyn, nfft=nfft_dyn)
            y_time = y_time[:n_out]
        else:
            y_time = ola_taper(Y_stft, nperseg_dyn, hop_length_dyn, self.synth, n_out)
        return y_time, weights


class NM_MVDR_DSM_FB:
    """
    NM_MVDR_DSM_BLIND con UN SOLO DTLN y el lazo cerrado FRAME A FRAME.

    Es la variante online de `NM_MVDR_DSM_BLIND`. La cadena de dos pasadas corre
    el DTLN dos veces sobre toda la senal, y la primera pasada (sobre el canal
    crudo) no llega al beamformer: es solo un BOOTSTRAP para poder estimar la
    RTF. Aca esa pasada desaparece, porque el lazo se bootstrapea solo: el
    estimador arranca con las dos mascaras en cero, con lo cual Phi_SS = 0,
    d = e_ref y el front-end ES el canal de referencia en el primer frame.

        mascara(t) = DTLN( w(d(t))^H x(t) ) ,   d(t) = RTF( Phi_SS(mascara(t-1)) )

    Ver `beamforming/mask/blind_feedback.py` para el detalle y la verificacion de
    equivalencia contra los cores batch.

    MODOS (`mode`)
    --------------
    Hay DOS cambios respecto de `NM_MVDR_DSM_BLIND` y conviene poder separarlos,
    porque no tienen por que aportar lo mismo:

      "fb"   : el lazo cerrado con UN DTLN. Los dos cambios juntos.
      "spec" : ABLACION. Mantiene las dos pasadas y el bootstrap por el canal
          crudo -- o sea, sin realimentacion -- pero calcula la mascara(2)
          alimentando al DTLN con el ESPECTRO conformado en vez de resintetizar
          y volver a enmarcar. Aisla el efecto de sacar el ida y vuelta al
          tiempo.

    POR QUE EL ESPECTRO SE PUEDE ALIMENTAR DIRECTO
    ----------------------------------------------
    Con analisis RECTANGULAR, el bloque que enmarcaria el DTLN y el frame de la
    STFT son las mismas muestras: rFFT(bloque i) == L * X(:, i-1), con error
    EXACTAMENTE 0 (verificado). Como el bloque i corresponde al frame i-1, la
    mascara sale ya alineada y `align_mask_frames` no hace falta.

    En el camino de dos pasadas, en cambio, la mascara(2) se calcula sobre y_fix
    RESINTETIZADA: como w varia con (k,t), Y_fix no es la STFT de ninguna senal,
    asi que resintetizar y re-analizar es una proyeccion que mezcla 4 frames
    filtrados. No es lo mismo, y por eso existe el modo "spec".

    EL LAZO CERRADO NO TIENE ANCLA
    ------------------------------
    En la cadena de dos pasadas la mascara(1) es independiente del lazo: pase lo
    que pase con la realimentacion, las SCM del estimador se pesan con algo que
    no se puede corromper. Aca no. Por eso `conf_gate` deja de ser opcional --
    es el seguro que corta la rama de senal mientras no haya evidencia -- y
    conviene prenderlo si la escena puede arrancar con ruido no estacionario.
    """

    def __init__(self, nperseg=512, noverlap=384, min_loading=1e-9, alpha=0.99,
                 sharpen_exp=8.0, win_type='rect', synth='hann', mu=0.0,
                 lambda_floor=1e-3, psd_project=True, rtf_alpha=0.999,
                 rtf_loading=1e-2, rtf_mode="cs", w_mode="ds", bf_loading=1e-6,
                 smooth=None, ban=False, mask_warp=None, conf_gate=None,
                 conf_band=(300.0, 3400.0), conf_smooth=0.9, conf_alpha=None,
                 sd_eps=1e-2, sd_field="spherical", mode="fb"):
        if mode not in ("fb", "spec"):
            raise ValueError(f"mode desconocido: {mode!r} ('fb' | 'spec')")
        self.mode = mode
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nperseg
        self.hop_length = nperseg - noverlap
        self.min_loading = min_loading
        self.alpha = alpha
        self.sharpen_exp = sharpen_exp
        self.win_type = win_type
        self.synth = synth
        self.mu = mu
        self.lambda_floor = lambda_floor
        self.psd_project = psd_project
        self.rtf_alpha = rtf_alpha
        self.rtf_loading = rtf_loading
        self.rtf_mode = rtf_mode
        self.w_mode = w_mode
        self.bf_loading = bf_loading
        self.smooth = smooth
        self.ban = ban
        self.mask_warp = mask_warp
        self.conf_gate = conf_gate
        self.conf_band = conf_band
        self.conf_smooth = conf_smooth
        self.conf_alpha = conf_alpha
        self.sd_eps = sd_eps
        self.sd_field = sd_field

    def process(self, mic_signals: np.ndarray, scene_config: dict) -> tuple:
        fs = scene_config['fs']
        model_path = scene_config.get('dtln_model_path',
                                      'dnn_denoise/models/model_quant_1.tflite')
        nperseg_dyn = scene_config.get('stft_window', self.nperseg)
        noverlap_dyn = scene_config.get('stft_overlap', self.noverlap)
        hop_dyn = nperseg_dyn - noverlap_dyn

        M_tot = mic_signals.shape[0]
        ref_mic_idx = int(scene_config.get('ref_mic_idx', M_tot // 2))
        sharpen_exp = scene_config.get('dtln_sharpen_exp', self.sharpen_exp)
        win_spec = resolve_stft_window(scene_config, self.win_type, nperseg_dyn)

        # UNA sola STFT en toda la cadena (analisis rectangular).
        freqs, _, Zxx = sig.stft(mic_signals, fs=fs, window=win_spec,
                                 nperseg=nperseg_dyn, noverlap=noverlap_dyn,
                                 nfft=nperseg_dyn)
        X_stft = np.transpose(Zxx, (1, 2, 0))                    # (K, T, M)

        conf_bins = None
        if self.conf_band is not None:
            conf_bins = (freqs >= self.conf_band[0]) & (freqs <= self.conf_band[1])

        common = dict(
            ref_mic_idx=ref_mic_idx, sharpen_exp=sharpen_exp,
            rtf_alpha=self.rtf_alpha, rtf_loading=self.rtf_loading,
            rtf_mode=self.rtf_mode, w_mode=self.w_mode,
            bf_loading=self.bf_loading, alpha=self.alpha,
            min_loading=self.min_loading, mu=self.mu,
            lambda_floor=self.lambda_floor, psd_project=self.psd_project,
            ban=self.ban, smooth=self.smooth, conf_gate=self.conf_gate,
            conf_bins=conf_bins, conf_smooth=self.conf_smooth,
            conf_alpha=self.conf_alpha,
            mic_coords=(scene_config.get('mic_coords')
                        if self.w_mode == "sd" else None),
            freqs=freqs, sd_eps=self.sd_eps, sd_field=self.sd_field,
            mask_warp=self.mask_warp,
        )

        if self.mode == "fb":
            Y_stft, weights = blind_feedback_stft(
                X_stft, model_path, nperseg_dyn, feedback=True, **common)
        else:
            Y_stft, weights = self._process_spec(
                X_stft, mic_signals, freqs, fs, win_spec, model_path,
                nperseg_dyn, noverlap_dyn, hop_dyn, ref_mic_idx, sharpen_exp,
                mic_coords=scene_config.get('mic_coords'))

        n_out = mic_signals.shape[1]
        if self.synth is None:
            _, y_time = sig.istft(Y_stft, fs=fs, window=win_spec,
                                  nperseg=nperseg_dyn, noverlap=noverlap_dyn,
                                  nfft=nperseg_dyn)
            y_time = y_time[:n_out]
        else:
            y_time = ola_taper(Y_stft, nperseg_dyn, hop_dyn, self.synth, n_out)
        return y_time, weights

    # -- ABLACION "spec": dos pasadas, pero sin ida y vuelta al tiempo ------
    def _process_spec(self, X_stft, mic_signals, freqs, fs, win_spec,
                      model_path, nperseg, noverlap, hop, ref_mic_idx,
                      sharpen_exp, mic_coords=None):
        from beamforming.mask.blind_feedback import SoudenSubtractCore

        # Pasada 1: el bootstrap de siempre, sobre el canal crudo.
        m_s1, m_n1 = get_dtln_masks_sharpen(
            mic_signals, ref_mic_idx, model_path, block_len=nperseg,
            block_shift=hop, sharpen_exp=sharpen_exp, peak_norm=1.0, stretch=False)
        m_s1, m_n1 = align_mask_frames((m_s1, m_n1), None)

        K, T, M = X_stft.shape

        def _fit_T(m):
            m = np.asarray(m, dtype=np.float64)
            if m.shape[1] >= T:
                return m[:, :T]
            return np.concatenate(
                [m, np.repeat(m[:, -1:], T - m.shape[1], axis=1)], axis=1)

        m_s1, m_n1 = _fit_T(m_s1), _fit_T(m_n1)

        conf_bins = None
        if self.conf_band is not None:
            conf_bins = (freqs >= self.conf_band[0]) & (freqs <= self.conf_band[1])
        Gamma = None
        if self.w_mode == "sd":
            if mic_coords is None:
                raise ValueError("w_mode='sd' necesita scene_config['mic_coords'].")
            Gamma = diffuse_coherence(np.asarray(mic_coords, dtype=np.float64),
                                      freqs, field=self.sd_field)

        W_fix, _ = estimate_rtf_recursive(
            X_stft, m_s1, m_n1, ref_mic_idx=ref_mic_idx, rtf_alpha=self.rtf_alpha,
            rtf_loading=self.rtf_loading, rtf_mode=self.rtf_mode,
            w_mode=self.w_mode, bf_loading=self.bf_loading,
            conf_gate=self.conf_gate, conf_bins=conf_bins,
            conf_smooth=self.conf_smooth, conf_alpha=self.conf_alpha,
            Gamma=Gamma, sd_eps=self.sd_eps)

        # Pasada 2: el DTLN come el ESPECTRO conformado, sin resintetizar. Al
        # corresponder el bloque t+1 al frame t, la mascara sale ya alineada.
        Y_fix = np.einsum("ktm,ktm->kt", W_fix.conj(), X_stft)
        dtln = DTLNStream(model_path)
        p = float(sharpen_exp)
        core = SoudenSubtractCore(K, M, ref_mic_idx, alpha=self.alpha,
                                  min_loading=self.min_loading, mu=self.mu,
                                  lambda_floor=self.lambda_floor,
                                  psd_project=self.psd_project, ban=self.ban)
        Y_stft = np.zeros((K, T), dtype=np.complex128)
        W_out = np.zeros((K, T, M), dtype=np.complex128)
        for t in range(T):
            if t % 32 == 0 or t == T - 1:
                print(f"\r  [spec] frame {t+1}/{T}", end="")
            m_raw = np.clip(np.asarray(dtln.step(np.abs(nperseg * Y_fix[:, t])),
                                       dtype=np.float64), 0.0, 1.0)
            if self.mask_warp is None:
                m_s, m_n = m_raw ** p, (1.0 - m_raw) ** p
            else:
                a_s, b_s, a_n, b_n = self.mask_warp
                m_s = np.clip(a_s * m_raw + b_s, 1e-4, 1.0)
                m_n = np.clip(a_n * (1.0 - m_raw) + b_n, 1e-4, 1.0)
            Y, w = core.step(X_stft[:, t, :], m_s, m_n)
            if self.smooth is not None:
                Y = Y * (self.smooth + (1.0 - self.smooth) * m_raw)
            Y_stft[:, t] = Y
            W_out[:, t, :] = w
        print()
        return Y_stft, W_out
