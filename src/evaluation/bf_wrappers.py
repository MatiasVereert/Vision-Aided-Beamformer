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

from beamforming.mask.dtln_masks import get_dtln_masks, get_dtln_masks_sharpen
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
                 win_type=None, alpha_lf=None, alpha_fsplit_hz=300.0):
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
                 gate_fmax_hz=None, smooth=None, alpha_lf=None, alpha_fsplit_hz=300.0):
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
