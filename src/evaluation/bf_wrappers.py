import numpy as np
import scipy.signal as sig
# Assuming compute_rtf_steering_vector is imported in your module
from beamforming.signal_model import compute_rtf_steering_vector
from beamforming.MVDR.base import MVDR_recursive
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
)

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

        # Compute exact steering vector
        # Expected output shape: (K, M)
        sv = compute_rtf_steering_vector(
            freqs, source_pos, mic_coords,
            ref_mic_idx=0, mode="near_field", squeeze=True
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


class DTLN_MB_MVDR_SOUDEN_BAN:
    """
    Wrapper for the DTLN mask-based MVDR beamformer.
    Integrates offline neural mask estimation with recursive spatial filtering.
    """
    def __init__(self, nperseg=512, noverlap=384, min_loading=1e-6):
        # STFT configuration aligned with DTLN (block_len=512, block_shift=128)
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nperseg
        self.hop_length = nperseg - noverlap
        self.min_loading = min_loading

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
        Y_stft, weights = MVDR_Souden_recursive_mask_BAN(
            X_stft,
            mask_s,
            mask_n,
            min_loading= self.min_loading,
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
        Y_stft, weights = MVDR_Souden_recursive_mask_slow(
            X_stft,
            mask_s,
            mask_n,
            min_loading= self.min_loading,
            save_weights=True,
            alpha= self.alpha
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
    def __init__(self, nperseg=512, noverlap=384, min_loading=1e-6, alpha=0.99, sharpen_exp=4.0):
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
        Y_stft, weights = MVDR_Souden_recursive_mask(
            X_stft,
            mask_s,
            mask_n,
            min_loading= self.min_loading,
            save_weights=True,
            alpha= self.alpha
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

        # Define Reference Microphone Index as middle index (same as DTLN wrappers)
        M_tot = mic_signals.shape[0]
        ref_mic_idx = M_tot // 2

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
            alpha=self.alpha
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
            alpha=self.alpha
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
        Y_stft, weights_rec = MVDR_recursive(
            X_stft=X_stft,
            vad=vad_padded,
            fs=fs,
            array_geometry=mic_coords,
            source_pos=source_pos,
            length_fft=nperseg_dyn,
            hop_length_fft=hop_length_dyn,
            min_loading=self.min_loading,
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
            save_weights=True
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

        M_tot = mic_signals.shape[0]
        ref_mic_idx = M_tot // 2

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
            smooth=self.smooth, save_weights=True
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

        M_tot = mic_signals.shape[0]
        ref_mic_idx = M_tot // 2

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
            smooth=self.smooth, save_weights=True
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