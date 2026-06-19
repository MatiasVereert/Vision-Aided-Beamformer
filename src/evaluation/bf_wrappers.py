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

from beamforming.mask.single_dtln_mvdr_exp import get_dtln_masks, MVDR_recursive_mask_based
from beamforming.mask.single_dtln_mvdr_Souden import  MVDR_l_recursive_mask_based
from beamforming.mask.single_dtln_mvdr_Souden_BAN import MVDR_l_recursive_mask_based_BAN


from beamforming.mask.single_dtln_mvdr_rtf_geo import MVDR_recursive_mask_based as MVDR_recursive_mask_based_rtf




class DS_Processor:
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



class DTLN_RTF_MVDR_Processor:
    """
    Wrapper for the single-channel DTLN mask-based MVDR beamformer
    with empirical RTF estimation mixed with a geometric Steering Vector.
    """
    def __init__(self, nperseg=512, noverlap=384, min_loading=1e-3, lamda=0.99, alpha=0.8):
        # STFT configuration ideally aligned with DTLN (block_len=512, block_shift=128)
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nperseg
        self.hop_length = nperseg - noverlap

        # Algorithm hyper-parameters
        self.min_loading = min_loading
        self.lamda = lamda
        self.alpha = alpha

    def process(self, mic_signals: np.ndarray, scene_config: dict) -> tuple:
        # 1. Extract physical and operational configurations
        fs = scene_config['fs']
        source_pos = scene_config['source_pos'].reshape(1, 3)
        mic_coords = scene_config['mic_coords']

        # Dynamically extract DTLN model path from the benchmark config
        model_path = scene_config.get('dtln_model_path', r'dnn_denoise\models\model_quant_1.tflite')

        # Dynamic STFT configuration
        nperseg_dyn = scene_config.get('stft_window', self.nperseg)
        noverlap_dyn = scene_config.get('stft_overlap', self.noverlap)
        nfft_dyn = nperseg_dyn
        hop_length_dyn = nperseg_dyn - noverlap_dyn

        # Warning to use the same windows as the DTLN model was trained on
        if nperseg_dyn != 512 or hop_length_dyn != 128:
            print(f"[Warning]: Window length ({nperseg_dyn}) and hop length ({hop_length_dyn}) should ideally match DTLN training (512/128).")

        # Define Reference Microphone Index
        M_tot = mic_signals.shape[0]
        ref_mic_idx = scene_config.get('ref_mic', M_tot // 2)

        # 2. Extract masks using block_shift on a single channel
        mask_s, mask_n = get_dtln_masks(
            mic_signals,
            ref_mic_idx,
            model_path,
            block_len=nperseg_dyn,
            block_shift=hop_length_dyn
        )

        # 3. Compute Forward STFT
        # Input shape: (M, N_samples). Output shape Zxx: (M, K, T)
        freqs, times, Zxx = sig.stft(
            mic_signals, fs=fs, window='hamming',
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )

        # Transpose to (K, T, M) for spatial frequency-domain processing
        X_stft = np.transpose(Zxx, (1, 2, 0))

        # 4. Ensure time dimensions match strictly between STFT and neural masks
        min_frames = min(X_stft.shape[1], mask_s.shape[1])
        X_stft = X_stft[:, :min_frames, :]
        mask_s = mask_s[:, :min_frames]
        mask_n = mask_n[:, :min_frames]

        # 5. Execute the core mathematical function for RTF + Geo MVDR
        # Passing geometric parameters and alpha
        Y_stft, weights = MVDR_recursive_mask_based_rtf(
            X_stft=X_stft,
            mask_s=mask_s,
            mask_n=mask_n,
            fs=fs,
            array_geometry=mic_coords,
            source_pos=source_pos,
            alpha=self.alpha,
            min_loading=self.min_loading,
            lamda=self.lamda,
            save_weights=True
        )

        # 6. Compute Inverse STFT to return to the time domain
        _, y_time = sig.istft(
            Y_stft, fs=fs, window='hamming',
            nperseg=nperseg_dyn, noverlap=noverlap_dyn, nfft=nfft_dyn
        )

        # 7. Ensure the output length exactly matches the original input signal length
        original_length = mic_signals.shape[1]
        y_time = y_time[:original_length]

        return y_time, weights

class DTLN_MB_MVDR_Processor:
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
        Y_stft, weights = MVDR_recursive_mask_based(
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


class DTLN_MB_MVDR_SOUDEN_BAN_Processor:
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
        Y_stft, weights = MVDR_l_recursive_mask_based_BAN(
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



class DTLN_MB_MVDR_soft_Processor:
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
        Y_stft, weights = MVDR_l_recursive_mask_based(
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


class MVDR_Recursive_Processor:
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

class RTF_MVDR_Recursive_Processor:
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


class SPP_MVDR_Recursive_Processor:
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


class SPP_mono_MVDR_Recursive_Processor:
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


class KMVDR_Recursive_Processor:
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


class SDW_MWF_Processor:
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


class MPDR_Recursive_Processor:
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