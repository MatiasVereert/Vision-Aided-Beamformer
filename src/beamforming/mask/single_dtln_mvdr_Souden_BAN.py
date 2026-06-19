import os
import numpy as np
from plotly import tools
import scipy.signal as signal
import tensorflow as tf

# Imports for data simulation
from propagation.simulate_acoustics import SimAcoustic
from utils.audio import save_wav, normalize_signal
from ai_edge_litert.interpreter import Interpreter

# =====================================================================
# 1. ONLINE MVDR BEAMFORMER (MASK-BASED) - PAPER ACCUMULATIVE VERSION
# =====================================================================
import numpy as np


def MVDR_Souden_recursive_mask_BAN(X_stft, mask_s, mask_n, min_loading=1e-6, save_weights=False):
    K, T, M = X_stft.shape

    Y_stft = np.zeros((K, T), dtype=np.complex128)

    # Accumulators for matrix estimation
    Num_XX = np.zeros((K, M, M), dtype=np.complex128)
    Num_NN = np.zeros((K, M, M), dtype=np.complex128)

    Den_XX = np.zeros((K, 1, 1), dtype=np.float64)
    Den_NN = np.zeros((K, 1, 1), dtype=np.float64)

    # Reference microphone index (equivalent to the one-hot vector 'r' in Souden's formula)
    ref_mic = 0

    if save_weights:
        weights_rec = np.zeros((K, T, M), dtype=np.complex128)

    for m in range(T):
        print(f"\rProcessing frame {m} of {T}", end="")
        X_frame = X_stft[:, m, :]
        m_s_frame = mask_s[:, m, np.newaxis, np.newaxis]
        m_n_frame = mask_n[:, m, np.newaxis, np.newaxis]

        R_instant = np.einsum("fm,fn->fmn", X_frame, X_frame.conj())

        # Strict cumulative sum (Equation 4)
        Num_XX += m_s_frame * R_instant
        Den_XX += m_s_frame

        Num_NN += m_n_frame * R_instant
        Den_NN += m_n_frame

        # Calculate matrices by dividing the accumulators
        Phi_XX = Num_XX / (Den_XX + 1e-15)
        Phi_NN = Num_NN / (Den_NN + 1e-15)

        # Diagonal loading for numerical stability before inversion
        tr_Phi = np.real(np.trace(Phi_NN, axis1=1, axis2=2))
        adaptive_load = min_loading * (tr_Phi / M)
        loading_matrix = np.eye(M)[np.newaxis, :, :] * np.maximum(adaptive_load, 1e-9)[:, np.newaxis, np.newaxis]

        Phi_NN_stable = Phi_NN + loading_matrix
        Phi_NN_inv = np.linalg.inv(Phi_NN_stable)

        # -----------------------------------------------------------------
        # SOUDEN'S MVDR FORMULATION
        # -----------------------------------------------------------------
        # 1. Compute the product of Phi_NN_inv and Phi_XX
        Phi_inv_Phi_X = np.einsum("fmn,fnk->fmk", Phi_NN_inv, Phi_XX)

        # 2. Calculate the normalization factor lambda (trace of the product)
        lambda_norm = np.real(np.trace(Phi_inv_Phi_X, axis1=1, axis2=2))

        # 3. Extract the unnormalized weights
        weights_unnorm = Phi_inv_Phi_X[:, :, ref_mic]

        # 4. Normalize the weights using lambda
        weights = weights_unnorm / (lambda_norm[:, np.newaxis] + 1e-15)

        # -----------------------------------------------------------------
        # BLIND ANALYTICAL NORMALIZATION (BAN)
        # -----------------------------------------------------------------
        # Step A: Compute Phi_NN * w
        Phi_NN_w = np.einsum("fmn,fn->fm", Phi_NN_stable, weights)

        # Step B: Denominator = w^H * Phi_NN * w
        # Forcing real output to prevent accumulation of floating-point imaginary noise
        ban_denom = np.real(np.einsum("fm,fm->f", weights.conj(), Phi_NN_w))

        # Step C: Numerator = sqrt( (w^H * Phi_NN^2 * w) / M )
        # w^H * Phi_NN^2 * w is equivalent to the squared L2 norm of (Phi_NN * w)
        w_Phi_NN_sq = np.real(np.einsum("fm,fm->f", Phi_NN_w.conj(), Phi_NN_w))
        ban_num = np.sqrt(w_Phi_NN_sq / M)

        # Step D: Compute BAN scalar per frequency bin
        ban_factor = ban_num / (ban_denom + 1e-15)

        # Step E: Apply BAN scaling to the current MVDR weights
        weights = weights * ban_factor[:, np.newaxis]
        # -----------------------------------------------------------------

        if save_weights:
            weights_rec[:, m, :] = weights

        # Apply the weights to the current STFT frame
        Y_stft[:, m] = np.einsum("fm,fm->f", weights.conj(), X_frame)

    if save_weights:
        return Y_stft, weights_rec
    else:
        return Y_stft

# =====================================================================
# 2. OFFLINE MASK ESTIMATION WRAPPER (DTLN) - SINGLE MIC OPTIMIZED
# =====================================================================
def get_dtln_masks_soft(time_domain_input, ref_mic, model1_path, block_len=512, block_shift=128):
    """
    Processes a single reference channel offline to extract neural masks
    using the DTLN STFT-based model. This avoids computing masks for all M channels.
    """
    M, samples = time_domain_input.shape
    num_blocks = (samples - (block_len - block_shift)) // block_shift

    interpreter_1 = Interpreter(model_path=model1_path)
    interpreter_1.allocate_tensors()

    input_details_1 = interpreter_1.get_input_details()
    output_details_1 = interpreter_1.get_output_details()

    # Isolate the audio from the selected reference microphone
    audio_mono = time_domain_input[ref_mic, :]

    # Normalize channel audio to prevent DTLN saturation
    max_val = np.max(np.abs(audio_mono))
    if max_val > 0:
        audio_mono = audio_mono / max_val

    # Initialize LSTM states for the reference channel
    states_1 = np.zeros(input_details_1[1]['shape'], dtype=np.float32)
    in_buffer = np.zeros((block_len), dtype=np.float32)

    ch_mask = np.zeros((block_len // 2 + 1, num_blocks), dtype=np.float32)

    print(f"\r -> Computing DTLN mask ONLY for reference channel {ref_mic}...", end="")

    for idx in range(num_blocks):
        # Shift buffer and load new audio samples
        in_buffer[:-block_shift] = in_buffer[block_shift:]
        start_idx = idx * block_shift
        in_buffer[-block_shift:] = audio_mono[start_idx : start_idx + block_shift]

        # Compute FFT and magnitude
        in_block_fft = np.fft.rfft(in_buffer)
        in_mag = np.abs(in_block_fft)
        in_mag = np.reshape(in_mag, (1, 1, -1)).astype(np.float32)

        # Predict mask
        interpreter_1.set_tensor(input_details_1[1]['index'], states_1)
        interpreter_1.set_tensor(input_details_1[0]['index'], in_mag)
        interpreter_1.invoke()

        out_mask = interpreter_1.get_tensor(output_details_1[0]['index'])
        states_1 = interpreter_1.get_tensor(output_details_1[1]['index'])

        ch_mask[:, idx] = np.squeeze(out_mask)

    print() # New line after processing

    # Skip median pooling as we only have one channel mask now
    sp_mask_condensed = ch_mask

    # Contrast patch: stretch values between 0 and 1
    def stretch_mask(m):
        return (m - np.min(m)) / (np.max(m) - np.min(m) + 1e-12)


    # Calculate noise mask mathematically (linear complement)
    n_mask_condensed = 1.0 - sp_mask_condensed

    # The squaring operation was removed here to keep the soft, linear continuous mask
    # This prevents harsh zeros in the STFT domain and reduces potential artifacts

    return sp_mask_condensed, n_mask_condensed

# =====================================================================
# 3. PIPELINE INTEGRATION
# =====================================================================
def apply_hybrid_pipeline(time_domain_input, fs, model1_path, ref_mic=None, length_fft=512, hop_length_fft=128):

    # Calculate the middle microphone index if not explicitly provided
    M_total = time_domain_input.shape[0]
    if ref_mic is None:
        ref_mic = M_total // 2

    print(f" -> Computing offline neural masks with DTLN (using reference mic {ref_mic})...")
    mask_s, mask_n = get_dtln_masks_soft(time_domain_input, ref_mic, model1_path, block_len=length_fft, block_shift=hop_length_fft)

    print(" -> Computing STFT of the mixture...")
    freqs, times, Zxx = signal.stft(
        time_domain_input, fs=fs, nperseg=length_fft, noverlap=length_fft - hop_length_fft
    )
    X_stft = np.transpose(Zxx, (1, 2, 0)) # Transpose to (K, T, M)

    # Ensure time dimensions match between STFT and neural masks
    min_frames = min(X_stft.shape[1], mask_s.shape[1])
    X_stft = X_stft[:, :min_frames, :]
    mask_s = mask_s[:, :min_frames]
    mask_n = mask_n[:, :min_frames]

    # ==============================================================
    # DEBUG DUMP: Masked reference signal
    # ==============================================================
    print(f" -> [DEBUG] Generating single-channel masked dump for mic {ref_mic}...")
    X_ref = X_stft[:, :, ref_mic] # Take the reference microphone instead of 0
    Y_debug_stft = X_ref * mask_s

    _, y_debug_time = signal.istft(
        Y_debug_stft, fs=fs, nperseg=length_fft, noverlap=length_fft - hop_length_fft
    )

    save_wav(f"DEBUG_mask_only_mic{ref_mic}.wav", fs, normalize_signal(y_debug_time), "tests/data/hybrid_v2_mvdr_output")
    # ==============================================================

    print(" -> Running mask-based recursive MVDR...")
    Y_stft = MVDR_recursive_mask_based(X_stft, mask_s, mask_n)

    print("\n -> Reconstructing time-domain signal...")
    _, y_time = signal.istft(
        Y_stft, fs=fs, nperseg=length_fft, noverlap=length_fft - hop_length_fft
    )

    return y_time, mask_s

import matplotlib.pyplot as plt

def plot_results_comparison(input_signal, output_signal, mask, fs, length_fft, hop_length_fft):
    """
    Plots the spectrograms of input, output, and the estimated mask for comparison.
    """
    # Use a style for better visibility
    plt.style.use('seaborn-v0_8-muted')
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    cmap_choice = 'viridis' # Uniform colormap

    # Common parameters for spectrograms
    spec_params = dict(
        Fs=fs,
        NFFT=length_fft,
        noverlap=length_fft - hop_length_fft,
        cmap=cmap_choice
    )

    # 1. Input Spectrogram (Mic 0)
    print(" -> Plotting Input Spectrogram...")
    axes[0].specgram(input_signal[0], **spec_params)
    axes[0].set_title("Input Signal Spectrogram (Mic 0 - Mixture)")
    axes[0].set_ylabel("Frequency [Hz]")

    # 2. Output Spectrogram (MVDR Enhanced)
    print(" -> Plotting Output Spectrogram...")
    axes[1].specgram(output_signal, **spec_params)
    axes[1].set_title("Output Signal Spectrogram (Hybrid MVDR Enhanced)")
    axes[1].set_ylabel("Frequency [Hz]")

    # 3. Neural Mask (Speech Mask)
    # The mask is already in (K, T) format, we use imshow
    print(" -> Plotting Neural Mask...")
    # Extent is needed to align axes: [xmin, xmax, ymin, ymax]
    duration = len(output_signal) / fs
    im = axes[2].imshow(
        mask,
        aspect='auto',
        origin='lower',
        extent=[0, duration, 0, fs / 2],
        cmap=cmap_choice
    )
    axes[2].set_title("DTLN Speech Mask ($M_s$)")
    axes[2].set_ylabel("Frequency [Hz]")
    axes[2].set_xlabel("Time [s]")

    # Add a colorbar for the mask to show confidence levels
    fig.colorbar(im, ax=axes[2], label="Mask Intensity")

    plt.tight_layout()
    plt.show()
# =====================================================================
# 4. MAIN SCRIPT (SIMULATION & TESTING)
# =====================================================================
if __name__ == "__main__":

    FS = 16000
    M = 8
    iSIR_dB = 0

    # Use a raw string 'r' to prevent Windows backslash issues
    # Point directly to the first DTLN model
    MODEL_1_PATH = r'/home/matias/Documents/Tesis/Vision-Aided-Beamformer/tools/data/models/model_quant_1.tflite'

    print("=== HYBRID TEST: DTLN MASK-BASED ONLINE MVDR ===")

    output_folder = "tests/data/single_dtln_mvdr"
    os.makedirs(output_folder, exist_ok=True)
    # =================================================================
    # 3D LOGARITHMIC FIBONACCI SPHERE GEOMETRY
    # =================================================================
    r_min = 0.01  # Minimum radius (e.g., 4 cm for high frequencies)
    r_max = 0.30  # Maximum radius (e.g., 30 cm for low frequencies)

    if M > 1:
        indices = np.arange(M)

        # Logarithmic radial expansion
        r = r_min * (r_max / r_min)**(indices / (M - 1))

        # Golden ratio for uniform angular distribution
        golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0

        # Fibonacci sphere angles
        theta = 2.0 * np.pi * indices / golden_ratio
        phi = np.arccos(1.0 - 2.0 * (indices / (M - 1)))

        # Convert spherical to Cartesian coordinates (x, y, z)
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
    else:
        x, y, z = np.array([0.0]), np.array([0.0]), np.array([0.0])

    # Assemble coordinates and align with the array center
    mic_coords = np.column_stack([x, y, z])
    array_center = np.array([1.25, 2.0, 1.25])
    mic_coords = mic_coords - np.mean(mic_coords, axis=0) + array_center

    # Positions
    r = 1.0
    ang_target = np.deg2rad(130)
    ang_interf = np.deg2rad(50)

    source_pos = array_center + np.array([r * np.cos(ang_target), r * np.sin(ang_target), 0.0])
    interf_pos1 = array_center + np.array([r * np.cos(ang_interf), r * np.sin(ang_interf), 0.0])
    source_pos_2d = source_pos.reshape(1, 3)

    print(" -> Initializing acoustic scene...")
    acoustic_scene = SimAcoustic(mic_coords, array_mismatch=0.0, duration=20, fs=FS)
    acoustic_scene.set_source(r"/home/matias/Documents/Tesis/Vision-Aided-Beamformer/tools/data/signals/p002_emo_adoration_sentences.wav", gain=1, position=source_pos_2d)
    acoustic_scene.set_interference(r"tools/data/signals/hairdryer_02_RHODE_NT5.wav", gain=1, position=interf_pos1.reshape(1,3))


    # =================================================================
    # PHASE: ROOM SIMULATION (Reverberant)
    # =================================================================
    cache_room_path = os.path.join(output_folder, "cache_room.npz")

    if os.path.exists(cache_room_path):
        print("\n--- LOADING ROOM SIMULATION FROM CACHE ---")
        cache_data = np.load(cache_room_path)
        room_input = cache_data['input']
    else:
        print("\n--- COMPUTING ROOM SIMULATION ---")
        room_dimensions = np.array([4.0, 5.0, 2.5])
        room_sim_dic = acoustic_scene.get_eval_scene(
            room_dimensions=room_dimensions, desire_RT=0.5, iSIR_dB=iSIR_dB, mode="ideal"
        )
        room_input = room_sim_dic["mic_signals"]
        np.savez(cache_room_path, input=room_input)

    save_wav("1_ROOM_input_mix_mic0.wav", FS, room_input[0], output_folder)

    # Execute Hybrid Pipeline
    print("\n--- APPLYING HYBRID MVDR (DTLN MASK + ONLINE FILTER) ---")
    # Updated FFT parameters to match DTLN natively
    output_rm, mask_rm = apply_hybrid_pipeline(room_input, FS, MODEL_1_PATH, length_fft=512, hop_length_fft=128)

    save_wav("2_ROOM_hybrid_output_final.wav", FS, normalize_signal(output_rm), output_folder)

    print("\n -> Processing complete. Check output folder.")

    plot_results_comparison(room_input, output_rm, mask_rm, FS, 1024, 256)
