import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from utils.audio import  normalize_signal


# Assuming the import works correctly in your local environment
from beamforming.signal_model import compute_rtf_steering_vector
import numpy as np

import numpy as np
import scipy.signal as signal
from beamforming.signal_model import compute_rtf_steering_vector

def RTF_MVDR_recursive(X_stft, vad, fs, array_geometry, source_pos, length_fft, hop_length_fft, alpha=1.0, save_weights=False, min_loading=1e-6):
    lamda = 0.99
    gamma_rtf = 0.90 # Smoothing factor for the RTF vector
    K, T, M = X_stft.shape

    frecs = np.linspace(0, fs/2, K)
    beta = 1e-3

    # Get geometric steering vectors as an initial fallback anchor, expected shape (K, M)
    sv = compute_rtf_steering_vector(frecs, source_pos, array_geometry, ref_mic_idx=0, mode="near_field", squeeze=True)

    # Initialize output complex STFT matrix
    Y_stft = np.zeros((K, T), dtype=np.complex128)

    # Initialize covariance matrices
    R_nn = np.tile(np.eye(M, dtype=np.complex128) * 1e-6, (K, 1, 1))
    R_xx = np.tile(np.eye(M, dtype=np.complex128) * 1e-6, (K, 1, 1))

    # Initialize smoothed RTF vector with the geometric steering vector
    rtf_smoothed = sv.copy()

    # Save weights
    weights_rec = np.zeros((K, T, M), dtype=np.complex128)

    for m in range(T):
        # Extract the current frame
        X_frame = X_stft[:, m, :]

        # Define VAD status
        vad_frame = vad[m * hop_length_fft : length_fft + m * hop_length_fft]
        vad_status = np.mean(vad_frame) > 0.1

        # Calculate instantaneous covariance matrix
        R_instant = np.einsum("fm,fn->fmn", X_frame, X_frame.conj())

        # Update matrices based on VAD status
        if not vad_status:
            # Update noise covariance ONLY when target speech is absent
            R_nn = lamda * R_nn + (1 - lamda) * R_instant
        else:
            # Update noisy speech covariance when target speech is present
            R_xx = lamda * R_xx + (1 - lamda) * R_instant

        # --- Covariance Subtraction (CS) for RTF Estimation ---
        R_ss = R_xx - R_nn

        rtf_nom = R_ss[:, :, 0]
        rtf_den = np.real(R_ss[:, 0, 0])

        # STABILIZATION 1: Dynamic floor based on R_xx energy to prevent massive spikes
        floor_energy = 1e-3 * np.real(R_xx[:, 0, 0])
        rtf_den_safe = np.maximum(rtf_den, floor_energy)
        # Absolute fallback
        rtf_den_safe = np.maximum(rtf_den_safe, 1e-10)

        rtf_empirical = rtf_nom / rtf_den_safe[:, np.newaxis]

        # STABILIZATION 2: Smooth the RTF vector over time to prevent rapid frame-to-frame transients
        rtf_smoothed = gamma_rtf * rtf_smoothed + (1.0 - gamma_rtf) * rtf_empirical

        # Mix Empirical RTF with Geometric Steering Vector
        # If alpha=1.0, it is 100% blind (uses only rtf_smoothed)
        rtf_mixed = alpha * rtf_smoothed + (1.0 - alpha) * sv

        # --- DYNAMIC DIAGONAL LOADING ---
        tr_R = np.real(np.trace(R_nn, axis1=1, axis2=2))
        adaptive_load = beta * (tr_R[:, None, None] / M)
        loading = np.maximum(adaptive_load, min_loading)

        # Apply stabilization and invert
        R_nn_stable = R_nn + np.eye(M)[None, :, :] * loading
        R_nn_inv = np.linalg.inv(R_nn_stable)

        # --- Calculate MVDR Weights ---
        weights_nom = np.einsum("fmn,fn->fm", R_nn_inv, rtf_mixed)
        weights_den = np.einsum("fm,fm->f", rtf_mixed.conj(), weights_nom)

        # STABILIZATION 3: Ensure denominator for weights is strictly positive and bounded
        weights_den_safe = np.maximum(np.real(weights_den), 1e-10)
        weights = weights_nom / weights_den_safe[:, np.newaxis]

        # Save weights
        weights_rec[:, m, :] = weights

        # Apply weights
        Y_stft[:, m] = np.einsum("fm,fm->f", weights.conj(), X_frame)

    if save_weights:
        return Y_stft, weights_rec
    else:
        return Y_stft

# Normalize signals to range [-0.99, 0.99] to prevent clipping when saving as WAV
def normalize_signal(sig):
    max_abs = np.max(np.abs(sig))
    if max_abs > 0:
        return sig * (0.99 / max_abs)
    return sig

import numpy as np
import scipy.signal as signal

# Assuming these are available from your local environment modules
from beamforming.signal_model import compute_rtf_steering_vector

import os
from propagation.simulate_acoustics import SimAcoustic
from utils.audio import save_wav
import os
import numpy as np
import scipy.signal as signal

# Assuming these are available from your local environment modules
from beamforming.signal_model import compute_rtf_steering_vector
# from simulation_module import SimAcoustic
# from utils import save_wav, normalize_signal
# from your_mvdr_module import RTF_MVDR_recursive
# from your_wpe_module import process_wpe_online

def apply_mvdr_stft_bridge(time_domain_input, vad_oracle, mic_coords, source_pos_2d, fs, length_fft=512, hop_length_fft=256):
    """
    Helper function to wrap the STFT -> MVDR -> ISTFT process.
    """
    # Compute STFT
    freqs, times, Zxx = signal.stft(
        time_domain_input,
        fs=fs,
        nperseg=length_fft,
        noverlap=length_fft - hop_length_fft
    )

    # Transpose Zxx from (M, K, T) to (K, T, M)
    X_stft = np.transpose(Zxx, (1, 2, 0))

    # Pad VAD to avoid index out of bounds during the last STFT frames
    vad_padded = np.pad(vad_oracle, (0, length_fft + hop_length_fft), mode='constant')

    # Execute the Recursive MVDR
    Y_stft = RTF_MVDR_recursive(
        X_stft=X_stft,
        vad=vad_padded,
        fs=fs,
        array_geometry=mic_coords,
        source_pos=source_pos_2d,
        length_fft=length_fft,
        hop_length_fft=hop_length_fft
    )

    # Compute Inverse STFT
    _, y_time = signal.istft(
        Y_stft,
        fs=fs,
        nperseg=length_fft,
        noverlap=length_fft - hop_length_fft
    )

    # Truncate to original length
    original_length = time_domain_input.shape[1]
    return y_time[:original_length]



from beamforming.MWF.SP_SDW_MWF_base import process_wpe_online




if __name__ == "__main__":
    # Basic simulation parameters
    FS = 16000
    M1, M2 = 12, 1
    M = M1 * M2
    speed_of_sound = 343.0

    iSIR_dB = 0

    print("=== INTEGRATION TEST: PIPELINE (FREE-FIELD, ROOM, WPE+ROOM) ===")

    output_folder = "tests/data/rtf_mvdr_output"
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
    acoustic_scene = SimAcoustic(mic_coords, array_mismatch=0.0, duration=40, fs=FS)
    acoustic_scene.set_source(r"/home/matias/Documents/Tesis/Vision-Aided-Beamformer/tools/data/signals/p002_emo_adoration_sentences.wav", gain=1, position=source_pos_2d)
    acoustic_scene.set_interference(r"/home/matias/Documents/Tesis/Vision-Aided-Beamformer/tools/data/signals/ruido_rosa_16k.wav", gain=1, position=interf_pos1.reshape(1,3))

    # -------------------------------------------------------------------
    # PHASE 1: FREE FIELD SIMULATION (Anechoic)
    # -------------------------------------------------------------------
    cache_ff_path = os.path.join(output_folder, "cache_free_field.npz")

    if os.path.exists(cache_ff_path):
        print("\n--- PHASE 1: LOADING FREE FIELD SIMULATION FROM CACHE ---")
        cache_data = np.load(cache_ff_path)
        free_field_input = cache_data['input']
        vad_oracle_ff = cache_data['vad']
    else:
        print("\n--- PHASE 1: COMPUTING FREE FIELD SIMULATION ---")
        free_field_input, vad_oracle_ff = acoustic_scene.free_field(iSIR_dB=iSIR_dB, normalize=True, mode="ideal", VAD=True)
        # Save to cache
        np.savez(cache_ff_path, input=free_field_input, vad=vad_oracle_ff)

    save_wav("1_FF_input_mix_mic0.wav", FS, free_field_input[0], output_folder)

    print(" -> Applying Recursive MVDR...")
    output_ff = apply_mvdr_stft_bridge(free_field_input, vad_oracle_ff, mic_coords, source_pos_2d, FS)

    save_wav("2_FF_output_final.wav", FS, normalize_signal(output_ff), output_folder)

    # -------------------------------------------------------------------
    # PHASE 2: ROOM SIMULATION (Reverberant)
    # -------------------------------------------------------------------
    cache_room_path = os.path.join(output_folder, "cache_room.npz")

    if os.path.exists(cache_room_path):
        print("\n--- PHASE 2: LOADING ROOM SIMULATION FROM CACHE ---")
        cache_data = np.load(cache_room_path)
        room_input = cache_data['input']
        vad_oracle_room = cache_data['vad']
    else:
        print("\n--- PHASE 2: COMPUTING ROOM SIMULATION ---")
        room_dimensions = np.array([4.0, 5.0, 2.5])
        room_sim_dic = acoustic_scene.get_eval_scene(
            room_dimensions=room_dimensions, desire_RT=0.5, iSIR_dB=iSIR_dB, mode="ideal"
        )
        room_input = room_sim_dic["mic_signals"]
        vad_oracle_room = room_sim_dic["VAD"]
        # Save to cache
        np.savez(cache_room_path, input=room_input, vad=vad_oracle_room)

    save_wav("3_ROOM_input_mix_mic0.wav", FS, room_input[0], output_folder)

    print(" -> Applying Recursive MVDR (Without WPE)...")
    output_rm = apply_mvdr_stft_bridge(room_input, vad_oracle_room, mic_coords, source_pos_2d, FS)

    save_wav("4_ROOM_output_final.wav", FS, normalize_signal(output_rm[100:]), output_folder)

    # -------------------------------------------------------------------
    # PHASE 3: WPE DEREVERBERATION + RECURSIVE MVDR
    # -------------------------------------------------------------------
    print("\n--- PHASE 3: WPE + MVDR PIPELINE ---")
    print(" -> Applying Online WPE Dereverberation on Room Simulation...")

    wpe_output = process_wpe_online(room_input)

    save_wav("5_WPE_input_mix_mic0.wav", FS, wpe_output[0], output_folder)

    print(" -> Applying Recursive MVDR on Dereverberated Signals...")
    output_wpe = apply_mvdr_stft_bridge(wpe_output, vad_oracle_room, mic_coords, source_pos_2d, FS)

    save_wav("6_WPE_ROOM_output_final.wav", FS, normalize_signal(output_wpe), output_folder)

    print("\n -> Pipeline completed successfully.")