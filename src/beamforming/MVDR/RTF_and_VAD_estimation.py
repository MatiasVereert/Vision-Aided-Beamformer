import numpy as np 
import scipy.signal as signal
import matplotlib.pyplot as plt
from utils.audio import  normalize_signal


# Assuming the import works correctly in your local environment
from beamforming.signal_model import compute_rtf_steering_vector
import numpy as np

import numpy as np
import scipy.signal as signal
import pyroomacoustics as pra
import numpy as np
import scipy.signal as signal
import pyroomacoustics as pra
import numpy as np
import scipy.signal as signal
import pyroomacoustics as pra
import numpy as np
import scipy.signal as signal

# We import your validated steering vector function
from beamforming.signal_model import compute_rtf_steering_vector

def compute_spatial_vad_offline(signals, mic_coords, source_pos, fs=16000, frame_len=512, hop_len=256, coherence_threshold=0.30, **kwargs):
    """
    Ultra-fast single-point SRP-PHAT using the system's validated Steering Vector model.
    Includes magnitude weighting to prevent noise bins from dragging down the coherence score.
    """
    M, sig_len = signals.shape

    # 1. Compute STFT
    f_axis, t_axis, Zxx = signal.stft(signals, fs=fs, nperseg=frame_len, noverlap=frame_len - hop_len)
    K, T = Zxx.shape[1], Zxx.shape[2]

    # 2. Get geometric steering vectors directly from your validated function
    sv = compute_rtf_steering_vector(f_axis, source_pos, mic_coords, ref_mic_idx=0, mode="near_field", squeeze=True)
    sv_transposed = sv.T # Shape: (M, K)

    # 3. PHAT Normalization (Phase isolation)
    eps = 1e-10
    Zxx_phat = Zxx / (np.abs(Zxx) + eps)
    sv_phat = sv_transposed / (np.abs(sv_transposed) + eps) 

    # 4. Align the phases using the CONJUGATE of the normalized Steering Vector
    sv_expanded = sv_phat[:, :, np.newaxis] 
    Zxx_aligned = Zxx_phat * np.conj(sv_expanded)

    # 5. Coherent sum across microphones
    coherent_sum = np.sum(Zxx_aligned, axis=0) # Shape: (K, T)

    # 6. Calculate coherence score (Filtered and Energy-Weighted)
    idx_min = int(500 / (fs / frame_len))  
    idx_max = int(4000 / (fs / frame_len)) 
    
    # Extract magnitude weights from the reference microphone to emphasize speech bins
    mag_weights = np.abs(Zxx[0, idx_min:idx_max, :])
    
    # Compute weighted mean across frequency bins instead of a flat unweighted mean
    weighted_coherent_sum = np.sum(mag_weights * np.abs(coherent_sum[idx_min:idx_max, :]), axis=0)
    sum_weights = np.sum(mag_weights, axis=0) + eps
    
    # Normalize by M to keep score strictly between 0.0 and 1.0
    coherence_score = (weighted_coherent_sum / sum_weights) / M

    # 7. Print debug info
    print(f"      [VAD DEBUG] Mean Coherence: {np.mean(coherence_score):.3f} | Max Peak: {np.max(coherence_score):.3f}")

    # 8. Energy threshold (Safely increased to 1e-4)
    frame_energies = np.sum(np.abs(Zxx[0, :, :])**2, axis=0)
    energy_thresh = np.max(frame_energies) * 1e-4

    # 9. Build Binary VAD Mask
    vad_time = np.zeros(sig_len)
    
    for t_idx in range(T):
        if frame_energies[t_idx] > energy_thresh and coherence_score[t_idx] > coherence_threshold:
            start_sample = t_idx * hop_len
            end_sample = min(start_sample + frame_len, sig_len)
            vad_time[start_sample:end_sample] = 1.0

    return vad_time


def RTF_MVDR_recursive(X_stft, vad, fs, array_geometry, source_pos, length_fft, hop_length_fft, alpha=0.85, save_weights=False, min_loading = 1e-6):
    lamda = 0.99
    K, T, M = X_stft.shape  

    frecs = np.linspace(0, fs/2, K)
    beta = 1e-3
    # Get geometric steering vectors as an initial fallback anchor, expected shape (K, M)
    sv = compute_rtf_steering_vector(frecs, source_pos, array_geometry, ref_mic_idx=0, mode="near_field", squeeze=True)
    
    # Initialize output complex STFT matrix
    Y_stft = np.zeros((K, T), dtype=np.complex128)
    
    # Initialize covariance matrices for all frequencies (K, M, M)
    # R_nn tracks noise, R_xx tracks noisy speech
    R_nn = np.tile(np.eye(M, dtype=np.complex128) * 1e-6, (K, 1, 1))
    R_xx = np.tile(np.eye(M, dtype=np.complex128) * 1e-6, (K, 1, 1))

    #save weights
    weights_rec = np.zeros((K,T,M), dtype=np.complex128)

    for m in range(T):
        # Extract the current frame across all frequencies, shape (K, M)
        X_frame = X_stft[:, m, :]

        # Define VAD frame state (mapping STFT frame to time-domain VAD)
        vad_frame = vad[m * hop_length_fft : length_fft + m * hop_length_fft]
        vad_status = np.mean(vad_frame) > 0.1

        # Calculate instantaneous covariance matrix: (K, M) and (K, M) -> (K, M, M)
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
        rtf_den_safe = np.maximum(rtf_den, 1e-10)
        rtf_empirical = rtf_nom / rtf_den_safe[:, np.newaxis]

        # Mix Empirical RTF with Geometric Steering Vector
        rtf_mixed = alpha * rtf_empirical + (1.0 - alpha) * sv

        # --- DYNAMIC DIAGONAL LOADING ---
        # Calculate dynamic loading based on the noise covariance trace
        tr_R = np.real(np.trace(R_nn, axis1=1, axis2=2))
        
        adaptive_load = beta * (tr_R[:, None, None] / M)
        loading = np.maximum(adaptive_load, min_loading)

        # Apply stabilization
        R_nn_stable = R_nn + np.eye(M)[None, :, :] * loading

        # Invert the stable covariance matrix
        R_nn_inv = np.linalg.inv(R_nn_stable)

        # --- Calculate MVDR Weights using the MIXED RTF ---
        # Numerator: R_nn_inv * rtf_mixed -> (K, M, M) * (K, M) -> (K, M)
        weights_nom = np.einsum("fmn,fn->fm", R_nn_inv, rtf_empirical)
        
        # Denominator: rtf_mixed^H * numerator -> (K, M) * (K, M) -> (K,)
        weights_den = np.einsum("fm,fm->f", rtf_mixed.conj(), weights_nom)
        
        # Divide numerator by denominator. Add small epsilon to prevent NaNs
        weights = weights_nom / (weights_den[:, np.newaxis] + 1e-10)

        #save weights
        weights_rec[:,m,:] = weights

        # Apply weights to the current observation to get the clean output
        # Output is shape (K,) for the current frame
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


def plot_vad_comparison(vad_oracle, vad_estimated, fs, title="VAD Comparison"):
    """
    Superpone el VAD oráculo y el VAD estimado en el tiempo.
    """
    # Eje temporal en segundos
    time_axis = np.arange(len(vad_oracle)) / fs
    
    plt.figure(figsize=(12, 3))
    # VAD oráculo con offset 1, VAD estimado con offset 0
    plt.plot(time_axis, vad_oracle + 0.05, label='VAD Oracle (offset +0.05)', color='green', alpha=0.7)
    plt.plot(time_axis, vad_estimated, label='VAD Estimado', color='blue', alpha=0.7)
    
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Decisión')
    plt.yticks([0, 1], ['Silencio', 'Voz'])
    plt.title(title)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_az_estimate(azimuth_est, azimuth_real, fs, hop_len=256, title="DOA Estimation"):
    """
    Plots the expected azimuth versus the multi-source estimated azimuths over time.
    """
    # Calculate correct time axis based on the STFT frames and hop length
    T = azimuth_est.shape[0]
    time_axis = np.arange(T) * hop_len / fs

    # Reference vector for the expected real target
    azimuth_vector = np.full(T, azimuth_real)
    
    plt.figure(figsize=(12, 3))
    
    # Plot real target azimuth
    plt.plot(time_axis, azimuth_vector, label='Real Target', color='blue', alpha=0.7, linestyle='--')
    
    # Plot estimated sources (ignoring NaNs from silent frames)
    num_src = azimuth_est.shape[1]
    colors = ['green', 'orange', 'purple', 'red']
    
    for i in range(num_src):
        plt.plot(time_axis, azimuth_est[:, i], 
                 label=f'Estimate Source {i+1}', 
                 marker='.', 
                 linestyle='None', 
                 markersize=3,
                 color=colors[i % len(colors)], 
                 alpha=0.6)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Azimuth (Radians)')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.ylim(0, 2 * np.pi)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


from beamforming.MWF.SP_SDW_MWF_base import process_wpe_online


if __name__ == "__main__":
    # Basic simulation parameters
    FS = 16000
    M1, M2 = 12, 1          
    M = M1 * M2
    speed_of_sound = 343.0 

    iSIR_dB = 0
    
    print("=== INTEGRATION TEST: PIPELINE WITH SRP-PHAT SPATIAL VAD ===")
    
    # New output folder to avoid overwriting previous tests
    output_folder = "tests/data/rtf_mvdr_output_srp_vad"
    os.makedirs(output_folder, exist_ok=True)
    
    # Create logarithmic spacing for the microphone array
    max_length = 0.30
    if M > 1:
        base = 2.0
        indices = np.arange(M)
        x_norm = (base**indices - 1) / (base**(M - 1) - 1)
        x = x_norm * max_length
    else:
        x = np.array([0.0])

    mic_coords = np.column_stack([x, np.zeros(M), np.zeros(M)])
    array_center = np.array([1.25, 2.0, 1.25])
    mic_coords = mic_coords - np.mean(mic_coords, axis=0) + array_center
    
    r = 1.0 
    ang_target = np.deg2rad(130)
    ang_interf = np.deg2rad(50)
    
    source_pos = array_center + np.array([r * np.cos(ang_target), r * np.sin(ang_target), 0.0])
    interf_pos1 = array_center + np.array([r * np.cos(ang_interf), r * np.sin(ang_interf), 0.0])
    source_pos_2d = source_pos.reshape(1, 3)

    print(" -> Initializing acoustic scene...")
    acoustic_scene = SimAcoustic(mic_coords, array_mismatch=0.0, duration=40, fs=FS)
    acoustic_scene.set_source("tools/data/signals/FA01_09.wav", gain=1, position=source_pos_2d)
    acoustic_scene.set_interference("tools/data/signals/MC15_03.wav", gain=1, position=interf_pos1.reshape(1,3))

    # -------------------------------------------------------------------
    # PHASE 1: FREE FIELD SIMULATION (Anechoic)
    # -------------------------------------------------------------------
    cache_ff_path = os.path.join(output_folder, "cache_free_field.npz")
    
    if os.path.exists(cache_ff_path):
        print("\n--- PHASE 1: LOADING FREE FIELD SIMULATION FROM CACHE ---")
        cache_data = np.load(cache_ff_path)
        free_field_input = cache_data['input']
        # We load the oracle VAD just for reference, but we won't use it for MVDR
        vad_oracle_ff = cache_data['vad'] 
    else:
        print("\n--- PHASE 1: COMPUTING FREE FIELD SIMULATION ---")
        free_field_input, vad_oracle_ff = acoustic_scene.free_field(iSIR_dB=iSIR_dB, normalize=True, mode="ideal", VAD=True)
        np.savez(cache_ff_path, input=free_field_input, vad=vad_oracle_ff)
        
    save_wav("1_FF_input_mix_mic0.wav", FS, free_field_input[0], output_folder)
    
    # Compute Spatial VAD offline using SRP-PHAT (20cm tolerance sphere)
    print(" -> Estimating Spatial VAD offline using SRP-PHAT (Free Field)...")
    # Reemplaza tu línea actual por esta:
    spatial_vad_ff = compute_spatial_vad_offline(free_field_input, mic_coords, source_pos, fs=16000, frame_len=512, hop_len=256, coherence_threshold=0.45)




    plot_vad_comparison(vad_oracle_ff, spatial_vad_ff, FS, title="Free Field VAD: Oracle vs Estimado (SRP-PHAT)")
    
    print(" -> Applying Recursive MVDR...")
    # Pass our estimated spatial VAD instead of vad_oracle_ff
    output_ff = apply_mvdr_stft_bridge(free_field_input, spatial_vad_ff, mic_coords, source_pos_2d, FS)
    
    save_wav("2_FF_output_final.wav", FS, normalize_signal(output_ff), output_folder)

    # -------------------------------------------------------------------
    # PHASE 2: ROOM SIMULATION (Reverberant)
    # -------------------------------------------------------------------
    cache_room_path = os.path.join(output_folder, "cache_room.npz")
    
    if os.path.exists(cache_room_path):
        print("\n--- PHASE 2: LOADING ROOM SIMULATION FROM CACHE ---")
        cache_data = np.load(cache_room_path)
        room_input = cache_data['input']
    else:
        print("\n--- PHASE 2: COMPUTING ROOM SIMULATION ---")
        room_dimensions = np.array([4.0, 5.0, 2.5])
        room_sim_dic = acoustic_scene.get_eval_scene(
            room_dimensions=room_dimensions, desire_RT=0.5, iSIR_dB=iSIR_dB, mode="ideal"
        )
        room_input = room_sim_dic["mic_signals"]
        vad_oracle_room = room_sim_dic["VAD"]
        np.savez(cache_room_path, input=room_input, vad=vad_oracle_room)
    
    save_wav("3_ROOM_input_mix_mic0.wav", FS, room_input[0], output_folder)

    # Compute Spatial VAD offline using SRP-PHAT on reverberant signals
    print(" -> Estimating Spatial VAD offline using SRP-PHAT (Room)...")
    spatial_vad_room = compute_spatial_vad_offline(room_input, mic_coords, source_pos_2d, radius_m=0.20, fs=FS)

    print(" -> Applying Recursive MVDR (Without WPE)...")
    output_rm = apply_mvdr_stft_bridge(room_input, spatial_vad_room, mic_coords, source_pos_2d, FS)
    
    save_wav("4_ROOM_output_final.wav", FS, normalize_signal(output_rm), output_folder)

    # -------------------------------------------------------------------
    # PHASE 3: WPE DEREVERBERATION + RECURSIVE MVDR
    # -------------------------------------------------------------------
    print("\n--- PHASE 3: WPE + MVDR PIPELINE ---")
    print(" -> Applying Online WPE Dereverberation on Room Simulation...")
    
    wpe_output = process_wpe_online(room_input)
    
    save_wav("5_WPE_input_mix_mic0.wav", FS, wpe_output[0], output_folder)

    # Compute Spatial VAD offline using SRP-PHAT on WPE-cleaned signals (highly accurate)
    print(" -> Estimating Spatial VAD offline using SRP-PHAT (WPE)...")
    spatial_vad_wpe = compute_spatial_vad_offline(wpe_output, mic_coords, source_pos_2d, radius_m=0.20, fs=FS)

    print(" -> Applying Recursive MVDR on Dereverberated Signals...")
    output_wpe = apply_mvdr_stft_bridge(wpe_output, spatial_vad_wpe, mic_coords, source_pos_2d, FS)
    
    save_wav("6_WPE_ROOM_output_final.wav", FS, normalize_signal(output_wpe), output_folder)

    print("\n -> Pipeline completed successfully. Check the new folder for results.")