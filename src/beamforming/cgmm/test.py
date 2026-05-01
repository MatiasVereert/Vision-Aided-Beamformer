
import numpy as np 
import scipy.signal as signal
import matplotlib.pyplot as plt

# Assuming the import works correctly in your local environment
from beamforming.signal_model import compute_rtf_steering_vector
import numpy as np 
import scipy.signal as signal
import matplotlib.pyplot as plt
from utils.audio import  normalize_signal


# Assuming the import works correctly in your local environment
from beamforming.signal_model import compute_rtf_steering_vector
from beamforming.MWF.SP_SDW_MWF_base import process_wpe_online

import os
import numpy as np
import scipy.signal as sps
from beamforming.cgmm.cgmm import PriorCGMM
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

# Initialize your custom functions here (save_wav, normalize_signal, SimAcoustic, process_wpe_online)

def get_principal_eigenvector(matrix):
    """
    Computes the principal eigenvector of a given matrix, 
    which corresponds to the maximum eigenvalue.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return eigenvectors[:, np.argmax(eigenvalues)]

def apply_online_cgmm_mvdr(mic_signals, fs, nperseg=512, noverlap=384, chunk_size_frames=20, init_frames=40):
    """
    Applies Online CGMM-based MVDR beamforming to multichannel microphone signals.
    Replaces the need for oracle VAD or SPP by estimating spatial covariances directly.
    """
    M = mic_signals.shape[0]
    
    # Compute STFT for all microphone channels
    f, t, Zxx = sps.stft(mic_signals, fs=fs, nperseg=nperseg, noverlap=noverlap)
    F_bins = Zxx.shape[1]
    T_frames = Zxx.shape[2]
    
    # Ensure we do not exceed available frames for initialization
    init_frames = min(init_frames, max(1, T_frames // 10))
    stft_mat_for_init = Zxx[:, :, :init_frames]
    
    # Initialize PriorCGMM for each frequency bin using the init chunk
    cgmmEngine = [PriorCGMM(stft_mat_for_init[:, i, :], K=2) for i in range(F_bins)]
    
    # Prepare output STFT container
    stft_out = np.zeros((F_bins, T_frames), dtype=complex)
    
    # Calculate total number of chunks for online processing
    num_chunks = int(np.ceil(T_frames / chunk_size_frames))
    
    for c in range(num_chunks):
        start_idx = c * chunk_size_frames
        end_idx = min(start_idx + chunk_size_frames, T_frames)
        
        stft_mat_chunk = Zxx[:, :, start_idx:end_idx]
        
        for i in range(F_bins):
            # Run MAP estimation for the current chunk
            cgmmEngine[i].run(stft_mat_chunk[:, i, :], itr_num=3)
            
            # Get the spatial covariance matrices
            R = cgmmEngine[i].getR() 
            
            # Assuming K=2: index 0 is noise (Rv), index 1 is speech (Rx)
            Rv = R[0, :, :]
            Rx = R[1, :, :]
            
            # Calculate steering vector (principal eigenvector of speech covariance matrix)
            steer_vec = get_principal_eigenvector(Rx)
            
            # Add small epsilon to diagonal for stability before inversion
            Rv_stable = Rv + 1e-6 * np.eye(M)
            try:
                inv_Rv = np.linalg.inv(Rv_stable)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if matrix is singular
                inv_Rv = np.linalg.pinv(Rv_stable)
            
            # Calculate MVDR weights
            numerator = inv_Rv @ steer_vec
            denominator = steer_vec.conj().T @ numerator
            w = numerator / (denominator + 1e-10)
            
            # Apply weights to the current chunk
            stft_out[i, start_idx:end_idx] = w.conj().T @ stft_mat_chunk[:, i, :]
            
    # Reconstruct the time-domain signal via ISTFT
    _, wav_out = sps.istft(stft_out, fs=fs, nperseg=nperseg, noverlap=noverlap)
    
    return wav_out

import numpy as np
import scipy.signal as signal
from beamforming.signal_model import compute_rtf_steering_vector
import numpy as np
import scipy.signal as signal
from beamforming.signal_model import compute_rtf_steering_vector
import numpy as np
import scipy.signal as signal
import scipy.linalg
from cgmm import CGMM

def apply_offline_cgmm_mvdr(input_signals, mic_coords, source_pos_2d, FS):
    """
    Applies offline CGMM clustering and MVDR beamforming.
    
    Args:
        input_signals: Array of shape (M, samples) containing microphone signals.
        mic_coords: Microphone coordinates (not strictly needed for blind MVDR, kept for signature compatibility).
        source_pos_2d: Target source position (not strictly needed for blind MVDR).
        FS: Sampling frequency.
        
    Returns:
        enhanced_audio: 1D array containing the beamformed time-domain signal.
    """
    # 1. STFT parameters
    nperseg = 512
    noverlap = 256
    
    # Compute STFT
    # Zxx shape: (M, F, T) where F is freq bins, T is time frames
    f, t, Zxx = signal.stft(input_signals, fs=FS, nperseg=nperseg, noverlap=noverlap)
    M, F, T = Zxx.shape
    
    # 2. Offline CGMM
    # R array to store spatial covariance matrices for all freq bins
    R_matrices = np.zeros((F, 2, M, M), dtype=complex)
    
    for i in range(F):
        # Extract the specific frequency bin across all channels and frames
        # Shape: (M, T)
        freq_bin_data = Zxx[:, i, :]
        
        # Initialize CGMM. K=2 (1 for noise, 1 for speech)
        cgmm_engine = CGMM(freq_bin_data, K=2)
        
        # Run Expectation-Maximization
        cgmm_engine.run(itr_num=10)
        
        # Store the spatial covariance matrix for this frequency
        R_matrices[i] = cgmm_engine.getR()

    # 3. MVDR Beamforming
    enhanced_stft = np.zeros((F, T), dtype=complex)
    
    for i in range(F):
        # From cgmm.py init: index 0 is noise (1e-6*eye), index 1 is speech (data covariance)
        Rv = R_matrices[i, 0, :, :]
        Rx = R_matrices[i, 1, :, :]
        
        # Compute the principal eigenvector of the speech covariance matrix Rx
        # to use as the blind steering vector (Relative Transfer Function)
        eigenvalues, eigenvectors = scipy.linalg.eigh(Rx)
        
        # The principal eigenvector corresponds to the largest eigenvalue (last column)
        steering_vector = eigenvectors[:, -1]
        
        # Normalize with respect to the reference microphone (channel 0)
        # to avoid arbitrary scaling and preserve the target signal level
        if steering_vector[0] != 0:
            steering_vector = steering_vector / steering_vector[0]
            
        # Calculate MVDR weights: w = (Rv^-1 @ v) / (v^H @ Rv^-1 @ v)
        # We use pinv for numerical stability
        inv_Rv = np.linalg.pinv(Rv)
        numerator = inv_Rv @ steering_vector
        denominator = steering_vector.conj().T @ numerator
        
        # Avoid division by zero
        if np.abs(denominator) > 1e-12:
            w = numerator / denominator
        else:
            w = np.zeros(M, dtype=complex)
            w[0] = 1.0 # Fallback to reference mic
            
        # Apply the weights to the current frequency bin
        # w is (M,), freq_bin_data is (M, T). w^H @ X -> (T,)
        enhanced_stft[i, :] = w.conj().T @ Zxx[:, i, :]

    # 4. Inverse STFT to get back to the time domain
    _, enhanced_audio = signal.istft(enhanced_stft, fs=FS, nperseg=nperseg, noverlap=noverlap)
    
    return enhanced_audio

def normalize_signal(sig):
    """
    Helper function to prevent clipping when saving the wav file.
    """
    # Prevents division by zero
    max_val = np.max(np.abs(sig))
    if max_val > 0:
        return sig / max_val
    return sig


if __name__ == "__main__":
    # Basic simulation parameters
    FS = 16000
    M1, M2 = 12, 1          
    M = M1 * M2
    speed_of_sound = 343.0 

    iSIR_dB = 0
    
    print("=== INTEGRATION TEST: PIPELINE (FREE-FIELD, ROOM, WPE+ROOM) ===")
    print("=== USING: OFFLINE CGMM-BASED MVDR ===")
    
    output_folder = "tests/data/mvdr_cgmm_offline_output"
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
        # The oracle VAD is loaded but ignored, as the CGMM operates blindly
    else:
        print("\n--- PHASE 1: COMPUTING FREE FIELD SIMULATION ---")
        free_field_input, vad_oracle_ff = acoustic_scene.free_field(iSIR_dB=iSIR_dB, normalize=True, mode="ideal", VAD=True)
        # Save to cache
        np.savez(cache_ff_path, input=free_field_input, vad=vad_oracle_ff)
        
    save_wav("1_FF_input_mix_mic0.wav", FS, free_field_input[0], output_folder)
    
    print(" -> Applying Offline CGMM-MVDR...")
    # Notice the removal of vad_oracle. The CGMM clusters the data itself.
    output_ff = apply_offline_cgmm_mvdr(free_field_input, mic_coords, source_pos_2d, FS)
    
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
        # Save to cache
        np.savez(cache_room_path, input=room_input, vad=vad_oracle_room)
    
    save_wav("3_ROOM_input_mix_mic0.wav", FS, room_input[0], output_folder)

    print(" -> Applying Offline CGMM-MVDR (Without WPE)...")
    output_rm = apply_offline_cgmm_mvdr(room_input, mic_coords, source_pos_2d, FS)
    
    save_wav("4_ROOM_output_final.wav", FS, normalize_signal(output_rm), output_folder)

    # -------------------------------------------------------------------
    # PHASE 3: WPE DEREVERBERATION + OFFLINE CGMM-MVDR
    # -------------------------------------------------------------------
    print("\n--- PHASE 3: WPE + CGMM-MVDR PIPELINE ---")
    print(" -> Applying Online WPE Dereverberation on Room Simulation...")
    
    wpe_output = process_wpe_online(room_input)
    
    save_wav("5_WPE_input_mix_mic0.wav", FS, wpe_output[0], output_folder)

    print(" -> Applying Offline CGMM-MVDR on Dereverberated Signals...")
    output_wpe = apply_offline_cgmm_mvdr(wpe_output, mic_coords, source_pos_2d, FS)
    
    save_wav("6_WPE_ROOM_output_final.wav", FS, normalize_signal(output_wpe), output_folder)

    print("\n -> Pipeline completed successfully.")