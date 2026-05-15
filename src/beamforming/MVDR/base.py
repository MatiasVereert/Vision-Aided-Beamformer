import numpy as np 
import scipy.signal as signal
import matplotlib.pyplot as plt
from utils.audio import  normalize_signal


# Assuming the import works correctly in your local environment
from beamforming.signal_model import compute_rtf_steering_vector

def MVDR_recursive_rtf_subtraction(X_stft, vad, fs, array_geometry, source_pos, length_fft, hop_length_fft, min_loading=1e-6, save_weights=False):
    lamda = 0.99
    beta = 1e-3 # Relative loading
    K, T, M = X_stft.shape  
    
    # Initialize output complex STFT matrix
    Y_stft = np.zeros((K, T), dtype=np.complex128)
    
    # Initialize covariance matrices
    # R_xx is initialized slightly larger to avoid negative subtraction at the very beginning
    R_nn = np.tile(np.eye(M, dtype=np.complex128) * 1e-6, (K, 1, 1))
    R_xx = np.tile(np.eye(M, dtype=np.complex128) * 1e-5, (K, 1, 1))
    
    # Save weights array
    weights_rec = np.zeros((K, T, M), dtype=np.complex128)
    
    for m in range(T):
        # Extract the current frame across all frequencies, shape (K, M)
        X_frame = X_stft[:, m, :]

        # Define VAD frame state (mapping STFT frame to time-domain VAD)
        vad_frame = vad[m * hop_length_fft : length_fft + m * hop_length_fft]
        vad_status = np.mean(vad_frame) > 0.1

        # Calculate instantaneous covariance of the current frame
        R_instant = np.einsum("fm,fn->fmn", X_frame, X_frame.conj())

        # Update matrices based on VAD oracle
        if vad_status:
            # Update noisy mixture covariance when speech is present
            R_xx = lamda * R_xx + (1 - lamda) * R_instant
        else:
            # Update noise covariance when speech is absent
            R_nn = lamda * R_nn + (1 - lamda) * R_instant

        # --- RTF Estimation via Covariance Subtraction ---
        # Estimate the pure speech covariance matrix
        R_ss = R_xx - R_nn
        
        # Extract the column corresponding to the reference microphone (index 0)
        # R_ss has shape (K, M, M), taking slice [:, :, 0] yields (K, M)
        h_raw = R_ss[:, :, 0]
        
        # Normalize with respect to the reference microphone to obtain the RTF (h)
        # We add a small epsilon to the denominator to prevent division by zero
        h = h_raw / (h_raw[:, 0:1] + 1e-10)

        # --- Dynamic Loading ---
        tr_R = np.real(np.trace(R_nn, axis1=1, axis2=2))
        adaptive_load = beta * (tr_R[:, None, None] / M)
        loading = np.maximum(adaptive_load, min_loading)
        
        R_nn_stable = R_nn + np.eye(M)[None, :, :] * loading
        
        # Invert the covariance matrices for all frequencies simultaneously
        R_nn_inv = np.linalg.inv(R_nn_stable)

        # --- Calculate MVDR Weights ---
        # Numerator: R_nn_inv * h -> (K, M, M) * (K, M) -> (K, M)
        weights_nom = np.einsum("fmn,fn->fm", R_nn_inv, h)
        
        # Denominator: h^H * numerator -> (K, M) * (K, M) -> (K,)
        weights_den = np.einsum("fm,fm->f", h.conj(), weights_nom)
        
        # Divide numerator by denominator
        # Expand dims of denominator to allow broadcasting from (K,) to (K, M)
        weights = weights_nom / (weights_den[:, np.newaxis] + 1e-10)

        weights_rec[:, m, :] = weights
        
        # Apply weights to the current observation to get the clean output
        Y_stft[:, m] = np.einsum("fm,fm->f", weights.conj(), X_frame)

    if save_weights:
        return Y_stft, weights_rec
    else:
        return Y_stft
    

def MVDR_recursive(X_stft, vad, fs, array_geometry, source_pos, length_fft, hop_length_fft, min_loading = 1e-6, save_weights=False):
    lamda = 0.99
    beta = 1e-2 # relative loading
    K, T, M = X_stft.shape  
    frecs = np.linspace(0, fs/2, K)

    # Get steering vectors, expected shape (K, M)
    sv = compute_rtf_steering_vector(frecs, source_pos, array_geometry, ref_mic_idx=0, mode="near_field", squeeze=True)
    
    # Initialize output complex STFT matrix
    Y_stft = np.zeros((K, T), dtype=np.complex128)
    
    # Initialize noise covariance matrix R_nn for all frequencies (K, M, M)
    # We use a small diagonal loading to prevent singularities from the start
    R_nn = np.tile(np.eye(M, dtype=np.complex128) * 1e-6, (K, 1, 1))
    count = 0
    

    #save weights
    weights_rec = np.zeros((K,T,M), dtype=np.complex128)
    for m in range(T):
        # Extract the current frame across all frequencies, shape (K, M)
        X_frame = X_stft[:, m, :]

        # Define VAD frame state (mapping STFT frame to time-domain VAD)
        vad_frame = vad[m * hop_length_fft : length_fft + m * hop_length_fft]
        vad_status = np.mean(vad_frame) > 0.3

        # Update noise covariance ONLY when target speech is absent
        if not vad_status:
            # Outer product of the frame: (K, M) and (K, M) -> (K, M, M)
            R_nn_instant = np.einsum("fm,fn->fmn", X_frame, X_frame.conj())
            R_nn = lamda * R_nn + (1 - lamda) * R_nn_instant

        # Cálculo dinámico parametrizado
        tr_R = np.real(np.trace(R_nn, axis1=1, axis2=2))
        adaptive_load = beta * (tr_R[:, None, None] / M)
        loading = np.maximum(adaptive_load, min_loading)
        
        R_nn_stable = R_nn + np.eye(M)[None, :, :] * loading
        
        # Invert the covariance matrices for all frequencies simultaneously
        R_nn_inv = np.linalg.inv(R_nn_stable)

        # Calculate MVDR Weights
        # Numerator: R_nn_inv * d -> (K, M, M) * (K, M) -> (K, M)
        weights_nom = np.einsum("fmn,fn->fm", R_nn_inv, sv)
        
        # Denominator: d^H * numerator -> (K, M) * (K, M) -> (K,)
        weights_den = np.einsum("fm,fm->f", sv.conj(), weights_nom)
        
        # Divide numerator by denominator. 
        # Expand dims of denominator to allow broadcasting from (K,) to (K, M)
        weights = weights_nom / weights_den[:, np.newaxis]

        weights_rec[:,m,:] = weights
        # Apply weights to the current observation to get the clean output
        # Output is shape (K,) for the current frame
        Y_stft[:, m] = np.einsum("fm,fm->f", weights.conj(), X_frame)

    if save_weights:
        return Y_stft, weights_rec
    else:
        return Y_stft

# Normalize signals to range [-0.99, 0.99] to prevent clipping when saving as WAV
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
# from your_mvdr_module import MVDR_recursive 
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
    Y_stft = MVDR_recursive(
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



if __name__ == "__main__":
    from beamforming.MWF.SP_SDW_MWF_base import process_wpe_online

    # Basic simulation parameters
    FS = 16000
    M1, M2 = 12, 1          
    M = M1 * M2
    speed_of_sound = 343.0 

    iSIR_dB = 0
    
    print("=== INTEGRATION TEST: PIPELINE (FREE-FIELD, ROOM, WPE+ROOM) ===")
    
    output_folder = "tests/data/mvdr_base_output"
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
    acoustic_scene = SimAcoustic(mic_coords, array_mismatch=0.0, duration=20, fs=FS)
    acoustic_scene.set_source(r"data/audio/input/p002_emo_adoration_sentences.wav", gain=1, position=source_pos_2d)
    acoustic_scene.set_interference(r"data/audio/input/hairdryer_07_SH_MKH800.wav", gain=1, position=interf_pos1.reshape(1,3))
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
    
    save_wav("4_ROOM_output_final.wav", FS, normalize_signal(output_rm), output_folder)

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