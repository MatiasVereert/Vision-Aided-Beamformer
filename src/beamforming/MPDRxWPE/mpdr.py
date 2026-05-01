import numpy as np 
import scipy.signal as signal
import os

# Assuming the import works correctly in your local environment
from beamforming.signal_model import compute_rtf_steering_vector
from propagation.simulate_acoustics import SimAcoustic
from utils.audio import save_wav, normalize_signal
from beamforming.MWF.SP_SDW_MWF_base import process_wpe_online
from beamforming.MPDRxWPE.mpdr import MPDR_recursive


def MPDR_recursive(X_stft, fs, array_geometry, source_pos, beta=1e-3, min_loading=1e-6, save_weights=False):
    """
    MPDR implementation: Updates the covariance matrix continuously using the 
    observed signal, without relying on a Voice Activity Detector (VAD).
    """
    lamda = 0.999
    K, T, M = X_stft.shape  
    frecs = np.linspace(0, fs/2, K)

    # Get steering vectors, expected shape (K, M)
    sv = compute_rtf_steering_vector(
        frecs, source_pos, array_geometry, 
        ref_mic_idx=0, mode="near_field", squeeze=True
    )
    
    # Initialize output complex STFT matrix
    Y_stft = np.zeros((K, T), dtype=np.complex128)
    
    # Initialize observation covariance matrix R_yy for all frequencies (K, M, M)
    R_yy = np.tile(np.eye(M, dtype=np.complex128) * 1e-6, (K, 1, 1))
    
    # Initialize tracking weight matrix
    weights_rec = np.zeros((K, T, M), dtype=np.complex128)
    
    for m in range(T):
        # Extract the current frame across all frequencies, shape (K, M)
        X_frame = X_stft[:, m, :]

        # Unconditional update for MPDR (Uses observed signal directly)
        # Outer product of the frame: (K, M) and (K, M) -> (K, M, M)
        R_yy_instant = np.einsum("fm,fn->fmn", X_frame, X_frame.conj())
        R_yy = lamda * R_yy + (1 - lamda) * R_yy_instant

        # Dynamic parameterization for Diagonal Loading
        tr_R = np.real(np.trace(R_yy, axis1=1, axis2=2))
        adaptive_load = beta * (tr_R[:, None, None] / M)
        loading = np.maximum(adaptive_load, min_loading)
        
        R_yy_stable = R_yy + np.eye(M)[None, :, :] * loading
        
        # Invert the covariance matrices for all frequencies simultaneously
        R_yy_inv = np.linalg.inv(R_yy_stable)

        # Calculate MPDR Weights
        # Numerator: R_yy_inv * sv -> (K, M, M) * (K, M) -> (K, M)
        weights_nom = np.einsum("fmn,fn->fm", R_yy_inv, sv)
        
        # Denominator: sv^H * numerator -> (K, M) * (K, M) -> (K,)
        weights_den = np.einsum("fm,fm->f", sv.conj(), weights_nom)
        
        # Divide numerator by denominator. 
        # Expand dims of denominator to allow broadcasting from (K,) to (K, M)
        weights = weights_nom / weights_den[:, np.newaxis]
        
        # Save weights for analysis
        if save_weights:
            weights_rec[:, m, :] = weights

        # Apply weights to the current observation to get the clean output
        # Output is shape (K,) for the current frame
        Y_stft[:, m] = np.einsum("fm,fm->f", weights.conj(), X_frame)

    if save_weights:
        return Y_stft, weights_rec
    else:
        return Y_stft


def apply_mpdr_stft_bridge(time_domain_input, mic_coords, source_pos_2d, fs, length_fft=512, hop_length_fft=256, beta=1e-3, min_loading=1e-6):
    """
    Helper function to wrap the STFT -> MPDR -> ISTFT process.
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

    # Execute the Recursive MPDR (No VAD required)
    Y_stft = MPDR_recursive(
        X_stft=X_stft, 
        fs=fs, 
        array_geometry=mic_coords, 
        source_pos=source_pos_2d, 
        beta=beta,
        min_loading=min_loading
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
    # Basic simulation parameters
    FS = 16000
    M1, M2 = 12, 1          
    M = M1 * M2
    speed_of_sound = 343.0 

    iSIR_dB = 0
    
    print("=== INTEGRATION TEST: MPDR PIPELINE (FREE-FIELD, ROOM, WPE+ROOM) ===")
    
    output_folder = "tests/data/mpdr_base_output"
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
        # VAD is ignored for MPDR but we can load it if cached by other tests
    else:
        print("\n--- PHASE 1: COMPUTING FREE FIELD SIMULATION ---")
        free_field_input, vad_oracle_ff = acoustic_scene.free_field(iSIR_dB=iSIR_dB, normalize=True, mode="ideal", VAD=True)
        np.savez(cache_ff_path, input=free_field_input, vad=vad_oracle_ff)
        
    save_wav("1_FF_input_mix_mic0.wav", FS, free_field_input[0], output_folder)
    
    print(" -> Applying Recursive MPDR...")
    output_ff = apply_mpdr_stft_bridge(free_field_input, mic_coords, source_pos_2d, FS)
    
    save_wav("2_FF_output_MPDR.wav", FS, normalize_signal(output_ff), output_folder)

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

    print(" -> Applying Recursive MPDR (Without WPE)...")
    output_rm = apply_mpdr_stft_bridge(room_input, mic_coords, source_pos_2d, FS)
    
    save_wav("4_ROOM_output_MPDR.wav", FS, normalize_signal(output_rm), output_folder)

    # -------------------------------------------------------------------
    # PHASE 3: WPE DEREVERBERATION + RECURSIVE MPDR
    # -------------------------------------------------------------------
    print("\n--- PHASE 3: WPE + MPDR PIPELINE ---")
    print(" -> Applying Online WPE Dereverberation on Room Simulation...")
    
    wpe_output = process_wpe_online(room_input)
    
    save_wav("5_WPE_input_mix_mic0.wav", FS, wpe_output[0], output_folder)

    print(" -> Applying Recursive MPDR on Dereverberated Signals...")
    output_wpe = apply_mpdr_stft_bridge(wpe_output, mic_coords, source_pos_2d, FS)
    
    save_wav("6_WPE_ROOM_output_MPDR.wav", FS, normalize_signal(output_wpe), output_folder)

    print("\n -> MPDR Pipeline completed successfully.")