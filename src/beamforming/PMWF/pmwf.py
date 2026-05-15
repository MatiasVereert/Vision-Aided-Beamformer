import numpy as np 
import scipy.signal as signal
import matplotlib.pyplot as plt

# Assuming the import works correctly in your local environment
from beamforming.signal_model import compute_rtf_steering_vector
import numpy as np


def pmwf_recursive(X_stft, vad, fs, array_geometry, source_pos, length_fft, hop_length_fft, alpha=0.98):
    K, T, M = X_stft.shape  
    frecs = np.linspace(0, fs/2, K)
    
    # Tuning parameter for the PMWF (Beta = 1 corresponds to Wiener, Beta = 0 to MVDR)
    Beta = 0

    # Reference mic index
    ref_mic_idx = 0
    warmup_frames = 30

    # Initialize output complex STFT matrix
    Y_stft = np.zeros((K, T), dtype=np.complex128)
    
    # Initialize covariance matrices for all frequencies (K, M, M)
    R_vv = np.tile(np.eye(M, dtype=np.complex128) * 1e-6, (K, 1, 1))
    R_yy = np.tile(np.eye(M, dtype=np.complex128) * 1e-6, (K, 1, 1))
    
    # Pre-create diagonal loading matrix to be used inside the loop
    diag_load = np.tile(np.eye(M, dtype=np.complex128) * 1e-3, (K, 1, 1))

    # Selection vector u to extract the reference microphone column
    u = np.zeros(M, dtype=int)
    u[ref_mic_idx] = 1

    # Identity matrix for the alternative numerator formula
    I_matrix = np.eye(M, dtype=np.complex128)[np.newaxis, :, :]

    for m in range(T):
        # Extract the current frame across all frequencies, shape (K, M)
        X_frame = X_stft[:, m, :]

        # Define VAD frame state (mapping STFT frame to time-domain VAD)
        vad_frame = vad[m * hop_length_fft : length_fft + m * hop_length_fft]
        vad_status = np.mean(vad_frame) > 0.1

        # Calculate instantaneous covariance matrix
        R_instant = np.einsum("fm,fn->fmn", X_frame, X_frame.conj())

        # FIX 1: Observation covariance matrix R_yy must be updated unconditionally
        # to ensure it tracks the true signal statistics, preventing stale speech 
        # energy during noise-only periods.
        R_yy = alpha * R_yy + (1 - alpha) * R_instant

        if not vad_status:
            # Update noise covariance ONLY when target speech is absent
            R_vv = alpha * R_vv + (1 - alpha) * R_instant

        # Dynamic diagonal loading for matrix inversion stability
        # ... (cálculo de R_yy y R_vv como antes) ...

        # 1. Aplicar la MISMA regularización a ambas matrices
        eps = 1e-2
        trace_R_vv = np.trace(R_vv, axis1=1, axis2=2) 
        dynamic_diag = I_matrix * trace_R_vv[:, np.newaxis, np.newaxis] * eps
        
        R_vv_stable = R_vv + dynamic_diag + diag_load  
        R_yy_stable = R_yy + dynamic_diag + diag_load  # <--- CORRECCIÓN CRÍTICA
        
        R_vv_inv = np.linalg.inv(R_vv_stable)

        # Batch matrix multiplication: R_vv_inv * R_yy_stable
        R_vv_inv_R_yy = R_vv_inv @ R_yy_stable
        
        lambda_eig = np.trace(R_vv_inv_R_yy, axis1=1, axis2=2) - M
        
        # 2. Piso de seguridad para evitar divisiones peligrosas
        # En lugar de 0.0, usamos un valor pequeño positivo
        lambda_eig = np.maximum(np.real(lambda_eig), 1e-3) # <--- CORRECCIÓN CRÍTICA

        if m < warmup_frames:
            Y_stft[:, m] = X_frame[:, ref_mic_idx]
        else:
            numerator_matrix = R_vv_inv_R_yy - I_matrix
            weights_nom = numerator_matrix @ u 
            
            # Al sumarle Beta (0) a lambda_eig (min 1e-3), el denominador es estable
            weights_den = Beta + lambda_eig
            weights = weights_nom / weights_den[:, np.newaxis]
            
            Y_stft[:, m] = np.einsum("fm,fm->f", weights.conj(), X_frame)

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

from beamforming.signal_model import compute_rtf_steering_vector


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
    Y_stft = pmwf_recursive(
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
    
    output_folder = "tests/data/mvdr_output"
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