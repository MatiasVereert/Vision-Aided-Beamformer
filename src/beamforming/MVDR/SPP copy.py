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

def SPP_MVDR_recursive(X_stft, fs, array_geometry, source_pos, save_weights = False):
    # but it is completely ignored in favor of the Spatial SPP.
    lamda = 0.99
    K, T, M = X_stft.shape  
    frecs = np.linspace(0, fs/2, K)

    # Get steering vectors, expected shape (K, M)
    sv = compute_rtf_steering_vector(frecs, source_pos, array_geometry, ref_mic_idx=0, mode="near_field", squeeze=True)
    
    # Initialize output complex STFT matrix
    Y_stft = np.zeros((K, T), dtype=np.complex128)
    
    # Pre-create diagonal loading matrix to be used inside the loop
    diag_load = np.tile(np.eye(M, dtype=np.complex128) * 1e-6, (K, 1, 1))

    # Initialize noise covariance matrix R_nn
    R_nn = np.tile(np.eye(M, dtype=np.complex128) * 1e-6, (K, 1, 1))
    
    # Initialize noisy signal covariance matrix R_xx using the steering vector
    # We assume an initial PSD (phi_s) for the source to seed the matrix
    phi_s = 1e-3 
    R_ss_init = phi_s * np.einsum("fm,fn->fmn", sv, sv.conj())
    R_xx = R_ss_init + R_nn

    # Initialize the inverse of R_nn for the first frame's SPP calculation
    R_nn_inv = np.linalg.inv(R_nn + diag_load)

    # SPP Hyperparameters
    # Threshold for spatial SNR (gamma). 3.0 linear is approx 4.7 dB
    gamma_th = 3.0 
    spp_slope = 2.0 

    # weights recording
    weights_rec = np.zeros((K,T,M), dtype=np.complex128)


    for m in range(T):
        # Extract the current frame across all frequencies, shape (K, M)
        # Extract the current frame across all frequencies, shape (K, M)
        X_frame = X_stft[:, m, :]

        # --- NEW: Instantaneous Gain Normalization ---
        # Calculate the magnitude of the signal at each microphone
        mags = np.abs(X_frame) + 1e-12
        
        # Calculate a correction factor relative to the reference microphone (idx 0)
        # This preserves the delicate phase differences but crushes gain mismatches
        gain_correction = mags[:, 0:1] / mags
        X_frame_norm = X_frame * gain_correction

        # --- 1. Calculate A Posteriori Spatial SNR (gamma) ---
        # USE THE NORMALIZED FRAME HERE instead of raw X_frame
        R_nn_inv_x = np.einsum("fmn,fn->fm", R_nn_inv, X_frame_norm)
        num_complex = np.einsum("fm,fm->f", sv.conj(), R_nn_inv_x)
        num = np.abs(num_complex)**2
        
        R_nn_inv_v = np.einsum("fmn,fn->fm", R_nn_inv, sv)
        den_complex = np.einsum("fm,fm->f", sv.conj(), R_nn_inv_v)
        den = np.real(den_complex) 
        
        gamma = num / (den + 1e-10)
        
        # ... [El resto de tu código de actualización P, R_xx y R_nn sigue usando X_frame normal] ...

        # --- 2. Map Gamma to Spatial Presence Probability (SPP) ---
        # Using a sigmoid function to map SNR to a probability [0, 1]
        P = 1.0 / (1.0 + np.exp(-spp_slope * (gamma - gamma_th)))
        
        # Clip probabilities to prevent matrices from completely freezing
        P = np.clip(P, 0.05, 0.95)
        P_expand = P[:, np.newaxis, np.newaxis]

        # --- 3. Update Covariance Matrices ---
        # Calculate instantaneous covariance of the observation
        R_instant = np.einsum("fm,fn->fmn", X_frame, X_frame.conj())
        
        # Update R_xx weighted by the probability of target speech presence
        R_xx = lamda * R_xx + (1 - lamda) * P_expand * R_instant
        
        # Update R_nn weighted by the probability of target speech absence
        R_nn = lamda * R_nn + (1 - lamda) * (1 - P_expand) * R_instant

        # --- 4. Compute MVDR Weights ---
        # Prepare stable R_nn and invert it for the current frame
        R_nn_stable = R_nn + diag_load
        R_nn_inv = np.linalg.inv(R_nn_stable)

        # Numerator: R_nn_inv * v
        weights_nom = np.einsum("fmn,fn->fm", R_nn_inv, sv)
        
        # Denominator: v^H * numerator
        weights_den = np.einsum("fm,fm->f", sv.conj(), weights_nom)
        
        # Final weights calculation
        weights = weights_nom / (weights_den[:, np.newaxis] + 1e-10)
        weights_rec[:,m,:] = weights
        # --- 5. Apply Filter ---
        Y_stft[:, m] = np.einsum("fm,fm->f", weights.conj(), X_frame)

    if save_weights:
        return Y_stft, weights_rec
    else:
        return Y_stft




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
# from your_mvdr_module import SPP_SPP_MVDR_recursive 
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
    Y_stft = SPP_MVDR_recursive(
        X_stft=X_stft, 
        fs=fs, 
        array_geometry=mic_coords, 
        source_pos=source_pos_2d, 

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
    
    output_folder = "tests/data/mvdr_SPP0_output"
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