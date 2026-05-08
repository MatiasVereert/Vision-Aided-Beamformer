import os
import numpy as np
import scipy.signal as signal

# We keep the import just in case, but GEV won't use it because it's blind
from beamforming.signal_model import compute_rtf_steering_vector
from propagation.simulate_acoustics import SimAcoustic
from utils.audio import save_wav, normalize_signal
from dereverberation.nara_wrappers import process_wpe_online


def Blind_GEV_recursive(X_stft, beta=1e-3, min_loading=1e-6, save_weights=False):
    """
    Blind Generalized Eigenvalue (GEV) Beamformer with BAN post-filter.
    Uses a normal energy-based SPP mask instead of a spatial one.
    """
    lamda = 0.99
    K, T, M = X_stft.shape  

    Y_stft = np.zeros((K, T), dtype=np.complex128)
    
    # --- 1. Compute Normal SPP Mask (Blind, based on Mic 0) ---
    print(" -> [Blind SPP] Estimating normal SPP mask from reference mic...")
    X_ref = X_stft[:, :, 0]
    power_ref = np.abs(X_ref)**2
    
    SPP = np.zeros((K, T))
    noise_psd = np.copy(power_ref[:, 0]) # Initialize noise PSD
    alpha_n = 0.95
    gamma_th = 3.0  
    spp_slope = 2.0 
    
    for m in range(T):
        # A posteriori SNR
        gamma = power_ref[:, m] / (noise_psd + 1e-10)
        
        # Sigmoid mapping to probability
        P = 1.0 / (1.0 + np.exp(-spp_slope * (gamma - gamma_th)))
        SPP[:, m] = np.clip(P, 0.05, 0.95)
        
        # Update noise PSD recursively only when speech is likely absent
        is_noise = gamma < 2.0
        noise_psd = np.where(is_noise, alpha_n * noise_psd + (1 - alpha_n) * power_ref[:, m], noise_psd)

    # --- 2. Initialize Covariance Matrices ---
    # Since it's blind, we initialize with identity matrices instead of steering vectors
    R_xx = np.tile(np.eye(M, dtype=np.complex128) * 1e-3, (K, 1, 1))
    R_nn = np.tile(np.eye(M, dtype=np.complex128) * 1e-6, (K, 1, 1))
    
    weights_rec = np.zeros((K, T, M), dtype=np.complex128)

    print(" -> [GEV Beamformer] Computing recursive weights...")
    for m in range(T):
        X_frame = X_stft[:, m, :]
        P_expand = SPP[:, m, np.newaxis, np.newaxis]

        # --- Covariance Updates ---
        R_instant = np.einsum("fm,fn->fmn", X_frame, X_frame.conj())
        R_xx = lamda * R_xx + (1 - lamda) * P_expand * R_instant
        R_nn = lamda * R_nn + (1 - lamda) * (1 - P_expand) * R_instant

        # --- Dynamic Diagonal Loading ---
        tr_R = np.real(np.trace(R_nn, axis1=1, axis2=2))
        adaptive_load = beta * (tr_R[:, None, None] / M)
        loading = np.maximum(adaptive_load, min_loading)
        R_nn_stable = R_nn + np.eye(M)[None, :, :] * loading

        R_nn_inv = np.linalg.inv(R_nn_stable)

        # --- GEV Calculation ---
        # Solve the eigenvalue problem: R_nn^-1 * R_xx
        matrix_for_eig = np.matmul(R_nn_inv, R_xx)
        
        # Calculate eigenvalues and eigenvectors for all frequency bins
        eigenvalues, eigenvectors = np.linalg.eig(matrix_for_eig)
        
        # Extract the eigenvector corresponding to the maximum eigenvalue
        max_idx = np.argmax(np.real(eigenvalues), axis=1)
        w_gev = eigenvectors[np.arange(K), :, max_idx] # Shape: (K, M)

        # --- BAN Post-filter (Blind Analytic Normalization) ---
        # Calculate expected output noise power: w^H * R_nn * w
        w_H_R_w = np.real(np.einsum('km,kmn,kn->k', w_gev.conj(), R_nn_stable, w_gev))
        
        # Calculate BAN scaling factor to match input average noise power
        ban_scale = np.sqrt(tr_R / M) / (np.sqrt(w_H_R_w) + 1e-10)
        w_final = w_gev * ban_scale[:, np.newaxis]
        
        # --- Phase Alignment ---
        # GEV phase is arbitrary. We align it to the reference microphone (Mic 0) 
        # to prevent destructive phase jumps between STFT frames.
        phase_ref = np.exp(-1j * np.angle(w_final[:, 0]))
        w_final = w_final * phase_ref[:, np.newaxis]
        
        weights_rec[:, m, :] = w_final
        
        # --- Apply Filter ---
        Y_stft[:, m] = np.einsum("fm,fm->f", w_final.conj(), X_frame)

    if save_weights:
        return Y_stft, weights_rec
    else:
        return Y_stft
    

def apply_gev_stft_bridge(time_domain_input, fs, length_fft=512, hop_length_fft=256):
    """
    Helper function to wrap the STFT -> GEV -> ISTFT process.
    Notice that vad_oracle, mic_coords, and source_pos are no longer passed
    to the beamformer because it is completely blind.
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
    
    # Execute the Recursive Blind GEV
    Y_stft = Blind_GEV_recursive(
        X_stft=X_stft, 
 
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
    M1, M2 = 8, 1          
    M = M1 * M2
    speed_of_sound = 343.0 

    iSIR_dB = 0
    
    print("=== INTEGRATION TEST: BLIND GEV PIPELINE (FREE-FIELD, ROOM, WPE+ROOM) ===")
    
    output_folder = "tests/data/gev_blind_output"
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
    acoustic_scene = SimAcoustic(mic_coords, array_mismatch=0.0, duration=15, fs=FS)
    
    # Note: Use raw strings or forward slashes for paths to avoid escape sequence errors
    acoustic_scene.set_source(r"tools\data\signals\p002_emo_adoration_sentences.wav", gain=1, position=source_pos_2d)
    acoustic_scene.set_interference(r"tools\data\signals\hairdryer_07_SH_MKH800.wav", gain=1, position=interf_pos1.reshape(1,3))

    # -------------------------------------------------------------------
    # PHASE 1: FREE FIELD SIMULATION (Anechoic)
    # -------------------------------------------------------------------
    cache_ff_path = os.path.join(output_folder, "cache_free_field.npz")
    
    if os.path.exists(cache_ff_path):
        print("\n--- PHASE 1: LOADING FREE FIELD SIMULATION FROM CACHE ---")
        cache_data = np.load(cache_ff_path)
        free_field_input = cache_data['input']
    else:
        print("\n--- PHASE 1: COMPUTING FREE FIELD SIMULATION ---")
        free_field_input, vad_oracle_ff = acoustic_scene.free_field(iSIR_dB=iSIR_dB, normalize=True, mode="ideal", VAD=True)
        # Save to cache
        np.savez(cache_ff_path, input=free_field_input, vad=vad_oracle_ff)
        
    save_wav("1_FF_input_mix_mic0.wav", FS, free_field_input[0], output_folder)
    
    print(" -> Applying Blind GEV...")
    output_ff = apply_gev_stft_bridge(free_field_input, FS)
    
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

    print(" -> Applying Blind GEV (Without WPE)...")
    output_rm = apply_gev_stft_bridge(room_input, FS)
    
    save_wav("4_ROOM_output_final.wav", FS, normalize_signal(output_rm), output_folder)

    # -------------------------------------------------------------------
    # PHASE 3: WPE DEREVERBERATION + RECURSIVE GEV
    # -------------------------------------------------------------------
    print("\n--- PHASE 3: WPE + BLIND GEV PIPELINE ---")
    print(" -> Applying Online WPE Dereverberation on Room Simulation...")
    
    wpe_output = process_wpe_online(room_input, delay=1, taps=12)
    
    save_wav("5_WPE_input_mix_mic0.wav", FS, wpe_output[0], output_folder)

    print(" -> Applying Blind GEV on Dereverberated Signals...")
    output_wpe = apply_gev_stft_bridge(wpe_output, FS)
    
    save_wav("6_WPE_ROOM_output_final.wav", FS, normalize_signal(output_wpe), output_folder)

    print("\n -> Pipeline completed successfully.")