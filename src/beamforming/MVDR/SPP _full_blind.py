# Assuming the import works correctly in your local environment
from beamforming.signal_model import compute_rtf_steering_vector
import numpy as np 
import scipy.signal as signal
from dereverberation.nara_wrappers import process_wpe_online
from propagation.simulate_acoustics import SimAcoustic
from utils.audio import normalize_signal, save_wav
import os


def SPP_MVDR_recursive(X_stft, fs, array_geometry, source_pos, beta=1e-3, min_loading=1e-6, save_weights=False, save_mask=False):
    """
    Recursive MVDR Beamformer using Covariance Subtraction (CS) for data-driven RTF estimation.
    """
    # Forgetting factor for covariance matrix smoothing
    lamda = 0.99
    K, T, M = X_stft.shape  
    frecs = np.linspace(0, fs / 2, K)

    # Get geometric steering vectors as an initial seed and spatial backup
    # Expected shape: (K, M)
    sv_geom = compute_rtf_steering_vector(frecs, source_pos, array_geometry, ref_mic_idx=0, mode="near_field", squeeze=True)
    
    # Initialize output complex STFT matrices
    Y_stft = np.zeros((K, T), dtype=np.complex128)
    Y_spp_stft = np.zeros((K, T), dtype=np.complex128) 
    
    # Initialize noise covariance matrix R_nn
    R_nn = np.tile(np.eye(M, dtype=np.complex128) * 1e-6, (K, 1, 1))
    
    # Initialize noisy signal covariance matrix R_xx using the geometric steering vector seed
    phi_s = 1e-3 
    R_ss_init = phi_s * np.einsum("fm,fn->fmn", sv_geom, sv_geom.conj())
    R_xx = R_ss_init + R_nn

    # Initialize the inverse of R_nn for the first frame's SPP calculation
    initial_diag_load = np.tile(np.eye(M, dtype=np.complex128) * min_loading, (K, 1, 1))
    R_nn_inv = np.linalg.inv(R_nn + initial_diag_load)

    # SPP Hyperparameters
    gamma_th = 6
    spp_slope = 6

    weights_rec = np.zeros((K, T, M), dtype=np.complex128)
    
    # Selection vector i_0 to extract the reference microphone (mic 0) component
    i_0 = np.zeros(M, dtype=np.complex128)
    i_0[0] = 1.0

    # Initialize data-driven RTF tracking array
    rtf_cs = np.copy(sv_geom)

    for m in range(T):
        X_frame = X_stft[:, m, :]

        # --- 1. Calculate Robust A Posteriori Spatial SNR (gamma) ---
        # We use the current RTF estimate to evaluate instantaneous spatial SNR
        num_complex = np.einsum("fm,fm->f", rtf_cs.conj(), X_frame)
        num = np.abs(num_complex)**2
        
        v_norm = np.real(np.einsum("fm,fm->f", rtf_cs.conj(), rtf_cs))
        frame_power = np.real(np.einsum("fm,fm->f", X_frame.conj(), X_frame)) / M
        den = v_norm * frame_power
        
        gamma = num / (den + 1e-10)

        # --- 2. Map Gamma to Speech Presence Probability (SPP) ---
        P = 1.0 / (1.0 + np.exp(-spp_slope * (gamma - gamma_th)))
        
        P_expand = P[:, np.newaxis, np.newaxis]
        Y_spp_stft[:, m] = P * X_frame[:, 0]

        # --- 3. Update Covariance Matrices ---
        R_instant = np.einsum("fm,fn->fmn", X_frame, X_frame.conj())

        # Update R_xx weighted by target speech presence probability
        R_xx = lamda * R_xx + (1 - lamda) * P_expand * R_instant
        
        # Update R_nn weighted by target speech absence probability
        R_nn = lamda * R_nn + (1 - lamda) * (1 - P_expand) * R_instant

        # --- 4. DATA-DRIVEN RTF ESTIMATION (Covariance Subtraction) ---
        # Subtract noise covariance from mixture covariance: R_s = R_xx - R_nn
        R_sub = R_xx - R_nn
        
        # Extract the column corresponding to the reference microphone: (R_xx - R_nn) * i_0
        rtf_nom = R_sub[:, :, 0]
        
        # Extract the normalization factor (reference mic auto-power subtraction): i_0^H * (R_xx - R_nn) * i_0
        rtf_den = np.real(R_sub[:, 0, 0])
        
        # To avoid instabilities during early frames or extreme noise periods, 
        # apply a soft threshold and fall back smoothly to the geometric vector
        valid_bins = rtf_den > 1e-9
        
        if np.any(valid_bins):
            rtf_cs[valid_bins, :] = rtf_nom[valid_bins, :] / rtf_den[valid_bins, np.newaxis]
            
        # Fallback for unstable frequency bins
        rtf_cs[~valid_bins, :] = sv_geom[~valid_bins, :]

        # --- 5. Compute MVDR Weights with Dynamic Diagonal Loading ---
        tr_R = np.real(np.trace(R_nn, axis1=1, axis2=2))
        adaptive_load = beta * (tr_R[:, None, None] / M)
        
        loading = np.maximum(adaptive_load, min_loading)
        R_nn_stable = R_nn + np.eye(M)[None, :, :] * loading

        R_nn_inv = np.linalg.inv(R_nn_stable)

        # Numerator: R_nn_inv * rtf_cs
        weights_nom = np.einsum("fmn,fn->fm", R_nn_inv, rtf_cs)
        
        # Denominator: rtf_cs^H * numerator
        weights_den = np.einsum("fm,fm->f", rtf_cs.conj(), weights_nom)
        
        weights = weights_nom / (weights_den[:, np.newaxis] + 1e-10)
        weights_rec[:, m, :] = weights
        
        # --- 6. Apply Filter ---
        Y_stft[:, m] = np.einsum("fm,fm->f", weights.conj(), X_frame)

    if save_weights:
        if save_mask:
            return Y_stft, Y_spp_stft, weights_rec
        else:
            return Y_stft, weights_rec
    else:
        if save_mask:
            return Y_stft, Y_spp_stft
        else: 
            return Y_stft


def apply_mvdr_stft_bridge(time_domain_input, vad_oracle, mic_coords, source_pos_2d, fs, length_fft=512, hop_length_fft=256):
    """
    Helper function to wrap the STFT -> MVDR -> ISTFT process.
    """
    freqs, times, Zxx = signal.stft(
        time_domain_input, 
        fs=fs, 
        nperseg=length_fft, 
        noverlap=length_fft - hop_length_fft
    )
    
    X_stft = np.transpose(Zxx, (1, 2, 0))
    vad_padded = np.pad(vad_oracle, (0, length_fft + hop_length_fft), mode='constant')

    Y_stft = SPP_MVDR_recursive(
        X_stft=X_stft, 
        fs=fs, 
        array_geometry=mic_coords, 
        source_pos=source_pos_2d, 
    )
    
    _, y_time = signal.istft(
        Y_stft, 
        fs=fs, 
        nperseg=length_fft, 
        noverlap=length_fft - hop_length_fft
    )
    
    original_length = time_domain_input.shape[1]
    return y_time[:original_length]


if __name__ == "__main__":
    # Basic simulation parameters
    FS = 16000
    M1, M2 = 12, 1          
    M = M1 * M2
    speed_of_sound = 343.0 

    iSIR_dB = 0
    
    print("=== INTEGRATION TEST: PIPELINE (FREE-FIELD, ROOM, WPE+ROOM) ===")
    
    output_folder = "tests/data/mvdr_SPP_output"
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
    
    wpe_output = process_wpe_online(room_input, delay=3, taps=7)
    
    save_wav("5_WPE_input_mix_mic0.wav", FS, wpe_output[0], output_folder)

    print(" -> Applying Recursive MVDR on Dereverberated Signals...")
    output_wpe = apply_mvdr_stft_bridge(wpe_output, vad_oracle_room, mic_coords, source_pos_2d, FS)
    
    save_wav("6_WPE_ROOM_output_final.wav", FS, normalize_signal(output_wpe), output_folder)

    print("\n -> Pipeline completed successfully.")
    