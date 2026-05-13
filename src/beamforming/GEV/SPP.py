# Assuming the import works correctly in your local environment
from beamforming.signal_model import compute_rtf_steering_vector
import numpy as np 
import scipy.signal as signal
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from utils.audio import normalize_signal, save_wav
import os
from propagation.simulate_acoustics import SimAcoustic
from dereverberation.nara_wrappers import process_wpe_online


def SPP_GEVD_recursive(X_stft, fs, array_geometry, source_pos, beta=1e-3, min_loading=1e-6, save_weights=False):
    # Forgetting factor for covariance matrix smoothing
    lamda = 0.99
    K, T, M = X_stft.shape  
    frecs = np.linspace(0, fs/2, K)

    # Get steering vectors, expected shape (K, M)
    sv = compute_rtf_steering_vector(frecs, source_pos, array_geometry, ref_mic_idx=0, mode="near_field", squeeze=True)
    
    # Initialize output complex STFT matrices
    Y_stft = np.zeros((K, T), dtype=np.complex128)
    Y_spp_stft = np.zeros((K, T), dtype=np.complex128) 
    
    # Initialize noise covariance matrix R_nn
    R_nn = np.tile(np.eye(M, dtype=np.complex128) * 1e-6, (K, 1, 1))
    
    # Initialize noisy signal covariance matrix R_xx using the steering vector
    # We assume an initial PSD (phi_s) for the source to seed the matrix
    phi_s = 1e-3 
    R_ss_init = phi_s * np.einsum("fm,fn->fmn", sv, sv.conj())
    R_xx = R_ss_init + R_nn

    # --- BLIND SPP INITIALIZATION ---
    # Estimate initial noise power spectral density (PSD) averaging all channels 
    # assuming the first frame is mostly noise/silence
    noise_psd = np.mean(np.abs(X_stft[:, 0, :])**2, axis=1) 
    alpha_n = 0.95 
    
    # SPP Hyperparameters
    gamma_th = 4
    spp_slope = 8.0 

    weights_rec = np.zeros((K, T, M), dtype=np.complex128)

    for m in range(T):
        X_frame = X_stft[:, m, :]

        # --- 1. Blind Multichannel A Posteriori SNR Calculation ---
        inst_power = np.mean(np.abs(X_frame)**2, axis=1) 
        gamma = inst_power / (noise_psd + 1e-10)

        # --- 2. Map Gamma to Speech Presence Probability (SPP) ---
        P = 1.0 / (1.0 + np.exp(-spp_slope * (gamma - gamma_th)))
        P = np.clip(P, 0.05, 0.95)
        
        # --- 2b. Blind Noise Floor Tracking Update ---
        prob_noise = 1.0 - P
        noise_psd = prob_noise * (alpha_n * noise_psd + (1.0 - alpha_n) * inst_power) + P * noise_psd

        P_expand = P[:, np.newaxis, np.newaxis]
        Y_spp_stft[:, m] = P * X_frame[:, 0]

        # --- 3. Update Covariance Matrices ---
        R_instant = np.einsum("fm,fn->fmn", X_frame, X_frame.conj())

        # Update R_xx weighted by the probability of target speech presence
        R_xx = lamda * R_xx + (1 - lamda) * P_expand * R_instant
        
        # Update R_nn weighted by the probability of target speech absence
        R_nn = lamda * R_nn + (1 - lamda) * (1 - P_expand) * R_instant

        # --- 4. Compute GEVD Weights with Dynamic Diagonal Loading & BAN ---
        tr_R = np.real(np.trace(R_nn, axis1=1, axis2=2))
        adaptive_load = beta * (tr_R / M)
        loading = np.maximum(adaptive_load, min_loading)
        
        # Apply stabilization loading
        R_nn_stable = R_nn + np.eye(M)[None, :, :] * loading[:, None, None]

        weights = np.zeros((K, M), dtype=np.complex128)

        # Solve the Generalized Eigenvalue Problem per frequency bin
        for f in range(K):
            # Ensure strict Hermiticity for numerical stability in eigh
            A = (R_xx[f] + R_xx[f].conj().T) / 2.0
            B = (R_nn_stable[f] + R_nn_stable[f].conj().T) / 2.0
            
            try:
                # eigh returns eigenvalues in ascending order; the last is the maximum
                evals, evecs = linalg.eigh(A, B)
                w_gev = evecs[:, -1]
            except linalg.LinAlgError:
                # Fallback to pure reference channel selection if decomposition fails
                w_gev = np.zeros(M, dtype=np.complex128)
                w_gev[0] = 1.0

            # --- Apply Blind Analytic Normalization (BAN) Post-filter ---
            # Numerator: sqrt(w^H * R_nn * R_nn * w / M)
            R_nn_sq = B @ B
            num_ban = np.sqrt(np.abs(w_gev.conj().T @ R_nn_sq @ w_gev) / M)
            
            # Denominator: w^H * R_nn * w
            den_ban = np.abs(w_gev.conj().T @ B @ w_gev) + 1e-10
            
            g_ban = num_ban / den_ban
            w_final = g_ban * w_gev
            
            # --- Phase Alignment ---
            # Align the arbitrary eigenvector phase to the target steering vector
            phase_alignment = np.exp(-1j * np.angle(w_final.conj() @ sv[f]))
            weights[f] = w_final * phase_alignment

        weights_rec[:, m, :] = weights
        
        # --- 5. Apply Filter ---
        Y_stft[:, m] = np.einsum("fm,fm->f", weights.conj(), X_frame)

    if save_weights:
        return Y_stft, Y_spp_stft, weights_rec
    else:
        return Y_stft, Y_spp_stft
    

def apply_gevd_stft_bridge(time_domain_input, vad_oracle, mic_coords, source_pos_2d, fs, length_fft=512, hop_length_fft=256):
    """
    Helper function to wrap the STFT -> GEVD -> ISTFT process.
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

    # Execute the Recursive GEVD and get both standard output and SPP masked output
    Y_stft, Y_spp_stft = SPP_GEVD_recursive(
        X_stft=X_stft, 
        fs=fs, 
        array_geometry=mic_coords, 
        source_pos=source_pos_2d, 
    )
    
    # Compute Inverse STFT for GEVD output
    _, y_time = signal.istft(
        Y_stft, 
        fs=fs, 
        nperseg=length_fft, 
        noverlap=length_fft - hop_length_fft
    )

    # Compute Inverse STFT for SPP masked output
    _, y_spp_time = signal.istft(
        Y_spp_stft, 
        fs=fs, 
        nperseg=length_fft, 
        noverlap=length_fft - hop_length_fft
    )
    
    # Truncate to original length
    original_length = time_domain_input.shape[1]
    return y_time[:original_length], y_spp_time[:original_length]


if __name__ == "__main__":
    # Basic simulation parameters
    FS = 16000
    M1, M2 = 8, 1          
    M = M1 * M2
    speed_of_sound = 343.0 

    iSIR_dB = 0
    
    print("=== INTEGRATION TEST: PIPELINE (FREE-FIELD, ROOM, WPE+ROOM) ===")
    
    output_folder = "tests/data/gevd_SPP_spatial_debug_output"
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
    ang_interf2 = np.deg2rad(-50)
    
    source_pos = array_center + np.array([r * np.cos(ang_target), r * np.sin(ang_target), 0.0])
    interf_pos1 = array_center + np.array([r * np.cos(ang_interf), r * np.sin(ang_interf), 0.0])
    interf_pos2 = array_center + np.array([r * np.cos(ang_interf2), r * np.sin(ang_interf2), 0.0])
    source_pos_2d = source_pos.reshape(1, 3)

    print(" -> Initializing 16kHz acoustic scene...")
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
        # Save to cache
        np.savez(cache_ff_path, input=free_field_input, vad=vad_oracle_ff)
        
    save_wav("1_FF_input_mix_mic0.wav", FS, free_field_input[0], output_folder)
    
    print(" -> Applying Recursive GEVD...")
    output_ff, output_spp_ff = apply_gevd_stft_bridge(free_field_input, vad_oracle_ff, mic_coords, source_pos_2d, FS)
    
    save_wav("2_FF_output_final.wav", FS, normalize_signal(output_ff), output_folder)
    save_wav("2_FF_output_SPP_mask.wav", FS, normalize_signal(output_spp_ff), output_folder)

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

    print(" -> Applying Recursive GEVD (Without WPE)...")
    output_rm, output_spp_rm = apply_gevd_stft_bridge(room_input, vad_oracle_room, mic_coords, source_pos_2d, FS)
    
    save_wav("4_ROOM_output_final.wav", FS, normalize_signal(output_rm), output_folder)
    save_wav("4_ROOM_output_SPP_mask.wav", FS, normalize_signal(output_spp_rm), output_folder)

    # -------------------------------------------------------------------
    # PHASE 3: WPE DEREVERBERATION + RECURSIVE GEVD
    # -------------------------------------------------------------------
    print("\n--- PHASE 3: WPE + GEVD PIPELINE ---")
    print(" -> Applying Online WPE Dereverberation on Room Simulation...")
    
    wpe_output = process_wpe_online(room_input, delay=2, stft_size=1024, stft_shift=128)
    
    save_wav("5_WPE_input_mix_mic0.wav", FS, wpe_output[0], output_folder)

    print(" -> Applying Recursive GEVD on Dereverberated Signals...")
    output_wpe, output_spp_wpe = apply_gevd_stft_bridge(wpe_output, vad_oracle_room, mic_coords, source_pos_2d, FS)
    
    save_wav("6_WPE_ROOM_output_final.wav", FS, normalize_signal(output_wpe), output_folder)
    save_wav("6_WPE_ROOM_output_SPP_mask.wav", FS, normalize_signal(output_spp_wpe), output_folder)

    print("\n -> Pipeline completed successfully.")