# Assuming the import works correctly in your local environment
from beamforming.signal_model import compute_rtf_steering_vector
import numpy as np 
import scipy.signal as signal
import matplotlib.pyplot as plt
from utils.audio import normalize_signal, save_wav
import os
from propagation.simulate_acoustics import SimAcoustic
from dereverberation.nara_wrappers import process_wpe_online

import numpy as np
import numpy as np
import numpy as np
import numpy as np

class OnlineSingleChannelSPP:
    def __init__(self, K, phi_s_init=None, noise_init=None, alpha=0.98, q=0.3):
        # Number of frequency bins
        self.K = K
        
        # Hyperparameters
        self.alpha = alpha  # Decision-directed smoothing factor
        self.q = q          # A priori probability of speech absence
        self.lamda_n = 0.99 # Noise PSD smoothing factor
        
        # Internal state initialization (Lazy initialization driven by data)
        self.is_initialized = False
        self.sigma_n2 = np.zeros(K, dtype=np.float64)
        self.xi_prev = np.zeros(K, dtype=np.float64)
        self.S_prev2 = np.zeros(K, dtype=np.float64)
        
        # Precompute constant factor for likelihood
        self.prior_factor = (1.0 - self.q) / self.q

    def step(self, X_ch0_frame):
        """
        Processes a single STFT frame of the reference microphone online.
        """
        # 1. Compute instantaneous power
        X_power = np.abs(X_ch0_frame)**2
        
        # Data-driven initialization on the very first frame
        if not self.is_initialized:
            # Seed the noise floor directly with the initial frame energy
            self.sigma_n2 = X_power.copy() + 1e-12
            self.S_prev2 = X_power.copy() + 1e-12
            self.xi_prev = np.ones(self.K, dtype=np.float64)
            self.is_initialized = True
            # Return a neutral SPP mask for the initial frame
            return np.ones(self.K, dtype=np.float64) * 0.5
            
        # 2. Compute a posteriori SNR (gamma) safely
        gamma = X_power / np.maximum(self.sigma_n2, 1e-12)
        gamma = np.minimum(gamma, 1e6)
        
        # 3. Compute a priori SNR (xi) using Decision-Directed approach
        xi = self.alpha * (self.S_prev2 / np.maximum(self.sigma_n2, 1e-12)) + \
             (1.0 - self.alpha) * np.maximum(gamma - 1.0, 0.0)
        xi = np.clip(xi, 1e-4, 1e3)
        
        # 4. Compute Generalized Likelihood Ratio (Lambda) safely
        nu = (gamma * xi) / (1.0 + xi)
        nu = np.minimum(nu, 700.0)
        
        Lambda = self.prior_factor * (1.0 / (1.0 + xi)) * np.exp(nu)
        
        # 5. Compute Speech Presence Probability (SPP)
        P = Lambda / (1.0 + Lambda)
        P = np.clip(P, 1e-4, 0.98)
        
        # 6. Update estimated speech power for the next frame's DD step
        G_wiener = xi / (1.0 + xi)
        self.S_prev2 = (G_wiener**2) * X_power
        self.xi_prev = xi
        
        # 7. CRITICAL FIX: Mathematically correct unbiased noise PSD update
        # Compute adaptive smoothing factor to ensure weights sum exactly to 1.0
        # When P -> 1 (speech), alpha_adaptive -> 1.0 (freezes noise update)
        # When P -> 0 (noise), alpha_adaptive -> lamda_n (updates noise normally)
        alpha_adaptive = self.lamda_n + (1.0 - self.lamda_n) * P
        
        self.sigma_n2 = alpha_adaptive * self.sigma_n2 + (1.0 - alpha_adaptive) * X_power
        self.sigma_n2 = np.maximum(self.sigma_n2, 1e-12)
                        
        return P
    

def SPP_MVDR_recursive(X_stft, fs, array_geometry, source_pos, beta=1e-3, min_loading=1e-6, save_weights=False):
    # Forgetting factor for covariance matrix smoothing
    lamda = 0.99
    K, T, M = X_stft.shape  
    frecs = np.linspace(0, fs/2, K)

    # Get steering vectors, expected shape (K, M)
    sv = compute_rtf_steering_vector(frecs, source_pos, array_geometry, ref_mic_idx=0, mode="far_field", squeeze=True)
    
    # Initialize output complex STFT matrices
    Y_stft = np.zeros((K, T), dtype=np.complex128)
    Y_spp_stft = np.zeros((K, T), dtype=np.complex128) # New matrix for SPP-filtered output
    
    # Initialize noise covariance matrix R_nn
    R_nn = np.tile(np.eye(M, dtype=np.complex128) * 1e-6, (K, 1, 1))
    
    # Initialize noisy signal covariance matrix R_xx using the steering vector
    # We assume an initial PSD (phi_s) for the source to seed the matrix
    phi_s = 1e-3 
    R_ss_init = phi_s * np.einsum("fm,fn->fmn", sv, sv.conj())
    R_xx = R_ss_init + R_nn


    # Initialize the inverse of R_nn for the first frame's SPP calculation
    initial_diag_load = np.tile(np.eye(M, dtype=np.complex128) * min_loading, (K, 1, 1))
    R_nn_inv = np.linalg.inv(R_nn + initial_diag_load)

    # SPP Hyperparameters (May need slight tuning for spectral energy)
    gamma_th = 6
    spp_slope = 6

    weights_rec = np.zeros((K, T, M), dtype=np.complex128)
# Antes del bucle 'for m in range(T):', inicializas el estimador:
    spp_estimator = OnlineSingleChannelSPP(K=K, phi_s_init=1e-3, noise_init=1e-6)
    
    for m in range(T):
        X_frame = X_stft[:, m, :]

        # --- 1 y 2. Estimación Estadística Moderna de SPP (Monocanal) ---
        # Usamos solo el micrófono de referencia (índice 0)
        P = spp_estimator.step(X_frame[:, 0])
        
        P_expand = P[:, np.newaxis, np.newaxis]
        Y_spp_stft[:, m] = P * X_frame[:, 0]

        # --- 3. Update Covariance Matrices (El resto sigue igual) ---
        R_instant = np.einsum("fm,fn->fmn", X_frame, X_frame.conj())

        # ... (Rest of your loop remains exactly the same) ...
        # Update R_xx weighted by the probability of target speech presence
        R_xx = lamda * R_xx + (1 - lamda) * P_expand * R_instant
        
        # Update R_nn weighted by the probability of target speech absence
        R_nn = lamda * R_nn + (1 - lamda) * (1 - P_expand) * R_instant

        # --- 4. Compute MVDR Weights with Dynamic Diagonal Loading ---
        # Calculate dynamic loading based on the noise covariance trace
        tr_R = np.real(np.trace(R_nn, axis1=1, axis2=2))
        adaptive_load = beta * (tr_R[:, None, None] / M)
        
        # Apply the minimum loading threshold
        loading = np.maximum(adaptive_load, min_loading)
        
        # Apply stabilization
        R_nn_stable = R_nn + np.eye(M)[None, :, :] * loading

        # Invert the stable covariance matrix safely
        R_nn_inv = np.linalg.inv(R_nn_stable)

        # Numerator: R_nn_inv * v
        weights_nom = np.einsum("fmn,fn->fm", R_nn_inv, sv)
        
        # Denominator: v^H * numerator
        weights_den = np.einsum("fm,fm->f", sv.conj(), weights_nom)
        
        # Final weights calculation
        weights = weights_nom / (weights_den[:, np.newaxis] + 1e-10)
        weights_rec[:, m, :] = weights
        
        # --- 5. Apply Filter ---
        Y_stft[:, m] = np.einsum("fm,fm->f", weights.conj(), X_frame)

    if save_weights:
        return Y_stft, Y_spp_stft, weights_rec
    else:
        return Y_stft, Y_spp_stft
    

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

    # Execute the Recursive MVDR and get both standard output and SPP masked output
    Y_stft, Y_spp_stft = SPP_MVDR_recursive(
        X_stft=X_stft, 
        fs=fs, 
        array_geometry=mic_coords, 
        source_pos=source_pos_2d, 
    )
    
    # Compute Inverse STFT for MVDR output
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
    
    output_folder = "tests/data/mvdr_SPP_spatial_debug_output"
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
    
    # ------------------# -------------------------------------------------------------------
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
    output_ff, output_spp_ff = apply_mvdr_stft_bridge(free_field_input, vad_oracle_ff, mic_coords, source_pos_2d, FS)
    
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

    print(" -> Applying Recursive MVDR (Without WPE)...")
    output_rm, output_spp_rm = apply_mvdr_stft_bridge(room_input, vad_oracle_room, mic_coords, source_pos_2d, FS)
    
    save_wav("4_ROOM_output_final.wav", FS, normalize_signal(output_rm), output_folder)
    save_wav("4_ROOM_output_SPP_mask.wav", FS, normalize_signal(output_spp_rm), output_folder)

    # -------------------------------------------------------------------
    # PHASE 3: WPE DEREVERBERATION + RECURSIVE MVDR
    # -------------------------------------------------------------------
    print("\n--- PHASE 3: WPE + MVDR PIPELINE ---")
    print(" -> Applying Online WPE Dereverberation on Room Simulation...")
    
    wpe_output = process_wpe_online(room_input, delay=2, stft_size = 1024 , stft_shift=128)
    
    save_wav("5_WPE_input_mix_mic0.wav", FS, wpe_output[0], output_folder)

    print(" -> Applying Recursive MVDR on Dereverberated Signals...")
    output_wpe, output_spp_wpe = apply_mvdr_stft_bridge(wpe_output, vad_oracle_room, mic_coords, source_pos_2d, FS)
    
    save_wav("6_WPE_ROOM_output_final.wav", FS, normalize_signal(output_wpe), output_folder)
    save_wav("6_WPE_ROOM_output_SPP_mask.wav", FS, normalize_signal(output_spp_wpe), output_folder)

    print("\n -> Pipeline completed successfully.")