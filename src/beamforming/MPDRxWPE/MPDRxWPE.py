import numpy as np
from beamforming.signal_model import compute_rtf_steering_vector
import scipy.signal as signal
from numba import njit
from numba import prange

    
@njit(parallel=True, fastmath=True)
def MPDRxWPE_numba(y_stft, sv, alpha=0.994, L=20, Delta=6, epsilon=1e-3, save_weights=False):
    # Optimized MPDR-WPE bilinear framework using Numba.
    # Note default parameters updated for the second stage WPE.
    K, T, M = y_stft.shape
    epsilon_inv = 1.0 / epsilon
    
    # Pre-allocate the output array
    X_hat_out = np.zeros((K, T), dtype=np.complex128)
    
    buffer_len = (L + Delta) * M

    h_register = np.zeros((K, T, M), dtype=np.complex128) if save_weights else np.zeros((1, 1, 1), dtype=np.complex128)

    
    # Outer loop parallelized over frequency bins
    for k in prange(K):
        
        # Thread-local filter initialization
        h = sv[k] / M
        g = np.zeros(L, dtype=np.complex128)
        
        # Thread-local covariance matrices
        R_h_inv = np.eye(L, dtype=np.complex128) * 1e-4
        R_g_inv = np.eye(M, dtype=np.complex128) * 1e-2
        I_M = np.eye(M, dtype=np.complex128)
        
        # Thread-local buffers
        y_buffer = np.zeros(buffer_len, dtype=np.complex128)
        y_bar = np.zeros(L * M, dtype=np.complex128)
        y_bar_g = np.zeros(M, dtype=np.complex128)
        y_bar_h = np.zeros(L, dtype=np.complex128)
        
        # Temporal loop (sequential per frequency bin)
        for l_idx in range(T):
            y_frame = y_stft[k, l_idx, :]

            if save_weights:
                h_register[k, l_idx, :] = h
            
            # 1. Update buffer
            for i in range(buffer_len - 1, M - 1, -1):
                y_buffer[i] = y_buffer[i - M]
            
            for m in range(M):
                y_buffer[m] = y_frame[m]
                
            for i in range(L * M):
                y_bar[i] = y_buffer[i + (Delta * M)]
                
            # 2. Compute y_bar_g
            for m in range(M):
                summ = 0.0 + 0.0j
                for l in range(L):
                    summ += np.conj(g[l]) * y_bar[l * M + m]
                y_bar_g[m] = y_frame[m] - summ
                
            # 3. Compute y_bar_h
            for l in range(L):
                summ = 0.0 + 0.0j
                for m in range(M):
                    summ += np.conj(h[m]) * y_bar[l * M + m]
                y_bar_h[l] = summ
                
            # 4. Compute A Priori Estimate
            X_hat = np.vdot(h, y_bar_g)
            lambda_l = np.abs(X_hat)**2
            X_hat_out[k, l_idx] = X_hat
            
            # 5. Compute Kalman gain for g
            num_g = np.dot(R_g_inv, y_bar_g)
            den_g = alpha + np.vdot(y_bar_g, num_g)
            k_g = num_g / den_g
            
            # 6. Compute Kalman gain for h
            num_h = np.dot(R_h_inv, y_bar_h)
            den_h = alpha * lambda_l + np.vdot(y_bar_h, num_h)
            k_h = num_h / den_h
            
            # 7 & 8. Update Covariances
            update_g = np.dot(np.outer(k_g, np.conj(y_bar_g)), R_g_inv)
            R_g_inv = (R_g_inv - update_g) / alpha
            
            update_h = np.dot(np.outer(k_h, np.conj(y_bar_h)), R_h_inv)
            R_h_inv = (R_h_inv - update_h) / alpha
            
            # 9. Update temporal filter g
            g = g + k_h * np.conj(X_hat)
            
            # 10. Update regularized covariance
            inner_inv = np.linalg.inv((epsilon_inv * I_M) + R_g_inv)
            R_sigma_inv = R_g_inv - np.dot(R_g_inv, np.dot(inner_inv, R_g_inv))
            
            # 11. Update spatial filter h
            d = sv[k]
            num_h_update = np.dot(R_sigma_inv, d)
            den_h_update = np.vdot(d, num_h_update)
            h = num_h_update / den_h_update
            
    return X_hat_out, h_register





@njit(parallel=True, fastmath=True)
def MPDRxWPE_numba_scaled(y_stft, sv, T_init, alpha_steady=0.994, alpha_init=0.90, tau=20.0, L=20, Delta=6, beta=1e-3, min_loading=1e-10, save_weights=False):
    # Optimized MPDR-WPE bilinear framework using Numba.
    # Features dynamic initialization, scaled diagonal loading, and exponential forgetting factor.
    
    K, T, M = y_stft.shape
    
    # Pre-allocate the output array
    X_hat_out = np.zeros((K, T), dtype=np.complex128)
    
    buffer_len = (L + Delta) * M
    h_register = np.zeros((K, T, M), dtype=np.complex128) if save_weights else np.zeros((1, 1, 1), dtype=np.complex128)
    
    # Outer loop parallelized over frequency bins
    for k in prange(K):
        
        # Calculate initial energy power over the first T_init frames for bin k
        p_k = 0.0
        for t in range(T_init):
            for m in range(M):
                p_k += np.abs(y_stft[k, t, m])**2
        
        # Average power per frame and microphone
        p_k = p_k / (T_init * M)

        # Floor value to prevent numerical explosion in silent bins
        p_k = max(p_k, min_loading) 
        
        # Thread-local filter initialization
        h = sv[k] / M
        g = np.zeros(L, dtype=np.complex128)
        
        # Thread-local covariance matrices inversely proportional to initial energy
        R_h_inv = np.eye(L, dtype=np.complex128) / p_k
        R_g_inv = np.eye(M, dtype=np.complex128) / p_k
        I_M = np.eye(M, dtype=np.complex128)
        
        # Track the trace of R_g directly. 
        # Since R_g(0) = p_k * I_M, the initial trace is M * p_k
        tr_R_g = M * p_k
        
        # Thread-local buffers
        y_buffer = np.zeros(buffer_len, dtype=np.complex128)
        y_bar = np.zeros(L * M, dtype=np.complex128)
        y_bar_g = np.zeros(M, dtype=np.complex128)
        y_bar_h = np.zeros(L, dtype=np.complex128)

        
        
        # Temporal loop (sequential per frequency bin)
        for l_idx in range(T):
            # Calculate time-varying exponential forgetting factor
            alpha = alpha_steady - (alpha_steady - alpha_init) * np.exp(-l_idx / tau)
            
            y_frame = y_stft[k, l_idx, :]

            if save_weights:
                h_register[k, l_idx, :] = h
            
            # 1. Update buffer
            for i in range(buffer_len - 1, M - 1, -1):
                y_buffer[i] = y_buffer[i - M]
            
            for m in range(M):
                y_buffer[m] = y_frame[m]
                
            for i in range(L * M):
                y_bar[i] = y_buffer[i + (Delta * M)]
                
            # 2. Compute y_bar_g
            y_bar_g_sq_norm = 0.0
            for m in range(M):
                summ = 0.0 + 0.0j
                for l in range(L):
                    summ += np.conj(g[l]) * y_bar[l * M + m]
                y_bar_g[m] = y_frame[m] - summ
                # Accumulate squared norm for trace tracking
                y_bar_g_sq_norm += np.abs(y_bar_g[m])**2
                
            # 3. Compute y_bar_h
            for l in range(L):
                summ = 0.0 + 0.0j
                for m in range(M):
                    summ += np.conj(h[m]) * y_bar[l * M + m]
                y_bar_h[l] = summ
                
            # 4. Compute A Priori Estimate
            X_hat = np.vdot(h, y_bar_g)
            lambda_l = np.abs(X_hat)**2
            X_hat_out[k, l_idx] = X_hat
            
            # 5. Compute Kalman gain for g
            num_g = np.dot(R_g_inv, y_bar_g)
            den_g = alpha + np.vdot(y_bar_g, num_g)
            k_g = num_g / den_g
            
            # 6. Compute Kalman gain for h
            num_h = np.dot(R_h_inv, y_bar_h)
            den_h = alpha * lambda_l + np.vdot(y_bar_h, num_h)
            k_h = num_h / den_h
            
            # 7 & 8. Update Covariances
            update_g = np.dot(np.outer(k_g, np.conj(y_bar_g)), R_g_inv)
            R_g_inv = (R_g_inv - update_g) / alpha
            
            update_h = np.dot(np.outer(k_h, np.conj(y_bar_h)), R_h_inv)
            R_h_inv = (R_h_inv - update_h) / alpha
            
            # Dynamic Trace Update
            tr_R_g = alpha * tr_R_g + y_bar_g_sq_norm
            
            # 9. Update temporal filter g
            g = g + k_h * np.conj(X_hat)
            
            # 10. Update regularized covariance with Scaled Diagonal Loading
            # epsilon is now a function of the matrix energy trace
            epsilon = beta * (tr_R_g / M)
            epsilon = max(epsilon, 1e-12) # Safeguard against total silence
            epsilon_inv = 1.0 / epsilon
            
            inner_inv = np.linalg.inv((epsilon_inv * I_M) + R_g_inv)
            R_sigma_inv = R_g_inv - np.dot(R_g_inv, np.dot(inner_inv, R_g_inv))
            
            # 11. Update spatial filter h
            d = sv[k]
            num_h_update = np.dot(R_sigma_inv, d)
            den_h_update = np.vdot(d, num_h_update)
            h = num_h_update / den_h_update
            
    return X_hat_out, h_register



# Import the new algorithm and the steering vector function
# from beamforming.MPDRxWPE import MPDRxWPE, compute_rtf_steering_vector 
import os
# Adjust imports based on your environment
from propagation.simulate_acoustics import SimAcoustic
from utils.audio import save_wav
from beamforming.signal_model import compute_rtf_steering_vector
from beamforming.MWF.WPE_SP_SDW_MWF import process_wpe_online

# Paste your MPDRxWPE_numba_scaled function here 
# (assuming it is imported or defined above this block in your real script)
# from beamforming.MPDRxWPE import MPDRxWPE_numba_scaled

def apply_mpdr_wpe_stft_bridge(time_domain_input, mic_coords, source_pos_2d, fs, 
                               nperseg=1024, noverlap=768, T_init=20):
    """
    Helper function to wrap the STFT -> MPDRxWPE -> ISTFT process.
    Uses longer windows (1024) and 75% overlap as preferred by the joint algorithm.
    """
    # Compute STFT: output shape (Mics, Freqs, Frames)
    freqs, times, Zxx = signal.stft(
        time_domain_input, 
        fs=fs, 
        window='hamming',
        nperseg=nperseg, 
        noverlap=noverlap,
        nfft=nperseg
    )
    
    # Transpose Zxx from (M, K, T) to (K, T, M)
    X_stft = np.transpose(Zxx, (1, 2, 0))
    # Numba parallel loops strongly benefit from contiguous arrays in memory
    X_stft = np.ascontiguousarray(X_stft, dtype=np.complex128)
    
    # Compute the Steering Vector
    sv = compute_rtf_steering_vector(
        freqs, 
        source_pos_2d, 
        mic_coords, 
        ref_mic_idx=0, 
        mode="near_field", 
        squeeze=True
    )
    
    # Ensure T_init does not exceed the total number of available frames
    actual_T_init = min(T_init, X_stft.shape[1])
    
    # Execute the Numba optimized Joint MPDR-WPE
    X_hat_stft, _ = MPDRxWPE_numba_scaled(
        y_stft=X_stft, 
        sv=sv, 
        T_init=actual_T_init
    )
    
    # Compute Inverse STFT
    _, y_time = signal.istft(
        X_hat_stft, 
        fs=fs, 
        window='hamming',
        nperseg=nperseg, 
        noverlap=noverlap,
        nfft=nperseg
    )
    
    # Truncate to original length to ensure consistency
    original_length = time_domain_input.shape[1]
    return y_time[:original_length]

def normalize_signal(sig):
    """Normalizes the audio array to avoid clipping."""
    max_abs = np.max(np.abs(sig))
    if max_abs > 0:
        return sig * (0.99 / max_abs)
    return sig


if __name__ == "__main__":
    # Basic simulation parameters identical to base.py
    FS = 16000
    M1, M2 = 12, 1          
    M = M1 * M2
    speed_of_sound = 343.0 

    iSIR_dB = 0
    
    print(f"=== INTEGRATION TEST: MPDRxWPE SCALED PIPELINE (M={M}) ===")
    
    # Dedicated output folder for this algorithm
    output_folder = "tests/data/mpdr_wpe_scaled_output"
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
    else:
        print("\n--- PHASE 1: COMPUTING FREE FIELD SIMULATION ---")
        # Note: MPDR doesn't explicitly need an oracle VAD like MVDR does, 
        # so we extract the signals directly.
        free_field_input, _ = acoustic_scene.free_field(iSIR_dB=iSIR_dB, normalize=True, mode="ideal", VAD=True)
        np.savez(cache_ff_path, input=free_field_input)
        
    save_wav("1_FF_input_mix_mic0.wav", FS, free_field_input[0], output_folder)
    
    print(" -> Applying Scaled MPDRxWPE...")
    output_ff = apply_mpdr_wpe_stft_bridge(free_field_input, mic_coords, source_pos_2d, FS)
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
        np.savez(cache_room_path, input=room_input)
    
    save_wav("3_ROOM_input_mix_mic0.wav", FS, room_input[0], output_folder)

    print(" -> Applying Scaled MPDRxWPE on reverberant scene...")
    # Expect Numba compilation overhead on the very first run if not cached
    output_rm = apply_mpdr_wpe_stft_bridge(room_input, mic_coords, source_pos_2d, FS)
    save_wav("4_ROOM_output_final.wav", FS, normalize_signal(output_rm), output_folder)

    # -------------------------------------------------------------------
    # PHASE 3: EXTERNAL WPE + MPDRxWPE (Comparison Phase)
    # -------------------------------------------------------------------
    print("\n--- PHASE 3: EXTERNAL WPE + JOINT MPDRxWPE PIPELINE ---")
    print(" -> Applying Online WPE Dereverberation on Room Simulation...")
    
    wpe_output = process_wpe_online(room_input)
    save_wav("5_WPE_input_mix_mic0.wav", FS, wpe_output[0], output_folder)

    print(" -> Applying Scaled MPDRxWPE on externally dereverberated signals...")
    output_wpe = apply_mpdr_wpe_stft_bridge(wpe_output, mic_coords, source_pos_2d, FS)
    save_wav("6_WPE_ROOM_output_final.wav", FS, normalize_signal(output_wpe), output_folder)

    print("\n -> Pipeline completed successfully.")


    