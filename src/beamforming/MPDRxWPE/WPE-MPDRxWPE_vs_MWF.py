import os
import numpy as np
import scipy.signal as sig
import scipy.linalg
from numba import njit, prange

# Import nara_wpe online utilities
try:
    from nara_wpe.online_wpe import OnlineWPE
except ImportError:
    OnlineWPE = None
    print("Warning: Ensure nara_wpe is installed and supports the OnlineWPE module.")

# Adjust imports based on your exact file structure
from propagation.simulate_acoustics import SimAcoustic
from utils.audio import save_wav

def get_gev_vector(sv, Phi_n):
    # Calculates the Generalized Eigenvalue Decomposition (GEVD) spatial filter weights.
    # sv shape: (F_bins, M_ch)
    # Phi_n shape: (F_bins, M_ch, M_ch)
    F_bins, M_ch = sv.shape
    w_gevd = np.zeros((F_bins, M_ch), dtype=np.complex128)
    
    for f in range(F_bins):
        # Regularize the noise covariance matrix to prevent singularity
        Rn = Phi_n[f] + np.eye(M_ch) * 1e-6
        
        try:
            # 1. Compute the Rank-1 spatial covariance matrix for the target
            # Outer product of the steering vector for frequency f
            Phi_s = np.outer(sv[f], sv[f].conj())

            # 2. Solve GEVD using scipy.linalg.eigh
            # This solves Phi_s * w = lambda * Rn * w
            eigenvalues, eigenvectors = scipy.linalg.eigh(Phi_s, Rn)
            
            # 3. Extract the eigenvector corresponding to the largest eigenvalue
            # eigh returns eigenvalues in ascending order, so we take the last one
            w_gevd[f, :] = eigenvectors[:, -1]
            
        except np.linalg.LinAlgError:
            # Fallback to the reference microphone if the matrix is completely unresolvable
            w_gevd[f, 0] = 1.0
            
    return w_gevd

def apply_beamformer(w, Y):
    # Applies spatial filter weights to the STFT observation matrix.
    # w shape: (F_bins, M_ch)
    # Y shape: (F_bins, M_ch, T_frames)
    Z_out = np.einsum('fm,fmt->ft', w.conj(), Y)
    return Z_out


def compute_rtf_steering_vector(f, Rs, mic_array, ref_mic_idx=0, c=343.0, mode="near_field", squeeze=True):
    # Computes the Relative Transfer Function (RTF) steering vector in the frequency domain.
    f = np.atleast_1d(f)
    Rs = np.atleast_2d(Rs)
    
    F = f.shape[0]
    P = Rs.shape[0]
    M = mic_array.shape[0]
    
    # Calculate Euclidean distance from each source point to each microphone
    mic_dist = np.linalg.norm(Rs[:, np.newaxis, :] - mic_array[np.newaxis, :, :], axis=2)
    
    # Extract the distance from each source to the designated fixed reference microphone
    ref_dist = mic_dist[:, ref_mic_idx, np.newaxis]
    
    # Calculate the path difference relative to the reference microphone
    delta_dist = mic_dist - ref_dist
    
    # Reshape arrays for correct NumPy broadcasting across frequencies (F), sources (P), and mics (M)
    f_bcast = f[:, np.newaxis, np.newaxis]
    delta_dist_bcast = delta_dist[np.newaxis, :, :]
    
    # Compute the relative phase delay
    phase_term = np.exp(-1j * 2 * np.pi * f_bcast * delta_dist_bcast / c)
    
    if mode == "near_field":
        # In near-field, amplitude decays with 1/r. 
        amp_ratio = ref_dist[np.newaxis, :, :] / mic_dist[np.newaxis, :, :]
        rtf_vector = amp_ratio * phase_term
    else:
        # In far-field, we assume plane waves where amplitude attenuation across the array is negligible
        rtf_vector = phase_term

    if squeeze:
        rtf_vector = np.squeeze(rtf_vector)

    return rtf_vector

@njit(parallel=True, fastmath=True)
def MPDRxWPE_numba(y_stft, sv, alpha=0.994, L=20, Delta=6, epsilon=1e-3):
    # Optimized MPDR-WPE bilinear framework using Numba.
    # Note default parameters updated for the second stage WPE.
    K, T, M = y_stft.shape
    epsilon_inv = 1.0 / epsilon
    
    # Pre-allocate the output array
    X_hat_out = np.zeros((K, T), dtype=np.complex128)
    
    buffer_len = (L + Delta) * M
    
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
            
    return X_hat_out

@njit(parallel=True, fastmath=True)
def online_mwf_numba(y_stft, sv, alpha=0.95, diag_load=1e-3):
    # Online Multichannel Wiener Filter (MWF) with Robust MVDR and Dynamic Noise Tracking
    K, T, M = y_stft.shape
    X_hat_out = np.zeros((K, T), dtype=np.complex128)
    
    for k in prange(K):
        # Initialize thread-local Spatial Covariance Matrix (SCM)
        R_yy = np.eye(M, dtype=np.complex128) * diag_load
        d = sv[k]
        
        # Initialize dynamic noise power tracker for this frequency bin
        noise_pow = diag_load
        
        for t in range(T):
            y = y_stft[k, t, :]
            
            # 1. Update Spatial Covariance Matrix recursively
            for i in range(M):
                for j in range(M):
                    R_yy[i, j] = alpha * R_yy[i, j] + (1.0 - alpha) * (y[i] * np.conj(y[j]))
            
            # 2. Apply Diagonal Loading for robustness against steering vector mismatch
            R_yy_reg = R_yy.copy()
            for i in range(M):
                R_yy_reg[i, i] += diag_load
                
            # 3. Invert the regularized SCM
            R_yy_inv = np.linalg.inv(R_yy_reg)
            
            # 4. Calculate Robust MVDR spatial weights
            num = np.dot(R_yy_inv, d)
            den = np.vdot(d, num)
            
            if np.abs(den) < 1e-12:
                w_mvdr = d / M
            else:
                w_mvdr = num / den
                
            # 5. Apply spatial filter to get the beamformed output
            y_out = np.vdot(w_mvdr, y)
            out_pow = np.abs(y_out)**2
            
            # 6. Track noise power using an asymmetric leaky integrator
            # Fast attack when power drops (noise floor), slow release when power rises (speech)
            if out_pow < noise_pow:
                noise_pow = 0.8 * noise_pow + 0.2 * out_pow
            else:
                noise_pow = 0.995 * noise_pow + 0.005 * out_pow
            
            # 7. Compute a priori SNR for the Wiener post-filter gain
            snr_prio = max(1e-6, (out_pow - noise_pow) / noise_pow)
            g_wiener = snr_prio / (snr_prio + 1.0)
            
            # 8. Apply MWF final gain
            X_hat_out[k, t] = g_wiener * y_out
            
    return X_hat_out

if __name__ == "__main__":
    FS = 16000
    M1, M2 = 12, 1          
    M = M1 * M2
    
    print("=== INTEGRATION TEST: HYBRID DECOUPLED MIMO-WPE + MPDR-WPE / ONLINE MWF ===")
    
    # We will save both outputs to a new comparison folder
    output_folder = os.path.join("tests", "data", "mwf_comparison_output")
    os.makedirs(output_folder, exist_ok=True)
    
    # 1. IDEAL GEOMETRY DEFINITION
    mic_spacing = 0.06
    x = np.linspace(0, (M1-1)*mic_spacing, M1)
    mic_coords_ideal = np.column_stack([x, np.zeros(M), np.zeros(M)])
    
    array_center = np.array([1.25, 2.0, 1.25])
    mic_coords_ideal = mic_coords_ideal - np.mean(mic_coords_ideal, axis=0) + array_center
    
    r = 1.0 
    ang_target = np.deg2rad(130)
    ang_interf = np.deg2rad(50)
    
    source_pos = array_center + np.array([r * np.cos(ang_target), r * np.sin(ang_target), 0.0])
    interf_pos1 = array_center + np.array([r * np.cos(ang_interf), r * np.sin(ang_interf), 0.0])

    # 2. ACOUSTIC SCENE WITH PHASE MISMATCH
    print(" -> Initializing acoustic scene with physical mismatch...")
    acoustic_scene = SimAcoustic(mic_coords_ideal, array_mismatch=0.005, duration=10, fs=FS)

    source_path = "tools/data/signals/FA01_09.wav"
    int_path1 = "tools/data/signals/MC15_03.wav"

    acoustic_scene.set_source(source_path, gain=1, position=source_pos.reshape(1,3))
    acoustic_scene.set_interference(int_path1, gain=1, position=interf_pos1.reshape(1,3))

    print(" -> Computing Room Impulse Responses (ISB)...")
    room_dimensions = np.array([4.0, 5.0, 2.5])
    
    room_input_real = acoustic_scene.compute_room_ISB(iSIR_dB=0, 
                                                    desire_RT=0.5,
                                                    room_dimensions=room_dimensions, 
                                                    mode="ideal")

    # 3. APPLY GAIN MISMATCH
    print(" -> Applying Gain Mismatch to microphones...")
    np.random.seed(42) 
    gain_mismatch = np.random.uniform(0.9, 1.1, (M, 1))
    room_input_mismatched = room_input_real * gain_mismatch
    
    # Save the input mixture using the correct custom function signature
    save_wav("1_input_mix_mic0_mismatched.wav", FS, room_input_mismatched[0], output_folder)

    # 4. SHORT-TIME FOURIER TRANSFORM
    nperseg = 1024
    noverlap = int(1024 * 0.75)
    nfft = 1024
    
    print(" -> Applying STFT...")
    freqs, times, X_stft = sig.stft(
        room_input_real, fs=FS, window='hamming', 
        nperseg=nperseg, noverlap=noverlap, nfft=nfft 
    )
    
    # 4.1 ONLINE MIMO WPE (NARA-WPE DECOUPLED)
    print(" -> Executing Online MIMO WPE (First Stage)...")
    
    # nara_wpe expects input shape (F, M, T)
    Y_wpe_in = np.transpose(X_stft, (1, 0, 2))
    F_bins, M_ch, T_frames = Y_wpe_in.shape
    
    taps_in = 4
    delay_in = 2
    
    # Initialize the output array for the first stage
    Z_wpe_out = np.zeros_like(Y_wpe_in)
    
    # Robustly initialize the Online WPE processor from the library
    if OnlineWPE is not None:
        wpe_processor = OnlineWPE(
            taps=taps_in,
            delay=delay_in,
            channels=M_ch,
            alpha=0.999 # Typical forgetting factor for nara_wpe
        )
        # Process frame by frame to simulate streaming
        for t in range(T_frames):
            Z_wpe_out[:, :, t] = wpe_processor.step(Y_wpe_in[:, :, t])
    else:
        print("OnlineWPE not available. Bypassing WPE stage.")
        Z_wpe_out = Y_wpe_in

    # Compute steering vector for both algorithms
    Rs_matrix = source_pos.reshape(1, 3)
    sv = compute_rtf_steering_vector(freqs, 
                                     Rs_matrix, 
                                     mic_coords_ideal, 
                                     ref_mic_idx=0, 
                                     mode="near_field", 
                                     squeeze=True)

    # Transpose back to (K, T, M) and make contiguous for Numba algorithms
    X_stft_processing = np.transpose(Z_wpe_out, (0, 2, 1))
    X_stft_processing = np.ascontiguousarray(X_stft_processing, dtype=np.complex128)

    # -------------------------------------------------------------
    # 5. BRANCH A: ORIGINAL MPDR-WPE OPTIMIZATION
    # -------------------------------------------------------------
    print(" -> Executing MPDR-WPE (Branch A)...")
    
    # Cascaded delay parameters: Output delay skips the input WPE window
    delta_out = delay_in + taps_in
    taps_out = 20

    X_hat_stft_mpdr = MPDRxWPE_numba(X_stft_processing, sv, L=taps_out, Delta=delta_out)
    
    print(" -> Reconstructing Branch A time-domain signal...")
    _, y_time_mpdr = sig.istft(
        X_hat_stft_mpdr, fs=FS, window='hamming', 
        nperseg=nperseg, noverlap=noverlap, nfft=nfft
    )
    
    # Save Branch A output
    save_wav("2_output_MIMOWPE_MPDR_WPE_ONLINE.wav", FS, y_time_mpdr, output_folder)

    # -------------------------------------------------------------
    # 6. BRANCH B: ONLINE MULTICHANNEL WIENER FILTER (MWF)
    # -------------------------------------------------------------
    print(" -> Executing Online MWF (Branch B)...")

    # The online_mwf_numba function processes shape (K, T, M) internally
    X_hat_stft_mwf = online_mwf_numba(X_stft_processing, sv, alpha=0.95)

    print(" -> Reconstructing Branch B time-domain signal...")

    # Reconstruct the single target source from MWF
    _, y_time_mwf = sig.istft(
        X_hat_stft_mwf, fs=FS, window='hamming', 
        nperseg=nperseg, noverlap=noverlap, nfft=nfft
    )

    # Save Branch B output
    save_wav("3_output_ONLINE_MWF.wav", FS, y_time_mwf, output_folder)
    print(" -> Processing complete.")