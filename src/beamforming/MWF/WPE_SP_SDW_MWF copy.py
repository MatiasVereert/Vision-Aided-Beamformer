# This model is based on the paper titled: 
# "Frequency-domain criterion for the speech distortion weighted multichannel Wiener filter for robust noise reduction"
# doi: 10.1016/j.specom.2007.02.001
# by Simon Doclo *, Ann Spriet, Jan Wouters, Marc Moonen

import numpy as np
from numba import njit
# Local 
from beamforming.signal_model import steering_vector, compute_rtf_steering_vector
from scipy.spatial.distance import pdist

from nara_wpe.wpe import OnlineWPE
from nara_wpe.utils import stft, istft
from nara_wpe.wpe import online_wpe_step, get_power_online, OnlineWPE

from nara_wpe.wpe import wpe # Importamos la versión Batch/Offline
from nara_wpe.utils import stft, istft

def process_wpe_offline(u, taps=10, delay=2, iterations=3, stft_size=256, stft_shift=64):
    """
    Offline (Batch) WPE wrapper.
    Processes the entire signal at once to compute a fixed optimal dereverberation filter.
    This guarantees a stationary output, avoiding conflicts with downstream adaptive filters.
    """
    # 1. Transform to STFT domain
    # nara_wpe stft returns (channels, frames, frequency_bins)
    Y = stft(u, size=stft_size, shift=stft_shift)
    
    # nara_wpe offline expects input shape: (frequency_bins, channels, frames)
    Y_wpe = Y.transpose(2, 0, 1)
    
    # 2. Apply Batch WPE (typically 3 to 5 iterations is enough)
    Z_wpe = wpe(Y_wpe, taps=taps, delay=delay, iterations=iterations)
    
    # 3. Reconstruct the time-domain signal
    # Transpose back to (channels, frames, frequency_bins)
    Z_out = Z_wpe.transpose(1, 2, 0)
    
    z_time = istft(Z_out, size=stft_size, shift=stft_shift)
    
    # Ensure output length matches input
    z_time = z_time[:, :u.shape[1]]
    
    return z_time

def process_wpe_online(u, taps=5, delay=1, alpha=0.9999, stft_size=256, stft_shift=64):
    """
    Online WPE wrapper (Functional Approach).
    Processes a multichannel time-domain signal frame by frame to simulate 
    online dereverberation. Bypasses the buggy OnlineWPE class state management 
    by handling the Q and G matrices directly.
    """
    # 1. Transform to STFT domain
    Y = stft(u, size=stft_size, shift=stft_shift)
    Y = Y.transpose(1, 2, 0)  # Shape: (frames, bins, channels)
    T, F, M = Y.shape
    
    buffer_target_size = taps + delay + 1
    if T < buffer_target_size:
        print("Warning: Signal is too short for WPE with given taps and delay.")
        return u
        
    # 2. Initialize Q (Inverse Correlation) and G (Filter) matrices manually
    # Q shape: (F, M*taps, M*taps) -> Identity matrices
    Q = np.stack([np.identity(M * taps) for _ in range(F)])
    # G shape: (F, M*taps, M) -> Zeros
    G = np.zeros((F, M * taps, M))
    
    Z_list = []
    
    # 3. Bypass the first unprocessed frames to maintain strict temporal alignment
    for i in range(taps + delay):
        Z_list.append(Y[i, :, :])
        
    # Initialize the sliding buffer with the first history chunk
    buffer = list(Y[:taps + delay, :, :])
    
    # 4. Process frame by frame
    for t in range(taps + delay, T):
        buffer.append(Y[t, :, :])
        
        # Convert buffer to numpy array: shape (buffer_target_size, F, M)
        Y_step = np.array(buffer)
        
        # Compute power. get_power_online expects (bins, channels, frames)
        power = get_power_online(Y_step.transpose(1, 2, 0))
        
        # Perform functional online dereverberation step
        Z_frame, Q, G = online_wpe_step(
            Y_step,
            power,
            Q,
            G,
            alpha=alpha,
            taps=taps,
            delay=delay
        )
        
        Z_list.append(Z_frame)
        
        # Discard the oldest frame to slide the window forward
        buffer.pop(0)
            
    # 5. Reconstruct the time-domain signal
    Z_stacked = np.stack(Z_list)
    
    # Transpose back to (channels, frames, frequency_bins) for istft
    Z_out = Z_stacked.transpose(2, 0, 1)
    
    # Inverse STFT to get the time-domain audio
    z_time = istft(Z_out, size=stft_size, shift=stft_shift)
    
    # Ensure the output length exactly matches the original input length 
    z_time = z_time[:, :u.shape[1]]
    
    return z_time


def compute_equivalent_weights(w_fixed, C_block, w_adaptive, L, frecs, fs):
    """
    Computes the equivalent global weights of the SP-SDW-MWF system 
    combining the fixed branch, blocking matrix, and adaptive weights.
    """
    # Number of frequency bins and microphones
    F, M = w_fixed.shape
    delay = L // 2
    
    # Calculate the delay phase shift for the fixed branch
    d_delta = np.exp(-1j * 2 * np.pi * frecs * (delay / fs))
    
    # Build the transformation matrix B(f) of shape (F, M, M)
    w_fixed_expanded = w_fixed[:, :, np.newaxis]
    B = np.concatenate((w_fixed_expanded, C_block), axis=2)
    
    # Multiply B(f) with w_adaptive(f) -> shape (F, M)
    # Using einsum: 'fmn, fn -> fm'
    adaptive_part = np.einsum('fmn,fn->fm', B, w_adaptive)
    
    # Calculate the equivalent global weights
    w_eq = d_delta[:, np.newaxis] * w_fixed - adaptive_part
    
    return w_eq


def get_fixed_weights(w_fixed_raw, L, frecs, fs, d_max):
    # Calculate the physical maximum delay in seconds
    max_delay_sec = d_max / 343.0
    
    # Convert to samples and add a small safety margin for the sinc tail
    alignment_samples = int(np.ceil(max_delay_sec * fs)) + 5 
    
    # Prevent severe truncation if L is physically too small for the array
    if alignment_samples >= L // 2:
        print(f"WARNING: alignment_samples ({alignment_samples}) is too large for L={L}.")
        
    alignment_phase = np.exp(-1j * 2 * np.pi * frecs * (alignment_samples / fs))
    
    # Broadcast phase across all M microphones
    w_fixed_aligned = w_fixed_raw * alignment_phase[:, None] 
    
    # Force DC to real
    w_fixed_aligned[0, :] = w_fixed_aligned[0, :].real + 0.0j
    
    # Force Nyquist to real
    # Force Nyquist to real without artificially zeroing it
    w_fixed_aligned[-1, :] = w_fixed_aligned[-1, :].real + 0.0j
    
    # Transform to time domain to truncate
    w_time = np.fft.irfft(w_fixed_aligned, n=2 * L, axis=0) 
    
    # Truncate to length L and pad with L zeros (Strict Overlap-Save rule)
    w_time_padded = np.zeros_like(w_time)
    w_time_padded[:L, :] = w_time[:L, :] 
    
    # Transform back to frequency domain
    w_fixed = np.fft.rfft(w_time_padded, axis=0)

    return w_fixed
import scipy.linalg

def get_blocking_matrix(w_fixed_freq, L):
    """
    Calculates an Orthogonal Blocking Matrix using the null space (SVD).
    This drastically improves White Noise Gain (WNG) and prevents spectral coloration.
    """
    F, M = w_fixed_freq.shape
    
    # 1. Calculate the purely spatial Orthogonal Null Space of the all-ones vector.
    # This matrix is strictly real, frequency-independent, and shape is (M, M-1).
    # Its columns are orthonormal: C^T * C = I
    C_ortho = scipy.linalg.null_space(np.ones((1, M)))
    
    # Initialize the raw frequency-domain Blocking Matrix
    Ca_raw = np.zeros((F, M, M - 1), dtype=complex)
    
    # 2. Combine the temporal alignment with the orthogonal spatial projection
    for n in range(M - 1):
        for m in range(M):
            # Multiply the causal alignment phase by the spatial orthogonal weight.
            # We multiply by M to undo the 1/M normalization specifically for the 
            # noise references, ensuring the white noise power remains exactly 1x.
            Ca_raw[:, m, n] = (w_fixed_freq[:, m] * M) * C_ortho[m, n]
            
    # 3. Apply the Strict Overlap-Save time constraint
    Ca_time = np.fft.irfft(Ca_raw, n=2 * L, axis=0)
    Ca_time_padded = np.zeros_like(Ca_time)
    
    # Preserve the causal bulk-delayed impulse
    Ca_time_padded[:L, :, :] = Ca_time[:L, :, :]
    
    # Transform back to the frequency domain for the main processing loop
    Ca_constrained = np.fft.rfft(Ca_time_padded, n=2 * L, axis=0)
    
    return Ca_constrained

def regularize_covariance_matrix(Q_x):
    """
    Applies eigenvalue regularization to a batch of Hermitian covariance matrices.
    Ensures that all matrices in the batch are positive semi-definite by 
    subtracting any negative minimum eigenvalue from the diagonal.
    
    Args:
        Q_x: Array of shape (F, M, M) containing F covariance matrices.
        
    Returns:
        Q_x_reg: Array of shape (F, M, M) with regularized matrices.
    """
    F, M, _ = Q_x.shape
    
    # 1. Calculate real eigenvalues for all frequency bins simultaneously.
    # np.linalg.eigvalsh is optimized for Hermitian/symmetric matrices.
    eig_vals = np.linalg.eigvalsh(Q_x)
    
    # 2. Find the minimum eigenvalue for each frequency bin. Shape: (F,)
    min_eig_vals = np.min(eig_vals, axis=1)
    
    # 3. Isolate the negative minimum eigenvalues. 
    # If the minimum is positive, it becomes 0.0.
    neg_min_eig_vals = np.minimum(min_eig_vals, 0.0)
    
    # 4. Subtract the negative minimum from the diagonal of Q_x.
    # Reshape to (F, 1, 1) for broadcasting with the (M, M) identity matrix.
    Q_x_reg = Q_x - (neg_min_eig_vals[:, np.newaxis, np.newaxis] * np.eye(M))
    
    return Q_x_reg

import scipy.linalg

def sdw_mwf(u, vad, array_pos, source_pos, fs, constrained=True, ouput_weights=False):
    """
    Computes the SP-SDW-MWF with dynamic RTF tracking and spatial regularization.
    u shape: (M, N_samples)
    """

    # Define constants
    Lambda = .95 
    mu = 8 
    diag_load = 1e-3
    rho = 2
    
    # NEW PARAMETER: Confidence in the statistical RTF vs Geometric Steering Vector
    # 0.0 = Pure Geometric GSC (no tracking)
    # 1.0 = Pure Blind RTF GSC (prone to drift)
    # 0.85 = Robust Hybrid
    alpha_rtf = 0.85 

    L = 128
    M = u.shape[0]
    mu_inv = 1 / mu

    # Max geo distance 
    max_dist = np.max(pdist(array_pos))

    # Define frequency bins vector 
    F = L + 1
    frecs = np.linspace(0, fs / 2, F)

    # Correct padding syntax
    u_padded = np.pad(u, ((0, 0), (L, 0)), mode='constant')
    
    # Calculate total frames
    tot_frames = (u_padded.shape[1] - L) // L

    # Initialize output array
    z = np.zeros(tot_frames * L, dtype=np.float64)
    z_fixed_branch = np.zeros(tot_frames * L, dtype=np.float64)
    z_noise = np.zeros(tot_frames * L, dtype=np.float64)
    post_block = np.zeros(tot_frames * L, dtype=np.float64)

    # Initial geometric steering vector (d)
    sv = compute_rtf_steering_vector(frecs, source_pos, array_pos, ref_mic_idx=0, mode="near_field", squeeze=True)
    
    # Pre-calculate the normalized geometric vector for later interpolation
    norm_sv = sv / (np.linalg.norm(sv, axis=1, keepdims=True) + 1e-12)

    # Initialize fixed weights and blocking matrix using geometric DOA
    w_fixed_raw = sv.conj() / M
    w_fixed = get_fixed_weights(w_fixed_raw, L, frecs, fs, d_max=max_dist)
    C_block = get_blocking_matrix(w_fixed, L)

    # Initialize adaptive variables
    w_adaptive = np.zeros((F, M), dtype=np.complex128)

    history_buffer = np.zeros(L, dtype=np.float64)
    y_buffer = np.zeros((L, M), dtype=np.float64) 
    delay = L // 2

    # Initialize Correlation Matrices with small diagonal loading for initial eigenvalue stability
    Q_instant = np.zeros((F, M, M), dtype=complex)
    initial_diag = np.tile(np.eye(M, dtype=complex) * 1e-6, (F, 1, 1))
    Q_y = initial_diag.copy()
    Q_v = initial_diag.copy()
    Q_x = np.zeros((F, M, M), dtype=complex)
    
    weights_rec = np.zeros((tot_frames, F, M), dtype=np.complex128)
    
    for m in range(tot_frames):
        # --- PROCESS FRAMES ---
        u_frame = u_padded[:, m * L : m * L + 2 * L]
        U_fram_rfft = np.fft.rfft(u_frame, axis=1)

        vad_frame = vad[m * L : m * L + L]
        vad_status = np.mean(vad_frame) > 0.1

        # --- PROCESS FIXED BRANCH (y_0) ---
        # Uses the dynamically updated w_fixed from the previous frame
        D = np.einsum('fm, mf->f', w_fixed, U_fram_rfft)
        d_time_2L = np.fft.irfft(D, n=2 * L)
        d_valid = d_time_2L[L:] 
        
        # --- APPLY DELAY FIXED BRANCH ---
        d_combined = np.concatenate((history_buffer, d_valid))
        start_idx = L - delay
        end_idx = 2 * L - delay
        d_delayed_valid = d_combined[start_idx : end_idx]
        history_buffer = d_valid.copy()
        z_fixed_branch[m * L : m * L + L] = d_valid

        # --- PROCESS BLOCKING MATRIX (y_1 to y_M-1) ---
        # Uses the dynamically updated C_block from the previous frame
        D_y_raw = np.einsum('fmn, mf->fn', C_block, U_fram_rfft)
        D_y_time = np.fft.irfft(D_y_raw, n=2 * L, axis=0) 
        D_y_valid = D_y_time[L:, :] 
        
        post_block[m * L : m * L + L] = D_y_valid[:, 0] 

        # --- PREPARE CLEAN OVERLAP-SAVE FRAME FOR ADAPTIVE BRANCH ---
        y_valid_time = np.column_stack((d_valid, D_y_valid)) 
        y_frame_2L = np.vstack((y_buffer, y_valid_time)) 
        y_buffer = y_valid_time.copy() 
        D_y = np.fft.rfft(y_frame_2L, axis=0)

        # --- PROCESS ADAPTIVE BRANCH ---
        noise = np.einsum('fm,fm->f', D_y, w_adaptive)
        noise_2L = np.fft.irfft(noise, n=2 * L)
        noise_valid = noise_2L[L:] 

        z_noise[m * L : m * L + L] = noise_valid

        # --- COMPUTE OUTPUT (in time domain) ---
        e = d_delayed_valid - noise_valid
        z[m * L : m * L + L] = e

        e_2L = np.concatenate((np.zeros(L, dtype=np.float64), e))
        e_undeline = np.fft.rfft(e_2L)
        
        # --- UPDATE CORRELATION MATRICES ---
        Q_instant = np.einsum('fm,fn->fmn', D_y.conj(), D_y)

        if vad_status: 
            Q_y = Lambda * Q_y + (1 - Lambda) * Q_instant / 2
        else:
            Q_v = Lambda * Q_v + (1 - Lambda) * Q_instant / 2
            
        # Recompute speech covariance every frame to ensure RTF tracks smoothly
        Q_x = Q_y - Q_v
        Q_x_reg = regularize_covariance_matrix(Q_x)

        # =====================================================================
        # --- DYNAMIC RTF TRACKING & REGULARIZATION (NEW BLOCK) ---
        # =====================================================================
        
        # 1. Extract raw RTF as the principal eigenvector of Q_x_reg
        # eigh returns eigenvalues in ascending order, last column is the principal eigenvector
        evals, evecs = np.linalg.eigh(Q_x_reg)
        h_raw = evecs[:, :, -1] 
        
        # 2. Phase alignment to reference microphone (index 0)
        phase_align = np.exp(-1j * np.angle(h_raw[:, 0]))
        h_raw = h_raw * phase_align[:, np.newaxis]
        
        # 3. Normalize the extracted RTF
        norm_h = np.linalg.norm(h_raw, axis=1, keepdims=True) + 1e-12
        h_raw_norm = h_raw / norm_h
        
        # 4. Interpolate with the geometric steering vector (Spatial Regularization)
        # This prevents acoustic drift while allowing reflection tracking
        h_updated = alpha_rtf * h_raw_norm + (1.0 - alpha_rtf) * norm_sv
        
        # 5. Re-normalize the final interpolated vector
        h_updated = h_updated / (np.linalg.norm(h_updated, axis=1, keepdims=True) + 1e-12)
        
        # 6. Update the Fixed Branch and Blocking Matrix for the NEXT frame
        w_fixed_raw = h_updated.conj() / M
        w_fixed = get_fixed_weights(w_fixed_raw, L, frecs, fs, d_max=max_dist)
        C_block = get_blocking_matrix(w_fixed, L)
        
        # =====================================================================

        # --- UPDATE ADAPTIVE WEIGHTS ---
        if not vad_status:
            Q_mix = Q_v + mu_inv * Q_x_reg 
            Q_mix = Q_mix + np.eye(M) * diag_load

            r_2NL = mu_inv * np.einsum('fmn, fn -> fm', Q_x, w_adaptive)
            gradient = np.einsum('fm, f -> fm', D_y.conj(), e_undeline) - r_2NL
            gradient_update = np.linalg.solve(Q_mix, gradient)

            w_adaptive_unconstrained = w_adaptive + rho * (1 - Lambda) * 0.5 * gradient_update

            if constrained:
                w_2L = np.fft.irfft(w_adaptive_unconstrained, axis=0) 
                w_2L[L:, :] = 0
                w_adaptive = np.fft.rfft(w_2L, n=2 * L, axis=0)
            else:
                w_adaptive = w_adaptive_unconstrained

            if ouput_weights:
                weights_rec[m, :, :] = compute_equivalent_weights(w_fixed, C_block, w_adaptive, L, frecs, fs)           
                
    if ouput_weights:
        return z_fixed_branch, post_block, z_noise, z, weights_rec
    else:
        return z_fixed_branch, post_block, z_noise, z



# Normalize signals to range [-0.99, 0.99] to prevent clipping when saving as WAV
def normalize_signal(sig):
    max_abs = np.max(np.abs(sig))
    if max_abs > 0:
        return sig * (0.99 / max_abs)
    return sig
    
import os
from propagation.simulate_acoustics import SimAcoustic
from utils.audio import save_wav
# Import the fixed branch function we built

if __name__ == "__main__":
    # Basic simulation parameters
    FS = 16000
    M1, M2 = 12, 1          
    M = M1 * M2
    speed_of_sound = 343.0 

    iSIR_dB = 0
    
    print("=== INTEGRATION TEST: PIPELINE (FREE-FIELD, ROOM, WPE+ROOM) ===")
    
    output_folder = "tests/data/sdw_mwf_output"
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
    
    print(" -> Applying SDW-MWF...")
    z_fixed_ff, post_block_ff, z_noise_ff, output_ff, _ = sdw_mwf(
        free_field_input, vad_oracle_ff, mic_coords, source_pos_2d, FS, ouput_weights=True
    )
    
    save_wav("2_FF_output_fixed.wav", FS, normalize_signal(z_fixed_ff), output_folder)
    save_wav("2_FF_output_noise.wav", FS, normalize_signal(z_noise_ff), output_folder)
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

    print(" -> Applying SDW-MWF (Without WPE)...")
    z_fixed_rm, post_block_rm, z_noise_rm, output_rm = sdw_mwf(
        room_input, vad_oracle_room, mic_coords, source_pos_2d, FS
    )
    
    save_wav("4_ROOM_output_fixed.wav", FS, normalize_signal(z_fixed_rm), output_folder)
    save_wav("4_ROOM_output_noise.wav", FS, normalize_signal(z_noise_rm), output_folder)
    save_wav("4_ROOM_output_final.wav", FS, normalize_signal(output_rm), output_folder)

    # -------------------------------------------------------------------
    # PHASE 3: WPE DEREVERBERATION + SDW-MWF
    # -------------------------------------------------------------------
    print("\n--- PHASE 3: WPE + SDW-MWF PIPELINE ---")
    print(" -> Applying Online WPE Dereverberation on Room Simulation...")
    
    wpe_output = process_wpe_online(room_input)
    
    save_wav("5_WPE_input_mix_mic0.wav", FS, wpe_output[0], output_folder)

    print(" -> Applying SDW-MWF on Dereverberated Signals...")
    z_fixed_wpe, post_block_wpe, z_noise_wpe, output_wpe = sdw_mwf(
        wpe_output, vad_oracle_room, mic_coords, source_pos_2d, FS
    )
    
    save_wav("6_WPE_ROOM_output_fixed.wav", FS, normalize_signal(z_fixed_wpe), output_folder)
    save_wav("6_WPE_ROOM_output_noise.wav", FS, normalize_signal(z_noise_wpe), output_folder)
    save_wav("6_WPE_ROOM_output_final.wav", FS, normalize_signal(output_wpe), output_folder)

    print("\n -> Pipeline completed successfully.")