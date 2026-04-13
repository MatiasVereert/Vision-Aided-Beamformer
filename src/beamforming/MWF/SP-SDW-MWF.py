# This model is based on the paper titled: 
# "Frequency-domain criterion for the speech distortion weighted multichannel Wiener filter for robust noise reduction"
# doi: 10.1016/j.specom.2007.02.001
# by Simon Doclo *, Ann Spriet, Jan Wouters, Marc Moonen

import numpy as np
from numba import njit
# Local 
from beamforming.signal_model import steering_vector, compute_rtf_steering_vector
from scipy.spatial.distance import pdist

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
    w_fixed_aligned[-1, :] = 0.0 + 0.0j
    
    # Transform to time domain to truncate
    w_time = np.fft.irfft(w_fixed_aligned, n=2 * L, axis=0) 
    
    # Truncate to length L and pad with L zeros (Strict Overlap-Save rule)
    w_time_padded = np.zeros_like(w_time)
    w_time_padded[:L, :] = w_time[:L, :] 
    
    # Transform back to frequency domain
    w_fixed = np.fft.rfft(w_time_padded, axis=0)

    return w_fixed

def get_blocking_matrix(w_fixed_freq, L):
    """
    Calculates a deterministic pairwise-subtraction Blocking Matrix (Griffiths-Jim style).
    Uses the causal fixed weights directly without conjugation to prevent destroying 
    the impulse response during the time-domain truncation.
    """
    F, M = w_fixed_freq.shape
    
    # Initialize the raw frequency-domain Blocking Matrix
    Ca_raw = np.zeros((F, M, M - 1), dtype=complex)
    
    # Populate the matrix to perform pairwise subtraction of ALIGNED signals.
    # CRITICAL FIX: Removed .conj() to maintain strict causality (bulk delay preserved)
    for n in range(M - 1):
        Ca_raw[:, n, n] = w_fixed_freq[:, n]
        Ca_raw[:, n + 1, n] = -w_fixed_freq[:, n + 1]
        
    # Apply the Overlap-Save time constraint
    Ca_time = np.fft.irfft(Ca_raw, n=2 * L, axis=0)
    Ca_time_padded = np.zeros_like(Ca_time)
    
    # This truncation now correctly captures the causal bulk-delayed impulse
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


def sdw_mwf(u, vad, target_pos, source_pos, fs):
    """
    Computes the fixed branch (speech reference) of the SP-SDW-MWF.
    u shape: (M, N_samples)
    """

    # Define constants
    Lambda = .9 # interval (0,1)
    mu = .5 #[0, 1] provides a trade-off between noise reduction and speech distortio
    diag_load = 10
    rho = 4
    

    L = 1024
    M = u.shape[0]
    mu_inv = 1 / mu

    # Max geo distance 
    max_dist = np.max(pdist(mic_coords))

    # Define frequency bins vector 
    F = L + 1
    frecs = np.linspace(0, fs / 2, F)

    # Correct padding syntax for 2D array: pad axis 1 (time) with L zeros at the beginning
    u_padded = np.pad(u, ((0, 0), (L, 0)), mode='constant')
    
    # Calculate total frames
    tot_frames = (u_padded.shape[1] - L) // L

    # Initialize output array
    z = np.zeros(tot_frames * L, dtype=np.float64)
    z_fixed_branch = np.zeros(tot_frames * L, dtype=np.float64)
    z_noise = np.zeros(tot_frames * L, dtype=np.float64)
    post_block = np.zeros(tot_frames * L, dtype=np.float64)

    # This aligns the phases relative to mic 0, keeping delays strictly local
    sv = compute_rtf_steering_vector(frecs, source_pos, target_pos, ref_mic_idx=0, mode="near_field", squeeze=True)
    
    # Normalize by M to prevent amplitude clipping
    w_fixed_raw= sv.conj() / M

    w_fixed = get_fixed_weights(w_fixed_raw, L, frecs, fs, d_max = max_dist)
    C_block = get_blocking_matrix(w_fixed, L    ) #Shape is (F, M, M-1)

    # Inicalice adaptive variables
    w_adaptive = np.zeros( (F, M ), dtype = np.complex128)

    print(np.shape(C_block))

    history_buffer = np.zeros(L, dtype=np.float64)
    e_buffer = np.zeros(L,dtype=np.float64 )
    delay = L // 2

    # Inicalizate Correlation Matirixes as Zero 
    Q_instant = np.zeros((F, M, M), dtype=complex)

    Q_y = np.zeros((F, M, M), dtype=complex)
    Q_v = np.zeros((F, M, M), dtype=complex)
    Q_x = np.zeros((F, M, M), dtype=complex)

    
    for m in range(tot_frames):
        # --- PROCESS FRAMES ---
        # Extract frame of size 2L
        u_frame = u_padded[: , m * L : m * L + 2 * L]

        # Apply rfft u->U from shape (M, 2L) -> (M, F)
        U_fram_rfft = np.fft.rfft(u_frame, axis=1)

        # Definie VAD frame state
        vad_frame = vad[m * L : m * L + L]
        vad_status = np.mean(vad_frame) > 0.1

        # --- PROCESS FIXED BRANCH ---
        # Apply the spatial fixed filter to obtain refrence Signal D, Sae (F)
        D = np.einsum('fm, mf->f', w_fixed, U_fram_rfft)

        # --- APPLY DELAY FIXED BRANCH ---
        # Convert an extract data with out circular conv. overlap residue 
        d_time_2L = np.fft.irfft(D, n=2 * L)
        d_valid = d_time_2L[L:] 
        d_combined = np.concatenate((history_buffer, d_valid))
        
        # Apply delay by sliding window
        start_idx = L - delay
        end_idx = 2 * L - delay
        d_delayed_valid = d_combined[start_idx : end_idx]

        # Save buffer for posterior frame
        history_buffer = d_valid.copy()

        # Save fixed output to debug
        z_fixed_branch[m * L : m * L + L] = d_valid

        # --- PROCCES ADAPTIVE BRANCH (with previus w) ---
        # Apply the spatial filter to noise reference signals (D_n)
        # Ca (F, M, M-1) and data ( M, F) -> expect (F, M-1)  (Contracting M dimention)
        D_y = np.einsum('fmn, mf->fn', C_block, U_fram_rfft )

        # Extract post block signal for debuging
        d_post_block_2L = np.fft.irfft(D_y[:, 0], n=2 * L)
        d_post_block_2L_valid = d_post_block_2L[L:]
        post_block[m * L : m * L + L] = d_post_block_2L_valid

        # Concatenate reference D signal without delay (F, M-1) -> (F, M) 
        D_y = np.column_stack((D, D_y))

        # Proccess spatial filter, convert to time delay, apply overlap save 
        noise = np.einsum('fm,fm->f', D_y, w_adaptive)
        noise_2L = np.fft.irfft(noise, n=2 * L )
        noise = noise_2L[L:]

        # Save to debug
        z_noise[ m * L : m * L + L] = noise

        # --- COMPUTE OUTPUT (in time domain) ---
        # 1. IMPORTANTE: Usar la señal retrasada como objetivo
        e = d_delayed_valid - noise
        
        # Save output
        z[m * L : m * L + L] = e

        # Zeropad
        e_2L = np.concatenate((np.zeros(L, dtype=np.float64), e))

        # Convert output to DFT (Esto es tu \underline{e}_{v,2L}[m] del paper)
        e_undeline = np.fft.rfft(e_2L)
        
        # --- UPDATE CORRELATION MATRICES ---
        # Compute instantaneous correlation matrix (X* X^T)
        Q_instant = np.einsum('fm,fn->fmn', D_y.conj(), D_y)

        # Update matrices based on VAD
        if vad_status: 
            Q_y = Lambda * Q_y + (1 - Lambda) * Q_instant / 2
            # Weights remain constant during speech
        else:
            Q_v = Lambda * Q_v + (1 - Lambda) * Q_instant / 2
            Q_x = Q_y - Q_v
            Q_x = regularize_covariance_matrix(Q_x)

            # --- UPDATE WEIGHTS ---
            # Mix covariance matrices and add diagonal loading for numerical stability
            Q_mix = Q_v + mu_inv * Q_x 
            
            # np.eye(M) broadcasts perfectly to (F, M, M)
            Q_mix = Q_mix + np.eye(M) * diag_load

            # Regularization term (Q_x is Hermitian, direct multiplication is safe)
            # Q_x: (F, M, M), w_adaptive: (F, M) -> Result: (F, M)
            r_2NL = mu_inv * np.einsum('fmn, fn -> fm', Q_x, w_adaptive)

            # Gradient calculation
            # D_y.conj() is (F, M) and e_undeline is (F,). We scale each row by the scalar error.
            gradient = np.einsum('fm, f -> fm', D_y.conj(), e_undeline) - r_2NL
            
            # Efficiently solve Q_mix * update = gradient across all frequency bins
            gradient_update = np.linalg.solve(Q_mix, gradient)

            # Apply update with the 0.5 unconstrained scalar factor (Table 2)
            w_adaptive = w_adaptive + rho * (1 - Lambda) * 0.5 * gradient_update

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
    FS = 48000
    M1, M2 = 12, 1          
    M = M1 * M2
    speed_of_sound = 343.0 
    
    print("=== INTEGRATION TEST: FIXED BRANCH (SDW-MWF) ===")
    
    output_folder = "tests/data/sdw_mwf_output"
    os.makedirs(output_folder, exist_ok=True)
    
    mic_spacing = 0.05
    x = np.linspace(0, (M1-1)*mic_spacing, M1)
    mic_coords = np.column_stack([x, np.zeros(M), np.zeros(M)])
    
    array_center = np.array([1.25, 2.0, 1.25])
    mic_coords = mic_coords - np.mean(mic_coords, axis=0) + array_center
    
    r = 1.0 
    ang_target = np.deg2rad(130)
    ang_interf = np.deg2rad(50)
    
    source_pos = array_center + np.array([r * np.cos(ang_target), r * np.sin(ang_target), 0.0])
    interf_pos1 = array_center + np.array([r * np.cos(ang_interf), r * np.sin(ang_interf), 0.0])

    print(" -> Initializing acoustic scene...")
    acoustic_scene = SimAcoustic(mic_coords, array_mismatch=0.0, duration=10, fs=FS)

    source_path = "tools/data/signals/FA01_09.wav"
    int_path1 = "tools/data/signals/MC15_03.wav"

    acoustic_scene.set_source(source_path, gain=1, position=source_pos.reshape(1,3))
    acoustic_scene.set_interference(int_path1, gain=1, position=interf_pos1.reshape(1,3))

    print(" -> Computing free field simulation...")
    # room_input_ideal shape is expected to be (M, N_samples)
    room_input_ideal, vad_oracle = acoustic_scene.free_field(iSIR_dB=0, normalize=True, mode="ideal", VAD = True)
    save_wav("1_input_mix_mic0.wav", FS, room_input_ideal[0], output_folder)
    
    print(" -> Applying Fixed Branch (SDW-MWF Delay-and-Sum)...")
    # Ensure source_pos is shape (1, 3) for the steering vector broadcasting
    source_pos_2d = source_pos.reshape(1, 3)
    
    # Execute the delay-and-sum fixed branch in frequency domain
    # Mapping: u = room_input_ideal, target_pos = mic_coords, source_pos = source_pos_2d
    z_fixed, post_block , z_noise, output = sdw_mwf(room_input_ideal,
                               vad_oracle, 
                               mic_coords, 
                               source_pos_2d, 
                               FS)
    
    print(" -> Normalizing and saving reconstructed time-domain signals...")

    print(z_noise)
    


    z_fixed_norm = normalize_signal(z_fixed)
    post_block_norm = normalize_signal(post_block)
    z_noise_norm = normalize_signal(z_noise)
    output_norm = normalize_signal(output)
    

    save_wav("2_output_SDW_MWF_fixed.wav", FS, z_fixed_norm, output_folder)
    save_wav("2_output_SDW_MWF_post_block.wav", FS, post_block_norm, output_folder)
    save_wav("2_output_SDW_MWF_noise.wav", FS, z_noise_norm, output_folder)
    save_wav("2_output_SDW_MWF_output.wav", FS, output_norm, output_folder)
    
    print(" -> Pipeline completed.")
