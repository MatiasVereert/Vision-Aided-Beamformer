# This model is based on the paper titled: 
# "Frequency-domain criterion for the speech distortion weighted multichannel Wiener filter for robust noise reduction"
# doi: 10.1016/j.specom.2007.02.001
# by Simon Doclo *, Ann Spriet, Jan Wouters, Marc Moonen

import numpy as np
from numba import njit
from scipy.linalg import null_space



# Local 
from beamforming.signal_model import steering_vector, compute_rtf_steering_vector


def get_fixed_weights( w_fixed_raw, L , frecs, fs):
    # Make filter causal and strictly bounded to length L
    # Apply a bulk delay (Delta = L/2) to shift advances into causal delays
    bulk_delay_samples = L // 2
    bulk_phase = np.exp(-1j * 2 * np.pi * frecs * (bulk_delay_samples / fs))
    
    # Broadcast phase across all M microphones
    w_fixed_delayed = w_fixed_raw * bulk_phase[:, None] 
    
    # --- LA CURA DEL SISEO ---
    # 1. El bin de continua (DC) no puede tener fase, tomamos solo su parte real
    w_fixed_delayed[0, :] = w_fixed_delayed[0, :].real + 0.0j
    
    # 2. El bin de Nyquist (último) debe ser estrictamente real. 
    # Lo más seguro es apagarlo a cero para matar cualquier artefacto de alta frecuencia.
    w_fixed_delayed[-1, :] = 0.0 + 0.0j
    # -------------------------
    
    # Transform to time domain to truncate the infinite sinc response
    w_time = np.fft.irfft(w_fixed_delayed, n=2 * L, axis=0) 
    
    # Truncate to length L and pad with L zeros (Strict Overlap-Save rule)
    w_time_padded = np.zeros_like(w_time)
    
    # Copy causal part
    w_time_padded[:L, :] = w_time[:L, :] 
    
    # Transform back to frequency domain to use inside the processing loop
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
"""
def get_blocking_matrix(w_fixed_freq, L):

    Calculates the SVD-based Blocking Matrix and applies the Overlap-Save 
    time-domain constraint to prevent circular convolution wrap-around.
    
    Args:
        w_fixed_freq: Array of shape (F, M) with the fixed beamformer weights.
        L: Frame size (the true time-domain constraint length).
        
    Returns:
        Ca_constrained: Array of shape (F, M, M-1) ready for frequency multiplication.
    
    F, M = w_fixed_freq.shape
    
    # 1. Initialize the raw frequency-domain Blocking Matrix
    Ca_raw = np.zeros((F, M, M - 1), dtype=complex)
    
    # 2. Calculate the Null Space (SVD) for each frequency bin independently
    for f in range(F):
        # Extract the fixed filter for this bin as a column vector (M, 1)
        C_bin = w_fixed_freq[f, :].reshape(M, 1)
        
        # The null space of C^H guarantees that C^H * Ca = 0
        Ca_raw[f, :, :] = null_space(C_bin.conj().T)
        
    # 3. Apply the Overlap-Save time constraint
    # Transform back to the time domain (axis=0 is the frequency axis)
    Ca_time = np.fft.irfft(Ca_raw, n=2 * L, axis=0)
    
    # Create a padded array of zeros
    Ca_time_padded = np.zeros_like(Ca_time)
    
    # Keep only the first L taps (strict causal constraint without cheating)
    Ca_time_padded[:L, :, :] = Ca_time[:L, :, :]
    
    # Transform back to the frequency domain for the main processing loop
    Ca_constrained = np.fft.rfft(Ca_time_padded, n=2 * L, axis=0)
    
    return Ca_constrained
"""

import numpy as np


def sdw_mwf(u, vad, target_pos, source_pos, fs):
    """
    Computes the SP-SDW-MWF adaptive filter utilizing the Algorithm 1 
    frequency-domain constrained update with block-structured step size.
    Input 'u' shape: (M, N_samples)
    """
    # Define constants and adaptive parameters
    L = 128
    M = u.shape[0]
    lmbda = 0.995        # Exponential forgetting factor (lambda)
    inv_mu = 1         # Trade-off parameter (1/mu)
    rho = 2.0       # Step size parameter

    # Define frequency bins vector 
    F = L + 1
    frecs = np.linspace(0, fs / 2, F)

    # Pad axis 1 (time) with L zeros at the beginning for Overlap-Save
    u_padded = np.pad(u, ((0, 0), (L, 0)), mode='constant')
    
    # Calculate total frames
    tot_frames = (u_padded.shape[1] - L) // L

    # Initialize output arrays
    z = np.zeros(tot_frames * L, dtype=np.float64)
    z_noise = np.zeros(tot_frames * L, dtype=np.float64)
    reference_sig = np.zeros(tot_frames * L, dtype=np.float64)
    noise_sig = np.zeros(tot_frames * L, dtype=np.float64)

    # Compute RTF steering vector to eliminate global coordinate shifts
    sv = compute_rtf_steering_vector(frecs, source_pos, target_pos, ref_mic_idx=0, mode="near_field", squeeze=True)
    
    # Normalize by M to prevent amplitude clipping and compute static matrices
    w_fixed_raw = sv.conj() / M
    w_fixed = get_fixed_weights(w_fixed_raw, L, frecs, fs)
    C_block = get_blocking_matrix(w_fixed, L) 

    # --- ADAPTIVE STAGE INITIALIZATION ---
    w_adapt = np.zeros((F, M - 1), dtype=np.complex128)
    
    # CRITICAL FIX 1: Initialize with 0.1 instead of 1e-6 to prevent "Cold Start Explosion"
    # This ensures the inverse matrix starts very small, keeping initial steps safe.
    Q_y = np.tile(0.1 * np.eye(M - 1, dtype=np.complex128), (F, 1, 1))
    Q_v = np.tile(0.1 * np.eye(M - 1, dtype=np.complex128), (F, 1, 1))

    # Performance counters
    noise = 0
    speech = 0

    for m in range(tot_frames):
        # Extract frame of size 2L
        u_frame = u_padded[:, m * L : m * L + 2 * L]

        # Apply rfft u->U from shape (M, 2L) -> (M, F)
        U_fram_rfft = np.fft.rfft(u_frame, axis=1)

        # --- PROCESS FIXED BRANCH ---
        # Apply the spatial fixed filter to obtain reference Speech Signal (D_0)
        D_0 = np.einsum('fm, mf->f', w_fixed, U_fram_rfft)

        # --- PROCESS ADAPTIVE BRANCH ---
        # Apply the blocking matrix to obtain noise reference signals (D_y)
        # Expected shape: (F, M-1)
        D_y = np.einsum('fmn, mf->fn', C_block, U_fram_rfft)
        
        # Extract 1 signal noise reference to listen as debug
        D_1 = D_y[:, 0]

        # 1. Circular convolution in frequency domain using ONLY D_y
        Y_w_freq = np.einsum('fm, fm -> f', D_y, w_adapt)
        
        # 2. Transform back to time domain
        y_w_time = np.fft.irfft(Y_w_freq, n=2 * L)
        
        # 3. Constrain valid output (Overlap-Save artifact removal)
        valid_y_w = y_w_time[L:]
        noise_sig[m * L : m * L + L] = valid_y_w
        
        # 4. Target reference is the delayed speech reference
        d_2L_0 = np.fft.irfft(D_0)
        d = d_2L_0[L:]
        reference_sig[m * L : m * L + L] = d
        
        # 5. Compute enhanced error signal (final output)
        e_m = d - valid_y_w

        # --- CORRELATION MATRICES UPDATE ---
        # Outer product uses strictly D_y (M-1 components)
        inst_corr = np.einsum('fm, fn -> fmn', D_y.conj(), D_y) / 2.0

        # Evaluate VAD for current frame (Tolerance for slightly noisy boolean masks)
        vad_frame = vad[m * L : m * L + L]
        vad_status = np.mean(vad_frame) > 0.1 
        
        if vad_status:
            speech += 1
            # Update speech+noise covariance matrix
            Q_y = lmbda * Q_y + (1.0 - lmbda) * inst_corr
        else:
            noise += 1
            # Update noise-only covariance matrix
            Q_v = lmbda * Q_v + (1.0 - lmbda) * inst_corr

        Q_x = Q_y - Q_v

        # --- UPDATE FILTER WEIGHTS ---
        # Ensure Q_x is Hermitian to avoid complex eigenvalues due to precision
        Q_x_herm = (Q_x + Q_x.conj().transpose(0, 2, 1)) / 2.0
        
        # Compute eigenvalues and regularize Q_x to be strictly positive definite
        eigvals = np.linalg.eigvalsh(Q_x_herm)
        min_eig = np.min(eigvals, axis=1)
        shift = np.maximum(0.0, -min_eig) + 1e-6
        shift_mat = shift[:, np.newaxis, np.newaxis] * np.eye(M - 1)[np.newaxis, :, :]
        Q_x_reg = Q_x_herm + shift_mat

        # Compute block-structured step size matrix with robust diagonal loading
        reg_matrix = 1e-6 * np.eye(M - 1)[np.newaxis, :, :]
        step_matrix = Q_v + inv_mu * Q_x_reg + reg_matrix
        
        # Force perfect symmetry to avoid complex precision errors
        step_matrix = (step_matrix + step_matrix.conj().transpose(0, 2, 1)) / 2.0
        
        # Now the inverse is mathematically safe
        inv_step = np.linalg.inv(step_matrix)

        # Pad valid error with zeros for frequency transform
        e_padded = np.zeros(2 * L)
        e_padded[L:] = e_m
        E_freq = np.fft.rfft(e_padded)

        # Compute regularization vector and gradient using D_y
        r_2NL = inv_mu * np.einsum('fmn, fn -> fm', Q_x_reg, w_adapt)
        D_H_E = D_y.conj() * E_freq[:, np.newaxis]
        raw_grad = D_H_E - r_2NL

        # Unconstrained delta weights
        delta_w_unconstrained = rho * np.einsum('fmn, fn -> fm', inv_step, raw_grad)

        # --- CRITICAL FIX 3: GRADIENT CLIPPING (Safety Net) ---
        # Prevent the filter from taking dangerously large steps that cause NaN
        max_step = 1.0
        step_magnitudes = np.abs(delta_w_unconstrained)
        # Find where the step is larger than the maximum allowed
        excessive_steps = step_magnitudes > max_step
        # Scale down only the components that exploded, preserving their phase
        delta_w_unconstrained[excessive_steps] = (delta_w_unconstrained[excessive_steps] / step_magnitudes[excessive_steps]) * max_step
        # ------------------------------------------------------        

        # Apply G10 causality constraint (Time-domain truncation)
        delta_w_time = np.fft.irfft(delta_w_unconstrained, n=2 * L, axis=0)
        delta_w_time[L:, :] = 0.0 # Strict causal constraint
        delta_w_constrained = np.fft.rfft(delta_w_time, n=2 * L, axis=0)

        # Update adaptive filter
        w_adapt += delta_w_constrained

        # --- PREVENT EXTREME FREQUENCY INSTABILITY ---
        # The DC bin must be strictly real to prevent low-frequency drift
        w_adapt[0, :] = w_adapt[0, :].real + 0.0j
        
        # The Nyquist bin is highly susceptible to high-frequency ringing; kill it safely
        w_adapt[-1, :] = 0.0 + 0.0j

        # --- SAVE OUTPUTS ---
        z[m * L : m * L + L] = e_m
        
        # Save one noise reference channel for debugging
        d_2L_1 = np.fft.irfft(D_1, n=2 * L) 
        z_noise[m * L : m * L + L] = d_2L_1[L:]

    print(f"Noise frames: {noise}, Speech frames: {speech}, Ratio N/S: {noise/speech:.2f}")

    return z, z_noise, reference_sig, noise_sig

import os
import numpy as np
# scipy.signal is no longer strictly needed for STFT/ISTFT, but you might need it elsewhere

# Adjust imports based on your exact file structure
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
    acoustic_scene = SimAcoustic(mic_coords, array_mismatch=0.0, duration=20, fs=FS)

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
    z, z_noise, reference_sig, noise_sig = sdw_mwf(u =room_input_ideal,
                               vad = vad_oracle,
                                target_pos = mic_coords,
                                source_pos = source_pos_2d, 
                               fs = FS)
    # Normalize the audio array to strictly fit within the safe range [-target_peak, target_peak]
    def normalize_peak(audio_array, target_peak=0.95):
        # Find the maximum absolute value in the array
        max_abs = np.max(np.abs(audio_array))
        
        # Avoid division by zero in case of pure silence
        if max_abs > 0.0:
            # Scale the array and apply the headroom margin
            normalized_audio = (audio_array / max_abs) * target_peak
            return normalized_audio
        else:
            return audio_array

    # Apply normalization directly to your output signals before saving
    z_norm = normalize_peak(z)
    z_noise_norm = normalize_peak(z_noise)
    ref_sig_norm = normalize_peak(reference_sig)
    noise_sig_norm = normalize_peak(noise_sig)

    # Save the safely normalized signals
    save_wav("2_output_SDW_MWF_output.wav", FS, z_norm, output_folder)
    save_wav("2_output_SDW_MWF_sum_delay.wav", FS, ref_sig_norm, output_folder)
    save_wav("2_output_SDW_MWF_noise_sig.wav", FS, noise_sig_norm, output_folder)
        
    
    from matplotlib import pyplot as plt

    # VAD PLOT (NEEDS FURTHER ANALISIS)

    time = np.linspace(0, len(vad_oracle)/FS, len(vad_oracle)) 
    plt.figure()
    plt.plot( time, vad_oracle , color = 'r')
