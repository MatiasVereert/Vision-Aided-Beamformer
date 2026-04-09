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

def sdw_mwf(u, target_pos, source_pos, fs):
    """
    Computes the fixed branch (speech reference) of the SP-SDW-MWF.
    u shape: (M, N_samples)
    """
    # Define constants
    L = 128
    M = u.shape[0]

    # Define frequency bins vector 
    F = L + 1
    frecs = np.linspace(0, fs / 2, F)

    # Correct padding syntax for 2D array: pad axis 1 (time) with L zeros at the beginning
    u_padded = np.pad(u, ((0, 0), (L, 0)), mode='constant')
    
    # Calculate total frames
    tot_frames = (u_padded.shape[1] - L) // L

    # Initialize matrices (for future adaptive stages)
    F_2L = np.zeros((2 * L, 2 * L), dtype=np.complex128)
    Zero_L = np.zeros((L, L), dtype=np.complex128)
    I_L = np.identity(L, dtype=np.complex128)

    # Initialize output array
    z = np.zeros(tot_frames * L, dtype=np.float64)
    z_noise = np.zeros(tot_frames * L, dtype=np.float64)

    # --- FIX: Use RTF steering vector to eliminate global coordinate shifts ---
    # This aligns the phases relative to mic 0, keeping delays strictly local
    sv = compute_rtf_steering_vector(frecs, source_pos, target_pos, ref_mic_idx=0, mode="near_field", squeeze=True)
    
    # Normalize by M to prevent amplitude clipping
    w_fixed_raw= sv.conj() / M

    w_fixed = get_fixed_weights(w_fixed_raw, L, frecs, fs)
    C_block = get_blocking_matrix(w_fixed, L    ) #Shape is (F, M, M-1)

    print(np.shape(C_block))

    for m in range(tot_frames):
        # Extract frame of size 2L
        u_frame = u_padded[: , m * L : m * L + 2 * L]

        # Apply rfft u->U from shape (M, 2L) -> (M, F)
        U_fram_rfft = np.fft.rfft(u_frame, axis=1)

        # --- PROCESS FIXED BRANCH ---
        # Apply the spatial fixed filter to obtain refrence Signal (D_0)
        D_0 = np.einsum('fm, mf->f', w_fixed, U_fram_rfft)

        # Apply the spatial filter to noise reference signals (D_n)

        # Ca (F, M, M-1) and data ( M, F) -> expect (F, M-1)  (calapse M)
        D_n = np.einsum('fmn, mf->fn', C_block, U_fram_rfft )
        
        # Extract 1 signal noise reference to listen as reference and debub
        D_1 = D_n[:,0]








        # Transform back to time domain
        d_2L_0 = np.fft.irfft(D_0) 
        d_2L_1 = np.fft.irfft(D_1) 
        
        # Overlap-Save: discard the first L samples and save the last L samples
        z[m * L : m * L + L] = d_2L_0[L:]
        z_noise[m * L : m * L + L] = d_2L_1[L:]

    return z, z_noise


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
    z_fixed, z_noise = sdw_mwf(room_input_ideal, mic_coords, source_pos_2d, FS)
    
    print(" -> Saving reconstructed time-domain signal...")
    save_wav("2_output_SDW_MWF_fixed.wav", FS, z_fixed, output_folder)
    save_wav("2_output_SDW_MWF_fixed_block_matrix.wav", FS, z_noise, output_folder)
    print(" -> Pipeline completed.")

    from matplotlib import pyplot as plt
    # VAD PLOT

    time = np.linspace(0, len(vad_oracle)/FS, len(vad_oracle)) 
    plt.figure()
    plt.plot( time, vad_oracle , color = 'r')
    plt.show()