# This model is based on the paper titled: 
# "Frequency-domain criterion for the speech distortion weighted multichannel Wiener filter for robust noise reduction"
# doi: 10.1016/j.specom.2007.02.001
# by Simon Doclo *, Ann Spriet, Jan Wouters, Marc Moonen

import numpy as np
from numba import njit
# Local 
from beamforming.signal_model import steering_vector, compute_rtf_steering_vector

def get_fixed_weights(w_fixed_raw, L, frecs, fs):
    # Apply ONLY a small alignment delay if necessary to make the IR causal,
    # NOT the algorithmic Delta = L/2. For near-field relative to mic 0, 
    # phase shifts might already be well-behaved.
    # If a centering shift is needed, keep it minimal (e.g., center of the L filter).
    alignment_samples = L // 4  # # In te future this could be minimize by using the max lenght inter microphone
    alignment_phase = np.exp(-1j * 2 * np.pi * frecs * (alignment_samples / fs))
    
    # Broadcast phase across all M microphones
    w_fixed_aligned = w_fixed_raw * alignment_phase[:, None] 
    
    # Force DC to real
    w_fixed_aligned[0, :] = w_fixed_aligned[0, :].real + 0.0j
    
    # Force Nyquist to real (or zero to prevent HF artifacts)
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

    


def sdw_mwf(u, vad, target_pos, source_pos, fs):
    """
    Computes the fixed branch (speech reference) of the SP-SDW-MWF.
    u shape: (M, N_samples)
    """

    # Define constants
    lamda = 1
    L = 128
    M = u.shape[0]

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

    # This aligns the phases relative to mic 0, keeping delays strictly local
    sv = compute_rtf_steering_vector(frecs, source_pos, target_pos, ref_mic_idx=0, mode="near_field", squeeze=True)
    
    # Normalize by M to prevent amplitude clipping
    w_fixed_raw= sv.conj() / M

    w_fixed = get_fixed_weights(w_fixed_raw, L, frecs, fs)
    C_block = get_blocking_matrix(w_fixed, L    ) #Shape is (F, M, M-1)


    # Inicalice adaptive variables
    w_adaptive = np.zeros( (F, M ), dtype = np.complex128)

    print(np.shape(C_block))

    history_buffer = np.zeros(L, dtype=np.float64)
    delay = L // 2

    for m in range(tot_frames):
        # --- PROCESS FRAMES ---
        # Extract frame of size 2L
        u_frame = u_padded[: , m * L : m * L + 2 * L]

        # Apply rfft u->U from shape (M, 2L) -> (M, F)
        U_fram_rfft = np.fft.rfft(u_frame, axis=1)

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

        # Concatenate reference D signal without delay
        D_y = np.column_stack((D, D_y))

        # Proccess spatial filter, convert to time delay, apply overlap save 
        noise = np.einsum('fm,fm->f', D_y, w_adaptive)
        noise_2L = np.fft.irfft(noise, n=2 * L )
        noise = noise_2L[L:]

        # Save to debug
        z_noise[ m * L : m * L + L] = noise

        # --- COMPUTE OUTPUT (in time domain) ---
        e = d_valid - noise

        #   Save ouput
        z[m * L : m * L + L] = e




    return z_fixed_branch, z_noise, z 


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
    z_fixed, z_noise, output = sdw_mwf(room_input_ideal,
                               vad_oracle, 
                               mic_coords, 
                               source_pos_2d, 
                               FS)
    
    print(" -> Saving reconstructed time-domain signal...")
    save_wav("2_output_SDW_MWF_fixed.wav", FS, z_fixed, output_folder)
    save_wav("2_output_SDW_MWF_fixed_block_matrix.wav", FS, z_noise, output_folder)
    print(" -> Pipeline completed.")

    from matplotlib import pyplot as plt

    # VAD PLOT (NEEDS FURTHER ANALISIS)
    """
    time = np.linspace(0, len(vad_oracle)/FS, len(vad_oracle)) 
    plt.figure()
    plt.plot( time, vad_oracle , color = 'r')
    plt.show()
    """