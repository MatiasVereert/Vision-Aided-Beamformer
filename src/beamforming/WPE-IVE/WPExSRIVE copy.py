import numpy as np
from matplotlib import pyplot as plt 
from numba import njit


@njit
def process_frame(x_frame, x_bar, W, W_inv_H, G_1, G_2, 
                  Sigma_1, Sigma_2, R_inv_1, R_inv_2, 
                  a_1, a_2, alpha, beta, lambda_unit, lambda_null, lambda_scale, N_Iter, frame_idx):
    
    F = x_frame.shape[0]
    M = x_frame.shape[1]
    
    # Pre-allocate temporary arrays
    y_1 = np.zeros((F, M), dtype=np.complex128)
    y_2 = np.zeros((F, M), dtype=np.complex128)
    y = np.zeros((F, M), dtype=np.complex128)
    s_hat = np.zeros(F, dtype=np.complex128)
    
    temp_vec = np.zeros(M, dtype=np.complex128)
    w_1_contig = np.zeros(M, dtype=np.complex128)
    yyH = np.zeros((M, M), dtype=np.complex128)
    Pi_1 = np.zeros((M, M), dtype=np.complex128)
    new_W_inv_H = np.zeros((M, M), dtype=np.complex128)
    
    for Iter in range(N_Iter):
        
        # Calculate dereverberated signals
        for f in range(F):
            y_1[f] = x_frame[f] - np.dot(G_1[f].conj().T, x_bar[f])
            y_2[f] = x_frame[f] - np.dot(G_2[f].conj().T, x_bar[f])
            
            w_1_contig[:] = W[f, :, 0]
            W_Z_contig = np.ascontiguousarray(W[f, :, 1:])
            
            temp_vec[0] = np.dot(w_1_contig.conj(), y_1[f])
            temp_vec[1:] = np.dot(W_Z_contig.conj().T, y_2[f])
            
            y[f] = np.dot(W_inv_H[f], temp_vec)
                
            if Iter == N_Iter - 1:
                # Apply projection back to reference microphone 0
                s_hat[f] = W_inv_H[f, 0, 0] * temp_vec[0]
                
        # Calculate source variance
        v_1 = 0.0
        for f in range(F):
            w_1_contig[:] = W[f, :, 0]
            s_hat_tmp = np.dot(w_1_contig.conj(), y_1[f])
            v_1 += np.abs(s_hat_tmp)**2
        v_1 = v_1 / F
        v_1 = max(v_1, 1e-15) 
        
        for f in range(F):
            
            # Update EWMA Covariances
            for i in range(M):
                for j in range(M):
                    yyH[i, j] = y[f, i] * np.conj(y[f, j])
            
            Sigma_1[f] = alpha * Sigma_1[f] + (1.0 - alpha) * yyH / v_1
            Sigma_2[f] = alpha * Sigma_2[f] + (1.0 - alpha) * yyH 
            
            # --- WARM-UP STABILIZATION ---
            # Do not update separation matrices in the first frames
            if frame_idx < M * 2: 
                pass 
            else:
                # --- Exact Spatial Regularization ---
                # Reconstruct Pi_1 entirely to prevent exponential decay of spatial prior
                for i in range(M):
                    for j in range(M):
                        Pi_1[i, j] = Sigma_1[f, i, j] \
                                     + lambda_unit * (a_1[f, i] * np.conj(a_1[f, j])) \
                                     + lambda_null * (a_2[f, i] * np.conj(a_2[f, j]))
                        if i == j:
                            Pi_1[i, j] += lambda_scale
                            
                # Calculate exact inverse to eliminate Sherman-Morrison drift
                Pi_inv_1_f = np.linalg.inv(Pi_1)
                
                w_old = np.ascontiguousarray(W[f, :, 0])
                W_inv_e1 = np.ascontiguousarray(W_inv_H[f, :, 0])
                
                # VCD Update for w_1
                w_tmp = np.dot(Pi_inv_1_f, W_inv_e1)
                w_hat = lambda_unit * np.dot(Pi_inv_1_f, a_1[f])
                
                h_1 = np.dot(np.conj(w_tmp), np.dot(Pi_1, w_tmp)).real
                h_hat = np.dot(np.conj(w_tmp), np.dot(Pi_1, w_hat))
                
                if np.abs(h_hat) == 0.0:
                    w_new = (1.0 / np.sqrt(h_1)) * w_tmp + w_hat
                else:
                    h_tilde = (h_hat / (2.0 * h_1)) * (-1.0 + np.sqrt(1.0 + 4.0 * h_1 / (np.abs(h_hat)**2)))
                    w_new = h_tilde * w_tmp + w_hat
                    
                W[f, :, 0] = w_new
                
                # Sherman-Morrison for W^{-H}
                delta_w = w_new - w_old
                denom_W = 1.0 + np.dot(np.conj(delta_w), W_inv_e1)
                
                for i in range(M):
                    for j in range(M):
                        yyH[i, j] = W_inv_e1[i] * np.dot(np.conj(delta_w), W_inv_H[f])[j]
                        
                W_inv_H[f] = W_inv_H[f] - (yyH / denom_W)
                
                # W_Z Update
                W_S_H_Sigma = np.dot(np.conj(w_new), Sigma_2[f])
                scalar_part = W_S_H_Sigma[0]
                vector_part = W_S_H_Sigma[1:]
                
                top_row = - (1.0 / scalar_part) * vector_part
                W[f, 0, 1:] = top_row
                
                for m_idx in range(1, M):
                    for m_idx2 in range(1, M):
                        W[f, m_idx, m_idx2] = 1.0 + 0.0j if m_idx == m_idx2 else 0.0 + 0.0j
                        
                # Block inversion for new W^{-H}
                X = w_new[0]
                Y = top_row
                Z = w_new[1:]
                X_tilde = X - np.dot(Y, Z)
                inv_X_tilde = 1.0 / X_tilde
                
                new_W_inv_H[0, 0] = inv_X_tilde
                new_W_inv_H[0, 1:] = -inv_X_tilde * Y
                new_W_inv_H[1:, 0] = -Z * inv_X_tilde
                
                for i in range(M - 1):
                    for j in range(M - 1):
                        new_W_inv_H[i + 1, j + 1] = (1.0 + 0.0j if i == j else 0.0 + 0.0j) + Z[i] * (inv_X_tilde * Y)[j]
                
                W_inv_H[f] = new_W_inv_H.conj().T
            
            # --- WPE Dereverberation Update ---
            if Iter == 0:
                # Target source filtering
                denom_R1 = beta * v_1 + np.dot(np.conj(x_bar[f]), np.dot(R_inv_1[f], x_bar[f])).real
                K_1 = np.dot(R_inv_1[f], x_bar[f]) / denom_R1
                
                K_1_err = np.dot(np.conj(x_bar[f]), R_inv_1[f])
                for i in range(M * x_bar.shape[1] // M):
                    for j in range(M * x_bar.shape[1] // M):
                        R_inv_1[f, i, j] = (R_inv_1[f, i, j] - K_1[i] * K_1_err[j]) / beta
                        
                err_1 = x_frame[f] - np.dot(G_1[f].conj().T, x_bar[f])
                
                for i in range(M * x_bar.shape[1] // M):
                    for j in range(M):
                        G_1[f, i, j] = G_1[f, i, j] + K_1[i] * np.conj(err_1[j])
                
                # Background noise filtering
                denom_R2 = beta * 1.0 + np.dot(np.conj(x_bar[f]), np.dot(R_inv_2[f], x_bar[f])).real
                K_2 = np.dot(R_inv_2[f], x_bar[f]) / denom_R2
                
                K_2_err = np.dot(np.conj(x_bar[f]), R_inv_2[f])
                for i in range(M * x_bar.shape[1] // M):
                    for j in range(M * x_bar.shape[1] // M):
                        R_inv_2[f, i, j] = (R_inv_2[f, i, j] - K_2[i] * K_2_err[j]) / beta
                        
                err_2 = x_frame[f] - np.dot(G_2[f].conj().T, x_bar[f])
                
                for i in range(M * x_bar.shape[1] // M):
                    for j in range(M):
                        G_2[f, i, j] = G_2[f, i, j] + K_2[i] * np.conj(err_2[j])
                
    return s_hat

@njit
def WPExSRIVE(x_stft, a_1, a_2, L=12, D=4):
    F, T, M = x_stft.shape

    # Optimal hyperparameters based on the reference literature
    lambda_scale = 1.0
    lambda_unit = 10.0
    lambda_null = 10.0
    alpha = 0.99
    beta = 0.9999

    W = np.zeros((F, M, M), dtype=np.complex128)
    W_inv_H = np.zeros((F, M, M), dtype=np.complex128)
    
    G_1 = np.zeros((F, L * M, M), dtype=np.complex128)
    G_2 = np.zeros((F, L * M, M), dtype=np.complex128)

    Sigma_1 = np.zeros((F, M, M), dtype=np.complex128)
    Sigma_2 = np.zeros((F, M, M), dtype=np.complex128)
    
    R_inv_1 = np.zeros((F, L * M, L * M), dtype=np.complex128)
    R_inv_2 = np.zeros((F, L * M, L * M), dtype=np.complex128)

    for f in range(F):
        W[f] = np.eye(M, dtype=np.complex128)
        W_inv_H[f] = np.eye(M, dtype=np.complex128)
        R_inv_1[f] = np.eye(L * M, dtype=np.complex128)
        R_inv_2[f] = np.eye(L * M, dtype=np.complex128)
        
        # Initialize with a small noise floor to prevent div by zero
        for i in range(M):
            Sigma_1[f, i, i] = 1e-6
            Sigma_2[f, i, i] = 1e-6

    buffer_len = L + D
    x_buffer = np.zeros((F, buffer_len, M), dtype=np.complex128)
    x_bar = np.zeros((F, M * L), dtype=np.complex128)

    s_hat_stft = np.zeros((F, T), dtype=np.complex128)

    for t in range(T):
        x_frame = np.ascontiguousarray(x_stft[:, t, :])

        for f in range(F):
            for m in range(M):
                for i in range(buffer_len - 1, 0, -1):
                    x_buffer[f, i, m] = x_buffer[f, i - 1, m]
                x_buffer[f, 0, m] = x_frame[f, m]

        for f in range(F):
            m_count = 0
            for i in range(D, D + L):
                for m in range(M):
                    x_bar[f, m_count] = x_buffer[f, i, m]
                    m_count += 1

        s_hat_frame = process_frame(
                    x_frame, x_bar, W, W_inv_H, G_1, G_2, 
                    Sigma_1, Sigma_2, R_inv_1, R_inv_2, 
                    Pi_inv_1_f,  # <-- Added missing argument here
                    a_1, a_2, alpha, beta, lambda_unit, lambda_null, lambda_scale, N_Iter=5, frame_idx=t
                )
        
        s_hat_stft[:, t] = s_hat_frame

    return s_hat_stft

import time
import os
import numpy as np
import scipy.signal as sig  

# Adjust imports based on your exact file structure
from propagation.simulate_acoustics import SimAcoustic
from utils.audio import save_wav
# from wpexsrive import WPExSRIVE, process_frame  # Import your algorithm here
import os
import numpy as np
import scipy.signal as sig

# Adjust imports based on your exact file structure
from propagation.simulate_acoustics import SimAcoustic
from utils.audio import save_wav
# from wpexsrive import WPExSRIVE  # Import your algorithm here

if __name__ == "__main__":
    # 1. GENERAL SETTINGS
    FS = 16000  # 16 kHz is standard for speech processing and matches the paper
    M = 8       # Number of microphones
    speed_of_sound = 343.0 
    
    print("=== INTEGRATION TEST: ONLINE WPExSRIVE ===")
    
    output_folder = "tests/data/wpexsrive_output"
    os.makedirs(output_folder, exist_ok=True)
    
    # 2. ARBITRARY ARRAY GEOMETRY DEFINITION (LOGARITHMIC SPACING)
    # Generate a logarithmic spacing along the X axis
    log_spacing = np.logspace(-2.0, -0.5, M) 
    
    # Shift coordinates so the first microphone (reference) is strictly at x=0
    x_coords = log_spacing - log_spacing[0]
    
    # Construct the M x 3 matrix for the microphone coordinates
    mic_coords_ideal = np.column_stack([x_coords, np.zeros(M), np.zeros(M)])
    
    # Center the entire array inside the simulated room
    array_center = np.array([2.0, 2.0, 1.25])
    mic_coords_ideal = mic_coords_ideal + array_center

    # Define source angles and distance
    r = 1.0 
    ang_target = np.deg2rad(130)
    ang_interf = np.deg2rad(50)
    
    # Calculate source positions relative to array center
    source_pos = array_center + np.array([r * np.cos(ang_target), r * np.sin(ang_target), 0.0])
    interf_pos = array_center + np.array([r * np.cos(ang_interf), r * np.sin(ang_interf), 0.0])

    # 3. ACOUSTIC SCENE SIMULATION (REVERBERANT ROOM)
    print(" -> Initializing reverberant acoustic scene...")
    # Add a small mismatch to make it realistic
    acoustic_scene = SimAcoustic(mic_coords_ideal, array_mismatch=0.002, duration=10, fs=FS)

    source_path = "tools/data/signals/FA01_09.wav"
    int_path = "tools/data/signals/MC15_03.wav"

    # Fix: Assign proper distinct positions to the sources, not the array coordinates
    acoustic_scene.set_source(source_path, gain=1.0, position=source_pos.reshape(1,3))
    acoustic_scene.set_interference(int_path, gain=1.0, position=interf_pos.reshape(1,3))

    print(" -> Computing Room Impulse Responses (ISB)...")
    room_dimensions = np.array([4.0, 5.0, 2.5])
    
    # T60 = 0.2s for a moderate room reverberation
    #room_input_mix = acoustic_scene.free_field( iSIR_dB=0, normalize=True, mode="ideal")
    room_input_mix = acoustic_scene.compute_room_ISB(iSIR_dB=0, 
                                                    desire_RT=0.2,
                                                    room_dimensions=room_dimensions, 
                                                    mode="ideal")
    
    save_wav("1_wpex_input_mix_mic0.wav", FS, room_input_mix[0], output_folder)
    
    # 4. SHORT-TIME FOURIER TRANSFORM (LOW LATENCY SETUP)
    # 8 ms window, 4 ms overlap
    nperseg = int(FS * 0.008)  
    noverlap = nperseg - int(FS * 0.004) 
    nfft = nperseg
    
    print(f" -> Applying STFT (Window: {nperseg} samples, Overlap: {noverlap} samples)...")
    freqs, times, X_stft = sig.stft(
        room_input_mix, fs=FS, window='hann', 
        nperseg=nperseg, noverlap=noverlap, nfft=nfft
    )
    
    F_bins = X_stft.shape[1]

    # 5. GENERALIZED STEERING VECTORS FOR ARBITRARY GEOMETRIES
    print(" -> Building Universal Steering Vectors (Target & Interference)...")
    a_1 = np.zeros((F_bins, M), dtype=np.complex128)
    a_2 = np.zeros((F_bins, M), dtype=np.complex128)
    
    # Define unit direction vectors pointing from the array center towards the sound sources
    u_target = np.array([np.cos(ang_target), np.sin(ang_target), 0.0])
    u_interf = np.array([np.cos(ang_interf), np.sin(ang_interf), 0.0])
    
    # Center all microphone coordinates strictly relative to the reference microphone (index 0).
    mic_coords_rel = mic_coords_ideal - mic_coords_ideal[0, :]
    
    # Compute exact physical time delays using the dot product for arbitrary 3D geometry
    tau_1 = -np.dot(mic_coords_rel, u_target) / speed_of_sound
    tau_2 = -np.dot(mic_coords_rel, u_interf) / speed_of_sound
    
    for f_idx, freq in enumerate(freqs):
        if freq == 0: 
            a_1[f_idx, :] = 1.0 / np.sqrt(M)
            a_2[f_idx, :] = 1.0 / np.sqrt(M)
        else:
            a_1[f_idx, :] = np.exp(-1j * 2 * np.pi * freq * tau_1) / np.sqrt(M)
            a_2[f_idx, :] = np.exp(-1j * 2 * np.pi * freq * tau_2) / np.sqrt(M)

    # Transpose to (F, T, M) and ensure contiguous memory layout for Numba
    X_stft_ive = np.transpose(X_stft, (1, 2, 0))
    X_stft_ive = np.ascontiguousarray(X_stft_ive, dtype=np.complex128)

    # 6. WPExSRIVE OPTIMIZATION
    print(" -> Executing WPExSRIVE online processing...")
    s_hat_stft = WPExSRIVE(X_stft_ive, a_1, a_2, L=12, D= 4)
    
    # 7. RECONSTRUCTION
    print(" -> Reconstructing time-domain signal (ISTFT)...")
    _, y_time = sig.istft(
        s_hat_stft, fs=FS, window='hann', 
        nperseg=nperseg, noverlap=noverlap, nfft=nfft
    )
    
    save_wav("2_wpex_output_target.wav", FS, y_time, output_folder)
    print(" -> Pipeline completed successfully.")