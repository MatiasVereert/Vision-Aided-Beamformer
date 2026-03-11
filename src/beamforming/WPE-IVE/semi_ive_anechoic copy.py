import numpy as np 
from numba import njit

@njit
def calculate_V_z(x_stft):
    F, T, M = x_stft.shape
    V_z = np.zeros((F, M, M), dtype=np.complex128)
    for f in range(F):
        for t in range(T):
            for m in range(M):
                for n in range(M):
                    V_z[f, m, n] += x_stft[f, t, m] * np.conjugate(x_stft[f, t, n])

    for f in range(F):
        for m in range(M):
            for n in range(M):
                V_z[f, m, n] /= T
    return V_z

@njit
def update_W_z_K1(W, V_z):
    """
    Updates the noise portion W_z of the matrix W for K=1.
    """
    F, M, _ = V_z.shape
    for f in range(F):
        # Ensure contiguous memory layout for Numba dot product efficiency
        w_1_h = np.ascontiguousarray(np.conjugate(W[f, :, 0]))
        
        temp = np.dot(w_1_h, V_z[f])
        A = temp[0]
        B = temp[1:]
        C = B / A
        
        W[f, 0, 1:] = C
        for i in range(1, M):
            for j in range(1, M):
                if i == j:
                    W[f, i, j] = -1.0 + 0.0j
                else:
                    W[f, i, j] = 0.0 + 0.0j
    return W

@njit
def update_W_IP2_K1(W, V_1, V_z, n_power_iter=3):
    """
    Updates the extraction filter w_1 for K=1 using IVE-IP2.
    """
    F, M, _ = V_1.shape
    for f in range(F):
        # CRITICAL FIX: Invert V_1, not V_z, to find the largest eigenvalue
        V_1_inv = np.linalg.inv(V_1[f])
        A = np.dot(V_1_inv, V_z[f])
        
        # Ensure contiguous memory layout
        u = np.ascontiguousarray(W[f, :, 0])
        
        for _ in range(n_power_iter):
            u = np.dot(A, u)
            norm_u = np.linalg.norm(u)
            u = u / norm_u
            
        temp = np.dot(np.conjugate(u), V_1[f])
        scalar_val = np.real(np.dot(temp, u))
        
        w_1 = u / np.sqrt(scalar_val)
        W[f, :, 0] = w_1
        
    return W

@njit 
def ive(x_stft, a_1_init=None , mode= "IVE-IP2"):
    # Parameters 
    beta = 0.1
    N_iter = 5
    F, T, M = x_stft.shape

    # Strict type allocation for W
    W = np.zeros((F, M, M), dtype=np.complex128)

    # Initialize W as negative identity
    for f in range(F):
        for m in range(M):
            W[f, m, m] = -1.0 + 0.0j
            
        # Optional: Initialize the first column with the steering vector 
        # to prevent permutation ambiguity
        if a_1_init is not None:
            W[f, :, 0] = a_1_init[f, :]

    V_z = calculate_V_z(x_stft)
    if mode == "IVE-IP2":
        W = update_W_z_K1(W, V_z)

    if mode == "Semi-IVE":
        

        
    
    s_1 = np.zeros((F, T), dtype=np.complex128)
    r_1 = np.zeros(T, dtype=np.float64)
    phi = np.zeros(T, dtype=np.float64)
    V_1 = np.zeros((F, M, M), dtype=np.complex128)
    
    # ---- Optimization Loop ----
    for iter in range(N_iter):
        for f in range(F):
            for m in range(M):
                for n in range(M):
                    V_1[f, m, n] = 0.0 + 0.0j
            
        for f in range(F):
            for t in range(T):
                suma_esp = 0.0 + 0.0j
                for m in range(M):
                    suma_esp += np.conjugate(W[f, m, 0]) * x_stft[f, t, m]
                s_1[f, t] = suma_esp

        r_sum = 0.0
        for t in range(T):
            suma_frecuencial = 0.0
            for f in range(F):
                re = s_1[f, t].real
                im = s_1[f, t].imag
                suma_frecuencial += (re * re) + (im * im)
            
            r_1[t] = np.sqrt(suma_frecuencial)
            r_sum += r_1[t] ** beta

        alpha_1_beta = (beta / (2.0 * F)) * (r_sum / T)
        
        for t in range(T):
            denominator = alpha_1_beta * (r_1[t] + 1e-15)**(2-beta)
            phi[t] = (beta / 2.0) / denominator

        phi_min = np.min(phi)
        max_threshold_max = 100000.0 * phi_min

        for t in range(T):
            if phi[t] > max_threshold_max:
                phi[t] = max_threshold_max

        for f in range(F):
            for t in range(T):
                for m in range(M):
                    for n in range(M):
                        V_1[f,m,n] += phi[t] * x_stft[f,t, m] * np.conjugate(x_stft[f,t, n]) / T
            
            tr_V_1 = 0.0      
            for m in range(M):
                tr_V_1 += V_1[f,m,m]

            for m in range(M):
                V_1[f, m, m] += 1e-3 * tr_V_1
        
        W = update_W_IP2_K1(W, V_1, V_z, n_power_iter=3)

    # ---- Post-Processing ----
    W = update_W_z_K1(W, V_z)
    x_1_out = np.zeros((F, T, M), dtype=np.complex128)

    for f in range(F):
        W_inv = np.linalg.inv(W[f])
        for t in range(T):
            scalar_signal = s_1[f, t]
            for m in range(M):
                x_1_out[f, t, m] = np.conjugate(W_inv[0, m]) * scalar_signal

    return x_1_out, W, s_1


import os
import numpy as np
import scipy.signal as sig

# Adjust imports based on your exact file structure
from propagation.simulate_acoustics import SimAcoustic
from utils.audio import save_wav
# from wpexsrive import WPExSRIVE, process_frame  # Import your algorithm here

if __name__ == "__main__":
    # 1. GENERAL SETTINGS
    FS = 16000  # 16 kHz is standard for speech processing and matches the paper
    M = 4       # Number of microphones
    speed_of_sound = 343.0 
    
    print("=== INTEGRATION TEST: ONLINE WPExSRIVE ===")
    
    output_folder = "tests/data/wpexsrive_output"
    os.makedirs(output_folder, exist_ok=True)
    
    # 2. ARRAY GEOMETRY DEFINITION (LINEAR ARRAY)
    mic_spacing = 0.021  # 2.1 cm spacing (car environment spacing from paper)
    x = np.linspace(0, (M-1)*mic_spacing, M)
    mic_coords_ideal = np.column_stack([x, np.zeros(M), np.zeros(M)])
    
    array_center = np.array([2.0, 2.0, 1.25])
    mic_coords_ideal = mic_coords_ideal - np.mean(mic_coords_ideal, axis=0) + array_center
    
    # Define angles (Target: 130 deg, Interference: 50 deg)
    r = 1.0 
    ang_target = np.deg2rad(130)
    ang_interf = np.deg2rad(50)
    
    source_pos = array_center + np.array([r * np.cos(ang_target), r * np.sin(ang_target), 0.0])
    interf_pos = array_center + np.array([r * np.cos(ang_interf), r * np.sin(ang_interf), 0.0])

    # 3. ACOUSTIC SCENE SIMULATION (REVERBERANT ROOM)
    print(" -> Initializing reverberant acoustic scene...")
    # Add a small mismatch to make it realistic
    acoustic_scene = SimAcoustic(mic_coords_ideal, array_mismatch=0.002, duration=5, fs=FS)

    source_path = "tools/data/signals/FA01_09.wav"
    int_path = "tools/data/signals/MC15_03.wav"

    acoustic_scene.set_source(source_path, gain=1.0, position=source_pos.reshape(1,3))
    acoustic_scene.set_interference(int_path, gain=1.0, position=interf_pos.reshape(1,3))

    print(" -> Computing Room Impulse Responses (ISB)...")
    room_dimensions = np.array([4.0, 5.0, 2.5])
    # T60 = 0.3s for a moderate office/room reverberation
    room_input_mix = acoustic_scene.compute_room_ISB(
        room_dimensions, 
        desire_RT=0.3, 
        iSIR_dB=0, 
        mode="real"
    )
    
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
    
    # 5. IDEAL STEERING VECTORS FOR SPATIAL REGULARIZATION
    print(" -> Building Ideal Steering Vectors (Target & Interference)...")
    a_1 = np.zeros((F_bins, M), dtype=np.complex128)
    a_2 = np.zeros((F_bins, M), dtype=np.complex128)
    
    for f_idx, freq in enumerate(freqs):
        if freq == 0: 
            a_1[f_idx, :] = 1.0 / np.sqrt(M)
            a_2[f_idx, :] = 1.0 / np.sqrt(M)
        else:
            # Calculate time delays based on linear array geometry
            tau_1 = - np.arange(M) * mic_spacing * np.cos(ang_target) / speed_of_sound
            tau_2 = - np.arange(M) * mic_spacing * np.cos(ang_interf) / speed_of_sound
            
            a_1[f_idx, :] = np.exp(1j * 2 * np.pi * freq * tau_1) / np.sqrt(M)
            a_2[f_idx, :] = np.exp(1j * 2 * np.pi * freq * tau_2) / np.sqrt(M)

    # Transpose to (F, T, M) and ensure contiguous memory layout for Numba
    X_stft_ive = np.transpose(X_stft, (1, 2, 0))
    X_stft_ive = np.ascontiguousarray(X_stft_ive, dtype=np.complex128)

    # 6. WPExSRIVE OPTIMIZATION
    print(" -> Executing WPExSRIVE online processing...")
    # NOTE: Ensure your WPExSRIVE signature matches this call, 
    # or inject a_1 and a_2 inside the algorithm initialization.
    # L=12, D=4 are common WPE parameters for T60 ~ 300ms
    s_hat_stft = WPExSRIVE(X_stft_ive, L=12, D=4) 
    
    # 7. RECONSTRUCTION
    print(" -> Reconstructing time-domain signal (ISTFT)...")
    # No need to transpose if WPExSRIVE returns (F, T) directly
    # If it returns multiple sources (F, T, N), adjust accordingly
    _, y_time = sig.istft(
        s_hat_stft, fs=FS, window='hann', 
        nperseg=nperseg, noverlap=noverlap, nfft=nfft
    )
    
    save_wav("2_wpex_output_target.wav", FS, y_time, output_folder)
    print(" -> Pipeline completed successfully.")