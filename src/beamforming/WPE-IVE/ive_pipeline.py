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

    # Normalize by T out of the time loop
    for f in range(F):
        for m in range(M):
            for n in range(M):
                V_z[f, m, n] /= T
    return V_z

@njit
def update_W_z_K1(W, V_z):
    """
    Updates the noise portion W_z of the matrix W for K=1.
    Equivalent to Step 4 of Algorithm 2 (IVE-IP1 / IVE-OC).
    Used during initialization and right before Projection Back.
    """
    F, M, _ = V_z.shape
    
    for f in range(F):
        # 1. Extract w_1^h (the current filter conjugate transposed, 1D array of size M)
        w_1_h = np.conjugate(W[f, :, 0])
        
        # 2. Multiply w_1^h * V_z(f)
        temp = np.dot(w_1_h, V_z[f])
        
        # 3. Apply E_s: extract the first element (A is a complex scalar)
        A = temp[0]
        
        # 4. Apply E_z: extract the remaining M-1 elements (B is a 1D array)
        B = temp[1:]
        
        # 5. Calculate (A^-1) * B
        C = B / A
        
        # 6. Build the W_z matrix in-place within W
        W[f, 0, 1:] = C
        
        # The remaining M-1 rows of W_z form the negative identity matrix (-I_{M-1})
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
    Updates the extraction filter w_1 for K=1 using IVE-IP2 (Algorithm 3).
    Solves the generalized eigenvalue problem via the power iteration method
    to maintain full compatibility with Numba's JIT compilation.
    """
    F, M, _ = V_1.shape
    
    for f in range(F):
        V_z_inv = np.linalg.inv(V_z[f])
        A = np.dot(V_z_inv, V_1[f])
        
        u = W[f, :, 0]
        
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
def ive(x_stft):
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

    # Compute static noise covariance
    V_z = calculate_V_z(x_stft)

    # Initial update of W_z (with K=1)
    W = update_W_z_K1(W, V_z)
    
    # Variables for K=1 (first source extraction)
    s_1 = np.zeros((F, T), dtype=np.complex128)
    r_1 = np.zeros(T, dtype=np.float64)
    phi = np.zeros(T, dtype=np.float64)
    V_1 = np.zeros((F, M, M), dtype=np.complex128)
    
    # ---- Optimization Loop (Majorization-Minimization) ----
    for iter in range(N_iter):

        # 1. Reset V_1 at the beginning of each iteration
        for f in range(F):
            for m in range(M):
                for n in range(M):
                    V_1[f, m, n] = 0.0 + 0.0j
            
        # 2. Extract raw signal: s_1(f, t) <- w_1(f)^h x(f, t)
        for f in range(F):
            for t in range(T):
                suma_esp = 0.0 + 0.0j
                for m in range(M):
                    suma_esp += np.conjugate(W[f, m, 0]) * x_stft[f, t, m]
                s_1[f, t] = suma_esp

        # 3. Compute amplitude: r_1(t) <- ||s_1(t)||    
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
        
        # 4. Calculate statistical weights: phi(t)
        for t in range(T):
            denominator = alpha_1_beta * (r_1[t] + 1e-15)**(2-beta)
            phi[t] = (beta / 2.0) / denominator

        phi_min = np.min(phi)
        max_threshold_max = 100000.0 * phi_min

        # Clipping Constraint for numerical stability
        for t in range(T):
            if phi[t] > max_threshold_max:
                phi[t] = max_threshold_max

        # 5. Compute time-weighted target covariance: V_1
        for f in range(F):
            for t in range(T):
                for m in range(M):
                    for n in range(M):
                        V_1[f,m,n] += phi[t] * x_stft[f,t, m] * np.conjugate(x_stft[f,t, n]) / T
            
            # Regularization: Add a fraction of the trace to the diagonal
            tr_V_1 = 0.0      
            for m in range(M):
                tr_V_1 += V_1[f,m,m]

            for m in range(M):
                V_1[f, m, m] += 1e-3 * tr_V_1
        
        # 6. Update the separation filter using the generalized eigenvalue problem
        W = update_W_IP2_K1(W, V_1, V_z, n_power_iter=3)

    # ---- Post-Processing (Source Extraction & Projection Back) ----
    
    # 1. Final update of W_z to ensure full matrix orthogonality
    W = update_W_z_K1(W, V_z)

    # 2. Allocate output array for the spatially projected signal
    x_1_out = np.zeros((F, T, M), dtype=np.complex128)

    # 3. Projection Back
    # The term W(f)^-h e_1 extracts the first column of the inverse conjugate transpose.
    # Mathematically, this is equivalent to the conjugate of the first row of W(f)^-1.
    for f in range(F):
        W_inv = np.linalg.inv(W[f])
        for t in range(T):
            scalar_signal = s_1[f, t]
            for m in range(M):
                # Apply the acoustic path mapping to restore scale and phase
                x_1_out[f, t, m] = np.conjugate(W_inv[0, m]) * scalar_signal

    return x_1_out, W, s_1


import numpy as np
import time

# (Insert here the previously defined functions: 
# calculate_V_z, update_W_z_K1, update_W_IP2_K1, and ive)

if __name__ == "__main__":
    # 1. Define dummy dimensions for the STFT spectrogram
    # F: Frequency bins (e.g., 257 for NFFT=512)
    # T: Time frames (e.g., 100 frames)
    # M: Microphones (e.g., 4 mics)
    F, T, M = 257, 100, 4
    
    print(f"Generating dummy STFT data with shape (F={F}, T={T}, M={M})...")
    
    # 2. Generate random complex data to simulate a microphone array mixture
    # We use normal distribution for real and imaginary parts
    np.random.seed(42)
    real_part = np.random.randn(F, T, M)
    imag_part = np.random.randn(F, T, M)
    x_stft_dummy = real_part + 1j * imag_part
    
    # Ensure strict dtype compatibility with Numba (complex128)
    x_stft_dummy = x_stft_dummy.astype(np.complex128)

    # 3. First execution (Includes Numba JIT compilation time)
    print("\nStarting IVE algorithm (First run includes JIT compilation overhead)...")
    start_time = time.time()
    
    # Call the main function
    x_1_out, W_final, s_1_raw = ive(x_stft_dummy)
    
    end_time = time.time()
    print(f"Execution finished in {end_time - start_time:.4f} seconds.")

    # 4. Dimension validation
    print("\n--- Shape Validation ---")
    print(f"Input mixture shape:       {x_stft_dummy.shape}")
    print(f"Separated raw signal s_1:  {s_1_raw.shape} -> Expected: ({F}, {T})")
    print(f"Separation matrix W:       {W_final.shape} -> Expected: ({F}, {M}, {M})")
    print(f"Projected output x_1_out:  {x_1_out.shape} -> Expected: ({F}, {T}, {M})")
    
    # 5. Sanity check for NaNs or Infs
    has_nans = np.isnan(x_1_out).any()
    print(f"Contains NaNs?:            {has_nans}")
    
    if not has_nans:
        print("\nSUCCESS: The pipeline ran without critical dimensional or memory errors!")
    else:
        print("\nWARNING: The output contains NaNs. Check numerical stability.")