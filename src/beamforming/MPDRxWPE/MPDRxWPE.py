import numpy as np
from beamforming.signal_model import compute_rtf_steering_vector
import scipy.signal as signal
from numba import njit
from numba import prange

def compute_rtf_steering_vector(f, Rs, mic_array, ref_mic_idx=0, c=343.0, mode="near_field", squeeze=True):
    """
    Computes the Relative Transfer Function (RTF) steering vector in the frequency domain.
    Aligns with the formulation d(l,k) = H_m(l,k) / H_ref(l,k) from the paper.
    """
    f = np.atleast_1d(f)
    Rs = np.atleast_2d(Rs)
    
    F = f.shape[0]
    P = Rs.shape[0]
    M = mic_array.shape[0]
    
    # Calculate Euclidean distance from each source point to each microphone
    # Shape: (P, M)
    mic_dist = np.linalg.norm(Rs[:, np.newaxis, :] - mic_array[np.newaxis, :, :], axis=2)
    
    # Extract the distance from each source to the designated fixed reference microphone
    # Shape: (P, 1)
    ref_dist = mic_dist[:, ref_mic_idx, np.newaxis]
    
    # Calculate the path difference relative to the reference microphone
    # Shape: (P, M)
    delta_dist = mic_dist - ref_dist
    
    # Reshape arrays for correct NumPy broadcasting across frequencies (F), sources (P), and mics (M)
    f_bcast = f[:, np.newaxis, np.newaxis]
    delta_dist_bcast = delta_dist[np.newaxis, :, :]
    
    # Compute the relative phase delay
    # phase = exp(-j * 2 * pi * f * (d_m - d_ref) / c)
    phase_term = np.exp(-1j * 2 * np.pi * f_bcast * delta_dist_bcast / c)
    
    if mode == "near_field":
        # In near-field, amplitude decays with 1/r. 
        # The relative amplitude ratio is (1/d_m) / (1/d_ref) = d_ref / d_m
        amp_ratio = ref_dist[np.newaxis, :, :] / mic_dist[np.newaxis, :, :]
        rtf_vector = amp_ratio * phase_term
    else:
        # In far-field, we assume plane waves where amplitude attenuation across the array is negligible
        rtf_vector = phase_term

    if squeeze:
        rtf_vector = np.squeeze(rtf_vector)

    return rtf_vector

"""

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

"""

    
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
        R_h_inv = np.eye(L, dtype=np.complex128) * 1e-1
        R_g_inv = np.eye(M, dtype=np.complex128) * 1e-0
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
def MPDRxWPE_numba_warm(y_stft, sv, Gamma_diffuse, alpha_steady=0.994, alpha_init=0.500, tau=20.0, L=20, Delta=6, epsilon=1e-3, save_weights=False):
    # Optimized MPDR-WPE bilinear framework using Numba.
    # Incorporates Diffuse Field Initialization and Dynamic Forgetting Factor (Warm-up).
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
        R_h_inv = np.eye(L, dtype=np.complex128) * 1e-0
        I_M = np.eye(M, dtype=np.complex128)
        
        # Diffuse Field Initialization for R_g
        # R_g_init = sigma^2 * Gamma_diffuse + epsilon * I_M
        R_g_init = 1e-2 * Gamma_diffuse[k] + epsilon * I_M
        R_g_inv = np.linalg.inv(R_g_init)
        
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
                
            # Dynamic Forgetting Factor (Exponential Warm-up)
            alpha_current = alpha_steady - (alpha_steady - alpha_init) * np.exp(-l_idx / tau)
            
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
            
            # 5. Compute Kalman gain for g (using dynamic alpha)
            num_g = np.dot(R_g_inv, y_bar_g)
            den_g = alpha_current + np.vdot(y_bar_g, num_g)
            k_g = num_g / den_g
            
            # 6. Compute Kalman gain for h (using dynamic alpha)
            num_h = np.dot(R_h_inv, y_bar_h)
            den_h = alpha_current * lambda_l + np.vdot(y_bar_h, num_h)
            k_h = num_h / den_h
            
            # 7 & 8. Update Covariances (using dynamic alpha)
            update_g = np.dot(np.outer(k_g, np.conj(y_bar_g)), R_g_inv)
            R_g_inv = (R_g_inv - update_g) / alpha_current
            
            update_h = np.dot(np.outer(k_h, np.conj(y_bar_h)), R_h_inv)
            R_h_inv = (R_h_inv - update_h) / alpha_current
            
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

from numba import njit, prange
import numpy as np

@njit(parallel=True, fastmath=True)
def MPDRxWPE_numba_warm_v1(y_stft, sv, Gamma_diffuse, alpha_steady=0.994, alpha_init=0.500, beta_init=1.0, tau=20.0, L=20, Delta=6, epsilon=1e-3, save_weights=False):
    # Optimized MPDR-WPE bilinear framework using Numba.
    # Incorporates Diffuse Field Initialization and Decoupled Dynamic Forgetting Factors.
    
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
        R_h_inv = np.eye(L, dtype=np.complex128) * 1e-0
        I_M = np.eye(M, dtype=np.complex128)
        
        # Diffuse Field Initialization for R_g
        R_g_init = 1e-2 * Gamma_diffuse[k] + epsilon * I_M
        
        # Force C-contiguous memory layout
        R_g_inv = np.ascontiguousarray(np.linalg.inv(R_g_init))
        
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
                
            # Decoupled Dynamic Forgetting Factors
            # alpha tracks fast initially, beta is frozen initially
            alpha_current = alpha_steady - (alpha_steady - alpha_init) * np.exp(-l_idx / tau)
            beta_current = alpha_steady + (beta_init - alpha_steady) * np.exp(-l_idx / tau)
            
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
            
            # 5. Compute Kalman gain for g (using alpha_current)
            num_g = np.dot(R_g_inv, y_bar_g)
            den_g = alpha_current + np.vdot(y_bar_g, num_g)
            k_g = num_g / den_g
            
            # 6. Compute Kalman gain for h (using beta_current for temporal stability)
            num_h = np.dot(R_h_inv, y_bar_h)
            den_h = beta_current * lambda_l + np.vdot(y_bar_h, num_h)
            k_h = num_h / den_h
            
            # 7. Update Covariance for spatial filter (using alpha_current)
            update_g = np.dot(np.outer(k_g, np.conj(y_bar_g)), R_g_inv)
            R_g_inv = np.ascontiguousarray((R_g_inv - update_g) / alpha_current)
            
            # 8. Update Covariance for temporal filter (using beta_current)
            update_h = np.dot(np.outer(k_h, np.conj(y_bar_h)), R_h_inv)
            R_h_inv = np.ascontiguousarray((R_h_inv - update_h) / beta_current)
            
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

import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def MPDRxWPE_numba_block_warmup(y_stft, sv, T_warmup=62, alpha=0.994, L=20, Delta=4, epsilon=1e-2, save_weights=False):
    # Optimized MPDR-WPE bilinear framework using Numba.
    # Implements Data-Driven Block Initialization (Warm-up).
    
    K, T, M = y_stft.shape
    epsilon_inv = 1.0 / epsilon
    
    # Pre-allocate the output array
    X_hat_out = np.zeros((K, T), dtype=np.complex128)
    buffer_len = (L + Delta) * M
    h_register = np.zeros((K, T, M), dtype=np.complex128) if save_weights else np.zeros((1, 1, 1), dtype=np.complex128)

    # RLS steady-state scaling factor for inverse covariance matrices
    scale_factor = 1.0 - alpha

    # Outer loop parallelized over frequency bins
    for k in prange(K):
        I_M = np.eye(M, dtype=np.complex128)
        I_L = np.eye(L, dtype=np.complex128)
        
        # ---------------------------------------------------------
        # PHASE 1: SPATIAL BLOCK INITIALIZATION
        # ---------------------------------------------------------
        R_g_empirical = np.zeros((M, M), dtype=np.complex128)
        
        # Accumulate empirical spatial covariance over the warmup block
        for l_idx in range(T_warmup):
            y_frame = y_stft[k, l_idx, :]
            for i in range(M):
                for j in range(M):
                    R_g_empirical[i, j] += y_frame[i] * np.conj(y_frame[j])
                    
        # Average and regularize
        R_g_empirical = (R_g_empirical / T_warmup) + (epsilon * I_M)
        R_g_inv_empirical = np.ascontiguousarray(np.linalg.inv(R_g_empirical))
        
        # Calculate optimal static spatial filter for the block
        d = sv[k]
        num_h_init = np.dot(R_g_inv_empirical, d)
        h = num_h_init / np.vdot(d, num_h_init)
        
        # Scale inverse matrix to transition into the RLS loop smoothly
        R_g_inv = np.ascontiguousarray(scale_factor * R_g_inv_empirical)
        
        # ---------------------------------------------------------
        # PHASE 2: TEMPORAL BLOCK INITIALIZATION
        # ---------------------------------------------------------
        R_h_empirical = np.zeros((L, L), dtype=np.complex128)
        p_h_empirical = np.zeros(L, dtype=np.complex128)
        
        y_buffer_init = np.zeros(buffer_len, dtype=np.complex128)
        y_bar_init = np.zeros(L * M, dtype=np.complex128)
        
        # Filter the warmup block using the static spatial filter to estimate temporal parameters
        for l_idx in range(T_warmup):
            y_frame = y_stft[k, l_idx, :]
            
            # Update buffer
            for i in range(buffer_len - 1, M - 1, -1):
                y_buffer_init[i] = y_buffer_init[i - M]
            for m in range(M):
                y_buffer_init[m] = y_frame[m]
                
            for i in range(L * M):
                y_bar_init[i] = y_buffer_init[i + (Delta * M)]
                
            # Compute filtered history y_bar_h
            y_bar_h = np.zeros(L, dtype=np.complex128)
            for l_tap in range(L):
                summ = 0.0 + 0.0j
                for m in range(M):
                    summ += np.conj(h[m]) * y_bar_init[l_tap * M + m]
                y_bar_h[l_tap] = summ
                
            # Compute static beamformer output
            Z = np.vdot(h, y_frame)
            
            # Accumulate temporal covariance and cross-correlation vector
            for i in range(L):
                p_h_empirical[i] += y_bar_h[i] * np.conj(Z)
                for j in range(L):
                    R_h_empirical[i, j] += y_bar_h[i] * np.conj(y_bar_h[j])
                    
        # Average and regularize temporal covariance
        R_h_empirical = (R_h_empirical / T_warmup) + (1e-4 * I_L)
        p_h_empirical = p_h_empirical / T_warmup
        
        R_h_inv_empirical = np.ascontiguousarray(np.linalg.inv(R_h_empirical))
        g = np.dot(R_h_inv_empirical, p_h_empirical)
        
        # Scale inverse matrix to transition into the RLS loop smoothly
        R_h_inv = np.ascontiguousarray(scale_factor * R_h_inv_empirical)
        
        # ---------------------------------------------------------
        # PHASE 3: BILINEAR RLS ONLINE PROCESSING
        # ---------------------------------------------------------
        # Reset buffer to process from the beginning
        y_buffer = np.zeros(buffer_len, dtype=np.complex128)
        y_bar = np.zeros(L * M, dtype=np.complex128)
        y_bar_g = np.zeros(M, dtype=np.complex128)
        y_bar_h = np.zeros(L, dtype=np.complex128)
        
        for l_idx in range(T):
            y_frame = y_stft[k, l_idx, :]

            if save_weights:
                h_register[k, l_idx, :] = h
                
            # Update buffer
            for i in range(buffer_len - 1, M - 1, -1):
                y_buffer[i] = y_buffer[i - M]
            for m in range(M):
                y_buffer[m] = y_frame[m]
            for i in range(L * M):
                y_bar[i] = y_buffer[i + (Delta * M)]
                
            # Compute y_bar_g
            for m in range(M):
                summ = 0.0 + 0.0j
                for l_tap in range(L):
                    summ += np.conj(g[l_tap]) * y_bar[l_tap * M + m]
                y_bar_g[m] = y_frame[m] - summ
                
            # Compute y_bar_h
            for l_tap in range(L):
                summ = 0.0 + 0.0j
                for m in range(M):
                    summ += np.conj(h[m]) * y_bar[l_tap * M + m]
                y_bar_h[l_tap] = summ
                
            # Compute A Priori Estimate
            X_hat = np.vdot(h, y_bar_g)
            lambda_l = np.abs(X_hat)**2
            X_hat_out[k, l_idx] = X_hat
            
            # ONLY update filters if past the warmup block
            if l_idx >= T_warmup:
                # Compute Kalman gains
                num_g = np.dot(R_g_inv, y_bar_g)
                den_g = alpha + np.vdot(y_bar_g, num_g)
                k_g = num_g / den_g
                
                num_h = np.dot(R_h_inv, y_bar_h)
                den_h = alpha * lambda_l + np.vdot(y_bar_h, num_h)
                k_h = num_h / den_h
                
                # Update Covariances
                update_g = np.dot(np.outer(k_g, np.conj(y_bar_g)), R_g_inv)
                R_g_inv = np.ascontiguousarray((R_g_inv - update_g) / alpha)
                
                update_h = np.dot(np.outer(k_h, np.conj(y_bar_h)), R_h_inv)
                R_h_inv = np.ascontiguousarray((R_h_inv - update_h) / alpha)
                
                # Update temporal filter g
                g = g + k_h * np.conj(X_hat)
                
                # Update regularized covariance and spatial filter h
                inner_inv = np.linalg.inv((epsilon_inv * I_M) + R_g_inv)
                R_sigma_inv = R_g_inv - np.dot(R_g_inv, np.dot(inner_inv, R_g_inv))
                
                num_h_update = np.dot(R_sigma_inv, d)
                h = num_h_update / np.vdot(d, num_h_update)
                
    return X_hat_out, h_register

import os
import scipy.signal as sig

# Adjust imports based on your exact file structure
from propagation.simulate_acoustics import SimAcoustic
from utils.audio import save_wav

# Import the new algorithm and the steering vector function
# from beamforming.MPDRxWPE import MPDRxWPE, compute_rtf_steering_vector 

if __name__ == "__main__":
    # The paper uses a sampling rate of 16 kHz 
    FS = 16000
    # The SPEAR challenge array uses 6 microphones 
    M1, M2 = 12, 1          
    M = M1 * M2
    speed_of_sound = 343.0 
    
    print("=== INTEGRATION TEST: MPDR-WPE BILINEAR FRAMEWORK ===")
    
    output_folder = "tests/data/mpdr_wpe_output"
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
    # We introduce a 5 mm random position error to simulate phase mismatch
    acoustic_scene = SimAcoustic(mic_coords_ideal, array_mismatch=0.005, duration=10, fs=FS)

    source_path = "tools/data/signals/FA01_09.wav"
    int_path1 = "tools/data/signals/MC15_03.wav"

    acoustic_scene.set_source(source_path, gain=1, position=source_pos.reshape(1,3))
    acoustic_scene.set_interference(int_path1, gain=1, position=interf_pos1.reshape(1,3))

    print(" -> Computing free field simulation (mode='real')...")
    # Using mode="real" forces the simulator to use the perturbed microphone positions
    #room_input_real = acoustic_scene.free_field(iSIR_dB=0, normalize=True, mode="ideal")

    print(" -> Computing Room Impulse Responses (ISB)...")
    room_dimensions = np.array([4.0, 5.0, 2.5])
    
    room_input_real = acoustic_scene.compute_room_ISB(iSIR_dB=0, 
                                                    desire_RT=0.5,
                                                    room_dimensions=room_dimensions, 
                                                    mode="ideal")
    


    # 3. APPLY GAIN MISMATCH
    print(" -> Applying Gain Mismatch to microphones...")
    np.random.seed(42) # For reproducibility in testing
    # Generate random gain variations between 0.9 and 1.1 (+/- 10%)
    gain_mismatch = np.random.uniform(0.9, 1.1, (M, 1))
    
    # Apply the gain mismatch to the propagated signals
    room_input_mismatched = room_input_real * gain_mismatch
    
    save_wav("1_input_mix_mic0_mismatched.wav", FS, room_input_mismatched[0], output_folder)
    
    # 4. SHORT-TIME FOURIER TRANSFORM
    # The paper uses 1024 samples with 75% overlap and a Hamming window 
    nperseg = 1024
    noverlap = int(1024 * 0.75)
    nfft = 1024
    
    print(" -> Applying STFT...")
    freqs, times, X_stft = sig.stft(
        room_input_mismatched, fs=FS, window='hamming', 
        nperseg=nperseg, noverlap=noverlap, nfft=nfft 
    )
    
    F_bins = X_stft.shape[1]
    
    # Transpose to (K, T, M) and make contiguous for Numba
    X_stft_mpdr = np.transpose(X_stft, (1, 2, 0))
    X_stft_mpdr = np.ascontiguousarray(X_stft_mpdr, dtype=np.complex128)

    # Steering Vector
    sv = compute_rtf_steering_vector(freqs, 
                                    source_pos, 
                                    mic_coords_ideal, 
                                    ref_mic_idx=0, 
                                    mode="near_field", 
                                    squeeze=True)
    # 5. MPDR-WPE OPTIMIZATION
    print(" -> Executing MPDR-WPE on mismatched signals...")
    Rs_matrix = source_pos.reshape(1, 3)
    
    # Calculate steering vector outside the Numba loop
    sv = compute_rtf_steering_vector(freqs, 
                                     Rs_matrix, 
                                     mic_coords_ideal, 
                                     ref_mic_idx=0, 
                                     mode="near_field", 
                                     squeeze=True)

    # Call the optimized numba function
    X_hat_stft = MPDRxWPE_numba(X_stft_mpdr, sv)
    # 6. RECONSTRUCTION
    print(" -> Reconstructing time-domain signal...")
    
    # Reconstruct the single enhanced channel
    _, y_time = sig.istft(
        X_hat_stft, fs=FS, window='hamming', 
        nperseg=nperseg, noverlap=noverlap, nfft=nfft
    )
    
    save_wav("2_output_MPDR_WPE.wav", FS, y_time, output_folder)
    print(" -> Pipeline completed.")

    
import os
import scipy.signal as sig

# Adjust imports based on your exact file structure
from propagation.simulate_acoustics import SimAcoustic
from utils.audio import save_wav

# Import the new algorithm and the steering vector function
# from beamforming.MPDRxWPE import MPDRxWPE, compute_rtf_steering_vector 

if __name__ == "__main__":
    # The paper uses a sampling rate of 16 kHz 
    FS = 16000
    # The SPEAR challenge array uses 6 microphones 
    M1, M2 = 6, 1          
    M = M1 * M2
    speed_of_sound = 343.0 
    
    print("=== INTEGRATION TEST: MPDR-WPE BILINEAR FRAMEWORK ===")
    
    output_folder = "tests/data/mpdr_wpe_output"
    os.makedirs(output_folder, exist_ok=True)
    
    # 1. IDEAL GEOMETRY DEFINITION
    mic_spacing = 0.021 
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
    # We introduce a 5 mm random position error to simulate phase mismatch
    acoustic_scene = SimAcoustic(mic_coords_ideal, array_mismatch=0.005, duration=10, fs=FS)

    source_path = "tools/data/signals/FA01_09.wav"
    int_path1 = "tools/data/signals/MC15_03.wav"

    acoustic_scene.set_source(source_path, gain=1, position=source_pos.reshape(1,3))
    acoustic_scene.set_interference(int_path1, gain=1, position=interf_pos1.reshape(1,3))

    print(" -> Computing free field simulation (mode='real')...")
    # Using mode="real" forces the simulator to use the perturbed microphone positions
    room_input_real = acoustic_scene.free_field(iSIR_dB=0, normalize=True, mode="real")


 
    
    # 3. APPLY GAIN MISMATCH
    print(" -> Applying Gain Mismatch to microphones...")
    np.random.seed(42) # For reproducibility in testing
    # Generate random gain variations between 0.9 and 1.1 (+/- 10%)
    gain_mismatch = np.random.uniform(0.9, 1.1, (M, 1))
    
    # Apply the gain mismatch to the propagated signals
    room_input_mismatched = room_input_real * gain_mismatch
    
    save_wav("1_input_mix_mic0_mismatched.wav", FS, room_input_mismatched[0], output_folder)
    
    # 4. SHORT-TIME FOURIER TRANSFORM
    # The paper uses 1024 samples with 75% overlap and a Hamming window 
    nperseg = 1024
    noverlap = int(1024 * 0.75)
    nfft = 1024
    
    print(" -> Applying STFT...")
    freqs, times, X_stft = sig.stft(
        room_input_mismatched, fs=FS, window='hamming', 
        nperseg=nperseg, noverlap=noverlap, nfft=nfft
    )
    
    F_bins = X_stft.shape[1]
    
    # Transpose to (K, T, M) and make contiguous for Numba
    X_stft_mpdr = np.transpose(X_stft, (1, 2, 0))
    X_stft_mpdr = np.ascontiguousarray(X_stft_mpdr, dtype=np.complex128)

    # Steering Vector
    sv = compute_rtf_steering_vector(freqs, 
                                    source_pos, 
                                    mic_coords_ideal, 
                                    ref_mic_idx=0, 
                                    mode="near_field", 
                                    squeeze=True)

    # 5. MPDR-WPE OPTIMIZATION
    print(" -> Executing MPDR-WPE on mismatched signals...")
    Rs_matrix = source_pos.reshape(1, 3)
    
    # The algorithm outputs a single enhanced channel of shape (K, T)
    X_hat_stft, _ = MPDRxWPE_numba(X_stft_mpdr, mic_coords_ideal, Rs_matrix, FS, freqs)
    
    # 6. RECONSTRUCTION
    print(" -> Reconstructing time-domain signal...")
    
    # Reconstruct the single enhanced channel
    _, y_time = sig.istft(
        X_hat_stft, fs=FS, window='hamming', 
        nperseg=nperseg, noverlap=noverlap, nfft=nfft
    )
    
    save_wav("2_output_MPDR_WPE.wav", FS, y_time, output_folder)
    print(" -> Pipeline completed.")