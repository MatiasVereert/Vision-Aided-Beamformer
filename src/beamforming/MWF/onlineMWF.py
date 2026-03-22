import numpy as np
from numba import njit

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
def online_mwf_numba(y_stft, sv, alpha=0.95, diag_load=1e-3):
    # Online Multichannel Wiener Filter (MWF) with Robust MVDR and Dynamic Noise Tracking
    K, T, M = y_stft.shape
    X_hat_out = np.zeros((K, T), dtype=np.complex128)
    # Output tensor to store the effective spatial weights for the visualization dashboard
    weights_out = np.zeros((K, T, M), dtype=np.complex128)
    
    for k in range(K):
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
            if out_pow < noise_pow:
                noise_pow = 0.8 * noise_pow + 0.2 * out_pow
            else:
                noise_pow = 0.995 * noise_pow + 0.005 * out_pow
            
            # 7. Compute a priori SNR for the Wiener post-filter gain
            snr_prio = max(1e-6, (out_pow - noise_pow) / noise_pow)
            g_wiener = snr_prio / (snr_prio + 1.0)
            
            # 8. Apply MWF final gain and save effective weights for visualization
            w_effective = w_mvdr
            weights_out[k, t, :] = w_effective
            X_hat_out[k, t] = g_wiener * y_out
            
    return X_hat_out, weights_out