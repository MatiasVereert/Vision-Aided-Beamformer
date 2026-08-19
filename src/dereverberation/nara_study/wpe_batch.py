import numpy as np
import scipy.signal as signal
import scipy.linalg as spla

def get_Y_obs(Y_stft, taps, delay):
    # Y_stft dimensions: (F: frequency bins, D: microphones, N: signal length)
    F, D, N_samples = Y_stft.shape

    # 1) Define tapped delay line 
    # Zero pad the time axis. We need (delay + taps - 1) zeros at the beginning
    pad_len = delay + taps - 1
    pad_width = ((0, 0), (0, 0), (pad_len, 0))
    Y_stft_padded = np.pad(Y_stft, pad_width, mode='constant', constant_values=0)

    # Tensor to hold the delayed frames before reshaping
    Y_stft_obs_tensor = np.zeros((F, D, N_samples, taps), dtype=np.complex128)

    # Stack windows
    for k in range(taps):
        # Extract window: shift the starting index backwards for older taps
        start_idx = (taps - 1) - k 
        end_idx = start_idx + N_samples
        Y_stft_obs_tensor[:, :, :, k] = Y_stft_padded[:, :, start_idx:end_idx]

    # Vectorize taps and D dimensions into one
    # Transpose to (F, taps, D, N_samples) to ensure taps are the outer block
    Y_stft_obs_tensor = Y_stft_obs_tensor.transpose(0, 3, 1, 2)
    
    # Reshape collapsing taps and D
    Y_stft_obs = Y_stft_obs_tensor.reshape(F, taps * D, N_samples)

    return Y_stft_obs # shape (F, D*K, N)

def get_X_hat(G, Y_obs, Y_in):
    # Calculate late reverberation using Einstein summation
    Y_late = np.einsum("fkd,fkn->fdn", G.conj(), Y_obs)

    # Subtract the estimated late reverberation to obtain the early signal
    X_hat = Y_in - Y_late

    return X_hat

def get_lamda(X_hat, delta):
    F, D, N_samples = X_hat.shape
    lamda = np.zeros((F, N_samples), dtype=np.float64)
    denom = (2 * delta + 1) * D

    # Nested summation across channels
    X_hat_2 = np.abs(X_hat) ** 2 
    X_hat_2_Dsum = np.sum(X_hat_2, axis=1) # Shape: (F, T)

    # Add zero-padding to evaluate the window at the boundaries
    pad_width = ((0, 0), (delta, delta))
    X_hat_2_Dsum_padded = np.pad(X_hat_2_Dsum, pad_width, mode='constant', constant_values=0)

    # Apply the sliding window summation
    for t in range(N_samples):
        lamda[:, t] = np.sum(X_hat_2_Dsum_padded[:, t : t + 2 * delta + 1], axis=1) / denom

    return lamda 

def get_R(Y_tilde, lamda):
    # Expand lamda to broadcast correctly against Y_tilde
    lamda_expanded = lamda[:, np.newaxis, :]
    
    # Divide the observation by the variance
    Y_tilde_div = Y_tilde / lamda_expanded
    
    # Perform the outer product and sum over the time axis simultaneously
    R = np.einsum('fin,fjn->fij', Y_tilde_div, Y_tilde.conj())
    
    return R 

def get_P(Y_tilde, Y_in, lamda):
    # Expand lamda to broadcast correctly
    lamda_expanded = lamda[:, np.newaxis, :]
    
    # Divide the delayed observation by the variance
    Y_tilde_div = Y_tilde / lamda_expanded
    
    # Perform the outer product and sum over time 
    P = np.einsum('fin,fdn->fid', Y_tilde_div, Y_in.conj())
    
    return P

def get_G_cholesky(R, P, eps=1e-6):
    F, DK, _ = R.shape
    D = P.shape[2]
    
    G = np.zeros((F, DK, D), dtype=complex)
    I = np.eye(DK, dtype=complex)
    
    for f in range(F):
        # Diagonal loading for numerical stability
        R_reg = R[f] + eps * I
        
        try:
            # Cholesky factorization and solve
            c, lower = spla.cho_factor(R_reg, lower=False)
            G[f] = spla.cho_solve((c, lower), P[f])
        except spla.LinAlgError:
            # Fallback to pseudo-inverse if heavily ill-conditioned
            G[f] = np.linalg.pinv(R_reg) @ P[f]
            
    return G

def batch_WPE(Y_in, taps, delay, delta, iterations):
    # Construct the stacked delayed observation matrix
    Y_obs = get_Y_obs(Y_in, taps, delay)
    
    # Initialize the early signal estimate as the input signal
    X_hat = Y_in 

    for i in range(iterations):
        print(f"Iteration numer: {i}")
        # Step 1: Estimate power
        lamda = get_lamda(X_hat, delta)

        # Step 2: Compute correlation matrices
        R = get_R(Y_obs, lamda)
        P = get_P(Y_obs, Y_in, lamda)

        # Compute filter weights using Cholesky decomposition
        G = get_G_cholesky(R, P, eps=1e-6)
        
        # Update the estimated early signal
        X_hat = get_X_hat(G, Y_obs, Y_in)

    return X_hat



def process_wpe_time_domain(audio_time, fs=16000, taps=10, delay=3, delta=2, iterations=3, nperseg=512, noverlap=384):
    """
    Wrapper function to apply WPE dereverberation directly on time-domain audio.
    """
    # Ensure input is at least 2D: (Channels, Samples)
    if audio_time.ndim == 1:
        # Add a channel dimension if it is a mono signal
        audio_time = audio_time[np.newaxis, :]
        
    # 1. Transform to Time-Frequency domain using STFT
    # SciPy's stft returns a tensor of shape (D, F, T) where:
    # D: Channels, F: Frequency bins, T: Time frames
    frequencies, times, Zxx = signal.stft(audio_time, fs=fs, nperseg=nperseg, noverlap=noverlap)
    
    # 2. Reshape for WPE
    # Our batch_WPE function strictly expects the shape (F, D, T)
    Y_in = Zxx.transpose(1, 0, 2)
    
    # 3. Apply the iterative Batch WPE algorithm
    X_hat = batch_WPE(Y_in, taps, delay, delta, iterations)
    
    # 4. Revert the shape for inverse STFT
    # Transpose back from (F, D, T) to (D, F, T) for SciPy compatibility
    Zxx_out = X_hat.transpose(1, 0, 2)
    
    # 5. Transform back to Time domain using iSTFT
    _, audio_dereverb = signal.istft(Zxx_out, fs=fs, nperseg=nperseg, noverlap=noverlap)
    
    return audio_dereverb


    