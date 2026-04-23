import numpy as np
from scipy import signal
from numpy.lib.stride_tricks import sliding_window_view

def create_tapped_delay_line(X, K, Delta_frames, axis=2):
    T = X.shape[axis]
    dim = X.ndim
    pad_with = [(Delta_frames + K - 1, 0) if i == axis else (0, 0) for i in range(dim)]
    
    Y = np.pad(X, pad_with, 'constant', constant_values=0)
    Y_delays = sliding_window_view(Y, K, axis=axis) 
    Y_delays_T = np.swapaxes(Y_delays, axis, -1)
    Y_window_view = np.flip(Y_delays_T, axis=axis)
    
    return Y_window_view[..., :T]

def get_L1_and_L2(M, L_target, max_diff_pct=0.3):
    L = L_target
    best_overall = None
    min_cost = float('inf')
    
    # Search loop with a safety limit to prevent infinite loops
    while L <= L_target + 50:
        L_M = M * L
        
        # Find factors closest to the square root of L_M
        l1 = int(np.sqrt(L_M))
        while L_M % l1 != 0:
            l1 -= 1
        l2 = L_M // l1
        
        diff_pct = abs(l1 - l2) / max(l1, l2)
        cost = l1**2 + l2**2
        
        if diff_pct <= max_diff_pct:
            return L, l1, l2
            
        # Track the best fallback configuration just in case
        if cost < min_cost:
            min_cost = cost
            best_overall = (L, l1, l2)
            
        L += 1
        
    return best_overall

def dereverb_kawpe_mimo(x, fs, L_target):
    # Base configuration
    n_window = 1024
    n_overlap = 768
    Delta_frames = 6
    P = 1   
    M = x.shape[0]  

    L, L1, L2 = get_L1_and_L2(M, L_target, max_diff_pct=0.3)
    print(f" Starting KAWPE with L = {L},L1 = {L1}, L2 = {L2} ")

    # HARDCODED
    n_window = 512
    n_overlap = 384
    Delta_frames = 7
    P = 2   
    M = x.shape[0]  

    L = 16
    L1 = 16
    L2 = 12

    # RLS e hiperparámetros estáticos 
    alpha = 0.99 
    eps = 1e-6
    bin_power_threshold = 1e-7 
    vad_power_threshold = 1e-5 
    
    # Límite de pre-entrenamiento para PTV-KAWPE (5 segundos) [cite: 460]
    freeze_frame = int(5.0 * fs / (n_window - n_overlap)) 

    original_length = x.shape[1]

    # STFT mapping
    f_bins, t, Y_stft = signal.stft(x, fs=fs, nperseg=n_window, 
                                    noverlap=n_overlap, window='hann', axis=1)
    Y = Y_stft.transpose(1, 0, 2)
    F, _, N = Y.shape

    # Tensor Initialization 
    G_1_underline = np.zeros((F, M, P * L1), dtype=complex)
    g_2_underline = np.zeros((F, M, P * L2), dtype=complex)
    
    epsilon_init = 0.5 # Valor de inicialización de g_2 según la literatura [cite: 363]
    for p in range(P):
        g_2_underline[:, :, p * L2 + p] = epsilon_init + 0j
        
    y_bar_4d = create_tapped_delay_line(Y, K=L, Delta_frames=Delta_frames, axis=2)

    # Inicialización directa de la inversa (Regularización de Tikhonov) 
    Phi_2_inv = np.zeros((F, M, P * L1, P * L1), dtype=complex)
    Phi_1_inv = np.zeros((F, M, P * L2, P * L2), dtype=complex)
    for m in range(M):
        for f in range(F):
            Phi_2_inv[f, m] = np.eye(P * L1)
            Phi_1_inv[f, m] = np.eye(P * L2)

    S_hat = np.zeros((F, M, N), dtype=complex)

    # Iterative RLS Loop
    for n in range(N):
        print(f"Step {n} of {N}", end = "\r")
        Y_n = Y[:, :, n]
        
        Y_multichannel = y_bar_4d[:, :, :, n] 
        Y_bar = Y_multichannel.transpose(0, 2, 1).reshape(F, L * M)
        Y_mat = Y_bar.reshape(F, L2, L1).transpose(0, 2, 1)

        # Energetic VAD and Sub-band Energy verification
        frame_power = np.mean(np.abs(Y_n)**2)
        is_active = frame_power > vad_power_threshold
        
        bin_energy = np.mean(np.abs(Y_n)**2, axis=1)
        valid_bins = bin_energy > bin_power_threshold



        # ==========================================
        # SUBFILTER 1
        # ==========================================
        g_2_reshaped = g_2_underline.reshape(F, M, P, L2)
        y_2_tensor = np.einsum('fij, fmpj -> fmpi', Y_mat, g_2_reshaped.conj())
        y_2_underline = y_2_tensor.reshape(F, M, P * L1) 

        y_2_pred = np.einsum('fmi, fmi -> fm', G_1_underline.conj(), y_2_underline)
        S_1_hat = Y_n - y_2_pred 

        # RLS updates conditionally bypassed by VAD
        if is_active:
            lambda_1 = np.maximum(np.abs(S_1_hat)**2, eps)
            inv_y_2 = np.einsum('fmij, fmj -> fmi', Phi_2_inv, y_2_underline)
            den_2 = alpha * lambda_1 + np.einsum('fmi, fmi -> fm', y_2_underline.conj(), inv_y_2).real
            kappa_2 = inv_y_2 / den_2[:, :, None]

            if np.any(valid_bins):
                k2_v, y2_v = kappa_2[valid_bins], y_2_underline[valid_bins]
                phi2_v, s1_v = Phi_2_inv[valid_bins], S_1_hat[valid_bins]
                
                y2_phi = np.einsum('vmi, vmij -> vmj', y2_v.conj(), phi2_v)
                Phi_2_inv[valid_bins] = (phi2_v - np.einsum('vmi, vmj -> vmij', k2_v, y2_phi)) / alpha
                G_1_underline[valid_bins] += np.einsum('vmi, vm -> vmi', k2_v, s1_v.conj())
        # SUBFILTER 2 (Estimación pre-congelada PTV-KAWPE) 
        # ==========================================
        if n < freeze_frame:
            G_1_reshaped = G_1_underline.reshape(F, M, P, L1)
            Y_mat_T = Y_mat.transpose(0, 2, 1) 
            
            y_1_tensor = np.einsum('fji, fmpi -> fmpj', Y_mat_T, G_1_reshaped.conj())
            y_1_underline = y_1_tensor.reshape(F, M, P * L2) 

            y_1_pred = np.einsum('fmi, fmi -> fm', g_2_underline.conj(), y_1_underline)
            S_2_hat = Y_n - y_1_pred 

            if is_active:
                lambda_2 = np.maximum(np.abs(S_2_hat)**2, eps)
                inv_y_1 = np.einsum('fmij, fmj -> fmi', Phi_1_inv, y_1_underline)
                den_1 = alpha * lambda_2 + np.einsum('fmi, fmi -> fm', y_1_underline.conj(), inv_y_1).real
                kappa_1 = inv_y_1 / den_1[:, :, None]

                if np.any(valid_bins):
                    k1_v, y1_v = kappa_1[valid_bins], y_1_underline[valid_bins]
                    phi1_v, s2_v = Phi_1_inv[valid_bins], S_2_hat[valid_bins]

                    y1_phi = np.einsum('vmi, vmij -> vmj', y1_v.conj(), phi1_v)
                    Phi_1_inv[valid_bins] = (phi1_v - np.einsum('vmi, vmj -> vmij', k1_v, y1_phi)) / alpha
                    g_2_underline[valid_bins] += np.einsum('vmi, vm -> vmi', k1_v, s2_v.conj())

            S_hat[:, :, n] = S_2_hat
        else:
            # PTV-KAWPE asume g_2_underline invariante, la predicción depende solo del Subfiltro 1
            S_hat[:, :, n] = S_1_hat

    # MIMO iSTFT
    _, s_hat_time = signal.istft(S_hat.transpose(1, 0, 2), fs=fs, nperseg=n_window, 
                                 noverlap=n_overlap, window='hann')
    
    return s_hat_time[:, :original_length]

if __name__ == "__main__":
    fs = 16000
    duration = 4.0
    M = 12
    T_samples = int(fs * duration)

    np.random.seed(42)
    t = np.arange(T_samples) / fs
    source_signal = signal.chirp(t, f0=100, f1=4000, t1=duration, method='logarithmic')
    
    x = np.zeros((M, T_samples))
    for m in range(M):
        delay = m * 5 
        attenuation = 1.0 / (m + 1)
        x[m, delay:] = source_signal[:T_samples - delay] * attenuation
        
    x += np.random.normal(0, 0.01, size=(M, T_samples))

    print(f"Running MIMO KAWPE. Input shape: {x.shape} at {fs} Hz")
    s_hat_time = dereverb_kawpe_mimo(x, fs, 12)
    print(f"Success. Output shape: {s_hat_time.shape}")