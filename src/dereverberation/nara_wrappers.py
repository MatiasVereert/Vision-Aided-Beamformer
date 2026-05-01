
from nara_wpe.wpe import OnlineWPE
from nara_wpe.utils import stft, istft
from nara_wpe.wpe import online_wpe_step, get_power_online, OnlineWPE
from nara_wpe.wpe import wpe # Importamos la versión Batch/Offline
from nara_wpe.utils import stft, istft
import numpy as np
import numpy as np
# Asumo que importas stft, istft, online_wpe_step y get_power de nara_wpe

def process_wpe_online(u, taps=5, delay=1, alpha=0.9999, stft_size=256, stft_shift=64):
    """
    Online WPE wrapper (Functional Approach).
    Processes a multichannel time-domain signal frame by frame to simulate 
    online dereverberation. Bypasses the buggy OnlineWPE class state management 
    by handling the Q and G matrices directly.
    """
    # 1. Transform to STFT domain
    Y = stft(u, size=stft_size, shift=stft_shift)
    Y = Y.transpose(1, 2, 0)  # Shape: (frames, bins, channels)
    T, F, M = Y.shape
    
    buffer_target_size = taps + delay + 1
    if T < buffer_target_size:
        print("Warning: Signal is too short for WPE with given taps and delay.")
        return u
        
    # 2. Initialize Q (Inverse Correlation) and G (Filter) matrices manually
    # Q shape: (F, M*taps, M*taps) -> Identity matrices
    Q = np.stack([np.identity(M * taps) for _ in range(F)])
    # G shape: (F, M*taps, M) -> Zeros
    G = np.zeros((F, M * taps, M))
    
    Z_list = []
    
    # 3. Bypass the first unprocessed frames to maintain strict temporal alignment
    for i in range(taps + delay):
        Z_list.append(Y[i, :, :])
        
    # Initialize the sliding buffer with the first history chunk
    buffer = list(Y[:taps + delay, :, :])
    
    # 4. Process frame by frame
    for t in range(taps + delay, T):
        buffer.append(Y[t, :, :])
        
        # Convert buffer to numpy array: shape (buffer_target_size, F, M)
        Y_step = np.array(buffer)
        
        # Compute power. get_power_online expects (bins, channels, frames)
        power = get_power_online(Y_step.transpose(1, 2, 0))
        
        # Perform functional online dereverberation step
        Z_frame, Q, G = online_wpe_step(
            Y_step,
            power,
            Q,
            G,
            alpha=alpha,
            taps=taps,
            delay=delay
        )
        
        Z_list.append(Z_frame)
        
        # Discard the oldest frame to slide the window forward
        buffer.pop(0)
            
    # 5. Reconstruct the time-domain signal
    Z_stacked = np.stack(Z_list)
    
    # Transpose back to (channels, frames, frequency_bins) for istft
    Z_out = Z_stacked.transpose(2, 0, 1)
    
    # Inverse STFT to get the time-domain audio
    z_time = istft(Z_out, size=stft_size, shift=stft_shift)
    
    # Ensure the output length exactly matches the original input length 
    z_time = z_time[:, :u.shape[1]]
    
    return z_time