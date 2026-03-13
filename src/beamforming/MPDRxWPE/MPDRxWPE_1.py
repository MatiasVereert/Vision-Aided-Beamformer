import numpy as np 
from matplotlib import pyplot as plt
from numba import njit
from beamforming.MPDRxWPE.signal_model import compute_rtf_steering_vector


#
@njit
def procces_frame(y_frame, fs,):

    return X_hat_frame


def MPDRxWPE(y_stft, mic_array, Rs, fs, frecs ):
    # Dimensions
    K, T, M = y_stft.shape

    # Lenght of the temopral Filter
    L = 12 
    # Delay
    Delta = 3 

    # Iniciate Filters
    h = np.zeros((K, M), dtype=np.complex128)
    g = np.zeros((K, M), dtype=np.complex128)

    # Obtain steering_vector
    sv = compute_rtf_steering_vector(frecs, Rs, mic_array, ref_mic_idx=0, mode="near_field", squeeze=True)
    sv = np.reshape(sv, (K,M))
    h = sv / M 
    
    # Covariance Matrix
    R_h_inv = np.eye((K, M), dtype=np.complex128)
    R_g_inv = np.eye((K, L), dtype=np.complex128)

    R_h_inv = R_h_inv * 1e-4
    R_g_inv = R_g_inv * 1e-2

    # Obvservarion Vector 
    y_bar = np.zeros((K, L * M), dtype=np.complex128)
    y_buffer = np.zeros((K, (L + Delta) * M), dtype=np.complex128) 


    # Temporal loop
    for l in range(T):
        y_frame = y_stft[:, l, :]

        # Shift buffer values
        for k in range(K):
            for i in range(len(y_buffer)-1, M -1 , -1 ):
                y_buffer[k, i] = y_buffer[k, i - 1]
            
            # Set actual frame 
            for m in range(M):
                y_buffer[k, m] = y_frame[k, m]









