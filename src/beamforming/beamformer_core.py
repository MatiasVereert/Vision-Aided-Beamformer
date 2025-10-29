import numpy as np
from scipy.constants import speed_of_sound
# No es necesario importar scipy.spatial.distance si no se usa.
# Se eliminaron importaciones no utilizadas como sys, os, matplotlib y scipy duplicado.
from numpy.lib.stride_tricks import sliding_window_view

def near_field_steering_vector(f, Rs, fs, mic_array, K=1, c=speed_of_sound):
    """
    Calculathes the steering vector for a specific frecuency
    norlmaliced in the origin of coords. 
        Args: 
        f (np.array or float): Frequency in Hz.
        fs (int): frecuency sample of the signal. (to determin tabs lenght)
        Rs (np.array): Source location (x, y, z), the focal point.
        K(int)_ number of tabs
        mic_array (np.array): Array of M microphones with 3D coordinates (x, y, z).
        c (float): Speed of sound in m/s.
    Returns:
        steering_vector (np.array): returns M.Kx1 array.
    """

    #Calculate the distance and fase relativet to the array, asuming is centered in the origin.
    source_distance = np.linalg.norm(Rs)
    phase_reference = np.exp(1j * 2 * np.pi * f * source_distance / c)
    normalization_factor = source_distance / phase_reference

    #Distances of each element respect to the source (shape Mx1)
    distances = np.linalg.norm(Rs - mic_array, axis=1).reshape(-1, 1)

    #Propagation delay for each microphone (shape Mx1)
    mic_delay = distances/c 

    #tabs delays shape 1xK  
    T = 1/fs
    tabs = np.arange(K)
    tab_delay = tabs * T

    #Defining the propagation vector (Brodcast (Mx1)(1xK)->(MxK))
    steering_vector = np.exp(1j * 2 * np.pi * f * (-mic_delay - tab_delay)) / distances
    normalized_steering_vector = normalization_factor * steering_vector

    #Colapsing the matriz into colums 

    normalized_steering_vector = normalized_steering_vector.reshape(-1,1)


    return normalized_steering_vector


def point_constraint( target_point , K_taps, mic_array , f, fs ):
    #steering vector of the specific point (shape (M.K , 1))
    steering_vector = near_field_steering_vector(f, target_point, fs, mic_array, K=K_taps)
    
    constrains_C = np.hstack([np.real(steering_vector), np.imag(steering_vector)]) 

    # 3. Definir el vector de respuesta deseada h. (Forma: 2 x 1)
    # [Ganancia Real Deseada (1)]
    # [Ganancia Imaginaria Deseada (0)]
    target_gain_h = np.vstack([1.0, 0.0]) # [1, 0]^T


    return constrains_C, target_gain_h


def compute_fixed_weights_optimized(Constrains, Target_gain):
    '''
    Fixed weights computation given by the equation w_q = C(C^[H]C)^-1 h.
    Uses np.linalg.solve for improved numerical stability and efficiency 
    compared to explicit matrix inversion.
    '''
    C_H = Constrains.conj().T
    
    # M = C^H C (La matriz que va a ser invertida)
    M = C_H @ Constrains
    
    # V = h (El vector de la derecha)
    V = Target_gain
    
    # X = (C^H C)^-1 h  --> Resuelve M X = V para X
    X = np.linalg.solve(M, V)
    
    # w_q = C X
    fixed_weights = Constrains @ X

    return fixed_weights

import numpy as np


def snapshots(array_signals, K):
    """
    Creates the snapshot matrix U for a multi-channel time-domain beamformer.

    This function efficiently transforms a 2D array of microphone signals 
    (M mics, N samples) into the 2D snapshot matrix required by a 
    tapped-delay-line (FIR) beamformer. Each column of the output matrix 
    represents a concatenated snapshot vector u(k).

    Args:
        array_signals (np.ndarray): 
            Input signal matrix of shape (M, N_samples). M is the number 
            of microphones, N_samples is the number of time samples.
        K (int): 
            The number of FIR taps per microphone (the window size).

    Returns:
        np.ndarray: 
            The snapshot matrix U of shape (M * K, N_snapshots), where 
            N_snapshots = N_samples - K + 1.
    """
    M, _ = array_signals.shape

    # Create a sliding window view of size K along the time axis (axis=1).
    # This results in a 3D tensor of shape (M, N_snapshots, K).
    signals_window = sliding_window_view(array_signals, window_shape=K, axis=1)

    # Reverse the tap axis to ensure a causal FIR filter structure [x(n), x(n-1), ...].
    # The steering vector is defined for this order.
    reversed_taps = signals_window[:, :, ::-1]

    # Reorder axes to group taps by microphone, then reshape to the final
    # (M*K, N_snapshots) matrix by concatenating all tap blocks.
    snapshot_matrix = reversed_taps.transpose(0, 2, 1).reshape(M * K, -1)

    return snapshot_matrix

import numpy as np

def beamforming(signals, weights):
    """
    Applies a time-domain beamformer to a snapshot matrix of signals.

    This function performs the core beamforming operation by taking the dot product
    of the Hermitian transpose of the weight vector with the snapshot matrix. This
    is equivalent to applying a multi-channel FIR filter and summing the result.

    Args:
        signals (np.ndarray): 
            The snapshot matrix U, where each column is a concatenated snapshot
            vector u(k). Expected shape: (N, T), where N = M * K is the total 
            number of weights, and T is the number of time snapshots.
        weights (np.ndarray): 
            The beamformer weight vector w. Expected shape: (N, 1).

    Returns:
        np.ndarray: 
            The single-channel output signal y. Shape: (1, T).
    """
    # Calculate the beamformer output via matrix multiplication (y = w^H * U).
    # The Hermitian transpose (.conj().T) is used for potentially complex weights,
    # but works correctly for real-valued weights as well (where it's just .T).
    y = weights.conj().T @ signals
    
    return y