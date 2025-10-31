import numpy as np
from scipy.constants import speed_of_sound
from beamforming.models import near_field_steering_vector
from numpy.lib.stride_tricks import sliding_window_view





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