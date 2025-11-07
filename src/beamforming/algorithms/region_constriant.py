import numpy as np 
from scipy.constants import speed_of_sound
from beamforming.signal_model import near_field_steering_vector_multi 
from utils.geometry import cartesian_to_spherical, spherical_to_cartesian


def set_domains(Rs, delta_r, delta_azimut, delta_elevation, fmin, fmax, P, J):

    #Frecuencya array 
    f_array = np.linspace(fmin, fmax, J)
    
    
    Rs_sferical = cartesian_to_spherical(Rs)

    #Generate P random points inside the domain
    radius = np.random.uniform(Rs_sferical[0] - delta_r/2,  Rs_sferical[0] + delta_r/2, P )
    azimut = np.random.uniform(Rs_sferical[1] - delta_azimut/2, Rs_sferical[1]+ delta_azimut/2, P)
    elevation = np.random.uniform( Rs_sferical[2] - delta_elevation/2, Rs_sferical[2]+ delta_elevation/2, P)

    #generate a single vector with all the points in cartesian coordinates 
    points = spherical_to_cartesian(radius, azimut, elevation)

    return f_array, points


def calculate_covariance_matrix(
    freqs: np.ndarray, 
    source_points: np.ndarray, 
    fs: int, 
    mic_array: np.ndarray, 
    K: int, 
    c: float = speed_of_sound
) -> np.ndarray:
    """
    Calculates the Source Sample Covariance Matrix (Rss).

    This function numerically approximates the integral from Equation 12 (Zheng paper),
    which models the average covariance of a signal originating from a
    defined spatial region and frequency band.

    Args:
        freqs (np.ndarray): Array of J frequency samples (Hz). Shape: (J,).
        source_points (np.ndarray): Array of I spatial sample points (m). Shape: (I, 3).
        fs (int): Sampling frequency (Hz), used to determine tap delays.
        mic_array (np.ndarray): Array of M microphone coordinates (m). Shape: (M, 3).
        K (int): Number of taps per microphone (defines the N = M * K dimension).
        c (float, optional): Speed of sound (m/s).

    Returns:
        np.ndarray: The calculated Rss covariance matrix. Shape: (N, N) where N = M * K.
    """
    
    # Get J (number of freqs) and I (number of points)
    num_freqs = len(freqs)
    num_points = len(source_points)
    
    if num_freqs == 0 or num_points == 0:
        raise ValueError("Frequency and source point arrays cannot be empty.")
        
    P_total = num_freqs * num_points 

    # 1. Calculate all steering vectors for all J frequencies and I points.
    #    The function should return a tensor of shape (J, I, N) where N = M * K
    steering_vectors = near_field_steering_vector_multi(
        f=freqs, 
        Rs=source_points, 
        fs=fs, 
        mic_array=mic_array, 
        K=K,  # <-- CORRECTED: The correct K is passed
        c=c
    )
    
    # 2. Calculate Rss using Einstein summation
    #    j: frequency index (J)
    #    i: spatial point index (I)
    #    n, m: indices of the flattened vector (N = M*K)
    #
    #    'jin,jim->nm' implements: Rss[n,m] = sum_j( sum_i( a[j,i,n] * conj(a[j,i,m]) ) )
    
    # CORRECTED: The correct notation for the outer product is '...n' and '...m'
    Rss = np.einsum('jin,jim->nm', steering_vectors, np.conj(steering_vectors))

    # 3. Normalize by the total number of points
    # CORRECTED: Divide by the total number of points
    return Rss / P_total

    