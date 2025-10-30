import numpy as np
from scipy.constants import speed_of_sound


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