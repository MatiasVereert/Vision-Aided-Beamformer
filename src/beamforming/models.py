import numpy as np
from scipy.constants import speed_of_sound

def near_field_steering_vector(f, Rs, fs, mic_array, K=1, c=speed_of_sound):
    """
    Calculates the near-field steering vector for an array of frequencies.

    This function is vectorized to efficiently compute steering vectors for multiple
    frequencies at once using NumPy broadcasting.

    Args:
        f (np.ndarray or float): Array of frequencies (shape F,) or a single frequency in Hz.
        fs (int): Sampling frequency of the signal (to determine tap length).
        Rs (np.ndarray): Source location (x, y, z), the focal point.
        K (int): Number of taps.
        mic_array (np.ndarray): Array of M microphones with 3D coordinates (x, y, z).
        c (float): Speed of sound in m/s.

    Returns:
        np.ndarray: Steering vectors array of shape (M*K, F). If f is a scalar,
                    the shape is (M*K, 1).
    """
    # 1. Asegurar que 'f' sea al menos un array 1D para consistencia
    f = np.atleast_1d(f)
    F = f.shape[0]

    # 2. Calcular factor de normalización (será un vector de forma (F,))
    source_distance = np.linalg.norm(Rs)
    phase_reference = np.exp(1j * 2 * np.pi * f * source_distance / c)
    normalization_factor = source_distance / phase_reference
    
    # Remodelar para broadcasting: (F, 1, 1)
    normalization_factor = normalization_factor.reshape(F, 1, 1)
    
    # 3. Calcular delays (esto no cambia, no dependen de la frecuencia)
    # Distances of each element respect to the source (shape M, 1)
    distances = np.linalg.norm(Rs - mic_array, axis=1).reshape(-1, 1)
    mic_delay = distances / c  # Shape (M, 1)

    # Tap delays (shape 1, K)
    T = 1 / fs
    tabs = np.arange(K)
    tab_delay = tabs * T # Shape (K,)
    
    # 4. Construir el steering vector usando broadcasting
    # Remodelar f para que tenga la forma (F, 1, 1)
    f_col = f.reshape(F, 1, 1)
    
    # El delay total tiene forma (M, K)
    total_delay = mic_delay + tab_delay.reshape(1, K)

    # Broadcasting: (F, 1, 1) * (M, K) -> (F, M, K)
    phase_term = np.exp(-1j * 2 * np.pi * f_col * total_delay)
    
    # Broadcasting: (F, M, K) / (M, 1) -> (F, M, K)
    steering_vector = phase_term / distances

    # 5. Aplicar la normalización
    # Broadcasting: (F, 1, 1) * (F, M, K) -> (F, M, K)
    normalized_sv = normalization_factor * steering_vector

    # 6. Colapsar las dimensiones M y K y transponer para la forma final (M*K, F)
    # Primero, se aplana la dimensión de M y K para cada frecuencia: (F, M*K)
    reshaped_sv = normalized_sv.reshape(F, -1)
    
    # Luego, se transpone para obtener la forma deseada: (M*K, F)
    final_steering_vector = reshaped_sv.T
    
    return final_steering_vector
'''
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

'''