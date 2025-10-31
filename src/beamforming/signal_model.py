import numpy as np
from scipy.constants import speed_of_sound

def near_field_steering_vector_multi(f, Rs, fs, mic_array, K=1, c=speed_of_sound):
    """
    Calcula los steering vectors de campo cercano para múltiples frecuencias y puntos de la fuente.
    Esta versión está corregida para manejar el broadcasting de dimensiones correctamente.
    """
    f = np.atleast_1d(f)
    Rs = np.atleast_2d(Rs)
    
    F = f.shape[0]
    P = Rs.shape[0]
    M = mic_array.shape[0]
    
    # --- Cálculos dependientes de Rs (dimensión de Puntos, P) ---
    source_distances = np.linalg.norm(Rs, axis=1) # Shape: (P,)
    mic_distances = np.linalg.norm(Rs[:, np.newaxis, :] - mic_array[np.newaxis, :, :], axis=2) # Shape: (P, M)
    
    # --- CORRECCIÓN: Construir el factor de normalización con la forma correcta ---
    
    # 1. Combinar frecuencias y distancias para obtener una matriz (F, P)
    #    f[:, np.newaxis] -> (F, 1)
    #    source_distances[np.newaxis, :] -> (1, P)
    #    El broadcasting resulta en (F, P)
    phase_ref_2d = np.exp(1j * 2 * np.pi * f[:, np.newaxis] * source_distances[np.newaxis, :] / c)
    
    # 2. Calcular el factor de normalización, que también será (F, P)
    norm_factor_2d = source_distances[np.newaxis, :] / phase_ref_2d
    
    # 3. ### LA CLAVE DE LA SOLUCIÓN ###
    #    Añadir dos dimensiones "dummy" al final para que tenga la forma (F, P, 1, 1)
    #    Esta forma SÍ es compatible para el broadcasting con (F, P, M, K)
    norm_factor = norm_factor_2d[:, :, np.newaxis, np.newaxis]

    # --- Delays (sin cambios) ---
    mic_delay = mic_distances / c # Shape: (P, M)
    T = 1 / fs
    tab_delay = np.arange(K) * T # Shape: (K,)
    total_delay = mic_delay[:, :, np.newaxis] + tab_delay[np.newaxis, np.newaxis, :] # Shape: (P, M, K)
    
    # --- Cálculo del Steering Vector (sin cambios) ---
    # Broadcasting de f (F,1,1,1) con total_delay (1,P,M,K) -> (F, P, M, K)
    f_bcast = f.reshape(F, 1, 1, 1)
    phase_term = np.exp(-1j * 2 * np.pi * f_bcast * total_delay[np.newaxis, ...])
    steering_vector = phase_term / mic_distances[np.newaxis, :, :, np.newaxis]
    
    # --- Aplicar la normalización (esta línea ahora funciona) ---
    # Broadcasting: (F, P, 1, 1) * (F, P, M, K) -> (F, P, M, K)
    normalized_sv = norm_factor * steering_vector
    
    # --- Reshape Final (sin cambios) ---
    final_sv = normalized_sv.reshape(F, P, M * K)
    
    return final_sv

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

