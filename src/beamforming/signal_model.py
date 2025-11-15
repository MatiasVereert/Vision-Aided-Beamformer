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
    # Distancia de cada punto de fuente al origen (d_s en el paper)
    source_dist_origin = np.linalg.norm(Rs, axis=1) # Shape: (P,)
    # Distancia de cada punto de fuente a cada micrófono (d_m en el paper)
    mic_distances = np.linalg.norm(Rs[:, np.newaxis, :] - mic_array[np.newaxis, :, :], axis=2) # Shape: (P, M)
    
    # --- Delays ---
    mic_delay = mic_distances / c # Shape: (P, M)
    source_delay_origin = source_dist_origin / c # Shape: (P,)
    T = 1/fs
    tap_delays = np.arange(K) * T # Shape: (K,)

    # --- CORRECCIÓN CLAVE: Añadir el retardo de referencia del centro del filtro ---
    ref_delay = (K - 1) / (2 * fs)
    
    # --- Cálculo del Steering Vector (CORREGIDO) ---
    # Fase = 2*pi*f * ( (d_m - d_s)/c + k/fs )
    # El paper no usa el ref_delay, pero es necesario para que g sea consistente.
    # La fase correcta que es consistente con g es: 2*pi*f * (ref_delay - (d_m - d_s)/c - k/fs)
    # Esto es equivalente a: 2*pi*f * (ref_delay + d_s/c - d_m/c - k/fs)
    f_bcast = f.reshape(F, 1, 1, 1)
    phase_term = np.exp(1j * 2 * np.pi * f_bcast * (ref_delay + source_delay_origin[np.newaxis, :, np.newaxis, np.newaxis] - mic_delay[np.newaxis, :, :, np.newaxis] - tap_delays[np.newaxis, np.newaxis, np.newaxis, :]))
    
    steering_vector = phase_term / mic_distances[np.newaxis, :, :, np.newaxis]
    
    # --- Reshape Final (sin cambios) ---
    final_sv = steering_vector.reshape(F, P, M * K)
    
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

    # Distancia de la fuente al origen (d_s)
    source_dist_origin = np.linalg.norm(Rs)
    # Distancia de la fuente a cada micrófono (d_m)
    distances = np.linalg.norm(Rs - mic_array, axis=1).reshape(-1, 1)

    # --- Delays ---
    mic_delay = distances/c 
    source_delay_origin = source_dist_origin / c
    T = 1/fs
    tap_delays = np.arange(K) * T

    # --- CORRECCIÓN CLAVE: Añadir el retardo de referencia del centro del filtro ---
    ref_delay = (K - 1) / (2 * fs)

    # --- Cálculo del Steering Vector (CORREGIDO según Ecuación 1 del paper) ---
    # Fase = 2*pi*f * (ref_delay + d_s/c - d_m/c - k/fs)
    phase_term = np.exp(1j * 2 * np.pi * f * (ref_delay + source_delay_origin - mic_delay - tap_delays))
    
    steering_vector = phase_term / distances

    #Colapsing the matriz into colums 
    steering_vector_flat = steering_vector.reshape(-1,1)

    return steering_vector_flat
