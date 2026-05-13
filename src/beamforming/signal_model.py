import numpy as np
from scipy.constants import speed_of_sound

def steering_vector(f, Rs, fs, mic_array, K=1, c=speed_of_sound, mode = "near_field" , squeeze = True):
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

    ref_delay = (K - 1) / (2 * fs)
    
    # --- Cálculo del Steering Vector (CORREGIDO) ---
    # Fase = 2*pi*f * ( (d_m - d_s)/c + k/fs )
    # El paper no usa el ref_delay, pero es necesario para que g sea consistente.
    # La fase correcta que es consistente con g es: 2*pi*f * (ref_delay - (d_m - d_s)/c - k/fs)
    # Esto es equivalente a: 2*pi*f * (ref_delay + d_s/c - d_m/c - k/fs)
    f_bcast = f.reshape(F, 1, 1, 1)
    phase_term = np.exp(1j * 2 * np.pi * f_bcast * (ref_delay + source_delay_origin[np.newaxis, :, np.newaxis, np.newaxis] - mic_delay[np.newaxis, :, :, np.newaxis] - tap_delays[np.newaxis, np.newaxis, np.newaxis, :]))
    
    if mode == "near_field":
        steering_vector = phase_term / mic_distances[np.newaxis, :, :, np.newaxis]
    else:#far field case
        steering_vector = phase_term

    # --- Reshape Final (sin cambios) ---
    final_sv = steering_vector.reshape(F, P, M * K)
    
    if squeeze:
        final_sv = np.squeeze(final_sv)

    return final_sv

def near_field_steering_vector(f, Rs, fs, mic_array, K=1, c=speed_of_sound, Normalize = False):
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
    
    steering_vector = phase_term 

    if not Normalize:
        steering_vector = steering_vector / distances

    #Colapsing the matriz into colums 
    steering_vector_flat = steering_vector.reshape(-1,1)

    return steering_vector_flat

def near_field_steering_vector(f, Rs, fs, mic_array, K=1, c=speed_of_sound, normalize=False):
    """
    Calculates the steering vector for a specific frequency
    normalized in the origin of coords. 
        Args: 
        f (np.array or float): Frequency in Hz.
        fs (int): frequency sample of the signal. (to determine taps length)
        Rs (np.array): Source location (x, y, z), the focal point.
        K(int): number of taps
        mic_array (np.array): Array of M microphones with 3D coordinates (x, y, z).
        c (float): Speed of sound in m/s.
        normalize (bool): If True, returns unit-norm vector. If False, includes 1/r attenuation.
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

    # --- Cálculo del Steering Vector ---
    # Fase = 2*pi*f * (ref_delay + d_s/c - d_m/c - k/fs)
    phase_term = np.exp(1j * 2 * np.pi * f * (ref_delay + source_delay_origin - mic_delay - tap_delays))
    
    # Aplicamos la lógica del flag
    if not normalize:
        steering_vector = phase_term / distances
    else:
        # Escala según el artículo para garantizar norma unitaria
        M = mic_array.shape[0]
        steering_vector = phase_term / np.sqrt(M)
    
    # Colapsing the matrix into columns 
    steering_vector_flat = steering_vector.reshape(-1,1)

    return steering_vector_flat

import numpy as np


def compute_rtf_steering_vector(f, Rs, mic_array, ref_mic_idx=0, c=343.0, mode="near_field", squeeze=True):
    """
    Computes the Relative Transfer Function (RTF) steering vector in the frequency domain.
    Aligns with the formulation d(l,k) = H_m(l,k) / H_ref(l,k) from the paper.
    """
    f = np.atleast_1d(f)
    Rs = np.atleast_2d(Rs)
    
    F = f.shape[0]
    P = Rs.shape[0]
    M = mic_array.shape[0]
    
    # Calculate Euclidean distance from each source point to each microphone
    # Shape: (P, M)
    mic_dist = np.linalg.norm(Rs[:, np.newaxis, :] - mic_array[np.newaxis, :, :], axis=2)
    
    # Extract the distance from each source to the designated fixed reference microphone
    # Shape: (P, 1)
    ref_dist = mic_dist[:, ref_mic_idx, np.newaxis]
    
    # Calculate the path difference relative to the reference microphone
    # Shape: (P, M)
    delta_dist = mic_dist - ref_dist
    
    # Reshape arrays for correct NumPy broadcasting across frequencies (F), sources (P), and mics (M)
    f_bcast = f[:, np.newaxis, np.newaxis]
    delta_dist_bcast = delta_dist[np.newaxis, :, :]
    
    # Compute the relative phase delay
    # phase = exp(-j * 2 * pi * f * (d_m - d_ref) / c)
    phase_term = np.exp(-1j * 2 * np.pi * f_bcast * delta_dist_bcast / c)
    
    if mode == "near_field":
        # In near-field, amplitude decays with 1/r. 
        # The relative amplitude ratio is (1/d_m) / (1/d_ref) = d_ref / d_m
        amp_ratio = ref_dist[np.newaxis, :, :] / mic_dist[np.newaxis, :, :]
        rtf_vector = amp_ratio * phase_term
    else:
        # In far-field, we assume plane waves where amplitude attenuation across the array is negligible
        rtf_vector = phase_term

    if squeeze:
        rtf_vector = np.squeeze(rtf_vector)

    return rtf_vector