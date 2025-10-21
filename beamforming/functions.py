import numpy as np
import scipy as sc
import sys
import os
from matplotlib import pyplot as plt

import scipy.signal as sc

# Add the parent directory to the system path
# so Python can find the 'signal_model' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from scipy.spatial.distance import cdist


def simular_propagacion_arreglo(signal, fs, source_pos, mic_positions_matrix):
    """
    Simula la propagación de una señal a un arreglo de micrófonos de forma optimizada.
    
    Calcula la FFT de la señal original una sola vez y utiliza el máximo retardo 
    para definir una longitud de salida uniforme (N_fft) para todas las señales.

    Args:
        signal (np.ndarray): Señal discreta de entrada (vector 1D).
        fs (float): Frecuencia de muestreo (Hz).
        source_pos (np.ndarray): Coordenadas (x, y, z) de la fuente (vector 1D).
        mic_positions_matrix (np.ndarray): Matriz de coordenadas de micrófonos (Mics x 3).

    Returns:
        tuple: 
            - array_retardado (np.ndarray): Matriz (Mics x N_fft) con las señales retardadas.
            - signal_referencia_padded (np.ndarray): Señal original rellenada con ceros a longitud N_fft.
            - tau_array (np.ndarray): Array con los retardos absolutos (en segundos).
    """
    
    N_original = len(signal)
    c =343
    # 1. CÁLCULO VECTORIAL DE DISTANCIAS Y RETARDOS
    # Calcula la distancia euclidiana de la fuente a cada micrófono
    distancias = np.linalg.norm(mic_positions_matrix - source_pos, axis=1)
    tau_array = distancias / c  # Array de retardos temporales (tau_m)
    
    # 2. DETERMINAR LA LONGITUD GLOBAL (N_fft)
    # Basado en el máximo retardo para el zero-padding (evitar aliasing)
    retardo_muestras_max = np.max(tau_array) * fs
    N_minimo = N_original + int(np.ceil(retardo_muestras_max))
    N_fft = 2**(int(np.ceil(np.log2(N_minimo)))) # Potencia de 2 para FFT eficiente
    
    # 3. PRE-PROCESAMIENTO GLOBAL (Hecho solo una vez)
    padding_length = N_fft - N_original
    signal_padded = np.pad(signal, (0, padding_length), 'constant')
    
    # 3b. Calcular la FFT global de la señal una sola vez
    X = np.fft.fft(signal_padded)
    
    # 3c. Vector de frecuencias k
    k = np.fft.fftfreq(N_fft, d=1/fs)
    
    # 4. CÁLCULO VECTORIAL DEL CAMBIO DE FASE
    # Multiplicar el vector de frecuencias k por cada tau en tau_array (Broadcasting)
    tau_matrix = tau_array[:, np.newaxis] # Forma (num_mics, 1)
    phase_shift_matrix = np.exp(-1j * 2 * np.pi * k * tau_matrix) # Forma (num_mics, N_fft)
    
    # 5. APLICAR FASE Y TRANSFORMAR (Vectorización)
    # Multiplicar la FFT de la señal X por la matriz de fases
    Y_matrix = X * phase_shift_matrix # Resultado: (num_mics, N_fft)
    
    # IFFT en el eje de las muestras (axis=1)
    array_retardado = np.fft.ifft(Y_matrix, axis=1).real
    
    # 6. SEÑAL DE REFERENCIA (Padded a la misma longitud N_fft)
    signal_referencia_padded = signal_padded 
    
    return array_retardado, signal_referencia_padded, tau_array


def near_field_steering_vector(f, Rs,fs, mic_array, K =1, c=343):
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
    phase_reference = np.exp( 1j * 2 * np.pi * f * source_distance / c)
    normalization_factor = source_distance / phase_reference

    #Distances of each element respect to the source (shape Mx1)
    distances = np.linalg.norm( Rs - mic_array, axis = 1).reshape(-1, 1)

    #Propagation delay for each microphone (shape Mx1)
    mic_delay = distances/c 

    #tabs delays shape 1xK  
    T = 1/fs
    tabs = np.arange(K)
    tab_delay = tabs * T

    #Defining the propagation vector (Brodcast (Mx1)(1xK)->(MxK))
    steering_vector = np.exp(1j * 2 *np.pi *f * (mic_delay -tab_delay) ) / distances
    normalized_steering_vector = normalization_factor * steering_vector

    #Colapsing the matriz into colums 

    normalized_steering_vector = normalized_steering_vector.reshape(-1,1)


    return normalized_steering_vector


def point_constraint( target_point , K_taps, mic_array , f, fs ):
    #steering vector of the specific point (shape (M.K , 1))
    steering_vector  = near_field_steering_vector(f, target_point,fs, mic_array, K = K_taps, c=343)
    
    constrains_C = np.hstack([np.real(steering_vector), np.imag(steering_vector)]) 

    # 3. Definir el vector de respuesta deseada h. (Forma: 2 x 1)
    # [Ganancia Real Deseada (1)]
    # [Ganancia Imaginaria Deseada (0)]
    target_gain_h = np.vstack([1.0, 0.0]) # [1, 0]^T


    return constrains_C, target_gain_h

import numpy as np

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

def beamforming(signals, weights):
    y = weights.conj().T @ signals
    return y 