import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.constants import speed_of_sound as c # Velocidad del sonido (m/s)


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
