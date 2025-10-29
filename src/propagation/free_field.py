import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.constants import speed_of_sound # Usamos el nombre completo para claridad

def space_delay(signal, fs, source_pos, mic_array):
    """
    Simula la propagación de una señal a un arreglo de micrófonos de forma optimizada.
    
    Calcula la FFT de la señal original una sola vez y utiliza el máximo retardo 
    para definir una longitud de salida uniforme (N_fft) para todas las señales.

    Args:
        signal (np.ndarray): Señal discreta de entrada (vector 1D).
        fs (float): Frecuencia de muestreo (Hz).
        source_pos (np.ndarray): Coordenadas (x, y, z) de la(s) fuente(s).
                                 Puede ser un vector 1D (3,) para una fuente,
                                 o una matriz (P, 3) para P fuentes.
        mic_array (np.ndarray): Matriz de coordenadas de micrófonos (Mics x 3).

    Returns:
        tuple: 
            - array_retardado (np.ndarray): Matriz con las señales retardadas.
                                            Si hay 1 fuente: (Mics, N_fft).
                                            Si hay P fuentes: (P, Mics, N_fft).
            - signal_referencia_padded (np.ndarray): Señal original rellenada con ceros a longitud N_fft.
            - tau_array (np.ndarray): Array con los retardos absolutos (en segundos).
    """
    
    N_original = len(signal)
    source_pos = np.atleast_2d(source_pos) # Asegura que source_pos sea al menos 2D (P, 3)
    num_sources = source_pos.shape[0]

    # 1. CÁLCULO VECTORIAL DE DISTANCIAS Y RETARDOS
    # Broadcasting: (P, 1, 3) - (M, 3) -> (P, M, 3)
    diff_vectors = source_pos[:, np.newaxis, :] - mic_array
    distancias = np.linalg.norm(diff_vectors, axis=2) # Resultado: (P, M)
    tau_array = distancias / speed_of_sound  # Array de retardos temporales (tau_m)
    
    # 2. DETERMINAR LA LONGITUD GLOBAL (N_fft)
    # Basado en el máximo retardo para el zero-padding (evitar aliasing)
    max_delay_samples = int(np.ceil(np.max(tau_array) * fs))
    # La longitud de la FFT debe ser suficiente para contener la señal original
    # más el máximo retardo para evitar el aliasing circular.
    N_fft = 2**(int(np.ceil(np.log2(N_original + max_delay_samples))))
    
    # 3. PRE-PROCESAMIENTO GLOBAL (Hecho solo una vez)
    # Se añade padding al final para acomodar la longitud de la FFT.
    signal_padded = np.pad(signal, (0, N_fft - N_original), 'constant')
    
    # Calcular la FFT global de la señal una sola vez
    X = np.fft.fft(signal_padded)
    
    # 3c. Vector de frecuencias k
    k = np.fft.fftfreq(N_fft, d=1/fs)
    
    # 4. CÁLCULO VECTORIAL DEL CAMBIO DE FASE
    # Broadcasting: k(N_fft) * tau_array(P, M) -> phase_shift_matrix(P, M, N_fft)
    # tau_array tiene forma (P, M), k tiene forma (N_fft)
    # Añadimos ejes para que las formas sean compatibles: (P, M, 1) y (N_fft)
    phase_shift_matrix = np.exp(-1j * 2 * np.pi * k * tau_array[..., np.newaxis])
    
    # 5. APLICAR FASE Y TRANSFORMAR (Vectorización)
    # Broadcasting: X(N_fft) * phase_shift_matrix(P, M, N_fft) -> Y_matrix(P, M, N_fft)
    Y_matrix = X * phase_shift_matrix
    
    # IFFT en el eje de las muestras (axis=1)
    # Usamos ifft en el eje correcto (el último)
    array_retardado_complex = np.fft.ifft(Y_matrix, axis=-1)
    # El resultado debe ser real, el componente imaginario es ruido numérico.
    array_retardado = array_retardado_complex.real
    
    # Si solo había una fuente, devolvemos el resultado con la forma original (M, N_fft)
    if num_sources == 1:
        array_retardado = array_retardado.squeeze(axis=0)

    # 6. SEÑAL DE REFERENCIA (Padded a la misma longitud N_fft)
    signal_referencia_padded = signal_padded 
    
    return array_retardado, signal_referencia_padded, tau_array
