import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.constants import speed_of_sound as c # Velocidad del sonido (m/s)

# ----------------------------------------------------------------------
# FUNCIONES DE MODELADO FÍSICO OPTIMIZADA
# ----------------------------------------------------------------------

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

# ----------------------------------------------------------------------
# SCRIPT DE EVALUACIÓN Y VISUALIZACIÓN ADAPTADO
# ----------------------------------------------------------------------

if __name__ == '__main__':
    # --- Parámetros de la Simulación ---
    fs = 44100.0  # Frecuencia de muestreo (Hz)
    duration = 0.01  # Duración de 10 ms
    t = np.arange(0, duration, 1/fs)

    # --- Señal de Entrada ---
    f1, f2, f3 = 500, 1000, 2000
    signal_original = (np.sin(2 * np.pi * f1 * t) + 
                       0.5 * np.sin(2 * np.pi * f2 * t) + 
                       0.25 * np.sin(2 * np.pi * f3 * t))

    # --- Geometría ---
    source_pos = np.array([1.5, 0.5, 1.0]) 
    mic_positions = np.array([
        [-0.1, 0.1, 0.0], # Mic 1
        [ 0.1, 0.1, 0.0], # Mic 2
        [-0.1, -0.1, 0.0], # Mic 3
        [ 0.1, -0.1, 0.0]  # Mic 4
    ])
    num_mics = len(mic_positions)

    # --- APLICACIÓN OPTIMIZADA DEL MODELO DE PROPAGACIÓN ---
    array_retardado, signal_referencia_padded, tau_array = simular_propagacion_arreglo(
        signal_original, fs, source_pos, mic_positions
    )

    # --- CÁLCULO DE RETARDOS RELATIVOS ---
    retardos_ms = tau_array * 1000
    tau_min = np.min(retardos_ms)
    retardos_relativos_us = (retardos_ms - tau_min) * 1000 # en microsegundos

    # --- GENERACIÓN DEL VECTOR DE TIEMPO PADDED ---
    N_fft_global = array_retardado.shape[1] # Longitud uniforme de la salida
    t_padded = np.arange(0, N_fft_global) / fs

    # ----------------------------------------------------------------------
    # VISUALIZACIÓN GRÁFICA 
    # ----------------------------------------------------------------------

    fig = plt.figure(figsize=(12, 10))

    # Subplot 1: Señales en el Dominio del Tiempo
    ax1 = fig.add_subplot(2, 1, 1)

    # Se utiliza signal_referencia_padded y t_padded para compatibilidad
    ax1.plot(t_padded * 1000, signal_referencia_padded, 
             label='Señal Original Padded (Referencia)', 
             color='gray', linestyle='--', alpha=0.7)

    for i in range(num_mics):
        ax1.plot(t_padded * 1000, array_retardado[i], 
                 label=f'Mic {i+1} (Retardo Relativo: {retardos_relativos_us[i]:.2f} $\mu$s)')

    ax1.set_title(f'Simulación de Propagación Optimizada (N_fft={N_fft_global})')
    ax1.set_xlabel('Tiempo (ms)')
    ax1.set_ylabel('Amplitud Normalizada')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)


    # Subplot 2: Mapeo Espacial 
    ax2 = fig.add_subplot(2, 1, 2)

    mic_x = mic_positions[:, 0]
    mic_y = mic_positions[:, 1]
    ax2.scatter(mic_x, mic_y, c='blue', marker='o', s=100, label='Micrófonos')
    ax2.scatter(source_pos[0], source_pos[1], c='red', marker='*', s=300, label='Fuente')

    for i in range(num_mics):
        ax2.text(mic_x[i] + 0.005, mic_y[i] + 0.005, f'M{i+1}', fontsize=9)
    ax2.text(source_pos[0] + 0.05, source_pos[1] + 0.05, f'Fuente ({source_pos[0]}m, {source_pos[1]}m, {source_pos[2]}m)', fontsize=9)

    ax2.set_title('Mapeo Espacial (Vista Superior XY)')
    ax2.set_xlabel('Eje X (m)')
    ax2.set_ylabel('Eje Y (m)')
    ax2.axis('equal') 
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()