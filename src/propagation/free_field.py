import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import speed_of_sound
from scipy.io import wavfile

# --- IMPORTACIONES ---
from beamforming.signal_model import steering_vector

def space_delay(signal_in, fs, source_pos, mic_array):
    """
    Simula la propagación usando FFT para retardos fraccionarios exactos.
    Retorna: Matriz (M, N_samples)
    """
    import numpy as np
    
    # Aseguramos dimensiones
    signal_in = np.array(signal_in)
    N_original = len(signal_in)
    source_pos = np.atleast_2d(source_pos) # (1, 3)
    mic_array = np.atleast_2d(mic_array)   # (M, 3)
    
    # 1. Cálculo de retardos y distancias
    # diff_vectors shape: (1, M, 3)
    diff_vectors = source_pos[:, np.newaxis, :] - mic_array[np.newaxis, :, :]
    distancias = np.linalg.norm(diff_vectors, axis=2) # (1, M)
    tau_array = distancias / speed_of_sound # Usamos la constante global C_SOUND
    
    # 2. Longitud FFT (Potencia de 2 para eficiencia)
    max_delay_samples = int(np.ceil(np.max(tau_array) * fs))
    N_fft = 2**(int(np.ceil(np.log2(N_original + max_delay_samples))))
    
    # 3. FFT
    # Paddeamos la señal al largo de la FFT
    signal_padded = np.pad(signal_in, (0, N_fft - N_original), 'constant')
    X = np.fft.fft(signal_padded)
    k = np.fft.fftfreq(N_fft, d=1/fs)
    
    # 4. Fase (Shift Theorem: x(t-tau) <-> X(f) * exp(-j*2pi*f*tau))
    # tau_array: (1, M), k: (N_fft) -> Broadcasting a (1, M, N_fft)
    phase_shift_matrix = np.exp(-1j * 2 * np.pi * k * tau_array[..., np.newaxis])
    
    # 5. IFFT
    # X es (N_fft,), phase es (1, M, N_fft) -> Result (1, M, N_fft)
    Y_matrix = X * phase_shift_matrix
    array_retardado_complex = np.fft.ifft(Y_matrix, axis=-1)
    
    # Volvemos a tiempo real
    array_retardado = array_retardado_complex.real
    
    # Quitamos dimensión extra de la fuente (1, M, N) -> (M, N)
    if array_retardado.shape[0] == 1:
        array_retardado = np.squeeze(array_retardado, axis=0)
        distancias = np.squeeze(distancias, axis=0) # También sacamos distancias para atenuar
    
    # --- CORRECCIÓN FÍSICA: ATENUACIÓN 1/r ---
    # Tu función original solo retardaba. Aquí aplicamos la pérdida de energía.
    # Broadcasting: (M, N) / (M, 1)
    array_retardado = array_retardado / distancias[:, np.newaxis]
    
    # Recortamos al largo útil (opcional, o dejamos el padding)
    # Para STFT conviene que sean todos iguales, cortamos al largo original + max delay
    return array_retardado[:, :N_original]