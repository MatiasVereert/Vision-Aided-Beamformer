import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import speed_of_sound
from scipy.io import wavfile

# --- IMPORTACIONES ---
from beamforming.signal_model import near_field_steering_vector_multi
from beamforming.algorithms.region_constriant import build_region_constraints
from beamforming.processors import gsc_adaptive_beamformer

# ------------------------------------------------------------------------------
# 1. TU FUNCIÓN DE PROPAGACIÓN (Copiada aquí para integración)
# ------------------------------------------------------------------------------
def space_delay(signal, fs, source_pos, mic_array):
    """
    Simula la propagación banda ancha usando FFT para retardos fraccionarios exactos.
    """
    N_original = len(signal)
    source_pos = np.atleast_2d(source_pos) 
    
    # 1. Cálculo de retardos
    diff_vectors = source_pos[:, np.newaxis, :] - mic_array[np.newaxis, :, :]
    distancias = np.linalg.norm(diff_vectors, axis=2) 
    tau_array = distancias / speed_of_sound 
    
    # 2. Longitud FFT (Potencia de 2)
    max_delay_samples = int(np.ceil(np.max(tau_array) * fs))
    N_fft = 2**(int(np.ceil(np.log2(N_original + max_delay_samples))))
    
    # 3. FFT
    signal_padded = np.pad(signal, (0, N_fft - N_original), 'constant')
    X = np.fft.fft(signal_padded)
    k = np.fft.fftfreq(N_fft, d=1/fs)
    
    # 4. Fase
    phase_shift_matrix = np.exp(-1j * 2 * np.pi * k * tau_array[..., np.newaxis])
    
    # 5. IFFT
    Y_matrix = X * phase_shift_matrix
    array_retardado_complex = np.fft.ifft(Y_matrix, axis=-1)
    array_retardado = array_retardado_complex.real
    
    return array_retardado, signal_padded, tau_array
