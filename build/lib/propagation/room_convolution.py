import numpy as np 
from matplotlib import pyplot as plt 
import scipy 

from utils.data_loader import load_sriracha_selection, get_rir_data_arrays



import numpy as np
from scipy import signal

import numpy as np
from scipy import signal

def generate_array_signals(
    source_audio: np.ndarray,
    interference_audios: np.ndarray, 
    rir_target: np.ndarray,          
    rirs_interference: np.ndarray    
) -> np.ndarray:
    """
    Genera las señales del arreglo convolucionando:
    - 1 fuente target con su RIR.
    - N fuentes de interferencia, CADA UNA con su propio audio distinto.
    
    Maneja automáticamente el broadcasting y casos de 1 sola interferencia.

    Args:
        source_audio: (L_samples,) Audio del locutor.
        interference_audios: (N_int, L_samples) Matriz con un audio por fila.
                             Si es (L_samples,), se asume 1 sola interferencia.
        rir_target: (M_mics, RIR_len) Respuesta al impulso del target.
        rirs_interference: (N_int, M_mics, RIR_len). 
                           Si es (M_mics, RIR_len), se asume 1 sola interferencia.
        
    Returns:
        np.ndarray: Señales en los micrófonos (M_mics, Total_samples)
    """
    
    # --- 1. BLINDAJE DE DIMENSIONES (Robustez) ---
    
    # Si rirs_interference es 2D (Mics, Samples), asumimos 1 sola fuente
    # y le agregamos la dimensión faltante al principio -> (1, Mics, Samples)
    if rirs_interference.ndim == 2:
        rirs_interference = rirs_interference[np.newaxis, :, :]
        # print("[Propagator] Aviso: rirs_interference era 2D, ajustado a 3D.")

    # Si interference_audios es 1D (Samples), asumimos 1 sola fuente
    # y le agregamos la dimensión faltante -> (1, Samples)
    if interference_audios.ndim == 1:
        interference_audios = interference_audios[np.newaxis, :]
        # print("[Propagator] Aviso: interference_audios era 1D, ajustado a 2D.")

    # --- 2. VALIDACIONES ---
    if rir_target.ndim != 2:
        raise ValueError(f"rir_target debe ser 2D (Mics, Samples), recibido: {rir_target.ndim}D")
    
    if rirs_interference.ndim != 3:
        raise ValueError(f"rirs_interference debe ser 3D (Fuentes, Mics, Samples), recibido: {rirs_interference.ndim}D")

    # Validar que la cantidad de fuentes coincida
    n_interf_rirs = rirs_interference.shape[0]
    n_interf_audios = interference_audios.shape[0]
    
    if n_interf_rirs != n_interf_audios:
        raise ValueError(f"Mismatch: Tienes {n_interf_rirs} RIRs de interferencia pero {n_interf_audios} audios.")

    # --- 3. PROCESAMIENTO DEL TARGET (Locutor) ---
    # Broadcasting: (M, N) * (1, L) -> Convolución por filas (Mics)
    target_mic_signals = signal.fftconvolve(
        rir_target, 
        source_audio[np.newaxis, :], 
        mode='full', 
        axes=1
    )

    # --- 4. PROCESAMIENTO DE INTERFERENCIAS (Distintos Audios) ---
    # rirs_interference:   (N_int, M_mics, RIR_len)
    # interference_audios: (N_int, L_samples)
    
    # Expansión para Broadcasting:
    # Queremos que interference_audios sea (N_int, 1, L_samples)
    # Así:
    #   - Eje 0 (N_int) coincide: Audio[i] se convolve con RIR_Interf[i]
    #   - Eje 1 (1 vs M_mics): Broadcast. El Audio[i] va a todos los mics de esa posición.
    #   - Eje 2: Convolución temporal.
    
    interf_components = signal.fftconvolve(
        rirs_interference, 
        interference_audios[:, np.newaxis, :], 
        mode='full', 
        axes=2
    )
    
    # --- 5. SUMA ESPACIAL (Colapso de fuentes) ---
    # Sumamos todas las interferencias ya propagadas para obtener la mezcla en cada mic
    # Resultado: (M_mics, Total_samples)
    total_interference = np.sum(interf_components, axis=0)

    # --- 6. MEZCLA FINAL ---
    # Ajustar longitudes si difieren por un sample (común en FFT)
    len_min = min(target_mic_signals.shape[1], total_interference.shape[1])
    
    array_output = target_mic_signals[:, :len_min] + total_interference[:, :len_min]
    
    return array_output