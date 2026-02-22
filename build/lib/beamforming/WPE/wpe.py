import numpy as np
from scipy import signal
from nara_wpe.wpe import wpe
from nara_wpe.utils import stft, istft  # Opcional: nara trae sus propias utils, pero scipy es más estándar

def apply_wpe(
    audio_multichannel: np.ndarray, 
    fs: int, 
    taps: int = 10, 
    delay: int = 3, 
    iterations: int = 3,
    n_fft: int = 512,
    hop_length: int = 128
) -> np.ndarray:
    """
    Aplica dereverberación WPE a una señal multicanal en el dominio del tiempo.
    
    Args:
        audio_multichannel (np.ndarray): Señal de entrada (M micrófonos, N muestras).
        fs (int): Frecuencia de muestreo.
        taps (int): Longitud del filtro de predicción (en frames de STFT).
        delay (int): Retardo de predicción (en frames de STFT).
        iterations (int): Número de iteraciones del algoritmo WPE.
        n_fft (int): Tamaño de la FFT.
        hop_length (int): Salto entre ventanas.

    Returns:
        np.ndarray: Señal dereverberada en el dominio del tiempo (M, N).
    """
    
    # 1. Validar dimensiones (M, N)
    if audio_multichannel.ndim != 2:
        raise ValueError("El audio debe tener forma (Canales, Muestras)")
        
    M, N = audio_multichannel.shape
    
    # 2. STFT (Short-Time Fourier Transform)
    # Scipy stft devuelve: (frecuencias, tiempos, canales) si pasamos eje correcto o 
    # (canales, frecuencias, tiempos). Scipy por defecto lo hace channel-last si no se especifica axis.
    # Pero nara_wpe espera: (bins_frecuencia, canales, frames_tiempo) -> (F, D, T)
    
    f, t, Y = signal.stft(audio_multichannel, fs=fs, nperseg=n_fft, noverlap=n_fft-hop_length, axis=1)
    
    # Y tiene forma (Canales, Frecuencias, Tiempo) -> (D, F, T)
    # Necesitamos transponer a (F, D, T) para nara_wpe
    Y_transposed = Y.transpose(1, 0, 2)
    
    # 3. Aplicar WPE
    # La librería procesa independientemente cada frecuencia.
    Z_transposed = wpe(
        Y_transposed,
        taps=taps,
        delay=delay,
        iterations=iterations,
        statistics_mode='full' # 'full' es más preciso, 'shorter' es más rápido
    )
    
    # 4. Transponer de vuelta a (D, F, T)
    Z = Z_transposed.transpose(1, 0, 2)
    
    # 5. iSTFT (Inverse Short-Time Fourier Transform)
    _, audio_dereverberated = signal.istft(Z, fs=fs, nperseg=n_fft, noverlap=n_fft-hop_length, axis=1)
    
    # Ajustar longitud por si la iSTFT añade o quita alguna muestra por el padding
    if audio_dereverberated.shape[1] > N:
        audio_dereverberated = audio_dereverberated[:, :N]
        
    return audio_dereverberated