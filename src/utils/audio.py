import os 
import numpy as np 
from scipy.io import wavfile
from scipy import signal

def normalize_signal(sig):
    max_abs = np.max(np.abs(sig))
    if max_abs > 0:
        return sig * (0.99 / max_abs)
    return sig


def save_wav(filename, rate, data, folder="resultados_test"):
    if not os.path.exists(folder): os.makedirs(folder)
    data = np.real(data)
    m = np.max(np.abs(data))
    if m > 0: 
        data = data / m * 0.9
    wavfile.write(os.path.join(folder, filename), rate, (data * 32767).astype(np.int16))
    print(f"-> Guardado: {filename}")


def load_audio_source(filename, target_fs, target_duration_sec):
    """
    Carga un archivo WAV, lo convierte a mono, lo normaliza y 
    lo remuestrea a la frecuencia del sistema (target_fs).
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No se encontró el archivo: {filename}")

    # 1. Cargar archivo
    fs_file, data = wavfile.read(filename)
    
    # 2. Convertir a float (-1.0 a 1.0) y Mono
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    
    # Si es estéreo (N, 2), promediar a mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # 3. Resampling (Si fs del archivo != fs del sistema)
    if fs_file != target_fs:
            print(f"[Loader] Remuestreando de {fs_file} Hz a {target_fs} Hz...")
            
            # Calculate the greatest common divisor to simplify the resampling ratio
            gcd_val = np.gcd(int(target_fs), int(fs_file))  
            
            # Determine the up and down factors for the polyphase filter
            up_factor = int(target_fs) // gcd_val
            down_factor = int(fs_file) // gcd_val
            
            # Apply the polyphase resampling
            data = signal.resample_poly(data, up=up_factor, down=down_factor)

    # 4. Ajustar duración
    target_samples = int(target_duration_sec * target_fs)
    
    if len(data) > target_samples:
        # Recortar si sobra
        data = data[:target_samples]
    else:
        # Loop (repetir) si falta
        print("[Loader] Audio corto, repitiendo en bucle para llenar duración...")
        repeats = int(np.ceil(target_samples / len(data)))
        data = np.tile(data, repeats)[:target_samples]

    # 5. Normalizar final
    data = data / np.max(np.abs(data))
    
    return data