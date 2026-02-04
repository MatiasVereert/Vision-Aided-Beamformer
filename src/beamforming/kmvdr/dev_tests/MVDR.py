import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.io import wavfile
import os

# --- 1. FUNCIONES DE BASE (IDÉNTICAS AL ANTERIOR) ---

def near_field_steering_vector_multi(f, Rs, fs, mic_array, K=1, c=343.0, squeeze=True):
    f = np.atleast_1d(f)
    Rs = np.atleast_2d(Rs)
    
    F = f.shape[0]
    P_sources = Rs.shape[0]
    M = mic_array.shape[0]
    
    source_dist_origin = np.linalg.norm(Rs, axis=1)
    mic_distances = np.linalg.norm(Rs[:, np.newaxis, :] - mic_array[np.newaxis, :, :], axis=2)
    
    mic_delay = mic_distances / c
    source_delay_origin = source_dist_origin / c
    T = 1/fs
    tap_delays = np.arange(K) * T
    ref_delay = (K - 1) / (2 * fs)
    
    f_bcast = f.reshape(F, 1, 1, 1)
    phase_arg = 2 * np.pi * f_bcast * (ref_delay + source_delay_origin[np.newaxis, :, np.newaxis, np.newaxis] 
                                       - mic_delay[np.newaxis, :, :, np.newaxis] 
                                       - tap_delays[np.newaxis, np.newaxis, np.newaxis, :])
    
    steering_vector = np.exp(1j * phase_arg) / (mic_distances[np.newaxis, :, :, np.newaxis] + 1e-6)
    
    final_sv = steering_vector.reshape(F, P_sources, M * K)
    if squeeze:
        final_sv = np.squeeze(final_sv)
    return final_sv

def space_delay(signal_in, fs, source_pos, mic_array, c=343.0):
    signal_in = np.array(signal_in)
    source_pos = np.atleast_2d(source_pos) 
    mic_array = np.atleast_2d(mic_array)
    N_original = len(signal_in)
    
    diff_vectors = source_pos[:, np.newaxis, :] - mic_array[np.newaxis, :, :]
    distancias = np.linalg.norm(diff_vectors, axis=2) 
    tau_array = distancias / c 
    
    max_delay_samples = int(np.ceil(np.max(tau_array) * fs))
    N_fft = 2**(int(np.ceil(np.log2(N_original + max_delay_samples))))
    
    signal_padded = np.pad(signal_in, (0, N_fft - N_original), 'constant')
    X_fft = np.fft.fft(signal_padded)
    k = np.fft.fftfreq(N_fft, d=1/fs)
    
    phase_shift = np.exp(-1j * 2 * np.pi * k * tau_array[..., np.newaxis])
    array_retardado = np.fft.ifft(X_fft * phase_shift, axis=-1).real
    
    if array_retardado.shape[0] == 1:
        array_retardado = np.squeeze(array_retardado, axis=0)
        distancias = np.squeeze(distancias, axis=0)
        
    return array_retardado[:, :N_original], distancias

def load_audio_track(path, target_fs):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
        
    fs_file, data = wavfile.read(path)
    if data.dtype == np.int16: data = data / 32768.0
    elif data.dtype == np.int32: data = data / 2147483648.0
    else: data = data.astype(float) / (np.max(np.abs(data)) + 1e-8)

    if data.ndim > 1: data = np.mean(data, axis=1)
    if fs_file != target_fs:
        num_samples = int(len(data) * target_fs / fs_file)
        data = signal.resample(data, num_samples)
    return data

# --- 2. CONFIGURACIÓN (MISMOS DATOS QUE KMVDR) ---

fs = 48000
C_SOUND = 343 
mic_spacing = 0.04 
M1, M2 = 3, 3  
M = M1 * M2

x = np.linspace(0, (M2-1)*mic_spacing, M2)
y = np.linspace(0, (M1-1)*mic_spacing, M1)
xv, yv = np.meshgrid(x, y, indexing='xy') 
mic_coords = np.column_stack([xv.flatten(), yv.flatten(), np.zeros(M)])

pos_src = [1, 1, 0.5]      
pos_noise = [-1.5, 2.5, 0.5]   

print("1. Cargando archivos de audio...")
file_voice = "tools/data/signals/MF31_03.wav"
file_interf = "tools/data/signals/FA01_09.wav"

try:
    raw_voice = load_audio_track(file_voice, fs)
    raw_interf = load_audio_track(file_interf, fs)
    min_len = min(len(raw_voice), len(raw_interf))
    limit_samples = min(min_len, 5 * fs) 
    source_signal = raw_voice[:limit_samples]
    noise_signal = raw_interf[:limit_samples] * 1.5 
    t = np.arange(limit_samples) / fs
except FileNotFoundError:
    print("Archivos no encontrados, usando sintéticos.")
    t = np.arange(0, 2.0, 1/fs)
    source_signal = np.sin(2 * np.pi * t * 440)
    noise_signal = np.random.normal(0, 1, len(t))

print("2. Simulando propagación acústica...")
src_raw, d_src = space_delay(source_signal, fs, pos_src, mic_coords, C_SOUND)
noise_raw, d_noise = space_delay(noise_signal, fs, pos_noise, mic_coords, C_SOUND)
array_input = (src_raw/d_src[:,None]) + (noise_raw/d_noise[:,None]) + np.random.normal(0, 0.0001, src_raw.shape)

# --- 3. PROCESAMIENTO MVDR ESTÁNDAR ---

n_window = 1024
n_overlap = 512
f_axis, t_axis, X = signal.stft(array_input, fs=fs, nperseg=n_window, noverlap=n_overlap, axis=1)
Y_stft = np.zeros_like(X[0,:,:], dtype=complex)

# Parámetros MVDR
alpha = 0.95    
diag_load = 1e-3

print(f"3. Ejecutando Standard MVDR (Full Rank)...")

for k, freq_val in enumerate(f_axis):
    if freq_val < 100 or freq_val > 8000: continue 

    # A. Steering Vector (Mismo que KMVDR)
    # d_vec es de tamaño (M, 1)
    d_vec = near_field_steering_vector_multi(freq_val, pos_src, fs, mic_coords, 1, squeeze=True)
    d_vec = d_vec.reshape(-1, 1)
    
    # B. Inicialización Covarianza
    # Matriz M x M completa
    R_curr = 1e-5 * np.eye(M, dtype=complex)

    # C. Bucle Temporal Frame-a-Frame
    for t_idx in range(X.shape[2]):
        x_t = X[:, k, t_idx].reshape(-1, 1)
        
        # Actualización de Covarianza (Recursiva)
        R_curr = alpha * R_curr + (1 - alpha) * (x_t @ x_t.conj().T)
        
        # Diagonal Loading (Crucial para robustez en MVDR estándar)
        tr_R = np.trace(R_curr).real
        R_loaded = R_curr + (diag_load * tr_R / M) * np.eye(M)
        
        # Solución MVDR Cerrada: w = (R^-1 * d) / (d^H * R^-1 * d)
        try:
            # Paso 1: Resolver R * z = d (equivale a z = R^-1 * d)
            # Usamos solve porque es más rápido que invertir explícitamente
            z = np.linalg.solve(R_loaded, d_vec)
            
            # Paso 2: Factor de normalización (Distortionless constraint)
            # den = d^H * z
            den = d_vec.conj().T @ z
            
            # Peso final
            w_mvdr = z / (den + 1e-12)
            
            # Filtrado
            Y_stft[k, t_idx] = (w_mvdr.conj().T @ x_t)[0,0]
            
        except np.linalg.LinAlgError:
            # Si la matriz es singular (pasa al inicio o con silencio puro), 
            # usar Delay-and-Sum simple como fallback
            w_ds = d_vec / M
            Y_stft[k, t_idx] = (w_ds.conj().T @ x_t)[0,0]

# --- 4. EXPORTACIÓN ---
print("4. Reconstruyendo y guardando...")
t_out, signal_out = signal.istft(Y_stft, fs=fs, nperseg=n_window, noverlap=n_overlap)

final_len = min(len(t), len(signal_out))
sig_out = signal_out[:final_len].real

def save_wav(filename, data, fs):
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data_norm = data / max_val
    else:
        data_norm = data
    wavfile.write(filename, fs, data_norm.astype(np.float32))
save_wav("input_standard_mvdr.wav", array_input[0,:], fs)

normalization_factor = 1/ np.max(sig_out)
save_wav("output_standard_mvdr.wav", sig_out * normalization_factor, fs)

plt.figure(figsize=(10, 6))
plt.title("Salida Standard MVDR (Full Rank)")
plt.plot(t[:final_len], sig_out)
plt.grid(True)
plt.show()

print("Proceso MVDR finalizado.")