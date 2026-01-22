import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.io import wavfile
import os

# --- 1. FUNCIONES DE BASE (MODELO FÍSICO) ---

def near_field_steering_vector_multi(f, Rs, fs, mic_array, K=1, c=343.0, squeeze=True):
    """
    Calcula el vector de dirección para campo cercano con atenuación 1/r y retardo exacto.
    """
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
    """Simula propagación acústica (retardo + atenuación)"""
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
    """Carga, normaliza y remuestrea un archivo de audio."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
        
    fs_file, data = wavfile.read(path)
    
    # Conversión a float [-1, 1]
    if data.dtype == np.int16:
        data = data / 32768.0
    elif data.dtype == np.int32:
        data = data / 2147483648.0
    else:
        data = data.astype(float)
        data = data / np.max(np.abs(data)) if np.max(np.abs(data)) > 0 else data

    # Conversión a Mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)
        
    # Resample si es necesario
    if fs_file != target_fs:
        num_samples = int(len(data) * target_fs / fs_file)
        data = signal.resample(data, num_samples)
        
    return data

# --- 2. CONFIGURACIÓN DEL ESCENARIO ---

fs = 48000
C_SOUND = 343 
mic_spacing = 0.04 
M1, M2 = 3, 3  
M = M1 * M2

# Geometría 'xy' indexing para compatibilidad Kronecker
x = np.linspace(0, (M2-1)*mic_spacing, M2)
y = np.linspace(0, (M1-1)*mic_spacing, M1)
xv, yv = np.meshgrid(x, y, indexing='xy') 
mic_coords = np.column_stack([xv.flatten(), yv.flatten(), np.zeros(M)])

# Posiciones en Campo Cercano
pos_src = [1, 1, 0.5]      # Voz deseada (cerca)
pos_noise = [-1.5, 2.5, 0.5]   # Voz interferente (cerca)

# --- CARGA DE SEÑALES ---
print("1. Cargando archivos de audio...")

file_voice = "tools/data/signals/MF31_03.wav"
file_interf = "tools/data/signals/FA01_09.wav"

try:
    raw_voice = load_audio_track(file_voice, fs)
    raw_interf = load_audio_track(file_interf, fs)
    
    # Ajustar longitudes (cortar al más corto)
    min_len = min(len(raw_voice), len(raw_interf))
    # Limitamos a 5 segundos máximo para no eternizar la simulación
    limit_samples = min(min_len, 5 * fs) 
    
    source_signal = raw_voice[:limit_samples]
    # Amplificamos la interferencia para desafiar al beamformer (SIR bajo)
    noise_signal = raw_interf[:limit_samples] * 1.5 
    
    t = np.arange(limit_samples) / fs
    print(f"   -> Señales cargadas: {limit_samples/fs:.2f} segundos.")
    
except FileNotFoundError as e:
    print(e)
    print("   -> Generando señales sintéticas de respaldo...")
    t = np.arange(0, 2.0, 1/fs)
    source_signal = np.sin(2 * np.pi * t * 440)
    noise_signal = np.random.normal(0, 1, len(t))

print("2. Simulando propagación acústica (Near Field)...")
src_raw, d_src = space_delay(source_signal, fs, pos_src, mic_coords, C_SOUND)
noise_raw, d_noise = space_delay(noise_signal, fs, pos_noise, mic_coords, C_SOUND)

# Mezcla
array_input = (src_raw/d_src[:,None]) + (noise_raw/d_noise[:,None]) + np.random.normal(0, 0.0001, src_raw.shape)

# --- 3. KMVDR ADAPTADO (P=2 para Near-Field) ---

# Configuración STFT
n_window = 1024
n_overlap = 512
f_axis, t_axis, X = signal.stft(array_input, fs=fs, nperseg=n_window, noverlap=n_overlap, axis=1)
Y_stft = np.zeros_like(X[0,:,:], dtype=complex)

# Parámetros Adaptativos
P = 1
alpha = 0.95    
als_iters = 2
diag_load = 1e-3

I1 = np.eye(M1)
I2 = np.eye(M2)

print(f"3. Ejecutando KMVDR (P={P})...")

for k, freq_val in enumerate(f_axis):
    # Procesamos rango típico de voz (100Hz - 8kHz)
    if freq_val < 100 or freq_val > 8000: continue 

    # A. Steering Vector
    d_vec = near_field_steering_vector_multi(freq_val, pos_src, fs, mic_coords, 1, squeeze=True)
    d_vec = d_vec.reshape(-1, 1)
    
    # B. Inicialización SVD (Clave para P=2 en campo cercano)
    d_matrix = d_vec.reshape(M1, M2)
    U, S, Vh = np.linalg.svd(d_matrix)
    
    h1_curr = np.zeros((M1, P), dtype=complex)
    h2_curr = np.zeros((M2, P), dtype=complex)
    
    for p in range(min(P, min(M1, M2))):
        h1_curr[:, p] = U[:, p]
        h2_curr[:, p] = Vh[p, :].conj()

    R_curr = 1e-5 * np.eye(M, dtype=complex)

    # C. Bucle Temporal
    for t_idx in range(X.shape[2]):
        x_t = X[:, k, t_idx].reshape(-1, 1)
        
        # Update R
        R_curr = alpha * R_curr + (1 - alpha) * (x_t @ x_t.conj().T)
        tr_R = np.trace(R_curr).real
        R_loaded = R_curr + (diag_load * tr_R / M) * np.eye(M)
        
        # ALS
        for _ in range(als_iters):
            # h1 dado h2
            V2 = np.zeros((M, P * M1), dtype=complex)
            for p in range(P):
                block = np.kron(I1, h2_curr[:, p].reshape(-1, 1)) 
                V2[:, p*M1 : (p+1)*M1] = block
                
            Phi_y2 = V2.conj().T @ R_loaded @ V2
            d_2 = V2.conj().T @ d_vec
            
            try:
                num = np.linalg.solve(Phi_y2, d_2)
                den = d_2.conj().T @ num
                h1_flat = num / den
                for p in range(P):
                    h1_curr[:, p] = h1_flat[p*M1 : (p+1)*M1].flatten()
            except np.linalg.LinAlgError: pass

            # h2 dado h1
            V1 = np.zeros((M, P * M2), dtype=complex)
            for p in range(P):
                block = np.kron(h1_curr[:, p].reshape(-1, 1), I2)
                V1[:, p*M2 : (p+1)*M2] = block
            
            Phi_y1 = V1.conj().T @ R_loaded @ V1
            d_1 = V1.conj().T @ d_vec
            
            try:
                num = np.linalg.solve(Phi_y1, d_1)
                den = d_1.conj().T @ num
                h2_flat = num / den
                for p in range(P):
                    h2_curr[:, p] = h2_flat[p*M2 : (p+1)*M2].flatten()
            except np.linalg.LinAlgError: pass
        
        # Aplicar
        h_total = np.zeros((M, 1), dtype=complex)
        for p in range(P):
            h_total += np.kron(h1_curr[:, p].reshape(-1,1), h2_curr[:, p].reshape(-1,1))
            
        Y_stft[k, t_idx] = (h_total.conj().T @ x_t)[0,0]

# --- 4. RESULTADOS Y EXPORTACIÓN ---
print("4. Reconstruyendo y guardando...")
t_out, signal_out = signal.istft(Y_stft, fs=fs, nperseg=n_window, noverlap=n_overlap)

# Trim final
final_len = min(len(t), len(signal_out))
input_ref = array_input[0, :final_len].real
sig_out = signal_out[:final_len].real

def save_wav(filename, data, fs):
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data_norm = data / max_val
    else:
        data_norm = data
    wavfile.write(filename, fs, data_norm.astype(np.float32))

save_wav("input_mix.wav", input_ref, fs)
save_wav("output_beamformed.wav", sig_out, fs)

plt.figure(figsize=(10, 6))
plt.subplot(2,1,1)
plt.title("Entrada Mezcla (Mic 1)")
plt.plot(t[:final_len], input_ref)
plt.subplot(2,1,2)
plt.title("Salida KMVDR")
plt.plot(t[:final_len], sig_out)
plt.tight_layout()
plt.show()

print("Proceso finalizado.")