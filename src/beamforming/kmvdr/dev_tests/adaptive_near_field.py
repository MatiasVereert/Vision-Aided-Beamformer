import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.io import wavfile

# --- 1. FUNCIONES DE BASE (MODELO FÍSICO) ---

def near_field_steering_vector_multi(f, Rs, fs, mic_array, K=1, c=343.0, squeeze=True):
    """
    Calcula el vector de dirección (Steering Vector) para campo cercano.
    Incluye atenuación por distancia (1/r) y retardo exacto.
    """
    f = np.atleast_1d(f)
    Rs = np.atleast_2d(Rs)
    
    F = f.shape[0]
    P_sources = Rs.shape[0]
    M = mic_array.shape[0]
    
    # Distancias Euclidianas exactas
    source_dist_origin = np.linalg.norm(Rs, axis=1)
    mic_distances = np.linalg.norm(Rs[:, np.newaxis, :] - mic_array[np.newaxis, :, :], axis=2)
    
    # Delays de propagación
    mic_delay = mic_distances / c
    source_delay_origin = source_dist_origin / c
    T = 1/fs
    tap_delays = np.arange(K) * T

    # Referencia temporal centrada para minimizar fase
    ref_delay = (K - 1) / (2 * fs)
    
    # Cálculo de Fase Esférica
    f_bcast = f.reshape(F, 1, 1, 1)
    phase_arg = 2 * np.pi * f_bcast * (ref_delay + source_delay_origin[np.newaxis, :, np.newaxis, np.newaxis] 
                                       - mic_delay[np.newaxis, :, :, np.newaxis] 
                                       - tap_delays[np.newaxis, np.newaxis, np.newaxis, :])
    
    # Steering vector: Amplitud (1/r) * Fase
    # Se añade 1e-6 para evitar división por cero
    steering_vector = np.exp(1j * phase_arg) / (mic_distances[np.newaxis, :, :, np.newaxis] + 1e-6)
    
    final_sv = steering_vector.reshape(F, P_sources, M * K)
    if squeeze:
        final_sv = np.squeeze(final_sv)
    return final_sv

def space_delay(signal_in, fs, source_pos, mic_array, c=343.0):
    """Generador de señal de micrófono simulando propagación real"""
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

# --- 2. CONFIGURACIÓN DEL ESCENARIO ---

fs = 48000
C_SOUND = 343 
mic_spacing = 0.04 # 4cm spacing para tener apertura decente
M1, M2 = 4, 4      
M = M1 * M2

# Definición de Geometría ('xy' indexing es clave para Kronecker)
x = np.linspace(0, (M2-1)*mic_spacing, M2)
y = np.linspace(0, (M1-1)*mic_spacing, M1)
xv, yv = np.meshgrid(x, y, indexing='xy') 
mic_coords = np.column_stack([xv.flatten(), yv.flatten(), np.zeros(M)])

# Fuente muy cercana (Campo cercano estricto)
# Distancia ~0.5m del centro del array (apertura array ~0.12m) -> Near Field
pos_src = [0.2, 0.2, 0.5]      
pos_noise = [-0.5, 0.5, 0.5]   

# Generación de Señales
duration = 1.0 
t = np.arange(0, duration, 1/fs)
source_signal = np.sin(2 * np.pi * t * 400) + 0.5*np.sin(2*np.pi*t*800) # Tono compuesto
noise_signal = np.random.normal(0, 1, len(t)) * 3.0 # Interferencia fuerte

print("1. Simulando propagación acústica (Near Field)...")
src_raw, d_src = space_delay(source_signal, fs, pos_src, mic_coords, C_SOUND)
noise_raw, d_noise = space_delay(noise_signal, fs, pos_noise, mic_coords, C_SOUND)

# Mezcla: Señal atenuada por distancia + Ruido + Piso térmico
array_input = (src_raw/d_src[:,None]) + (noise_raw/d_noise[:,None]) + np.random.normal(0, 0.001, src_raw.shape)

# --- 3. KMVDR ADAPTADO A CAMPO CERCANO ---

# Configuración STFT
n_window = 1024
n_overlap = 512
f_axis, t_axis, X = signal.stft(array_input, fs=fs, nperseg=n_window, noverlap=n_overlap, axis=1)
Y_stft = np.zeros_like(X[0,:,:], dtype=complex)

# Parámetros del Algoritmo
P = 2           # Rango 2 para aproximar curvatura
alpha = 0.95    # Forgetting factor (más alto para estabilidad)
als_iters = 2   
diag_load = 1e-3

I1 = np.eye(M1)
I2 = np.eye(M2)

print(f"2. Ejecutando KMVDR (P={P}) frame a frame...")

for k, freq_val in enumerate(f_axis):
    if freq_val < 200 or freq_val > 4000: continue 

    # --- A. Steering Vector de Campo Cercano ---
    d_vec = near_field_steering_vector_multi(freq_val, pos_src, fs, mic_coords, 1, squeeze=True)
    d_vec = d_vec.reshape(-1, 1) # Columna (M, 1)
    
    # --- B. Inicialización Optimizada para Campo Cercano ---
    # Aquí está la "magia" para P=2.
    # Descomponemos el Steering Vector MATRICIAL para encontrar
    # los mejores subfiltros iniciales que aproximan la fase esférica.
    d_matrix = d_vec.reshape(M1, M2) # Debe coincidir con 'xy' indexing
    U, S, Vh = np.linalg.svd(d_matrix)
    
    h1_curr = np.zeros((M1, P), dtype=complex)
    h2_curr = np.zeros((M2, P), dtype=complex)
    
    # Asignamos los P componentes principales del frente de onda
    # Esto inicializa el filtro "apuntando" a la curvatura correcta
    for p in range(min(P, min(M1, M2))):
        h1_curr[:, p] = U[:, p]
        h2_curr[:, p] = Vh[p, :].conj() # Vh es V^H, tomamos filas conjugadas -> columnas de V

    # Matriz de Covarianza Inicial
    R_curr = 1e-4 * np.eye(M, dtype=complex)

    # --- C. Bucle Temporal ---
    for t_idx in range(X.shape[2]):
        x_t = X[:, k, t_idx].reshape(-1, 1)
        
        # Update Covarianza
        R_curr = alpha * R_curr + (1 - alpha) * (x_t @ x_t.conj().T)
        tr_R = np.trace(R_curr).real
        R_loaded = R_curr + (diag_load * tr_R / M) * np.eye(M)
        
        # Algoritmo ALS (Igual que antes, pero operando sobre la inicialización correcta)
        for _ in range(als_iters):
            # 1. Update h1 dado h2
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

            # 2. Update h2 dado h1
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
        
        # Aplicar Filtro
        h_total = np.zeros((M, 1), dtype=complex)
        for p in range(P):
            h_term = np.kron(h1_curr[:, p].reshape(-1,1), h2_curr[:, p].reshape(-1,1))
            h_total += h_term
            
        Y_stft[k, t_idx] = (h_total.conj().T @ x_t)[0,0]

# --- 4. RESULTADOS ---
print("3. Reconstruyendo...")
t_out, signal_out = signal.istft(Y_stft, fs=fs, nperseg=n_window, noverlap=n_overlap)
min_len = min(len(t), len(signal_out), array_input.shape[1])

# Plot
plt.figure(figsize=(10, 6))
plt.subplot(2,1,1)
plt.title("Micrófono 1 (Señal + Interferencia)")
plt.plot(t[:min_len], array_input[0, :min_len].real, color='gray', alpha=0.6)
plt.grid(True)
plt.subplot(2,1,2)
plt.title(f"Salida KMVDR (P={P}, Near-Field Optimized)")
plt.plot(t[:min_len], signal_out[:min_len].real, color='blue')
plt.grid(True)
plt.tight_layout()
plt.show()


# --- 5. EXPORTACIÓN A WAV ---
print("4. Guardando archivos de audio...")

# Función auxiliar para normalizar a float32 [-1, 1]
def save_wav(filename, data, fs):
    # Normalizar al máximo absoluto para evitar clipping
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data_norm = data / max_val
    else:
        data_norm = data
    
    # Escribir archivo (float32 es estándar y de alta calidad)
    wavfile.write(filename, fs, data_norm.astype(np.float32))
    print(f"   -> Guardado: {filename}")

# 1. Guardar Entrada (Usamos el Micrófono 1 como referencia auditiva)
# array_input tiene forma (M, N_samples), tomamos la fila 0
input_ref = array_input[0, :min_len].real
save_wav("input_noisy_mic1.wav", input_ref, fs)

# 2. Guardar Salida (Señal procesada por KMVDR)
save_wav("output_kmvdr_clean.wav", signal_out[:min_len].real, fs)

# 2. Guardar Salida (Señal procesada por KMVDR)
save_wav("source_signal.wav", source_signal, fs)