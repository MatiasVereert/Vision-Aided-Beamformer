import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.io import wavfile

# --- 1. CONFIGURACIÓN Y MODELO DE SEÑAL ---

def near_field_steering_vector_multi(f, Rs, fs, mic_array, K=1, c=343.0, squeeze=True):
    """
    Calcula steering vectors. Adaptado para mantener compatibilidad con tu simulación.
    Nota: KMVDR asume teóricamente campo lejano, pero funcionará matemáticamente aquí.
    """
    f = np.atleast_1d(f)
    Rs = np.atleast_2d(Rs)
    
    F = f.shape[0]
    P_sources = Rs.shape[0]
    M = mic_array.shape[0]
    
    # Distancias
    source_dist_origin = np.linalg.norm(Rs, axis=1)
    mic_distances = np.linalg.norm(Rs[:, np.newaxis, :] - mic_array[np.newaxis, :, :], axis=2)
    
    # Delays
    mic_delay = mic_distances / c
    source_delay_origin = source_dist_origin / c
    T = 1/fs
    tap_delays = np.arange(K) * T

    ref_delay = (K - 1) / (2 * fs)
    
    # Fase
    f_bcast = f.reshape(F, 1, 1, 1)
    # Corrección de fase relativa al centro del array para consistencia
    phase_arg = 2 * np.pi * f_bcast * (ref_delay + source_delay_origin[np.newaxis, :, np.newaxis, np.newaxis] 
                                       - mic_delay[np.newaxis, :, :, np.newaxis] 
                                       - tap_delays[np.newaxis, np.newaxis, np.newaxis, :])
    
    steering_vector = np.exp(1j * phase_arg) / (mic_distances[np.newaxis, :, :, np.newaxis] + 1e-6)
    
    final_sv = steering_vector.reshape(F, P_sources, M * K)
    if squeeze:
        final_sv = np.squeeze(final_sv)
    return final_sv

def space_delay(signal_in, fs, source_pos, mic_array, c=343.0):
    """Simula el retardo de propagación para generar la señal de micrófono"""
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

# --- 2. PARAMETRIZACIÓN DEL ESCENARIO ---

fs = 48000
C_SOUND = 343 
mic_spacing = 0.03 # 3cm
M1, M2 = 4, 4      # 4x4 array (16 micrófonos)
M = M1 * M2

# GEOMETRÍA DEL ARRAY
# Importante: 'xy' indexing hace que x cambie rápido (columnas) y y lento (filas).
# Esto alinea mic_coords con el producto Kronecker h = h1 (y) (x) h2 (x).
x = np.linspace(0, (M2-1)*mic_spacing, M2)
y = np.linspace(0, (M1-1)*mic_spacing, M1)
xv, yv = np.meshgrid(x, y, indexing='xy') 
mic_coords = np.column_stack([xv.flatten(), yv.flatten(), np.zeros(M)])

# Generación de Señales
duration = 1.0 
t = np.arange(0, duration, 1/fs)

# Fuente deseada (Tono) e Interferencia (Ruido blanco)
source_signal = np.sin(2 * np.pi * t * 1000)
noise_signal = np.random.normal(0, 1, len(t)) * 2.5 # Interferencia fuerte

pos_src = [1.5, 1.5, 2.0]       # Frente al array
pos_noise = [-1.0, 1.5, 2.0]    # Lateral

print("1. Generando simulación acústica...")
src_raw, d_src = space_delay(source_signal, fs, pos_src, mic_coords, C_SOUND)
noise_raw, d_noise = space_delay(noise_signal, fs, pos_noise, mic_coords, C_SOUND)

# Mezcla en los micrófonos (Señal + Interferencia + Ruido de piso)
array_input = (src_raw/d_src[:,None]) + (noise_raw/d_noise[:,None]) + np.random.normal(0, 0.01, src_raw.shape)

# --- 3. PROCESAMIENTO KMVDR ---

# Parámetros STFT
n_window = 1024
n_overlap = 512
f_axis, t_axis, X = signal.stft(array_input, fs=fs, nperseg=n_window, noverlap=n_overlap, axis=1)

# Parámetros KMVDR
P = 2           # Rango de descomposición (Rank)
alpha = 0.92    # Forgetting factor para la covarianza
als_iters = 2   # Iteraciones por frame (suficiente para tracking)
diag_load = 1e-3

Y_stft = np.zeros((X.shape[1], X.shape[2]), dtype=complex)

print(f"2. Procesando Frame-by-Frame ({X.shape[2]} frames)...")

# Identidades para construcción de bloques
I1 = np.eye(M1)
I2 = np.eye(M2)

for k, freq_val in enumerate(f_axis):
    if freq_val < 100 or freq_val > 8000: continue # Procesar solo banda de interés

    # Vector de dirección (Steering Vector) para la frecuencia actual
    # d_vec debe ser (M, 1)
    d_vec = near_field_steering_vector_multi(freq_val, pos_src, fs, mic_coords, 1, squeeze=True)
    d_vec = d_vec.reshape(-1, 1) # Asegurar columna
    
    # Inicialización de Covarianza
    R_curr = 1e-6 * np.eye(M, dtype=complex)
    
    # Inicialización de subfiltros h1 y h2
    # Estrategia híbrida: P=1 usa SVD (mejor guess), P>1 usa identidad (independencia) 
    h1_curr = np.zeros((M1, P), dtype=complex)
    h2_curr = np.zeros((M2, P), dtype=complex)
    
    # Init P=0 (Principal)
    d_matrix = d_vec.reshape(M1, M2) # Ojo con el orden, debe coincidir con meshgrid 'xy'
    U, S, Vh = np.linalg.svd(d_matrix)
    h1_curr[:, 0] = U[:, 0]
    h2_curr[:, 0] = Vh[0, :].conj()
    
    # Init P>0 (Unit vectors para asegurar rango, según paper)
    for p in range(1, P):
        if p < M1: h1_curr[p, p] = 1.0
        if p < M2: h2_curr[p, p] = 1.0

    # --- BUCLE TEMPORAL ---
    for t_idx in range(X.shape[2]):
        
        # 1. Snapshot actual
        x_t = X[:, k, t_idx].reshape(-1, 1)
        
        # 2. Actualizar Covarianza R (Recursiva)
        R_curr = alpha * R_curr + (1 - alpha) * (x_t @ x_t.conj().T)
        
        # Diagonal loading para estabilidad numérica
        trace_R = np.trace(R_curr).real
        R_loaded = R_curr + (diag_load * trace_R / M) * np.eye(M)
        
        # 3. Algoritmo ALS (Alternating Least Squares)
        for _ in range(als_iters):
            
            # --- PASO A: Estimar h1 fijando h2 ---
            # Construir matriz de proyección V2 global
            # V2 concatena [H_{2,1} ... H_{2,P}]
            V2 = np.zeros((M, P * M1), dtype=complex)
            for p in range(P):
                # Eq 12: H_{2,p} = I_M1 (kron) h_{2,p}
                # h_{2,p} es el componente rápido (X)
                h2_p = h2_curr[:, p].reshape(-1, 1)
                block = np.kron(I1, h2_p) 
                V2[:, p*M1 : (p+1)*M1] = block
                
            # Proyección (Eq 21 y 23)
            Phi_y2 = V2.conj().T @ R_loaded @ V2
            d_2 = V2.conj().T @ d_vec
            
            # Resolver MVDR acoplado (Eq 24)
            try:
                # Usamos lstsq para mayor robustez que solve
                num = np.linalg.solve(Phi_y2, d_2)
                den = d_2.conj().T @ num
                h1_flat = num / den
                
                # Desempaquetar
                for p in range(P):
                    h1_curr[:, p] = h1_flat[p*M1 : (p+1)*M1].flatten()
            except np.linalg.LinAlgError:
                pass 

            # --- PASO B: Estimar h2 fijando h1 ---
            # Construir matriz de proyección V1 global
            V1 = np.zeros((M, P * M2), dtype=complex)
            for p in range(P):
                # Eq 11: H_{1,p} = h_{1,p} (kron) I_M2
                h1_p = h1_curr[:, p].reshape(-1, 1)
                block = np.kron(h1_p, I2)
                V1[:, p*M2 : (p+1)*M2] = block
            
            # Proyección (Eq 15 y 17)
            Phi_y1 = V1.conj().T @ R_loaded @ V1
            d_1 = V1.conj().T @ d_vec
            
            # Resolver MVDR acoplado (Eq 18)
            try:
                num = np.linalg.solve(Phi_y1, d_1)
                den = d_1.conj().T @ num
                h2_flat = num / den
                
                # Desempaquetar
                for p in range(P):
                    h2_curr[:, p] = h2_flat[p*M2 : (p+1)*M2].flatten()
            except np.linalg.LinAlgError:
                pass
        
        # 4. Construir Filtro Global y Aplicar
        # Eq 8: h = sum(h1_p (kron) h2_p)
        h_total = np.zeros((M, 1), dtype=complex)
        for p in range(P):
            h_term = np.kron(h1_curr[:, p].reshape(-1,1), h2_curr[:, p].reshape(-1,1))
            h_total += h_term
            
        Y_stft[k, t_idx] = (h_total.conj().T @ x_t)[0,0]

# --- 4. RECONSTRUCCIÓN Y RESULTADOS ---
print("3. Reconstruyendo señal de audio...")
t_out, signal_out = signal.istft(Y_stft, fs=fs, nperseg=n_window, noverlap=n_overlap)

# Ajuste de longitud
min_len = min(len(t), len(signal_out), array_input.shape[1])
sig_in_ref = array_input[0, :min_len].real # Micrófono 1 como referencia
sig_out = signal_out[:min_len].real
t_plot = t[:min_len]

# Gráficos
plt.figure(figsize=(12, 8))

plt.subplot(3,1,1)
plt.title("Señal en Micrófono 1 (Ruidosa)")
plt.plot(t_plot, sig_in_ref, color='gray', alpha=0.7)
plt.grid(True)

plt.subplot(3,1,2)
plt.title(f"Salida KMVDR (P={P})")
plt.plot(t_plot, sig_out, color='blue')
plt.grid(True)

plt.subplot(3,1,3)
plt.specgram(sig_out, NFFT=1024, Fs=fs, noverlap=512, cmap='inferno')
plt.title("Espectrograma de Salida")
plt.ylabel("Freq [Hz]")
plt.xlabel("Tiempo [s]")

plt.tight_layout()
plt.show()

# Normalizar y guardar
norm = lambda x: x / np.max(np.abs(x))
wavfile.write("kmvdr_in.wav", fs, norm(sig_in_ref).astype(np.float32))

wavfile.write("kmvdr_out.wav", fs, norm(sig_out).astype(np.float32))
print("Proceso finalizado.")