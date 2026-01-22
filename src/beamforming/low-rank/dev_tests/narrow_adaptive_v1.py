import numpy as np 
from matplotlib import pyplot as plt
from scipy import signal
from scipy.io import wavfile
from beamforming.signal_model import near_field_steering_vector_multi

# --- 1. CONFIGURACIÓN Y SIMULACIÓN (Igual que antes) ---
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

# Parámetros acústicos
fs = 48000
C_SOUND = 343 
mic_spacing = 0.02
M1, M2 = 3, 4
M = M1 * M2

x = np.linspace(0, (M2-1)*mic_spacing, M2)
y = np.linspace(0, (M1-1)*mic_spacing, M1)
xv, yv = np.meshgrid(x, y)
mic_coords = np.column_stack([xv.flatten(), yv.flatten(), np.zeros(M)])

# Generación de Señales Dinámicas
duration = 2.0 # 2 segundos
t = np.arange(0, duration, 1/fs)

# Fuente: Tono puro pulsante
source_signal = np.sin(2 * np.pi * t * 1000) * (np.sin(2 * np.pi * t * 3) > 0)

# Ruido: Ráfagas aleatorias rápidas (difíciles de seguir)
noise_env = np.abs(np.sin(2 * np.pi * t * 2)) + 0.5 
noise_signal = np.random.normal(0, 1, len(t)) *  2.0

pos_src = [1.5, 1.5, 1.0]
pos_noise = [-1.0, 1.5, 1.0]

print("Generando simulación acústica...")
src_raw, d_src = space_delay(source_signal, fs, pos_src, mic_coords, C_SOUND)
noise_raw, d_noise = space_delay(noise_signal, fs, pos_noise, mic_coords, C_SOUND)

# Entrada con mucho ruido
array_input = (src_raw/d_src[:,None]) + (noise_raw/d_noise[:,None])

# --- 2. PROCESAMIENTO RECURSIVO FRAME-A-FRAME ---

# STFT
n_window = 1024
n_overlap = 512 # 50% overlap
f_axis, t_axis, X = signal.stft(array_input, fs=fs, nperseg=n_window, noverlap=n_overlap, axis=1)

# Parámetros Recursivos
# lambda (forgetting factor): Determina la "memoria" efectiva.
# lambda = 0.95 -> Memoria aprox de 20 frames. 
# Si es muy bajo (0.5), R fluctúa mucho. Si es muy alto (0.999), tarda en adaptarse.
alpha = 0.9

P = 2
iterations_als_per_frame = 4 # Con 1 iteración por frame sobra, ya que venimos "calientes" del frame anterior
diag_load = 1e-3

Y_stft = np.zeros((X.shape[1], X.shape[2]), dtype=complex)
I1 = np.eye(M1)
I2 = np.eye(M2)

print(f"Procesando Frame-by-Frame (Total frames: {X.shape[2]})...")

for k, freq_val in enumerate(f_axis):
    if freq_val < 100: continue 

    # --- Inicialización (Frame 0) ---
    d_vec = near_field_steering_vector_multi(freq_val, pos_src, fs, mic_coords, 1)
    
    # R inicial (identidad para arrancar suave)
    R_curr = np.eye(M, dtype=complex) 
    
    # h1, h2 iniciales con SVD
    steering_m = d_vec.reshape(M1, M2)
    U_init, S_init, Vh_init = np.linalg.svd(steering_m, full_matrices=False)
    h1_curr = U_init[:, :P] * np.sqrt(S_init[:P])
    h2_curr = Vh_init[:P, :].conj().T * np.sqrt(S_init[:P])

    # --- BUCLE DE TIEMPO (FRAME A FRAME) ---
    for t_idx in range(X.shape[2]):
        
        # 1. Obtener snapshot actual (Vector columna)
        x_t = X[:, k, t_idx].reshape(-1, 1) # (M, 1)
        
        # 2. Actualización Recursiva de R (Exponential Moving Average)
        # R[t] = alpha * R[t-1] + (1-alpha) * x[t] * x[t]^H
        R_update = x_t @ x_t.conj().T
        R_curr = alpha * R_curr + (1 - alpha) * R_update
        
        # Diagonal Loading para robustez numérica en la inversión
        R_loaded = R_curr + diag_load * np.trace(R_curr).real * np.eye(M) / M

        # 3. ALS: Actualizar coeficientes (1 iteración es suficiente por frame)
        # Como h1 y h2 vienen del frame anterior, ya están casi convergidos.
        
        # --- Paso A: h1 dado h2 ---
        H2_underlined = np.zeros((M, P*M1), dtype=complex)
        for p in range(P):
            bloque = np.kron(h2_curr[:, p].conj().reshape(-1,1), I1)
            H2_underlined[:, p*M1:(p+1)*M1] = bloque
            
        R_red = H2_underlined.conj().T @ R_loaded @ H2_underlined
        d_proj = H2_underlined.conj().T @ d_vec
        
        try:
            h1_unscaled = np.linalg.solve(R_red, d_proj)
            factor = d_vec.conj().T @ H2_underlined @ h1_unscaled
            h1_vec_flat = h1_unscaled / factor
            for p in range(P):
                h1_curr[:, p] = h1_vec_flat[p*M1:(p+1)*M1].flatten()
        except np.linalg.LinAlgError:
            pass # Si falla, mantenemos el valor anterior

        # --- Paso B: h2 dado h1 ---
        H1_underlined = np.zeros((M, P*M2), dtype=complex)
        for p in range(P):
            bloque = np.kron(I2, h1_curr[:, p].reshape(-1,1))
            H1_underlined[:, p*M2:(p+1)*M2] = bloque
            
        R_red_1 = H1_underlined.conj().T @ R_loaded @ H1_underlined
        d_proj_1 = H1_underlined.conj().T @ d_vec
        
        try:
            h2_unscaled = np.linalg.solve(R_red_1, d_proj_1)
            factor = d_vec.conj().T @ H1_underlined @ h2_unscaled
            h2_vec_flat = h2_unscaled / factor
            for p in range(P):
                h2_curr[:, p] = h2_vec_flat[p*M2:(p+1)*M2].flatten()
        except np.linalg.LinAlgError:
            pass

        # 4. Aplicar Filtro (Instantáneo)
        out_sample = 0j
        
        # Reshape snapshot para filtro separable (M1, M2)
        x_matrix = x_t.reshape(M1, M2) 
        
        for p in range(P):
            w1 = h1_curr[:, p].conj()
            w2 = h2_curr[:, p].conj()
            # w1^T * X * w2
            # (M1) . (M1, M2) . (M2) -> Escalar
            intermedio = w1 @ x_matrix
            out_sample += intermedio @ w2
            
        Y_stft[k, t_idx] = out_sample

# --- 3. RESULTADOS ---
t_out, signal_out = signal.istft(Y_stft, fs=fs, nperseg=n_window, noverlap=n_overlap)
min_len = min(len(t), len(signal_out), array_input.shape[1])
sig_in = array_input[0, :min_len].real
sig_out = signal_out[:min_len].real
t_plot = t[:min_len]

# Gráficos Comparativos
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Entrada
Pxx, f, bins, im1 = ax1.specgram(sig_in, NFFT=1024, Fs=fs, noverlap=512, cmap='inferno')
ax1.set_title("Entrada (Ruido Dinámico)")
ax1.set_ylabel("Frecuencia [Hz]")
ax1.set_xlabel("Tiempo [s]")

vmax = im1.get_clim()[1]
vmin = vmax - 100
im1.set_clim(vmin, vmax)

# Salida
Pxx, f, bins, im2 = ax2.specgram(sig_out, NFFT=1024, Fs=fs, noverlap=512, cmap='inferno', vmin=vmin, vmax=vmax)
ax2.set_title("Salida (Adaptativa Frame-a-Frame)")
ax2.set_xlabel("Tiempo [s]")

# Barra de color
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
fig.colorbar(im1, cax=cbar_ax, label='dB')

plt.suptitle("Comparación Frame-by-Frame MVDR", fontsize=14)
plt.show()

# Exportar
norm_factor = 1.0 / np.max(np.abs(sig_out))
wavfile.write("input_recursive_mvdr.wav", fs, (sig_in * norm_factor * 0.9).astype(np.float32))
wavfile.write("output_recursive_mvdr.wav", fs, (sig_out * norm_factor * 0.9).astype(np.float32))