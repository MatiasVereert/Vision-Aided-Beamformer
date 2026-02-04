import numpy as np 
from matplotlib import pyplot as plt
from scipy import signal
from scipy.io import wavfile
from beamforming.signal_model import steering_vector

# --- 1. CONFIGURACIÓN Y FUNCIONES AUXILIARES ---

def space_delay(signal_in, fs, source_pos, mic_array, c=343.0):
    """Simula propagación con retardo fraccionario exacto y atenuación 1/r."""
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

# --- 2. ESCENARIO DE SIMULACIÓN ---
fs = 48000
C_SOUND = 343 

mic_spacing = 0.04  # 4cm (Optimizado para evitar aliasing hasta ~8.5kHz)
M1, M2 = 3, 4
M = M1 * M2

# Grilla 3x4
x = np.linspace(0, (M2-1)*mic_spacing, M2)
y = np.linspace(0, (M1-1)*mic_spacing, M1)
xv, yv = np.meshgrid(x, y)
mic_coords = np.column_stack([xv.flatten(), yv.flatten(), np.zeros(M)])

# Señales
t = np.arange(0, 0.5, 1/fs)
source_signal = np.sin(2 * np.pi * t * 1000) # Tono puro
noise_signal = np.random.normal(0, 1, len(t)) # Ruido blanco fuerte

# Posiciones
pos_src = [1.5, 1.5, 1.0]   # Fuente deseada
pos_noise = [-1.0, 1.5, 1.0] # Fuente de interferencia (La queremos cancelar)

print("Generando simulación acústica...")
src_raw, d_src = space_delay(source_signal, fs, pos_src, mic_coords, C_SOUND)
noise_raw, d_noise = space_delay(noise_signal, fs, pos_noise, mic_coords, C_SOUND)

# Mezcla (Señal + Ruido Potente)
# Bajamos un poco la señal para que el MVDR tenga que trabajar
array_input = (src_raw/d_src[:,None]) + 2.0 * (noise_raw/d_noise[:,None])

# --- 3. PROCESAMIENTO LOW-RANK MVDR (ALS) ---

# Parámetros STFT
n_window = 1024
n_overlap = 512
f_axis, t_axis, X = signal.stft(array_input, fs=fs, nperseg=n_window, noverlap=n_overlap, axis=1)

Y_stft = np.zeros((X.shape[1], X.shape[2]), dtype=complex)
P = 1 # Rango 1 (Aproximación fuerte pero eficiente)
iterations_als = 3 # Cantidad de rebotes entre h1 y h2
diagonal_loading = 1e-3 # Estabilización numérica

print(f"Procesando MVDR Iterativo (P={P}, Iters={iterations_als})...")

I1 = np.eye(M1)
I2 = np.eye(M2)

for k, freq_val in enumerate(f_axis):
    if freq_val < 50: continue # Ignorar DC/muy bajas
    
    # A. DATOS Y ESTADÍSTICA
    # ----------------------
    # Steering Vector Global (d)
    d_vec = steering_vector(freq_val, pos_src, fs, mic_coords, 1)
    
    # Matriz de Covarianza (Phi_v / R) estimada de los datos actuales
    # X_k: (M, TimeFrames)
    X_k = X[:, k, :] 
    R = (X_k @ X_k.conj().T) / X_k.shape[1]
    
    # Diagonal Loading (Crucial para robustez en MVDR)
    R = R + diagonal_loading * np.trace(R).real * np.eye(M)

    # B. INICIALIZACIÓN (Usamos SVD como punto de partida)
    # ----------------------------------------------------
    # Esto nos da un h1 y h2 iniciales razonables para empezar a iterar
    steering_m = d_vec.reshape(M1, M2)
    U, S, Vh = np.linalg.svd(steering_m, full_matrices=False)
    
    # Inicializamos vectores h (compactos)
    h1_compact = np.zeros((M1, P), dtype=complex)
    h2_compact = np.zeros((M2, P), dtype=complex)
    
    for p in range(P):
        h1_compact[:, p] = U[:, p] * np.sqrt(S[p])
        h2_compact[:, p] = Vh[p, :].conj() * np.sqrt(S[p]) # Vh es V^H, tomamos conj para tener v

    # C. BUCLE ALS (ALTERNATING LEAST SQUARES) - Ecs 5.50 y 5.51
    # ----------------------------------------------------------
    for it in range(iterations_als):
        
        # --- PASO 1: Calcular h1 óptimo dado h2 (Eq 5.50) ---
        
        # Construir Matriz H2_subrayada (M x P*M1)
        # H2 = [ H2_1* ... H2_P* ] donde H2_p* = h2_p* (kron) I1
        H2_underlined = np.zeros((M, P*M1), dtype=complex)
        for p in range(P):
            h2_p = h2_compact[:, p].reshape(-1, 1)
            # Nota: La imagen define H2 usando conjugados.
            # Bloque = h2* (kron) I1
            bloque = np.kron(h2_p.conj(), I1)
            H2_underlined[:, p*M1 : (p+1)*M1] = bloque
            
        # Aplicamos fórmula MVDR Eq 5.50
        # Numerador term: (H2^H * R * H2)^-1 * H2^H * d
        # Primero proyectamos R al subespacio reducido (Mucho más chico: P*M1 x P*M1)
        # R_red es la matriz entre paréntesis en el numerador
        R_reduced_2 = H2_underlined.conj().T @ R @ H2_underlined
        
        # Vector proyectado
        d_proj_2 = H2_underlined.conj().T @ d_vec
        
        # Solvemos el sistema lineal (equivale a invertir R_red y multiplicar)
        try:
            h1_new_unscaled = np.linalg.solve(R_reduced_2, d_proj_2)
        except np.linalg.LinAlgError:
            h1_new_unscaled = h1_compact.flatten('F').reshape(-1,1) # Fallback
            
        # Normalización (Denominador de Eq 5.50)
        # alpha = d^H * H2 * (inv) * H2^H * d = d^H * H2 * h1_unscaled
        alpha = d_vec.conj().T @ H2_underlined @ h1_new_unscaled
        h1_final_vec = h1_new_unscaled / alpha
        
        # Actualizamos h1_compact para la siguiente vuelta
        # Desempaquetamos el vector largo a matriz (M1, P)
        for p in range(P):
            h1_compact[:, p] = h1_final_vec[p*M1 : (p+1)*M1].flatten()


        # --- PASO 2: Calcular h2 óptimo dado h1 (Eq 5.51) ---
        
        # Construir Matriz H1_subrayada (M x P*M2)
        # H1 = [ H1_1 ... H1_P ] donde H1_p = I2 (kron) h1_p
        H1_underlined = np.zeros((M, P*M2), dtype=complex)
        for p in range(P):
            h1_p = h1_compact[:, p].reshape(-1, 1)
            # Bloque = I2 (kron) h1
            bloque = np.kron(I2, h1_p)
            H1_underlined[:, p*M2 : (p+1)*M2] = bloque
            
        # Aplicamos fórmula MVDR Eq 5.51
        # R_red es (P*M2 x P*M2)
        R_reduced_1 = H1_underlined.conj().T @ R @ H1_underlined
        d_proj_1 = H1_underlined.conj().T @ d_vec
        
        try:
            h2_new_unscaled = np.linalg.solve(R_reduced_1, d_proj_1)
        except np.linalg.LinAlgError:
            break
            
        # Normalización
        beta = d_vec.conj().T @ H1_underlined @ h2_new_unscaled
        h2_final_vec = h2_new_unscaled / beta
        
        # Actualizamos h2_compact
        for p in range(P):
            h2_compact[:, p] = h2_final_vec[p*M2 : (p+1)*M2].flatten()

    # D. APLICACIÓN DEL FILTRO OPTIMIZADO
    # -----------------------------------
    # Una vez convergido, usamos los h1 y h2 finales para filtrar
    output_frec_k = np.zeros(X_k.shape[1], dtype=complex)
    
    # Snapshot actual: (M1, M2, Time)
    X_snapshot_reshaped = X_k.reshape(M1, M2, -1)
    
    for p in range(P):
        # Tomamos los pesos optimizados. 
        # OJO: En beamforming w = h*. Como calculamos h_MVDR, usamos conj() para aplicar.
        w1 = h1_compact[:, p].conj()
        w2 = h2_compact[:, p].conj()
        
        # Filtrado separable eficiente
        # 1. Colapsar columnas con w2
        intermedio = np.einsum('ijk, j -> ik', X_snapshot_reshaped, w2)
        # 2. Colapsar filas con w1
        rama_p = w1 @ intermedio
        output_frec_k += rama_p
        
    Y_stft[k, :] = output_frec_k

# --- 4. RESULTADOS ---
t_out, signal_out = signal.istft(Y_stft, fs=fs, nperseg=n_window, noverlap=n_overlap)
min_len = min(len(t), len(signal_out), array_input.shape[1])
# --- BLOQUE DE VISUALIZACIÓN COMPARATIVA (Side-by-Side) ---

# 1. Preparación de datos (asegurar mismo largo y tipo)
min_len = min(len(array_input[0]), len(signal_out))
sig_in = array_input[0, :min_len].real
sig_out = signal_out[:min_len].real
t_plot = np.arange(min_len) / fs

# Configuración de rango dinámico visual (dB)
# Un rango de 100 dB es estándar para ver piso de ruido
DYNAMIC_RANGE_DB = 100 

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# --- GRÁFICO 1: ENTRADA (Input) ---
# Dejamos que matplotlib calcule el espectro, pero capturamos la imagen (im1)
Pxx, freqs, bins, im1 = ax1.specgram(sig_in, NFFT=1024, Fs=fs, noverlap=512, cmap='inferno')
ax1.set_title("Entrada (Con Ruido)")
ax1.set_ylabel("Frecuencia [Hz]")
ax1.set_xlabel("Tiempo [s]")

# TRUCO CLAVE: Obtener el máximo nivel detectado en la entrada para fijar la escala
# Esto asegura que el "rojo furioso" sea el mismo nivel absoluto en ambos gráficos.
vmax_in = im1.get_clim()[1] # El valor máximo de dB detectado
vmin_in = vmax_in - DYNAMIC_RANGE_DB

# Forzamos la escala en el primer gráfico
im1.set_clim(vmin_in, vmax_in)

# --- GRÁFICO 2: SALIDA (Beamformed) ---
# Usamos vmin y vmax explícitos obtenidos del gráfico 1
Pxx, freqs, bins, im2 = ax2.specgram(sig_out, NFFT=1024, Fs=fs, noverlap=512, 
                                     cmap='inferno', vmin=vmin_in, vmax=vmax_in)
ax2.set_title(f"Salida (Low-Rank MVDR)")
ax2.set_xlabel("Tiempo [s]")

# --- BARRA DE COLOR COMPARTIDA ---
# Ajustamos el layout para dejar lugar a la barra
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
cbar = fig.colorbar(im1, cax=cbar_ax)
cbar.set_label('Intensidad [dB]')

plt.suptitle("Comparación Espectral Directa (Misma Escala de Color)", fontsize=14)
plt.show()

# Exportar
def normalize_wav(data):
    return np.int16(data / np.max(np.abs(data)) * 32767)

wavfile.write("input_noisy.wav", fs, normalize_wav(array_input[0, :min_len]))
wavfile.write("output_mvdr_als.wav", fs, normalize_wav(signal_out[:min_len]))
print("Archivos de audio generados.")