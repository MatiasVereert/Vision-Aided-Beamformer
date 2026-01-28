import numpy as np 
from matplotlib import pyplot as plt
from scipy import signal

# --- 1. TU FUNCIÓN DE RETARDO EXACTO (INTEGRADA) ---
def space_delay(signal_in, fs, source_pos, mic_array):
    """
    Simula la propagación banda ancha usando FFT para retardos fraccionarios exactos.
    """
    # Aseguramos arrays
    signal_in = np.array(signal_in)
    source_pos = np.atleast_2d(source_pos) 
    mic_array = np.atleast_2d(mic_array)
    
    N_original = len(signal_in)
    
    # 1. Cálculo de retardos
    # diff_vectors: (1, M, 3)
    diff_vectors = source_pos[:, np.newaxis, :] - mic_array[np.newaxis, :, :]
    distancias = np.linalg.norm(diff_vectors, axis=2) 
    
    # CORRECCIÓN: Usamos la constante global C_SOUND definida abajo
    tau_array = distancias / C_SOUND 
    
    # 2. Longitud FFT (Potencia de 2)
    max_delay_samples = int(np.ceil(np.max(tau_array) * fs))
    N_fft = 2**(int(np.ceil(np.log2(N_original + max_delay_samples))))
    
    # 3. FFT
    signal_padded = np.pad(signal_in, (0, N_fft - N_original), 'constant')
    X = np.fft.fft(signal_padded)
    k = np.fft.fftfreq(N_fft, d=1/fs)
    
    # 4. Fase (Shift Theorem)
    # tau_array: (1, M), k: (N_fft) -> Broadcasting (1, M, N_fft)
    phase_shift_matrix = np.exp(-1j * 2 * np.pi * k * tau_array[..., np.newaxis])
    
    # 5. IFFT
    Y_matrix = X * phase_shift_matrix
    array_retardado_complex = np.fft.ifft(Y_matrix, axis=-1)
    array_retardado = array_retardado_complex.real

    if array_retardado.shape[0] == 1:
        array_retardado = np.squeeze(array_retardado, axis=0)
    
    # Devolvemos la señal completa (con cola) y las distancias para atenuar después
    return array_retardado, distancias.flatten()

# --- 2. CONSTANTES Y CONFIGURACIÓN ---
# Imports adicionales necesarios para tu script original
from beamforming.signal_model import steering_vector

fs = 48000
C_SOUND = 343 # Definido aquí para que lo use space_delay
fmin = 200
fmax = 10000

# Dimensiones del Array
mic_spacing = 0.05 # Aumenté un poco para el ejemplo (5cm), ajustá a tu gusto
M1 = 3
M2 = 4
M = M1 * M2

# Grilla rectangular
x = np.linspace(0, (M2-1)*mic_spacing, M2)
y = np.linspace(0, (M1-1)*mic_spacing, M1)
xv, yv = np.meshgrid(x, y)
mic_coords = np.column_stack([xv.flatten(), yv.flatten(), np.zeros(M)]) # (M, 3)

# Señales
f_source = 1000
time_duration = 0.5
t = np.arange(0, time_duration, 1/fs)
source_signal = np.sin(2 * np.pi * t * f_source)
noise_signal = np.random.rand(len(t)) - 0.5

source_pos = [1.5, 1.5, 1.0]
noise_pos = [-1.5, 1.5, 1.0]

# --- 3. GENERACIÓN CORRECTA DE ENTRADA (SOLUCIÓN DEL ERROR) ---
print("Generando señales con space_delay...")

# A. Fuente: Retardo + Atenuación (1/r)
raw_src, dists_src = space_delay(source_signal, fs, source_pos, mic_coords)
mics_src = raw_src[:, :len(t)] / dists_src[:, np.newaxis] # Recortamos y atenuamos

# B. Ruido: Retardo + Atenuación (1/r)
raw_noise, dists_noise = space_delay(noise_signal, fs, noise_pos, mic_coords)
mics_noise = raw_noise[:, :len(t)] / dists_noise[:, np.newaxis]

# C. Suma MATRICIAL (Ahora sí son arrays numéricos)
array_input = mics_src + mics_noise 
print(f"Input shape: {array_input.shape}") # Debería ser (12, N_muestras)

# --- 4. BEAMFORMING LOW RANK (TU LÓGICA) ---

n_window = 1024
n_overlap = 512

# STFT
f_axis, t_axis, X = signal.stft(x=array_input, 
                                fs=fs, 
                                nperseg=n_window, 
                                noverlap=n_overlap, 
                                window='hann', 
                                axis=1)

Y_stft = np.zeros((X.shape[1], X.shape[2]), dtype=complex)
P = 1 

print("Procesando señal Banda Ancha...")

for k, freq_val in enumerate(f_axis):
    
    if freq_val < 20:
        continue

    # A. SVD para esta frecuencia
    sv_k = steering_vector(freq_val, source_pos, fs, mic_coords, 1)
    steering_m = sv_k.reshape(M1, M2)
    U, S, Vh = np.linalg.svd(steering_m, full_matrices=False)
    
    # B. Aplicación Eficiente (Separable)
    X_k_snapshot = X[:, k, :].reshape(M1, M2, -1)
    output_frec_k = np.zeros(X_k_snapshot.shape[2], dtype=complex)
    
    for p in range(P):
        sigma_sqrt = np.sqrt(S[p])
        w1_p = (U[:, p] * sigma_sqrt).conj()
        w2_p = (Vh[p, :] * sigma_sqrt).conj()
        
        # Filtro Columnas -> Filas
        intermedio = np.einsum('ijk, j -> ik', X_k_snapshot, w2_p)
        salida_rama = w1_p @ intermedio
        output_frec_k += salida_rama
        
    Y_stft[k, :] = output_frec_k
from scipy.io import wavfile

# --- 5. RESULTADOS Y EXPORTACIÓN ---

# Recortamos para que tengan el mismo largo exacto
t_out, signal_out = signal.istft(Y_stft, fs=fs, nperseg=n_window, noverlap=n_overlap, window='hann')
min_len = min(len(t), len(signal_out), array_input.shape[1])

# Señales a comparar
# Tomamos la parte real (por si quedan residuos imaginarios ínfimos de la IFFT)
audio_in = array_input[0, :min_len].real
audio_out = signal_out[:min_len].real
t_plot = t[:min_len]

# --- A. GRÁFICOS COMPARATIVOS ---
plt.figure(figsize=(12, 10))

# 1. Tiempo - Entrada
plt.subplot(4, 1, 1)
plt.title("Entrada (Micrófono 1) - Dominio del Tiempo")
plt.plot(t_plot, audio_in, color='gray', alpha=0.8)
plt.ylabel("Amplitud")
plt.grid(True, alpha=0.3)

# 2. Espectrograma - Entrada
plt.subplot(4, 1, 2)
plt.title("Entrada (Micrófono 1) - Espectrograma")
plt.specgram(audio_in, Fs=fs, NFFT=1024, noverlap=512, cmap='inferno')
plt.ylabel("Frecuencia [Hz]")

# 3. Tiempo - Salida
plt.subplot(4, 1, 3)
plt.title("Salida Beamformer Low-Rank - Dominio del Tiempo")
plt.plot(t_plot, audio_out, color='tab:blue')
plt.ylabel("Amplitud")
plt.grid(True, alpha=0.3)

# 4. Espectrograma - Salida
plt.subplot(4, 1, 4)
plt.title("Salida Beamformer Low-Rank - Espectrograma")
plt.specgram(audio_out, Fs=fs, NFFT=1024, noverlap=512, cmap='inferno')
plt.ylabel("Frecuencia [Hz]")
plt.xlabel("Tiempo [s]")

plt.tight_layout()
plt.show()

# --- B. EXPORTACIÓN A WAV ---

def save_wav_normalized(filename, data, fs):
    """Normaliza a float32 entre -1 y 1 y guarda"""
    # Evitar división por cero
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data_norm = data / max_val
    else:
        data_norm = data
    
    # Aplicamos un factor de seguridad (0.95) para evitar clipping al reproducir
    data_norm = data_norm * 0.95
    wavfile.write(filename, fs, data_norm.astype(np.float32))
    print(f"Archivo guardado: {filename}")

print("\n--- Exportando Audio ---")
save_wav_normalized("input_mic1_noisy.wav", audio_in, fs)
save_wav_normalized("output_beamformed_clean.wav", audio_out, fs)