import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import speed_of_sound
from scipy.io import wavfile

# --- IMPORTACIONES ---
from beamforming.signal_model import near_field_steering_vector_multi
from beamforming.algorithms.region_constriant import build_region_constraints
from beamforming.processors import gsc_adaptive_beamformer

# ------------------------------------------------------------------------------
# 1. TU FUNCIÓN DE PROPAGACIÓN (Copiada aquí para integración)
# ------------------------------------------------------------------------------
def space_delay(signal, fs, source_pos, mic_array):
    """
    Simula la propagación banda ancha usando FFT para retardos fraccionarios exactos.
    """
    N_original = len(signal)
    source_pos = np.atleast_2d(source_pos) 
    
    # 1. Cálculo de retardos
    diff_vectors = source_pos[:, np.newaxis, :] - mic_array[np.newaxis, :, :]
    distancias = np.linalg.norm(diff_vectors, axis=2) 
    tau_array = distancias / speed_of_sound 
    
    # 2. Longitud FFT (Potencia de 2)
    max_delay_samples = int(np.ceil(np.max(tau_array) * fs))
    N_fft = 2**(int(np.ceil(np.log2(N_original + max_delay_samples))))
    
    # 3. FFT
    signal_padded = np.pad(signal, (0, N_fft - N_original), 'constant')
    X = np.fft.fft(signal_padded)
    k = np.fft.fftfreq(N_fft, d=1/fs)
    
    # 4. Fase
    phase_shift_matrix = np.exp(-1j * 2 * np.pi * k * tau_array[..., np.newaxis])
    
    # 5. IFFT
    Y_matrix = X * phase_shift_matrix
    array_retardado_complex = np.fft.ifft(Y_matrix, axis=-1)
    array_retardado = array_retardado_complex.real
    
    return array_retardado, signal_padded, tau_array

# ------------------------------------------------------------------------------
# 2. FUNCIÓN AUXILIAR DE EXPORTACIÓN
# ------------------------------------------------------------------------------
def save_normalized_wav(filename, rate, data, folder="resultados_audio"):
    if not os.path.exists(folder): os.makedirs(folder)
    filepath = os.path.join(folder, filename)
    data_real = np.real(data)
    max_val = np.max(np.abs(data_real))
    data_norm = (data_real / max_val * 0.9) if max_val > 0 else data_real
    wavfile.write(filepath, rate, (data_norm * 32767).astype(np.int16))
    print(f"  -> Guardado: {filepath}")

# ==============================================================================
# 3. CONFIGURACIÓN Y DISEÑO DEL BEAMFORMER (Igual que antes)
# ==============================================================================
print("\n--- 1. Configurando Parámetros ---")
fs = 48000
K = 25
C_SOUND = speed_of_sound

# Geometría
f_ref_geometry = 3000.0
LAMBDA_REF = C_SOUND / f_ref_geometry
M = 9
D = LAMBDA_REF / 2
mic_x = np.linspace(0, (M - 1) * D, M) - (M - 1) * D / 2
mic_array = np.stack([mic_x, np.zeros(M), np.zeros(M)], axis=1)

# Foco
radius = 5 * LAMBDA_REF
focal_angle_rad = np.deg2rad(90.0)
focal_point_cartesian = np.array([
    radius * np.cos(focal_angle_rad), radius * np.sin(focal_angle_rad), 0.0
])

print("\n--- 2. Diseñando Filtro Robusto (LCMV-RB) ---")
# Usamos parámetros banda ancha para el diseño de pesos w_q
C_rb, h_rb, Ca_rb = build_region_constraints(
    Rs=focal_point_cartesian, 
    delta_r=radius * 0.2, 
    delta_azimut=np.deg2rad(8.0),
    delta_elevation=np.deg2rad(2.0), 
    mic_array=mic_array, fs=fs, K=K,
    f_min=1000.0, f_max=4000.0, num_points=50, num_freqs=50
)
w_lcmv_rb = (C_rb @ h_rb.flatten())
print("Diseño listo.")

# ==============================================================================
# 4. GENERACIÓN DE SEÑALES (USANDO TU NUEVA FUNCIÓN)
# ==============================================================================
print("\n--- 3. Generando Señales (Propagación Banda Ancha) ---")
duration = 4.0 
t_src = np.arange(int(duration * fs)) / fs

# A. Señal Deseada (Fuente Mono 1D)
# "Beep" pulsante a 3kHz
target_mono = 0.05 * np.sin(2 * np.pi * 3000.0 * t_src) * (np.sin(2 * np.pi * 2.0 * t_src)**2)

# B. Interferencia (Fuente Mono 1D)
# Ruido + tono molesto
interf_mono = 0.3 * np.random.randn(len(t_src)) + 0.1 * np.sin(2 * np.pi * 1200.0 * t_src)

# C. Posición de Interferencia
angle_int = np.deg2rad(40.0)
pos_int = np.array([radius * np.cos(angle_int), radius * np.sin(angle_int), 0.0])


# --- APLICANDO TU FUNCIÓN SPACE_DELAY ---
print("  Propagando Target...")
# Retorna (1, M, N_fft) porque pasamos 1 posición
mics_target_prop, ref_target_padded, _ = space_delay(target_mono, fs, focal_point_cartesian, mic_array)
mics_target_prop = mics_target_prop[0] # Quitamos la dimensión de fuente (M, N_fft)

print("  Propagando Interferencia...")
mics_interf_prop, _, _ = space_delay(interf_mono, fs, pos_int, mic_array)
mics_interf_prop = mics_interf_prop[0] # (M, N_fft)

# --- ALINEACIÓN DE LONGITUDES ---
# Como space_delay usa potencias de 2, si las distancias son muy distintas podría
# dar longitudes distintas. Normalizamos al máximo.
len_t = mics_target_prop.shape[1]
len_i = mics_interf_prop.shape[1]
max_len = max(len_t, len_i)

# Padding si es necesario
if len_t < max_len:
    mics_target_prop = np.pad(mics_target_prop, ((0,0), (0, max_len-len_t)))
    ref_target_padded = np.pad(ref_target_padded, (0, max_len-len_t))
if len_i < max_len:
    mics_interf_prop = np.pad(mics_interf_prop, ((0,0), (0, max_len-len_i)))

# D. Mezcla Final
# Suma coherente + Ruido térmico no correlacionado
print("  Mezclando señales...")
mic_data = mics_target_prop + mics_interf_prop 
mic_data += 0.002 * np.random.randn(M, max_len) # Ruido térmico en cada mic

# Nuevo vector de tiempo para plotear (basado en N_fft)
t_sim = np.arange(max_len) / fs

print(f"Simulación lista. Muestras procesadas: {max_len}")

# ==============================================================================
# 5. PROCESAMIENTO (GSC)
# ==============================================================================
print("\n--- 4. Procesando Audio ---")

print("  -> Corriendo GSC Adaptativo (mu=0.05)...")
y_adaptive, w_log = gsc_adaptive_beamformer(
    input_signal=mic_data,
    w_q=w_lcmv_rb,    
    Ca=Ca_rb,         
    K=K,
    mu=0.05           
)

print("  -> Corriendo Beamformer Fijo...")
y_fixed, _ = gsc_adaptive_beamformer(
    input_signal=mic_data,
    w_q=w_lcmv_rb,
    Ca=Ca_rb,
    K=K,
    mu=0.0            
)

# ==============================================================================
# 6. PLOTEO Y EXPORTACIÓN
# ==============================================================================
print("\n--- 5. Resultados ---")

# Exportar WAVs
folder = "resultados_audio_broadband"
save_normalized_wav("0_target_ref.wav", fs, ref_target_padded, folder) # Usamos el padded como referencia
save_normalized_wav("1_mic0_input.wav", fs, mic_data[0, :], folder)
save_normalized_wav("2_output_fijo.wav", fs, y_fixed, folder)
save_normalized_wav("3_output_gsc.wav", fs, y_adaptive, folder)

# Gráficos
plt.figure(figsize=(10, 10))

# Zoom para ver detalles (aprox en el medio del audio)
zoom_s = int(max_len/2)
zoom_e = zoom_s + 2000
t_zoom = t_sim[zoom_s:zoom_e] * 1000

plt.subplot(3, 1, 1)
plt.title("Entrada Mic 0 (Interferencia domina)")
plt.plot(t_zoom, np.real(mic_data[0, zoom_s:zoom_e]), 'gray')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.title("Salida GSC vs Target")
plt.plot(t_zoom, np.real(y_adaptive[zoom_s:zoom_e]), 'green', label='GSC Output')
plt.plot(t_zoom, ref_target_padded[zoom_s:zoom_e], 'k--', alpha=0.8, label='Target Limpio')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.title("Evolución de Pesos (Adaptación)")
plt.plot(w_log)
plt.xlabel("Iteraciones"); plt.ylabel("Norma |wa|")
plt.grid(True)

plt.tight_layout()
plt.show()