import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import speed_of_sound
from scipy.io import wavfile

# --- IMPORTACIONES DE TUS MÓDULOS ---
# Asumo que el path es correcto y los archivos existen
from beamforming.signal_model import near_field_steering_vector_multi
from beamforming.algorithms.region_constriant import build_region_constraints
from beamforming.processors import gsc_adaptive_beamformer

# ==============================================================================
# 0. FUNCIÓN AUXILIAR PARA EXPORTAR AUDIO
# ==============================================================================
def save_normalized_wav(filename, rate, data, folder="resultados_audio"):
    """
    Normaliza el audio (float), lo convierte a int16 y lo guarda en una carpeta.
    """
    # Crear carpeta si no existe
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Carpeta creada: {folder}")
    
    filepath = os.path.join(folder, filename)
    
    # Asegurar parte real
    data_real = np.real(data)
    
    # Normalizar al pico máximo (evitar clipping)
    max_val = np.max(np.abs(data_real))
    if max_val > 0:
        data_norm = data_real / max_val
        # Bajamos un poquito el volumen (-1 dB) para margen de seguridad
        data_norm = data_norm * 0.9
    else:
        data_norm = data_real
        
    # Convertir a int16
    data_int16 = (data_norm * 32767).astype(np.int16)
    
    wavfile.write(filepath, rate, data_int16)
    print(f"  -> Guardado: {filepath}")

# ==============================================================================
# 1. CONFIGURACIÓN DEL ESCENARIO
# ==============================================================================
print("\n--- 1. Configurando Parámetros ---")
fs = 48000
K = 25
C_SOUND = speed_of_sound

# Banda de diseño (Broadband)
f_min_band = 1000.0
f_max_band = 4000.0
num_freqs_band = 50 
num_points_space = 50

# Geometría del Arreglo (9 micrófonos, ULA)
f_ref_geometry = 3000.0
LAMBDA_REF = C_SOUND / f_ref_geometry
M = 9
D = LAMBDA_REF / 2
mic_x = np.linspace(0, (M - 1) * D, M) - (M - 1) * D / 2
mic_array = np.stack([mic_x, np.zeros(M), np.zeros(M)], axis=1)

# Punto Focal (90 grados)
radius = 5 * LAMBDA_REF
focal_angle_rad = np.deg2rad(90.0)
focal_point_cartesian = np.array([
    radius * np.cos(focal_angle_rad), 
    radius * np.sin(focal_angle_rad), 
    0.0
])

# ==============================================================================
# 2. DISEÑO DEL BEAMFORMER (OFFLINE - REGIONAL ROBUSTO)
# ==============================================================================
print("\n--- 2. Diseñando Filtro Robusto (LCMV-RB) ---")

delta_r_val = radius * 0.2     # +/- 10% distancia
delta_azimut_val = np.deg2rad(8.0) # +/- 4 grados
delta_elev_val = np.deg2rad(2.0) 

# Obtenemos Restricciones (C), Respuesta Deseada (h) y Matriz de Bloqueo (Ca)
C_rb, h_rb, Ca_rb = build_region_constraints(
    Rs=focal_point_cartesian, 
    delta_r=delta_r_val, 
    delta_azimut=delta_azimut_val,
    delta_elevation=delta_elev_val, 
    mic_array=mic_array, 
    fs=fs, 
    K=K,
    f_min=f_min_band, 
    f_max=f_max_band, 
    num_points=num_points_space, 
    num_freqs=num_freqs_band
)

# Calculamos pesos fijos w_q = C * h
w_lcmv_rb = (C_rb @ h_rb.flatten())

print(f"Diseño listo. Rango L={C_rb.shape[1]}. Ca shape={Ca_rb.shape}")

# ==============================================================================
# 3. GENERACIÓN DE SEÑALES (SIMULACIÓN TEMPORAL)
# ==============================================================================
print("\n--- 3. Generando Señales (5 segundos) ---")
duration = 5.0 # 5 segundos para escuchar bien
t = np.arange(int(duration * fs)) / fs
n_samples = len(t)

# A. Señal Deseada (Target) en el FOCO (3000 Hz) - "Beep" limpio
freq_target = 3000.0
# Hacemos que pulse para distinguirla mejor
target_envelope = np.sin(2 * np.pi * 1.0 * t)**2 # Pulsa 2 veces por segundo
target_signal = 0.05 * np.sin(2 * np.pi * freq_target * t) # * target_envelope

# Steering vector focal (usamos narrowband para simular la propagación simple)
sv_target = near_field_steering_vector_multi(freq_target, focal_point_cartesian, fs, mic_array, 1).squeeze()
mic_target = np.outer(sv_target, target_signal)

# B. Interferencia (40 GRADOS) - Ruido + Tono molesto
angle_int = np.deg2rad(40.0)
pos_int = np.array([radius * np.cos(angle_int), radius * np.sin(angle_int), 0.0])
freq_int = 1200.0 # Tono molesto
# Ruido blanco fuerte + Tono
interference_signal = 0.3 * np.random.randn(n_samples) + 0.15 * np.sin(2 * np.pi * freq_int * t)

sv_interf = near_field_steering_vector_multi(freq_int, pos_int, fs, mic_array, 1).squeeze()
mic_interf = np.outer(sv_interf, interference_signal)

# C. Mezcla Total (Entrada a los micros) + Ruido Térmico de fondo
mic_data = mic_target + mic_interf + 0.002 * np.random.randn(M, n_samples)

print("Señales generadas. La interferencia es mucho más fuerte que el target.")

# ==============================================================================
# 4. PROCESAMIENTO (GSC y FIJO)
# ==============================================================================
print("\n--- 4. Procesando Audio ---")

# A. Caso Adaptativo (GSC)
print("  -> Corriendo GSC Adaptativo (mu=0.05)...")
y_adaptive, w_log = gsc_adaptive_beamformer(
    input_signal=mic_data,
    w_q=w_lcmv_rb,    
    Ca=Ca_rb,         
    K=K,
    mu=0.05           
)

# B. Caso Fijo (Referencia)
print("  -> Corriendo Beamformer Fijo (Solo w_q)...")
y_fixed, _ = gsc_adaptive_beamformer(
    input_signal=mic_data,
    w_q=w_lcmv_rb,
    Ca=Ca_rb,
    K=K,
    mu=0.0            
)

# ==============================================================================
# 5. EXPORTACIÓN DE AUDIO
# ==============================================================================
print("\n--- 5. Exportando WAVs ---")
out_folder = "resultados_audio"

# Guardar Target puro (Referencia de cómo debería sonar)
save_normalized_wav("0_target_limpio_ref.wav", fs, target_signal, out_folder)

# Guardar Entrada Mic 0 (Lo que se escucha sin procesar)
save_normalized_wav("1_entrada_mic0.wav", fs, mic_data[0, :], out_folder)

# Guardar Salida Fija (Solo atenuación espacial básica)
save_normalized_wav("2_salida_fija.wav", fs, y_fixed, out_folder)

# Guardar Salida Adaptativa (Cancelación activa)
save_normalized_wav("3_salida_gsc_adaptativo.wav", fs, y_adaptive, out_folder)


# ==============================================================================
# 6. PLOTEO RÁPIDO
# ==============================================================================
print("\n--- 6. Graficando ---")
# Zoom al principio (donde hay interferencia) y al medio (donde ya convergió)
t_zoom = t[:2000] * 1000 # Primeros 2000 samples

plt.figure(figsize=(10, 10))

plt.subplot(3, 1, 1)
plt.title("Entrada Mic 0 (Ruidosa)")
plt.plot(t_zoom, np.real(mic_data[0, :2000]), color='gray')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.title("Salida Fija (Atenuación parcial)")
plt.plot(t_zoom, np.real(y_fixed[:2000]), color='orange')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.title("Salida GSC (Convergencia y Limpieza)")
plt.plot(t_zoom, np.real(y_adaptive[:2000]), color='green', label='GSC')
plt.plot(t_zoom, target_signal[:2000], 'k--', alpha=0.5, label='Target')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"\nPROCESO TERMINADO. Revisa la carpeta '{out_folder}'.")