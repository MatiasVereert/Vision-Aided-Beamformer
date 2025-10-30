
import numpy as np
from beamforming.processors import point_constraint, compute_fixed_weights_optimized 
from beamforming.models import near_field_steering_vector
from beamforming.gain import calculate_gain_at_points, polar_gain
from utils.geometry import source_rotation
from utils.polar_plot import plot_polar_pattern

# --- 2. PARÁMETROS DEL ESCENARIO ---
fs = 48000
c = 343.0
M = 7      # Número de micrófonos
K = 50      # Número de taps
f_test = 3000.0 # Frecuencia de prueba para LCMV

# Geometría del Array (ULA en el eje X, centrado)
d = 0.04
mic_x = np.linspace(0, (M - 1) * d, M) - (M - 1) * d / 2
mic_array = np.stack([mic_x, np.zeros(M), np.zeros(M)], axis=1)

# Dirección y distancia de enfoque
look_direction_rad = np.deg2rad(70)
source_distance = 2.0
focal_point = source_distance * np.array([np.cos(look_direction_rad), np.sin(look_direction_rad), 0])

# --- 3. CÁLCULO DE PESOS LCMV ---
# Este paso no cambia: se calculan los pesos una sola vez.
print("--- Calculando pesos del beamformer LCMV ---")
C, h = point_constraint(focal_point, K, mic_array, f_test, fs)
w_lcmv = compute_fixed_weights_optimized(C, h)
print("Pesos calculados.")

# --- 4. VERIFICACIÓN DE LA DIRECTIVIDAD DEL BEAMFORMER (LÓGICA CORREGIDA) ---
print("\n--- Verificando la directividad del beamformer ---")

# 4.1. Definir una segunda dirección de prueba (un "nulo" a 90 grados del foco)
null_direction_rad = look_direction_rad + np.pi / 2 
null_point = source_distance * np.array([np.cos(null_direction_rad), np.sin(null_direction_rad), 0])

# 4.2. Puntos a probar: el foco y el punto nulo
points_to_test = np.array([focal_point, null_point])

# 4.3. Calcular la ganancia física en AMBOS puntos
gains_db = calculate_gain_at_points(
    f=f_test,
    fs=fs,
    mic_array=mic_array,
    weights=w_lcmv,
    source_points=points_to_test
)

gain_at_focal_db = gains_db[0]
gain_at_null_db = gains_db[1]

print(f"Ganancia de Arreglo en el foco ({np.rad2deg(look_direction_rad):.0f}°): {gain_at_focal_db:.4f} dB")
print(f"Ganancia de Arreglo en el nulo ({np.rad2deg(null_direction_rad):.0f}°): {gain_at_null_db:.4f} dB")

# 4.4. EL NUEVO TEST: Verificar que la atenuación es significativa
attenuation = gain_at_focal_db - gain_at_null_db
min_expected_attenuation = 15.0  # Umbral razonable: esperamos al menos 15 dB de atenuación

print(f"Atenuación entre foco y nulo: {attenuation:.2f} dB")

if attenuation > min_expected_attenuation:
    print(f"✅ TEST PASADO: La directividad es buena. La atenuación ({attenuation:.2f} dB) supera el umbral de {min_expected_attenuation} dB.")
else:
    print(f"❌ TEST FALLIDO: La directividad es pobre. La atenuación ({attenuation:.2f} dB) no alcanza el umbral de {min_expected_attenuation} dB.")

# --- El resto del script (sección 5) permanece exactamente igual ---
# --- 5. CÁLCULO Y GRÁFICO DEL PATRÓN POLAR (USANDO FUNCIONES MODULARES) ---
print("\n--- Calculando y graficando el Patrón Polar ---")

# 5.1. Generar la geometría de los puntos de prueba
points = 181
source_points_3d, angles_deg = source_rotation(radius=source_distance, samples=points)
source_points_transposed = source_points_3d.T # Convertir de (3, P) a (P, 3)

# 5.2. Calcular la ganancia para todos los puntos con UNA SOLA LLAMADA
gains_absolute_db = calculate_gain_at_points(
    f=f_test,
    fs=fs,
    mic_array=mic_array,
    weights=w_lcmv,
    source_points=source_points_transposed
)

# 5.3. Normalizar la ganancia para que el pico sea 0 dB (requerido para el patrón polar)
gains_normalized_db = gains_absolute_db - np.max(gains_absolute_db)

# 5.4. Graficar el resultado con UNA SOLA LLAMADA a la función de ploteo
plot_polar_pattern(
    gains_normalized_db,
    angles_deg,
    title=f"Patrón Polar LCMV a {f_test} Hz (Foco a {np.rad2deg(look_direction_rad):.0f}°)"
)

print("Script finalizado.")