import numpy as np
# Asumimos que los nombres de las funciones son los que definimos
from beamforming.processors import point_constraint, compute_fixed_weights_optimized 
from beamforming.models import near_field_steering_vector, near_field_steering_vector_multi # Usamos la versión vectorizada
from beamforming.evaluation.gain import synthetic_gain, analytical_gain # Nombres más claros y correctos
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

# --- 4. VERIFICACIÓN DE LA DIRECTIVIDAD DEL BEAMFORMER ---
# Esta sección no se modifica, sigue usando un método para la prueba.
print("\n--- Verificando la directividad del beamformer (método sintético) ---")
null_direction_rad = look_direction_rad + np.pi / 2 
null_point = source_distance * np.array([np.cos(null_direction_rad), np.sin(null_direction_rad), 0])
points_to_test = np.array([focal_point, null_point])

gains_db = synthetic_gain(
    f=f_test,
    fs=fs,
    mic_array=mic_array,
    weights=w_lcmv,
    source_points=points_to_test
)
attenuation = gains_db[0] - gains_db[1]
min_expected_attenuation = 15.0
print(f"Atenuación entre foco y nulo: {attenuation:.2f} dB")
if attenuation > min_expected_attenuation:
    print(f"✅ TEST PASADO: La directividad es buena.")
else:
    print(f"❌ TEST FALLIDO: La directividad es pobre.")


# --- 5. CÁLCULO Y GRÁFICO DE AMBOS PATRONES POLARES ---
print("\n--- Calculando y graficando Patrones Polares (Sintético vs. Analítico) ---")

# 5.1. Generar la geometría de los puntos de prueba (no cambia)
points = 181
source_points_3d, angles_deg = source_rotation(radius=source_distance, samples=points)
source_points_transposed = source_points_3d.T # Convertir de (3, P) a (P, 3)

# 5.2. Calcular la ganancia con AMBOS métodos
# Método 1: Simulación Sintética (como antes)
print("Calculando ganancia con método sintético...")
gains_synthetic_db = synthetic_gain(
    f=f_test,
    fs=fs,
    mic_array=mic_array,
    weights=w_lcmv,
    source_points=source_points_transposed
)

### NUEVO ###
# Método 2: Cálculo Analítico
print("Calculando ganancia con método analítico...")
# La función analítica espera un array de frecuencias, aunque sea una sola.
freqs_to_test = np.array([f_test])
gains_analytical_2d = analytical_gain(
    frecs=freqs_to_test,
    fs=fs,
    mic_array=mic_array,
    weights=w_lcmv,
    source_points=source_points_transposed
)
# El resultado tiene forma (F, P). Como F=1, seleccionamos la primera fila.
gains_analytical_db = gains_analytical_2d[0]
print("Cálculos finalizados.")

# 5.3. Normalizar AMBOS resultados para que el pico de cada uno sea 0 dB
gains_synthetic_normalized = gains_synthetic_db - np.max(gains_synthetic_db)
### NUEVO ###
gains_analytical_normalized = gains_analytical_db - np.max(gains_analytical_db)

# 5.4. Graficar los resultados superpuestos
### MODIFICADO ###
# Para lograr esto, tu función plot_polar_pattern debe ser modificada para
# aceptar una lista de ganancias y una lista de etiquetas.
# Ejemplo de la nueva firma esperada:
# def plot_polar_pattern(gains_list, angles_deg, labels_list, title=""):

plot_polar_pattern(
    gains_list=[gains_synthetic_normalized, gains_analytical_normalized],
    angles_deg=angles_deg,
    labels_list=["Sintético", "Analítico"],
    title=f"Patrón Polar LCMV a {f_test} Hz (Foco a {np.rad2deg(look_direction_rad):.0f}°)"
)

print("Script finalizado.")