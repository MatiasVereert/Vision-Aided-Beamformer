import numpy as np
from scipy.constants import speed_of_sound
from matplotlib import pyplot as plt 

# --- Importaciones de tus módulos (asume que están en el path) ---
from beamforming.signal_model import near_field_steering_vector_multi
from beamforming.evaluation.gain import analytical_gain
from beamforming.algorithms.lmcv import compute_fixed_weights_optimized
from utils.geometry import source_rotation

print("Configurando parámetros (MODIFICADOS PARA COINCIDIR CON FIG. 3)...")
fs = 48000

# Parámetros físicos
FS = 48000
C_SOUND = speed_of_sound
f_test = 3000.0 # Frecuencia de prueba (nuestra lambda de referencia)

# --- PARÁMETROS MODIFICADOS ---
# Lambda de referencia (lambda = c / f)
LAMBDA_REF = C_SOUND / f_test  # Aprox 0.114m

# K (Taps) como en el paper
K = 25  # 

# M (Micrófonos) como en el paper
M = 9   # 

# Geometría del Arreglo (D = lambda/2, R_a = 4*lambda)
D = LAMBDA_REF / 2  # 
mic_x = np.linspace(0, (M - 1) * D, M) - (M - 1) * D / 2
mic_array = np.stack([mic_x, np.zeros(M), np.zeros(M)], axis=1)

# Punto de enfoque central (r_F = 5*lambda, 90 grados)
radius = 5 * LAMBDA_REF  # 
focal_angle_rad = np.deg2rad(90.0) # 

focal_point_cartesian = np.array([
    radius * np.cos(focal_angle_rad), 
    radius * np.sin(focal_angle_rad), 
    0.0
])
# ------------------------- Define Point Constraint (LCMV) -----------------------
print("Calculando restricciones LCMV...")
sv = near_field_steering_vector_multi(f_test, 
                                      focal_point_cartesian, 
                                      fs ,mic_array, 
                                      K,
                                      C_SOUND)

sv = sv.squeeze()
C = np.stack([np.real(sv), np.imag(sv)], axis = 1)

tau_center = ((K - 1) / 2) / fs
phase = 2 * np.pi* f_test * tau_center
g_real = np.cos(phase)
g_im = np.sin(phase)
g = np.hstack([g_real, g_im])

print("Restricciones y ganancias definidas.")

# -------------------------- Optimize weights (LCMV) ---------------------------------
w_lcmv = compute_fixed_weights_optimized(C, g)
print("Pesos LCMV optimizados.")

# -------------------------- Compute DAS Weights -----------------------------------
print("Calculando pesos Delay-and-Sum (DAS)...")
w_das = np.zeros(M * K)
mic_distances = np.linalg.norm(focal_point_cartesian[np.newaxis, :] - mic_array, axis=1)
delays_sec = mic_distances / C_SOUND
ref_delay_sec = radius / C_SOUND
relative_delays_sec = ref_delay_sec - delays_sec
relative_delays_taps = np.round(relative_delays_sec * fs).astype(int)
k_center_tap = (K - 1) // 2

for m in range(M):
    k = k_center_tap + relative_delays_taps[m]
    if 0 <= k < K:
        w_das[m * K + k] = 1.0 / M 
    else:
        print(f"Advertencia: El tap del DAS para el mic {m} está fuera de rango (k={k})")
print("Pesos DAS calculados.")

# -------------------------- Calculate Gains (Angular Scan) --------------------------
print("Calculando ganancia (Escaneo Angular)...")
polar_points, deg = source_rotation(radius, 360, axis='h')
polar_points = polar_points.T
gains_db_lcmv_angular = analytical_gain(f_test, fs, mic_array, w_lcmv, polar_points)
gains_db_das_angular = analytical_gain(f_test, fs, mic_array, w_das, polar_points)

# -------------------------- Calculate Gains (Distance Scan) -------------------------
print("Calculando ganancia (Escaneo de Distancia)...")
focal_angle_rad = np.arctan2(focal_point_cartesian[1], focal_point_cartesian[0])
scan_distances = np.linspace(0.1, 2.0, 200) # Escanea de 0.1m a 2.0m

# Crea los puntos de escaneo (x, y, z) a lo largo del ángulo focal
dist_x = scan_distances * np.cos(focal_angle_rad)
dist_y = scan_distances * np.sin(focal_angle_rad)
dist_z = np.zeros_like(scan_distances)
distance_scan_points = np.stack([dist_x, dist_y, dist_z], axis=1)

gains_db_lcmv_dist = analytical_gain(f_test, fs, mic_array, w_lcmv, distance_scan_points)
gains_db_das_dist = analytical_gain(f_test, fs, mic_array, w_das, distance_scan_points)

# -------------------------- Plot (Comparativo) -----------------------------------
print("Graficando beampatterns...")

# --- 1. Squeezing y Normalización ---
# Colapsar formas
gains_lcmv_sq_ang = gains_db_lcmv_angular.squeeze()
gains_das_sq_ang = gains_db_das_angular.squeeze()
gains_lcmv_sq_dist = gains_db_lcmv_dist.squeeze()
gains_das_sq_dist = gains_db_das_dist.squeeze()

# Normalizar ambas curvas al PICO MÁXIMO ABSOLUTO del escaneo angular
# (Este pico es la ganancia en el punto focal)
abs_max_peak_lcmv = np.max([np.max(gains_lcmv_sq_ang)])
abs_max_peak_das =   np.max([np.max(gains_das_sq_ang)])
gains_lcmv_norm_ang = gains_lcmv_sq_ang - abs_max_peak_lcmv
gains_das_norm_ang = gains_das_sq_ang - abs_max_peak_das

# Normalizamos el escaneo de distancia con el MISMO pico para que sean comparables
gains_lcmv_norm_dist = gains_lcmv_sq_dist - abs_max_peak_lcmv
gains_das_norm_dist = gains_das_sq_dist - abs_max_peak_das

# --- 2. Filtrado Angular (0-180 grados) ---
filter_mask = (deg >= 0) & (deg <= 180) 
deg_filtered = deg[filter_mask]
gains_lcmv_filtered = gains_lcmv_norm_ang[filter_mask]
gains_das_filtered = gains_das_norm_ang[filter_mask]

# --- 3. Crear los subplots ---
# Creamos una figura con 2 filas, 1 columna
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# --- 4. Plot 1: Escaneo Angular (Tu gráfico original) ---
ax1.plot(deg_filtered, gains_lcmv_filtered, label="LCMV (Optimizado)")
ax1.plot(deg_filtered, gains_das_filtered, label="Delay-and-Sum (Simple)", linestyle='--', alpha=0.8)
ax1.set_ylim(-40, 5) 
ax1.set_xlim(0, 180) 
ax1.set_xlabel("Ángulo (grados)")
ax1.set_ylabel("Ganancia Normalizada (dB)")
ax1.grid(True)
focal_angle_deg = np.degrees(focal_angle_rad)
ax1.set_title(f"Beampattern @ {f_test} Hz (Focal point: {focal_angle_deg:.1f}°)")
ax1.legend()
ax1.axvline(x=focal_angle_deg, color='r', linestyle=':', label=f"Ángulo Focal ({focal_angle_deg:.1f}°) ")

# --- 5. Plot 2: Escaneo de Distancia (El nuevo gráfico) ---
ax2.plot(scan_distances, gains_lcmv_norm_dist, label="LCMV (Optimizado)")
ax2.plot(scan_distances, gains_das_norm_dist, label="Delay-and-Sum (Simple)", linestyle='--', alpha=0.8)
ax2.set_ylim(-40, 5)
ax2.set_xlabel("Distancia (metros)")
ax2.set_ylabel("Ganancia Normalizada (dB)")
ax2.grid(True)
ax2.set_title(f"Sensibilidad vs. Distancia (en {focal_angle_deg:.1f}°)")
ax2.axvline(x=radius, color='r', linestyle=':', label=f"Distancia Focal ({radius:.2f} m)")
ax2.legend()

# --- 6. Mostrar el gráfico ---
fig.tight_layout() # Ajusta el espacio entre plots
print("Mostrando gráfico.")
plt.show()# Delays relativos (en segundos)
# --- LÍNEA CORREGIDA ---
# El retardo debe ser (origen - mic) para ser consistente con el steering vector
relative_delays_sec = ref_delay_sec - delays_sec

# Convertir a taps relativos
relative_delays_taps = np.round(relative_delays_sec * fs).astype(int)

# Tap central del filtro como referencia
k_center_tap = (K - 1) // 2

# Construir el vector de pesos DAS (un '1' en el tap correcto por cada mic)
for m in range(M):
    k = k_center_tap + relative_delays_taps[m]
    if 0 <= k < K:
        w_das[m * K + k] = 1.0 / M 
    else:
        print(f"Advertencia: El tap del DAS para el mic {m} está fuera de rango (k={k})")

print("Pesos DAS calculados.")

# -------------------------- Calculate Gains ---------------------------------------
print("Generando puntos de escaneo...")
# Generamos 360 puntos y luego filtraremos para el plot
polar_points, deg = source_rotation(radius, 360, axis='h')
polar_points = polar_points.T

print("Calculando ganancia LCMV...")
gains_db_lcmv = analytical_gain(f_test, fs, mic_array, w_lcmv, polar_points)

print("Calculando ganancia DAS...")
gains_db_das = analytical_gain(f_test, fs, mic_array, w_das, polar_points)

# -------------------------- Plot (Comparativo) -----------------------------------
print("Graficando beampatterns...")

# 1. Colapsar formas (ej. de (1, 360) a (360,))
gains_lcmv_sq = gains_db_lcmv.squeeze()
gains_das_sq = gains_db_das.squeeze()

# 2. Normalizar ambas curvas al PICO MÁXIMO ABSOLUTO
abs_max_lcmv = np.max([np.max(gains_lcmv_sq)])
abs_max_das = np.max([ np.max(gains_das_sq)])

gains_lcmv_norm = gains_lcmv_sq - abs_max_lcmv
gains_das_norm = gains_das_sq - abs_max_das

# -------------------------- FILTRADO PARA PLOT DE 180 GRADOS --------------------------
# La función source_rotation devuelve ángulos de 0 a 360.
# Queremos plotear de 0 a 180 grados para evitar la ambigüedad.
filter_mask = (deg >= 0) & (deg <= 180) 

deg_filtered = deg[filter_mask]
gains_lcmv_filtered = gains_lcmv_norm[filter_mask]
gains_das_filtered = gains_das_norm[filter_mask]
# -------------------------------------------------------------------------------------

# 3. Crear el gráfico cartesiano
fig, ax = plt.subplots()

# Graficamos ambas curvas (usando los datos filtrados)
ax.plot(deg_filtered, gains_lcmv_filtered, label="LCMV (Optimizado)")
ax.plot(deg_filtered, gains_das_filtered, label="Delay-and-Sum (Simple)", linestyle='--', alpha=0.8)
# 4. Configurar la apariencia del gráfico
ax.set_ylim(-40, 5) # Damos 5dB de margen superior
ax.set_xlim(0, 180) # --- MANTENEMOS el xlim para 0-180 grados ---
ax.set_xlabel("Ángulo (grados)")
ax.set_ylabel("Ganancia Normalizada (dB)")
ax.grid(True)
# Ajustamos el título para indicar el ángulo focal
focal_angle_deg = np.degrees(np.arctan2(focal_point_cartesian[1], focal_point_cartesian[0]))
ax.set_title(f"Beampattern @ {f_test} Hz (Focal point: {focal_angle_deg:.1f}°)")
ax.legend() # Añadimos la leyenda
ax.axvline(x =focal_angle_deg, color = 'r') 

#Print data for GEMINI 
print("DATA OUTPUT [angle, lcmv_gain, das_gain]")

output_gains = np.stack([deg_filtered, gains_lcmv_filtered, gains_das_filtered], axis = 1)
print(output_gains)

# 5. Mostrar el gráfico
print("Mostrando gráfico.")
plt.show()