import numpy as np
from scipy.constants import speed_of_sound

# --- Importaciones de tu lista ---
from beamforming.algorithms.lmcv import  compute_fixed_weights_optimized
from beamforming.algorithms.constrains import point_constraint
from beamforming.signal_model import near_field_steering_vector_multi, near_field_steering_vector
from beamforming.evaluation.gain import analytical_gain
from utils.geometry import source_rotation, cartesian_to_spherical, spherical_to_cartesian
from utils.polar_plot import plot_polar_pattern
from beamforming.algorithms.region_constriant import build_region_constraints

# --- 1. CONFIGURACIÓN DEL ESCENARIO ---
# Parámetros físicos
FS = 48000
K = 50      # Número de taps
C_SOUND = speed_of_sound

# Geometría del Arreglo (ULA en el eje X, 7 micrófonos, 4cm de espacio)
M = 7
D = 0.04
mic_x = np.linspace(0, (M - 1) * D, M) - (M - 1) * D / 2
mic_array = np.stack([mic_x, np.zeros(M), np.zeros(M)], axis=1)

# Punto de enfoque central (Cartesiano)
focal_point_cartesian = np.array([0.8, .80, 0.0]) # 2 metros al frente
f_test = 3000.0 # Frecuencia de prueba

print("Escenario definido. Calculando pesos...")

# --- 2. PATH 1: Beamformer "Simple" (Point Constraint) ---
# Calcula las restricciones para un solo punto
C_point, h_point = point_constraint(
    target_point=focal_point_cartesian,
    K_taps=K,
    mic_array=mic_array,
    f=f_test,
    fs=FS
)
# Calcula los pesos usando la función que proporcionaste
w_point = compute_fixed_weights_optimized(C_point, h_point)
print("Pesos 'Simples' calculados.")

# --- 3. PATH 2: Beamformer "Robusto" (Region Constraint) ---
# Define la región de robustez alrededor del punto focal
delta_r = 0.2           # +/- 50 cm de robustez en distancia
delta_azimut = np.deg2rad(2) # +/- 10 grados de robustez en azimut
delta_elevation = np.deg2rad(2) # +/- 5 grados de robustez en elevación

# Frecuencias para la robustez de banda ancha
f_min = 250.0
f_max = 4000.0
num_freq_points = 50  # J
num_spatial_points = 100 # I

# Calcula las restricciones robustas
C_robust, h_robust = build_region_constraints(
    Rs=focal_point_cartesian,
    delta_r=delta_r,
    delta_azimut=delta_azimut,
    delta_elevation=delta_elevation,
    mic_array=mic_array,
    fs=FS,
    K=K,
    f_min=f_min,
    f_max=f_max,
    num_points=num_spatial_points,
    num_freqs=num_freq_points
)
# Calcula los pesos usando la MISMA función
w_robust = compute_fixed_weights_optimized(C_robust, h_robust)
print("Pesos 'Robustos' calculados.")

# --- 4. EVALUACIÓN Y GRÁFICO ---
print("Evaluando patrones polares...")
# Definir los puntos de ploteo (un círculo a 2m)
plot_radius = 2.0
plot_points, plot_angles = source_rotation(plot_radius, 360, 'h')
plot_points_cartesian = plot_points.T # (P, 3)

# Calcular la ganancia para ambos conjuntos de pesos
gain_point_2d = analytical_gain(
    frecs=np.array([f_test]),
    source_points=plot_points_cartesian,
    fs=FS,
    mic_array=mic_array,
    weights=w_point,
    
)
gain_robust_2d = analytical_gain(
    frecs=np.array([f_test]),
    source_points=plot_points_cartesian,
    fs=FS,
    mic_array=mic_array,
    weights=w_robust,
    
)

# Extraer el array 1D de ganancia
gain_point = gain_point_2d[0]
gain_robust = gain_robust_2d[0]

# Normalizar para el gráfico (pico en 0 dB)
gain_point_norm = gain_point - np.max(gain_point)
gain_robust_norm = gain_robust - np.max(gain_robust)

# --- 5. PLOTEO COMPARATIVO ---
plot_polar_pattern(
    gains_list=[gain_point_norm, gain_robust_norm],
    angles_deg=plot_angles,
    labels_list=["Simple (Point)", "Robusto (Region)"],
    title=f"Comparación de Beamformers a {f_test} Hz"
)
print("Script finalizado.")