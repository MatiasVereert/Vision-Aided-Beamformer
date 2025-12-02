import numpy as np
from scipy.constants import speed_of_sound
from matplotlib import pyplot as plt 

# --- Importaciones de tus módulos ---
from beamforming.signal_model import near_field_steering_vector_multi
# ESTA ES TU FUNCIÓN ORIGINAL (BEAMPATTERN)
from beamforming.evaluation.gain import analytical_gain 
from beamforming.gsc.weights import compute_fixed_weights_optimized
from utils.geometry import source_rotation, cartesian_to_spherical, spherical_to_cartesian
from typing import Tuple
from beamforming.evaluation.gain import compute_ags_vectorized
from beamforming.gsc.region_constriant import build_region_constraints

from utils.geometry import source_rotation


# ==============================================================================
# 2. SETUP DEL ESCENARIO (Idéntico al Paper)
# ==============================================================================
print("\n--- Configurando Escenario (Paper Fig 3) ---")
fs = 48000
K = 25
f_test = 3000.0 
C_SOUND = speed_of_sound

# Geometría Paper: 9 micros, espaciado lambda/2
LAMBDA_REF = C_SOUND / f_test
M = 9
D = LAMBDA_REF / 2
mic_x = np.linspace(0, (M - 1) * D, M) - (M - 1) * D / 2
mic_array = np.stack([mic_x, np.zeros(M), np.zeros(M)], axis=1)

# Foco Paper: 5 * lambda, 90 grados
radius = 5 * LAMBDA_REF
focal_angle_rad = np.deg2rad(90.0)
focal_point = np.array([
    radius * np.cos(focal_angle_rad), 
    radius * np.sin(focal_angle_rad), 
    0.0
])
N_total = M * K

print(f"  Target Freq: {f_test} Hz")
print(f"  Mic Array: {M} elementos, Apertura: {(M-1)*D:.2f} m")
print(f"  Foco: {radius:.2f} m @ 90 deg")


# ==============================================================================
# 3. DISEÑO 1: LCMV PUNTUAL (Point-Constraint)
# ==============================================================================
print("\n--- Diseñando LCMV Puntual ---")
# Construimos C y h manualmente para un solo punto (Bandwidth narrow para simplicidad de comparación pura)
sv_focal = near_field_steering_vector_multi(f_test, focal_point, fs, mic_array, K).squeeze()
# C puntual: Parte real e imaginaria del SV focal
C_point = np.stack([np.real(sv_focal), np.imag(sv_focal)], axis=1) # (N, 2)

# h puntual: Ganancia 1, fase ajustada al centro
tau_center = ((K - 1) / 2) / fs
phi = 2 * np.pi * f_test * tau_center
h_point = np.array([[np.cos(phi)], [np.sin(phi)]]) # (2, 1)

print(f"  C_point shape: {C_point.shape}")


# ==============================================================================
# 4. DISEÑO 2: LCMV REGIONAL (Regional-Constraint)
# ==============================================================================
print("\n--- Diseñando LCMV Regional (SVD) ---")
# Usamos tu módulo existente 'build_region_constraints'
# Parámetros de región similares al paper
delta_r = radius * 0.2     # +/- 10%
delta_az = np.deg2rad(8.0) # +/- 4 grados
f_min, f_max = 1000, 4000

# Llamamos a tu función del contexto
C_reg, h_reg, _ = build_region_constraints(
    Rs=focal_point,
    delta_r=delta_r,
    delta_azimut=delta_az,
    delta_elevation=np.deg2rad(2.0),
    mic_array=mic_array,
    fs=fs,
    K=K,
    f_min=f_min,
    f_max=f_max,
    num_points=50,
    num_freqs=50
)
# h debe ser columna para la función vectorizada
if h_reg.ndim == 1: h_reg = h_reg[:, None]
print(f"  C_reg shape: {C_reg.shape} (L={C_reg.shape[1]})")


# ==============================================================================
# 5. ESCANEO Y CÁLCULO DE AGS (FALLA)
# ==============================================================================
print("\n--- Calculando Array Gain Sensitivity (AGS) ---")

# A. Escaneo Angular (a distancia fija R)
scan_points_ang, deg_angles = source_rotation(radius, 360, axis='h')
scan_points_ang = scan_points_ang.T
# Filtramos 0-180
mask_ang = (deg_angles >= 0) & (deg_angles <= 180)
deg_plot = deg_angles[mask_ang]
points_ang_plot = scan_points_ang[mask_ang]

print("  -> Calculando AGS Angular Puntual...")
ags_point_ang = compute_ags_vectorized(f_test, fs, mic_array, K, points_ang_plot, C_point, h_point)
print("  -> Calculando AGS Angular Regional...")
ags_reg_ang = compute_ags_vectorized(f_test, fs, mic_array, K, points_ang_plot, C_reg, h_reg)


# B. Escaneo Distancia (a ángulo fijo 90)
dist_vals = np.linspace(radius * 0.2, radius * 2.0, 200)
points_dist_plot = np.zeros((len(dist_vals), 3))
points_dist_plot[:, 0] = dist_vals * np.cos(focal_angle_rad)
points_dist_plot[:, 1] = dist_vals * np.sin(focal_angle_rad)

print("  -> Calculando AGS Distancia Puntual...")
ags_point_dist = compute_ags_vectorized(f_test, fs, mic_array, K, points_dist_plot, C_point, h_point)
print("  -> Calculando AGS Distancia Regional...")
ags_reg_dist = compute_ags_vectorized(f_test, fs, mic_array, K, points_dist_plot, C_reg, h_reg)


# ==============================================================================
# 6. NORMALIZACIÓN Y GRÁFICO
# ==============================================================================
# Normalizamos AGS respecto al pico en el foco (definición del paper)
norm_point_ang = ags_point_ang - np.max(ags_point_ang)
norm_reg_ang = ags_reg_ang - np.max(ags_reg_ang)

norm_point_dist = ags_point_dist - np.max(ags_point_dist)
norm_reg_dist = ags_reg_dist - np.max(ags_reg_dist)

print("\n--- Graficando ---")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Plot Angular
ax1.plot(deg_plot, norm_point_ang, 'r', linewidth=1.5, label='Point-Constrained (Falla)')
ax1.plot(deg_plot, norm_reg_ang, 'b--', linewidth=2.5, label='Region-Constrained (Robusto)')
ax1.set_title("Array Gain Sensitivity vs Ángulo")
ax1.set_xlabel("Ángulo (Grados)")
ax1.set_ylabel("AGS Normalizada (dB)")
ax1.set_xlim(60, 120) # Zoom en la zona de interés
ax1.set_ylim(-40, 2)
ax1.grid(True, which='both', linestyle='--', alpha=0.7)
ax1.axvline(90, color='k', linestyle=':')
ax1.legend()

# Plot Distancia
ax2.plot(dist_vals, norm_point_dist, 'r', linewidth=1.5, label='Point-Constrained (Falla)')
ax2.plot(dist_vals, norm_reg_dist, 'b--', linewidth=2.5, label='Region-Constrained (Robusto)')
ax2.set_title("Array Gain Sensitivity vs Distancia")
ax2.set_xlabel("Distancia (m)")
ax2.set_ylabel("AGS Normalizada (dB)")
ax2.set_ylim(-40, 2)
ax2.grid(True, which='both', linestyle='--', alpha=0.7)
ax2.axvline(radius, color='k', linestyle=':', label=f"Foco ({radius:.2f}m)")
ax2.legend()

plt.tight_layout()
plt.show()