import numpy as np
from scipy.constants import speed_of_sound
from matplotlib import pyplot as plt 

# --- Importaciones de tus módulos ---
from beamforming.signal_model import near_field_steering_vector_multi
# ESTA ES TU FUNCIÓN ORIGINAL (BEAMPATTERN)
from beamforming.evaluation.gain import analytical_gain 
from beamforming.algorithms.lmcv import compute_fixed_weights_optimized
from utils.geometry import source_rotation, cartesian_to_spherical, spherical_to_cartesian
from typing import Tuple

# --------------------------------------------------------------------------
# --- TUS FUNCIONES SVD (Sin cambios) ---
# --------------------------------------------------------------------------
def set_domains(Rs, delta_r, delta_azimut, delta_elevation, fmin, fmax, P, J):
    f_array = np.linspace(fmin, fmax, J)
    Rs_sferical = cartesian_to_spherical(Rs)
    radius = np.random.uniform(Rs_sferical[0] - delta_r/2,  Rs_sferical[0] + delta_r/2, P )
    azimut = np.random.uniform(Rs_sferical[1] - delta_azimut/2, Rs_sferical[1]+ delta_azimut/2, P)
    elevation = np.random.uniform( Rs_sferical[2] - delta_elevation/2, Rs_sferical[2]+ delta_elevation/2, P)
    points = spherical_to_cartesian(radius, azimut, elevation)
    return f_array, points

def build_A_and_g(
    freqs: np.ndarray, 
    source_points: np.ndarray, 
    fs: int, 
    mic_array: np.ndarray, 
    K: int, 
    c: float = speed_of_sound
) -> Tuple[np.ndarray, np.ndarray]: 
    num_freqs = len(freqs)
    num_points = len(source_points)
    P_total = num_freqs * num_points
    a_tensor_complex = near_field_steering_vector_multi(
        f=freqs, Rs=source_points, fs=fs, mic_array=mic_array, K=K, c=c
    )
    N = a_tensor_complex.shape[2] 
    a_transpose_complex = np.transpose(a_tensor_complex, axes=(2, 0, 1))
    a_complex_flat = a_transpose_complex.reshape(N, -1)
    A = np.hstack([np.real(a_complex_flat), np.imag(a_complex_flat)])
    tau_center = ((K - 1) / 2) / fs
    gain_amplitude = 1.0
    freqs_P = np.repeat(freqs, num_points)
    phi = 2 * np.pi * freqs_P * tau_center
    g_real = gain_amplitude * np.cos(phi)
    g_imag = gain_amplitude * np.sin(phi)
    g = np.hstack([g_real, g_imag]).reshape(-1, 1)
    return A, g

def compute_svd_and_rank(
    A: np.ndarray, 
    energy_threshold: float = 0.9999
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, float]:
    N, two_P = A.shape
    P = two_P // 2
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    lambdas = (s**2) / P
    total_energy = np.sum(lambdas)
    if total_energy < 1e-12:
        return 0, U, s, Vh, 0.0
    cumulative_energy = np.cumsum(lambdas)
    L = np.searchsorted(cumulative_energy / total_energy, energy_threshold) + 1
    L = min(L, N)
    epsilon_L = np.sum(lambdas[L:])
    return L, U, s, Vh, epsilon_L

def build_region_constraints(
    Rs, delta_r, delta_azimut, delta_elevation, mic_array, fs, K,
    f_min, f_max, num_points, num_freqs, c=speed_of_sound
) -> Tuple[np.ndarray, np.ndarray, int]:
    freqs, points = set_domains(Rs, delta_r, delta_azimut, delta_elevation, f_min, f_max, num_points, num_freqs)
    A, g = build_A_and_g(freqs, points, fs, mic_array, K, c)
    L, U, s, Vh, epsilon = compute_svd_and_rank(A)
    C = U[:, :L]
    s_L_inv = 1.0 / (s[:L] + 1e-12)
    Sigma_L_inv = np.diag(s_L_inv)
    U_L_T = Vh[:L, :]
    h = Sigma_L_inv @ U_L_T @ g 
    return C, h, L

# --------------------------------------------------------------------------
# --- NUEVA FUNCIÓN PARA CALCULAR ARRAY GAIN SENSITIVITY (AGS) ---
# --------------------------------------------------------------------------
def calculate_array_gain(
    w_q: np.ndarray, 
    f_test: float, 
    fs: int, 
    mic_array: np.ndarray, 
    K: int, 
    scan_points: np.ndarray
) -> np.ndarray:
    """
    Calcula el Array Gain (AG) para un filtro fijo w_q.
    Asume Ruido Blanco (Rn = I).
    AG(x) = |w_q^H a(x)|^2 / (w_q^H w_q)
    """
    a_scan_complex = near_field_steering_vector_multi(
        f=f_test, Rs=scan_points, fs=fs, mic_array=mic_array, K=K
    )
    if a_scan_complex.shape[0] == 1:
        a_scan_complex = a_scan_complex.squeeze(0) 

    w_q_H = np.conj(w_q).T
    response_power = np.abs(a_scan_complex @ w_q_H)**2
    
    # Denominador: White Noise Gain (WNG)
    noise_power_out = np.dot(np.conj(w_q), w_q).real 
    
    if noise_power_out < 1e-12: noise_power_out = 1e-12
        
    array_gain = response_power / noise_power_out
    array_gain_db = 10 * np.log10(array_gain + 1e-12)
    
    return array_gain_db

#-------------------------- Define test setup -----------------------------------
print("Configurando parámetros (MODIFICADOS PARA COINCIDIR CON FIG. 3)...")
fs = 48000
K = 25
C_SOUND = speed_of_sound
f_test = 3000.0
f_min_band = 1000.0
f_max_band = 4000.0
num_freqs_band = 50 

LAMBDA_REF = C_SOUND / f_test
M = 9
D = LAMBDA_REF / 2
mic_x = np.linspace(0, (M - 1) * D, M) - (M - 1) * D / 2
mic_array = np.stack([mic_x, np.zeros(M), np.zeros(M)], axis=1)

radius = 5 * LAMBDA_REF
focal_angle_rad = np.deg2rad(90.0)
focal_point_cartesian = np.array([
    radius * np.cos(focal_angle_rad), 
    radius * np.sin(focal_angle_rad), 
    0.0
])
N_total = M * K
print(f"N total (M*K) = {N_total}")

# ------------------------- 1. Define Point-Narrowband Constraint (LCMV-PN) -----------------------
print("\n--- Calculando 1/4: LCMV-PN (Puntual, Narrowband) ---")
sv_focal_complex = near_field_steering_vector_multi(f_test, focal_point_cartesian, fs, mic_array, K, C_SOUND).squeeze()
C_pn = np.stack([np.real(sv_focal_complex), np.imag(sv_focal_complex)], axis = 1)
tau_center = ((K - 1) / 2) / fs
phase = 2 * np.pi * f_test * tau_center
g_pn = np.hstack([np.cos(phase), np.sin(phase)])
w_lcmv_pn = compute_fixed_weights_optimized(C_pn, g_pn) # w_q
print(f"Pesos LCMV-PN (L=2) optimizados.")

# -------------------------- 2. Compute DAS Weights -----------------------------------
print("\n--- Calculando 2/4: DAS (Delay-and-Sum) ---")
w_das = np.zeros(M * K)
mic_distances = np.linalg.norm(focal_point_cartesian[np.newaxis, :] - mic_array, axis=1)
delays_sec = mic_distances / C_SOUND
ref_delay_sec = radius / C_SOUND
relative_delays_sec = ref_delay_sec - delays_sec
relative_delays_taps = np.round(relative_delays_sec * fs).astype(int)
k_center_tap = (K - 1) // 2
for m in range(M):
    k = k_center_tap + relative_delays_taps[m]
    if 0 <= k < K: w_das[m * K + k] = 1.0 / M
print("Pesos DAS calculados.")

# -------------------------- 3. Define Point-Broadband Constraint (LCMV-PB) -------------------------
print("\n--- Calculando 3/4: LCMV-PB (Puntual, Broadband) ---")
freq_array_band = np.linspace(f_min_band, f_max_band, num_freqs_band)
point_array = focal_point_cartesian.reshape(1, 3) 
A_pb, g_pb = build_A_and_g(freq_array_band, point_array, fs, mic_array, K, C_SOUND)
L_pb, U_pb, s_pb, Vh_pb, epsilon_pb = compute_svd_and_rank(A_pb)
print(f"Rango L (Puntual, Broadband) calculado: {L_pb}")
C_pb = U_pb[:, :L_pb]
h_pb = (np.diag(1.0 / (s_pb[:L_pb] + 1e-12)) @ Vh_pb[:L_pb, :] @ g_pb).flatten()
w_lcmv_pb = (C_pb @ h_pb) # w_q
print(f"Pesos LCMV-PB (L={L_pb}) optimizados.")

# -------------------------- 4. Compute SVD-LCMV (Regional, Broadband) (LCMV-RB) -------------------------
print("\n--- Calculando 4/4: LCMV-RB (Regional, Broadband) ---")
delta_r_val = radius * 0.2
delta_azimut_val = np.deg2rad(8.0)
delta_elev_val = np.deg2rad(2.0) 
C_rb, h_rb, L_rb = build_region_constraints(
    Rs=focal_point_cartesian, delta_r=delta_r_val, delta_azimut=delta_azimut_val,
    delta_elevation=delta_elev_val, mic_array=mic_array, fs=fs, K=K,
    f_min=f_min_band, f_max=f_max_band, num_points=50, num_freqs=num_freqs_band
)
w_lcmv_rb = (C_rb @ h_rb).flatten() # w_q
print(f"Pesos LCMV-RB (L={L_rb}) optimizados.")


# -------------------------- Calculate Gains (Angular Scan) --------------------------
print("\nCalculando ganancia (Escaneo Angular)...")
angular_scan_points, deg = source_rotation(radius, 360, axis='h')
angular_scan_points = angular_scan_points.T
# Beampatterns (Usando tu función original)
bp_pn_ang = analytical_gain(f_test, fs, mic_array, w_lcmv_pn, angular_scan_points).squeeze()
bp_das_ang = analytical_gain(f_test, fs, mic_array, w_das, angular_scan_points).squeeze()
bp_pb_ang = analytical_gain(f_test, fs, mic_array, w_lcmv_pb, angular_scan_points).squeeze()
bp_rb_ang = analytical_gain(f_test, fs, mic_array, w_lcmv_rb, angular_scan_points).squeeze()
# Array Gain Sensitivity (Usando la nueva función)
ags_pn_ang = calculate_array_gain(w_lcmv_pn, f_test, fs, mic_array, K, angular_scan_points)
ags_das_ang = calculate_array_gain(w_das, f_test, fs, mic_array, K, angular_scan_points)
ags_pb_ang = calculate_array_gain(w_lcmv_pb, f_test, fs, mic_array, K, angular_scan_points)
ags_rb_ang = calculate_array_gain(w_lcmv_rb, f_test, fs, mic_array, K, angular_scan_points)

# -------------------------- Calculate Gains (Distance Scan) -------------------------
print("Calculando ganancia (Escaneo de Distancia)...")
scan_distances = np.linspace(radius * 0.1, radius * 3, 200) 
dist_x = scan_distances * np.cos(focal_angle_rad)
dist_y = scan_distances * np.sin(focal_angle_rad)
dist_z = np.zeros_like(scan_distances)
distance_scan_points = np.stack([dist_x, dist_y, dist_z], axis=1)
# Beampatterns
bp_pn_dist = analytical_gain(f_test, fs, mic_array, w_lcmv_pn, distance_scan_points).squeeze()
bp_das_dist = analytical_gain(f_test, fs, mic_array, w_das, distance_scan_points).squeeze()
bp_pb_dist = analytical_gain(f_test, fs, mic_array, w_lcmv_pb, distance_scan_points).squeeze()
bp_rb_dist = analytical_gain(f_test, fs, mic_array, w_lcmv_rb, distance_scan_points).squeeze()
# Array Gain Sensitivity
ags_pn_dist = calculate_array_gain(w_lcmv_pn, f_test, fs, mic_array, K, distance_scan_points)
ags_das_dist = calculate_array_gain(w_das, f_test, fs, mic_array, K, distance_scan_points)
ags_pb_dist = calculate_array_gain(w_lcmv_pb, f_test, fs, mic_array, K, distance_scan_points)
ags_rb_dist = calculate_array_gain(w_lcmv_rb, f_test, fs, mic_array, K, distance_scan_points)

# -------------------------- Plot (Comparativo) -----------------------------------
print("Graficando beampatterns...")
filter_mask = (deg >= 0) & (deg <= 180) 
deg_filtered = deg[filter_mask]
focal_angle_deg = np.degrees(focal_angle_rad)

# --- Crear 2 Figuras con 2 subplots cada una ---
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 14))

# --- FIGURA 1: BEAMPATTERN (Lo que estuvimos viendo) ---
fig1.suptitle("Métrica 1: Beampattern Response (Diseño de w_q)", fontsize=16)
# Normalización (cada uno a su pico)
bp_pn_norm_ang = bp_pn_ang - np.max(bp_pn_ang)
bp_das_norm_ang = bp_das_ang - np.max(bp_das_ang)
bp_pb_norm_ang = bp_pb_ang - np.max(bp_pb_ang)
bp_rb_norm_ang = bp_rb_ang - np.max(bp_rb_ang)
bp_pn_norm_dist = bp_pn_dist - np.max(bp_pn_dist)
bp_das_norm_dist = bp_das_dist - np.max(bp_das_dist)
bp_pb_norm_dist = bp_pb_dist - np.max(bp_pb_dist)
bp_rb_norm_dist = bp_rb_dist - np.max(bp_rb_dist)

# Plot 1: Angular Beampattern
ax1.plot(deg_filtered, bp_pn_norm_ang[filter_mask], label="1. LCMV-PN", linestyle=':', alpha=0.7)
ax1.plot(deg_filtered, bp_das_norm_ang[filter_mask], label="2. DAS", linestyle='--', alpha=0.7)
ax1.plot(deg_filtered, bp_pb_norm_ang[filter_mask], label="3. LCMV-PB", linewidth=2.5, color='red')
ax1.plot(deg_filtered, bp_rb_norm_ang[filter_mask], label="4. LCMV-RB", linewidth=2.5, color='blue')
ax1.set_title(f"Respuesta Angular (Diseño del Haz) @ {f_test} Hz")
ax1.set_ylim(-60, 5); ax1.set_xlim(0, 180); ax1.grid(True)
ax1.set_xlabel("Ángulo (grados)"); ax1.set_ylabel("Ganancia Normalizada (dB)")
ax1.axvline(x=focal_angle_deg, color='k', linestyle=':')
ax1.legend()

# Plot 2: Distance Beampattern
ax2.plot(scan_distances, bp_pn_norm_dist, label="1. LCMV-PN", linestyle=':', alpha=0.7)
ax2.plot(scan_distances, bp_das_norm_dist, label="2. DAS", linestyle='--', alpha=0.7)
ax2.plot(scan_distances, bp_pb_norm_dist, label="3. LCMV-PB", linewidth=2.5, color='red')
ax2.plot(scan_distances, bp_rb_norm_dist, label="4. LCMV-RB", linewidth=2.5, color='blue')
ax2.set_title(f"Respuesta a Distancia (Diseño del Haz) en {focal_angle_deg:.1f}°")
ax2.set_ylim(-60, 5); ax2.grid(True)
ax2.set_xlabel("Distancia (metros)"); ax2.set_ylabel("Ganancia Normalizada (dB)")
ax2.axvline(x=radius, color='k', linestyle=':')
ax2.legend()

# --- FIGURA 2: ARRAY GAIN SENSITIVITY (Lo que el paper ploteó) ---
fig2.suptitle("Métrica 2: Array Gain Sensitivity (Falla vs Ruido Blanco)", fontsize=16)
# Normalización (cada uno a su pico)
ags_pn_norm_ang = ags_pn_ang - np.max(ags_pn_ang)
ags_das_norm_ang = ags_das_ang - np.max(ags_das_ang)
ags_pb_norm_ang = ags_pb_ang - np.max(ags_pb_ang)
ags_rb_norm_ang = ags_rb_ang - np.max(ags_rb_ang)
ags_pn_norm_dist = ags_pn_dist - np.max(ags_pn_dist)
ags_das_norm_dist = ags_das_dist - np.max(ags_das_dist)
ags_pb_norm_dist = ags_pb_dist - np.max(ags_pb_dist)
ags_rb_norm_dist = ags_rb_dist - np.max(ags_rb_dist)

# Plot 3: Angular AGS
ax3.plot(deg_filtered, ags_pn_norm_ang[filter_mask], label="1. LCMV-PN (Puntual, Narrowband)", linestyle=':', alpha=0.7)
ax3.plot(deg_filtered, ags_das_norm_ang[filter_mask], label="2. DAS (Simple)", linestyle='--', alpha=0.7)
ax3.plot(deg_filtered, ags_pb_norm_ang[filter_mask], label="3. LCMV-PB (Puntual, Broadband)", linewidth=2.5, color='red')
ax3.plot(deg_filtered, ags_rb_norm_ang[filter_mask], label="4. LCMV-RB (Regional, Broadband)", linewidth=2.5, color='blue')
ax3.set_title(f"Sensibilidad Angular (Falla) @ {f_test} Hz")
ax3.set_ylim(-60, 5); ax3.set_xlim(0, 180); ax3.grid(True)
ax3.set_xlabel("Ángulo (grados)"); ax3.set_ylabel("Ganancia Normalizada (dB)")
ax3.axvline(x=focal_angle_deg, color='k', linestyle=':')
ax3.legend()

# Plot 4: Distance AGS
ax4.plot(scan_distances, ags_pn_norm_dist, label="1. LCMV-PN (Puntual, Narrowband)", linestyle=':', alpha=0.7)
ax4.plot(scan_distances, ags_das_norm_dist, label="2. DAS (Simple)", linestyle='--', alpha=0.7)
ax4.plot(scan_distances, ags_pb_norm_dist, label="3. LCMV-PB (Puntual, Broadband)", linewidth=2.5, color='red')
ax4.plot(scan_distances, ags_rb_norm_dist, label="4. LCMV-RB (Regional, Broadband)", linewidth=2.5, color='blue')
ax4.set_title(f"Sensibilidad a Distancia (Falla) en {focal_angle_deg:.1f}°")
ax4.set_ylim(-60, 5); ax4.grid(True)
ax4.set_xlabel("Distancia (metros)"); ax4.set_ylabel("Ganancia Normalizada (dB)")
ax4.axvline(x=radius, color='k', linestyle=':')
ax4.legend()

# --- 6. Mostrar el gráfico ---
fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
print("Mostrando gráficos (Figura 1 = Beampattern, Figura 2 = Array Gain Sensitivity)...")
plt.show()