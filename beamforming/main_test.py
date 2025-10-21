import numpy as np
from matplotlib import pyplot as plt 
import functions 
from functions import near_field_steering_vector, simular_propagacion_arreglo, beamforming
import sys

# --- 1. PARÁMETROS DEL SISTEMA Y ESCENARIO ---

# Constantes físicas y de muestreo
fs = 16000     # Frecuencia de muestreo (Hz)
c = 343.0      # Velocidad del sonido (m/s)

# Parámetros del Array
M_mic = 9      # Número de micrófonos
K_taps = 25    # Número de taps FIR por micrófono
N_total = M_mic * K_taps # Dimensión total del vector de pesos N

# Geometría del Array (Array Lineal Uniforme en el eje X)
d = 0.04       # Distancia entre micrófonos (ej: 4 cm)
mic_x = np.linspace(0, (M_mic - 1) * d, M_mic) - (M_mic - 1) * d / 2 # Centrado en el origen
mic_array_positions = np.stack([mic_x, np.zeros(M_mic), np.zeros(M_mic)], axis=1) # Forma (M, 3)

# Parámetros de la Señal y Foco (Narrowband Simulado)
f_test = 1000.0   # Frecuencia de prueba única (Hz)
R_focal = 0.5    # Distancia focal (0.5m)
source_angle_rad = np.deg2rad(90) # Azimut 90 grados (sobre el eje Y)

# Punto Focal Cartesiano (Target/Presumed Location)
x_focal = R_focal * np.cos(source_angle_rad)
y_focal = R_focal * np.sin(source_angle_rad)
z_focal = 0.0
target_point = np.array([x_focal, y_focal, z_focal])

# --- 2. CÁLCULO DE PESOS FIJOS (w_q) ---

# 2.1. Diseñar restricciones para el punto/frecuencia de prueba
# C tiene forma (N x 2), h tiene forma (2 x 1)
C, h = functions.point_constraint(target_point, K_taps, mic_array_positions, f_test, fs)
L_constraints = C.shape[1] # Debe ser 2

# 2.2. Calcular los pesos fijos w_q
# w_q tiene forma (N x 1)
w_fixed = functions.compute_fixed_weights_optimized(C, h)

print("--- Evaluación del Beamformer Fijo (Single Frequency) ---")
print(f"Dimensiones del vector de pesos N: {N_total}")
print(f"Número de restricciones L: {L_constraints}")

# --- 3. VERIFICACIÓN DE LA GANANCIA (PRUEBA CLAVE) ---

# La ganancia debe ser 1 (0 dB) en el punto focal para la frecuencia de prueba f_test.

# 3.1. Obtener el vector de dirección real a(x_F, f_test) para la verificación
a_test = functions.near_field_steering_vector(f_test, target_point, fs, mic_array_positions, K=K_taps, c=c)

# 3.2. Calcular la respuesta del beamformer: b(x_F, f) = w_q^H * a(x_F, f)
# La operación es una multiplicación matricial: (1 x N) * (N x 1)
# Utilizamos la conjugada transpuesta de w_fixed, ya que w_fixed debería ser real.
response_complex = w_fixed.conj().T @ a_test
response_magnitude = np.abs(response_complex[0, 0])
response_dB = 20 * np.log10(response_magnitude)

# 3.3. Evaluar el cumplimiento de la restricción C^H w_q = h
# C^H w_q (Forma 2 x 1)
constraint_check = C.T @ w_fixed

print("-" * 50)
print(f"Frecuencia de Prueba: {f_test} Hz")
print(f"Punto Focal: ({x_focal:.2f}, {y_focal:.2f}, {z_focal:.2f}) m")
print("-" * 50)

# Verificación de la respuesta compleja
print(f"Respuesta Compleja b(x_F, f_test): {response_complex[0, 0]:.4f}")
print(f"Ganancia en dB en el Foco (b(x_F, f_test)): {response_dB:.4f} dB")

# Verificación directa de las restricciones (parte real e imaginaria)
print(f"\nVerificación C^H w_q = h:")
print(f"C^H w_q (Resultado): \n{constraint_check}")
print(f"h (Deseado): \n{h}")

# --- 4. ANÁLISIS DEL RESULTADO ---

# Si el resultado es correcto, response_dB debe ser cercano a 0 dB y C^H w_q debe ser cercano a [1, 0]^T
if np.allclose(response_magnitude, 1.0, atol=1e-4):
    print("\n✅ Verificación exitosa: La ganancia es unitaria (0 dB) en el punto focal.")
    print("La Etapa 1 Fundamental (Restricción de Punto en Frecuencia Única) es correcta.")
else:
    print("\n❌ Error de verificación: La ganancia no es unitaria. Revise la implementación de near_field_steering_vector o compute_fixed_weights_optimized.")

plt.figure(figsize=(10, 5))
plt.plot(w_fixed)
plt.title(f'Coeficientes del Vector de Pesos Fijos w_q (f={f_test} Hz)')
plt.xlabel('Índice de Tap (m*K + k)')
plt.show()

# --- PARÁMETROS DEL SISTEMA (Reutilizados de tu código) ---
fs = 16000
f_test = 1000.0
R_focal = 0.5
M_mic = 9
K_taps = 25

# 1. Generar la Señal de Prueba (Tono puro a f_test)
duration_s = 0.1  # 100 ms de duración
N_samples = int(duration_s * fs)
t = np.arange(N_samples) / fs
signal_source = np.sin(2 * np.pi * f_test * t)

# --- 1. PARÁMETROS DE PRUEBA ---
angles_deg = np.linspace(0, 180, 91)
beampattern_dB = []

# --- 2. GENERACIÓN DE LA SEÑAL DE FUENTE (Tono puro a f_test) ---
duration_s = 0.1
N_samples = int(duration_s * fs)
t = np.arange(N_samples) / fs
signal_source = np.sin(2 * np.pi * f_test * t)
source_power_ref = np.mean(signal_source**2) # Potencia de referencia

# --- 3. BUCLE DEL PATRÓN POLAR ---
for angle_deg in angles_deg:
    angle_rad = np.deg2rad(angle_deg)
    
    # 3.1. Posición de la fuente de prueba (mismo radio R_focal)
    x_test = R_focal * np.cos(angle_rad)
    y_test = R_focal * np.sin(angle_rad)
    test_point = np.array([x_test, y_test, 0.0])

    # 3.2. Simular la señal recibida en el array Z (Mics x Tiempo)
    # Z (array_retardado) tiene la forma (M, N_fft)
    Z, _, _ = simular_propagacion_arreglo(signal_source, fs, test_point, mic_array_positions)
    N_fft = Z.shape[1] # Longitud total con padding

    # 3.3. Formatear la entrada U(t) para el beamformer con TAPS
    # La entrada U debe ser (M*K, T), donde T es el número de snapshots.
    # Esto es una implementación de filtro FIR multicanal
    
    T_snapshots = N_fft - K_taps + 1
    U_snapshots = np.zeros((M_mic * K_taps, T_snapshots))
    
    for t_idx in range(T_snapshots):
        u_k = np.zeros(M_mic * K_taps)
        for m in range(M_mic):
            # Obtener las K muestras temporales (taps) para el mic m
            taps_samples = Z[m, t_idx : t_idx + K_taps] 
            # Concatenar en el vector N x 1 (u_k)
            u_k[m*K_taps : (m+1)*K_taps] = taps_samples
        
        U_snapshots[:, t_idx] = u_k.T # Vector u(k) de dimensión (N)

    # 3.4. Aplicar el beamformer (w_q^H * U)
    # La salida y_snapshots tiene forma (1 x T_snapshots)
    y_snapshots = beamforming(U_snapshots, w_fixed) 
    
    # 3.5. Medir la Potencia (Media cuadrática de la salida)
    output_power = np.mean(y_snapshots**2)
    
    # 3.6. Calcular la Ganancia (relativa a la potencia de la fuente)
    # Ganancia_dB = 10 * log10(P_out / P_source)
    gain_dB = 10 * np.log10(output_power / source_power_ref)
    beampattern_dB.append(gain_dB)

# --- 4. VISUALIZACIÓN ---
plt.figure(figsize=(8, 6))
plt.plot(angles_deg, beampattern_dB, label=f'Ganancia medida a {R_focal} m')
plt.axvline(x=90, color='r', linestyle='--', label='Foco (90°)')
plt.title(f'Patrón Polar (Tiempo) del Beamformer Fijo a {f_test} Hz')
plt.xlabel('Ángulo de Azimut (grados)')
plt.ylabel('Ganancia (dB)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.show()

print("\n--- Resultado de la Prueba en Tiempo (Simulación) ---")
print(f"Ganancia en el Foco (90°): {beampattern_dB[angles_deg == 90][0]:.4f} dB")
print("El pico debería ser cercano a 0 dB.")