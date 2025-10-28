# --- 1. IMPORTACIONES ---
from beamforming.beamformer_core import beamforming, snapshots
from propagation.free_field import space_delay
import numpy as np
import matplotlib.pyplot as plt

# --- 2. PARÁMETROS DEL ESCENARIO ---
fs = 16000
c = 343.0
M = 9       # Número de micrófonos
K = 50      # Número de taps

# Geometría del Array (ULA en el eje X, centrado)
d = 0.04
mic_x = np.linspace(0, (M - 1) * d, M) - (M - 1) * d / 2
mic_array = np.stack([mic_x, np.zeros(M), np.zeros(M)], axis=1)

# Dirección de enfoque (p. ej., 45 grados) y distancia.
# NOTA: La distancia debe ser lo suficientemente corta para que los retardos
# sean mayores que el período de muestreo y se puedan distinguir entre taps.
look_direction_rad = np.deg2rad(45)
source_distance = 2.0 # Metros
source_pos = source_distance * np.array([np.cos(look_direction_rad), np.sin(look_direction_rad), 0])

print("--- Ejecutando Test de Delay-and-Sum ---")
print("Objetivo: Verificar que `snapshots` y `beamforming` suman coherentemente una señal.")

# --- 3. CREACIÓN DE PESOS "DELAY-AND-SUM" (w_ds) ---
# Estos pesos invierten el retardo de propagación para alinear las señales.
w_ds = np.zeros((M * K, 1))

# Calcular retardos de tiempo relativos al centro del array
center_pos = np.mean(mic_array, axis=0)
ref_dist = np.linalg.norm(source_pos - center_pos)
time_delays = (np.linalg.norm(source_pos - mic_array, axis=1) - ref_dist) / c

# Convertir retardos de tiempo a un índice de tap. Se usa un offset para centrar.
tap_offset = K // 2
delay_indices = np.round(time_delays * fs).astype(int) + tap_offset

# Asignar un '1' en el tap correcto para cada micrófono
for m in range(M):
    tap_index = delay_indices[m]
    if 0 <= tap_index < K:
        w_ds[m * K + tap_index] = 1.0
    else:
        print(f"ADVERTENCIA: Retardo para mic {m} (índice {tap_index}) fuera de rango [0, {K-1}]")

# Verificación de los pesos
non_zero_weights = np.count_nonzero(w_ds)
print(f"\nNúmero de pesos no nulos en w_ds: {non_zero_weights} (Esperado: {M})")
if non_zero_weights != M:
    print(">>> ❌ ALERTA: El vector de pesos w_ds no se está creando correctamente. Revise la geometría o el número de taps K.")

# --- 4. SIMULACIÓN DE LA SEÑAL RECIBIDA ---
# Usamos un impulso (delta de Kronecker) como señal de prueba.
source_signal = np.zeros(200)
source_signal[50] = 1.0 # Impulso en la muestra 50
Z_signals, _, _ = space_delay(source_signal, fs, source_pos, mic_array)

# --- 5. APLICACIÓN DEL BEAMFORMER ---
U_snapshots = snapshots(Z_signals, K)
y_output = beamforming(U_snapshots, w_ds)
y_output_flat = y_output.flatten()

# --- 6. VERIFICACIÓN DEL RESULTADO ---
peak_amplitude = np.max(y_output_flat)

print(f"\nAmplitud del pico de salida: {peak_amplitude:.4f}")
print(f"Amplitud esperada: {float(M)}")

if np.isclose(peak_amplitude, M, atol=1e-1):
    print("✅ TEST PASADO: La suma constructiva es correcta.")
else:
    print("❌ TEST FALLIDO: La amplitud de salida no coincide.")

# --- 7. VISUALIZACIÓN ---
plt.figure(figsize=(10, 7))
ax1 = plt.subplot(2, 1, 1)
im = ax1.imshow(Z_signals, aspect='auto', interpolation='nearest', origin='lower', extent=[0, Z_signals.shape[1]/fs*1000, -0.5, M-0.5])
ax1.set_title('Señales Recibidas en el Array (Z)')
ax1.set_ylabel('Índice de Micrófono')
ax1.set_xlabel('Tiempo (ms)')
plt.colorbar(im, ax=ax1, label='Amplitud')
ax2 = plt.subplot(2, 1, 2)
ax2.plot(np.arange(len(y_output_flat))/fs*1000, y_output_flat)
ax2.set_title(f'Salida del Beamformer Delay-and-Sum (Pico medido = {peak_amplitude:.2f})')
ax2.set_xlabel('Tiempo (ms)')
ax2.set_ylabel('Amplitud')
ax2.grid(True, linestyle=':')
plt.tight_layout()
plt.show()