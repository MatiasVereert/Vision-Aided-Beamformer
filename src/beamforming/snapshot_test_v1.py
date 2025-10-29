# --- 1. IMPORTACIONES ---
from beamforming.beamformer_core import beamforming, snapshots, near_field_steering_vector
from propagation.free_field import space_delay 
import numpy as np
import matplotlib.pyplot as plt

# --- 2. PARÁMETROS DEL ESCENARIO ---
fs = 48000
c = 343.0
M = 9       # Número de micrófonos
K = 50      # Número de taps
f_test = 1000.0 # Frecuencia de prueba para LCMV

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

print("--- Ejecutando Test de Beamformer LCMV (Tiempo) ---")
print(f"Objetivo: Verificar ganancia unitaria para un tono de {f_test} Hz desde {np.rad2deg(look_direction_rad):.0f} grados.")

# --- 3. CÁLCULO DE PESOS LCMV ---
# Para un caso sin ruido (matriz de covarianza = Identidad), el peso LCMV con una
# restricción de ganancia unitaria se simplifica a: w = a / (a^H * a)
# donde 'a' es el vector de dirección. Esto asegura que w^H * a = 1.

a = near_field_steering_vector(f_test, source_pos, fs, mic_array, K=K)
a_H_a = a.conj().T @ a
w_lcmv = a / a_H_a

# --- 4. SIMULACIÓN DE LA SEÑAL RECIBIDA ---
# Usamos un tono puro a f_test como señal de prueba.
duration_s = 0.1
N_samples = int(duration_s * fs)
t = np.arange(N_samples) / fs
source_signal = np.sin(2 * np.pi * f_test * t)

# Propagar la señal al array. Z_signals tendrá padding añadido por space_delay.
Z_signals, signal_padded_ref, _ = space_delay(source_signal, fs, source_pos, mic_array)

# Potencia de la señal de referencia para calcular la ganancia
# Se debe calcular sobre la señal con padding para que la duración coincida con la de la salida.
source_power = np.mean(signal_padded_ref**2)

# --- 5. APLICACIÓN DEL BEAMFORMER ---
U_snapshots = snapshots(Z_signals, K)
y_output = beamforming(U_snapshots, w_lcmv)
y_output_flat = y_output.flatten()

# --- 6. VERIFICACIÓN DEL RESULTADO ---
# La ganancia debe ser 1 (0 dB) para la señal objetivo.
output_power = np.mean(np.abs(y_output_flat)**2) # Corrected power calculation for complex output
gain_dB = 10 * np.log10(output_power / source_power)

print(f"\nGanancia de potencia en el foco: {gain_dB:.4f} dB")
print(f"Ganancia esperada: 0.0 dB")

if np.isclose(gain_dB, 0, atol=0.5): # Tolerancia de 0.5 dB
    print("✅ TEST PASADO: La ganancia en el foco es correcta.")
else:
    print("❌ TEST FALLIDO: La ganancia en el foco no es unitaria.")

# --- 7. VISUALIZACIÓN ---
plt.figure(figsize=(10, 7))
ax1 = plt.subplot(2, 1, 1)
im = ax1.imshow(Z_signals, aspect='auto', interpolation='nearest', origin='lower', extent=[0, Z_signals.shape[1]/fs*1000, -0.5, M-0.5])
ax1.set_title('Señales Recibidas en el Array (Z)')
ax1.set_ylabel('Índice de Micrófono')
ax1.set_xlabel('Tiempo (ms)')
plt.colorbar(im, ax=ax1, label='Amplitud')
ax2 = plt.subplot(2, 1, 2)
ax2.plot(np.arange(len(y_output_flat))/fs*1000, np.real(y_output_flat))
ax2.set_title(f'Salida del Beamformer LCMV (Ganancia medida = {gain_dB:.2f} dB)')
ax2.set_xlabel('Tiempo (ms)')
ax2.set_ylabel('Amplitud')
ax2.grid(True, linestyle=':')
plt.tight_layout()
plt.show()

# --- 8. CÁLCULO Y GRÁFICO DEL PATRÓN POLAR ---
print("\n--- Calculando Patrón Polar ---")

angles_deg = np.linspace(0, 180, 181) # Probar de 0 a 180 grados
angles_rad = np.deg2rad(angles_deg)
beampattern_dB = []

# Potencia de la señal de referencia (sin padding, ya que se recalculará en cada iteración)
source_power_ref = np.mean(source_signal**2)

for angle in angles_rad:
    # Posición de la fuente de prueba para el ángulo actual
    test_pos = source_distance * np.array([np.cos(angle), np.sin(angle), 0])
    
    # Simular la propagación desde la nueva posición
    Z_test, _, _ = space_delay(source_signal, fs, test_pos, mic_array)
    
    # Aplicar el beamformer con los mismos pesos w_lcmv
    U_test = snapshots(Z_test, K)
    y_test = beamforming(U_test, w_lcmv)
    
    # Calcular la potencia de salida
    output_power = np.mean(np.abs(y_test)**2) # Corrected power calculation for complex output
    
    # Calcular ganancia en dB. Usamos un valor pequeño para evitar log(0)
    gain = 10 * np.log10(output_power / source_power_ref + 1e-9)
    beampattern_dB.append(gain)

print("Cálculo finalizado.")

# Visualización en un gráfico polar
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(angles_rad, beampattern_dB)
ax.set_title(f'Patrón de Directividad del Beamformer LCMV (Foco a 45°)', va='bottom')
ax.set_xlabel('Ganancia (dB)')
ax.set_theta_zero_location("N") # 0 grados arriba
ax.grid(True)
plt.show()