import numpy as np
from matplotlib import pyplot as plt
from utils.data_loader import load_sriracha_selection, get_rir_data_arrays
from propagation.room_convolution import generate_array_signals
from utils.audio import load_audio_source, save_wav
from beamforming.array.mic_array import custom
from beamforming.gsc.system import AdaptiveBeamformer
# Importamos las herramientas para el cálculo directo
from beamforming.gsc.region_constriant import build_region_constraints
from beamforming.gsc.weights import compute_fixed_weights_optimized

data_set_path = r"tools/data/SR1-C1.h5"
signal_path = r"tools/data/signals/FA01_09.wav"
interference_path_1 = r"tools/data/signals/MC15_03.wav"
interference_path_2 = r"tools/data/signals/MF31_03.wav"

# --- 1. CARGA (Con inversión de eje X activada en data_loader) ---
setup_dictonary = load_sriracha_selection(data_set_path, n_mics_select=8)

mic_array_raw = setup_dictonary['mic_coords_final']
target_pos_raw = setup_dictonary['target_pos']
fs = setup_dictonary['fs']

# --- 2. PASO CRÍTICO: CENTRADO RELATIVO (LOCAL FRAME) ---
# Calculamos el centro geométrico de TUS 8 micrófonos
array_centroid = np.mean(mic_array_raw, axis=0)

# Movemos todo el universo para que el array quede en (0,0,0)
mic_array_centered = mic_array_raw - array_centroid
target_pos_centered = target_pos_raw - array_centroid

print(f"Centroide original: {array_centroid}")
print(f"Target (Global): {target_pos_raw}")
print(f"Target (Local):  {target_pos_centered}")

# --- 3. PREPARACIÓN DE SEÑALES (Igual que antes) ---
# ... (Carga de audios y RIRs igual que tu script anterior) ...
# OJO: Las RIRs NO cambian con el centrado de coordenadas, 
# porque la RIR ya captura la física relativa. Solo cambiamos las coordenadas 
# que usa el beamformer para calcular los retardos teóricos.

source_signal = load_audio_source(signal_path, fs, 1.8)
interference_signal_1 = load_audio_source(interference_path_1, fs, 1.8)
interference_signal_2 = load_audio_source(interference_path_2, fs, 1.8)
interference_signal = np.stack([interference_signal_1, interference_signal_2], axis=0)

rir_target, rirs_interference, _ = get_rir_data_arrays(data_set_path, setup_dictonary)
array_signals = generate_array_signals(source_signal, interference_signal, rir_target, rirs_interference)
array_signals = np.real(array_signals).astype(np.float32)

# --- 4. BEAMFORMER CON BYPASS ---

# Usamos las coordenadas CENTRADAS para inicializar el sistema
bf = AdaptiveBeamformer(
    mic_array=mic_array_centered, # <--- USAR ARRAY CENTRADO
    K=30,  # Recuerda el tema del K vs Apertura
    fs=fs, 
    fmin=100.0, 
    fmax=4000.0
)

print("\n--- INYECTANDO PESOS MANUALES (SIN BANCO) ---")

# Calculamos restricciones directamente para la posición exacta (centrada)
C, h, Ca = build_region_constraints(
    Rs=target_pos_centered,      # <--- TARGET EXACTO CENTRADO
    delta_r=0.2,                 # Región pequeña
    delta_azimut=np.deg2rad(5),  # Región pequeña
    delta_elevation=np.deg2rad(5),
    mic_array=mic_array_centered, # <--- ARRAY CENTRADO
    fs=fs,
    K=bf.K,
    f_min=bf.fmin,
    f_max=bf.fmax,
    num_points=50,  # Puntos para la integral de la región
    num_freqs=50
)

# Calculamos w_q (Fixed)
# IMPORTANTE: Usa el Diagonal Loading sugerido anteriormente si es posible
loading = 1e-3 
C_H = C.conj().T
M_mat = C_H @ C + np.eye(C.shape[1]) * loading 
X = np.linalg.solve(M_mat, h)
w_q_manual = C @ X

# --- INYECCIÓN QUIRÚRGICA ---
bf.current_wq = w_q_manual.flatten()
bf.current_Ca = Ca
bf.active_coords = target_pos_centered # Solo para referencia
bf.current_wa = np.zeros(Ca.shape[1], dtype=np.float32) # Inicializar adaptativo en 0

print(f"Norma de pesos inyectados: {np.linalg.norm(bf.current_wq):.2f}")

# --- 5. PROCESAMIENTO ---

# Prueba A: Solo Fijo (Debería enfocar la fuente, sonar 'reverberante' pero no 'roto')
bf.MU = 0.0
output_fixed = bf.process_block(array_signals)
save_wav('TEST_MANUAL_FIXED.wav', fs, output_fixed, folder='resultados_test')

# Prueba B: Adaptativo Suave
bf.buffer[:] = 0 # Limpiar buffer
bf.current_wa[:] = 0 # Reset pesos
bf.MU = 0.1
output_adaptive = bf.process_block(array_signals)
save_wav('TEST_MANUAL_ADAPTIVE.wav', fs, output_adaptive, folder='resultados_test')