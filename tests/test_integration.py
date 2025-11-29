import numpy as np
from matplotlib import pyplot as plt
from utils.data_loader import load_sriracha_selection, get_rir_data_arrays

from propagation.room_convolution import generate_array_signals
from utils.audio import load_audio_source, save_wav
from beamforming.array.mic_array import custom
from beamforming.system import AdaptiveBeamformer
from beamforming.algorithms.region_constriant import build_region_constraints
from beamforming.algorithms.weights import compute_fixed_weights_optimized


data_set_path = r"tools/data/SR1-C1.h5"

# --- Señales de prueba originales ---
signal_path = r"tools/data/signals/FA01_09.wav"
interference_path_1 = r"tools/data/signals/MC15_03.wav"
interference_path_2 = r"tools/data/signals/MF31_03.wav"

# --- 1. CONFIGURACIÓN Y CARGA DE DATOS ---
# Obtain setup data
setup_dictonary = load_sriracha_selection(
    data_set_path,
    n_mics_select=8
)

# Define setup variables
mic_array = setup_dictonary['mic_coords_final']
MicArrayObj = custom(mic_array)
fs = setup_dictonary['fs']
target_pos = setup_dictonary['target_pos']
f_min = 100.0
f_max = 4000.0
K = 30 #

print(f"La fuente se encuentra en {target_pos}")
print(f"Interferencias definidas en {setup_dictonary['interference_pos']}")

# --- 2. IMPORTACIÓN DE SEÑALES (ESCENARIO 1) ---
source_signal = load_audio_source(signal_path, fs, 1.8)
interference_signal_1 = load_audio_source(interference_path_1, fs, 1.8)
interference_signal_2 = load_audio_source(interference_path_2, fs, 1.8)

# Stack de dos interferencias de voz
interference_signal = np.stack([interference_signal_1, interference_signal_2], axis=0)

# Obtain Rirs
rir_target, rirs_interference, _ = get_rir_data_arrays(data_set_path, setup_dictonary)

# --- 3. CONVOLUCIÓN (ESCENARIO 1: VOZ + 2 INTERFERENCIAS) ---
array_signals = generate_array_signals(
    source_signal,
    interference_signal,
    rir_target,
    rirs_interference
)

array_signals = np.real(array_signals).astype(np.float32)

# --- 4. CONFIGURACIÓN DEL BEAMFORMER ---
bf = AdaptiveBeamformer(
    mic_array=mic_array,
    K=K,
    fs=fs,
    fmin=f_min,
    fmax=f_max
)

# Generar banco (Región muy pequeña/puntual según tus params)
bf.generate_bank(
    r_spam=0.001,
    az_spam=0.001,
    inc_spam=0.001,
    points=2,
    center=target_pos
)

bf.update_focal_point(target_pos)

# --- 5. PROCESAMIENTO ESCENARIO 1 ---
# A. Adaptativo (MU por defecto)
# Nota: Si no definiste bf.MU, usa el default de la clase. 
# Recomendación: bf.MU = 0.05 para estabilidad.
output = bf.process_block(array_signals)
save_wav('output_scenario1_adaptive.wav', fs, output, folder='resultados_test')

# B. Fijo (MU = 0)
bf.MU = 0
output_fixed = bf.process_block(array_signals)
save_wav('output_scenario1_fixed.wav', fs, output_fixed, folder='resultados_test')
save_wav('input_scenario1.wav', fs, array_signals[0], folder='resultados_test')


# ==============================================================================
# --- PRUEBA ADICIONAL: RUIDO BLANCO (SINGLE SOURCE) ---
# ==============================================================================
print("\n--- INICIANDO PRUEBA ADICIONAL: RUIDO BLANCO (1 INTERFERENCIA) ---")

# 1. Generar Ruido Blanco (Misma duración que la fuente)
# Amplitud 0.1 para que no sature
white_noise = np.random.normal(0, 0.1, len(source_signal)).astype(np.float32)

# Formatear para generate_array_signals: debe ser 2D (N_fuentes, N_samples)
white_noise_matrix = white_noise[np.newaxis, :] 

# 2. Seleccionar SOLO UNA RIR de interferencia
# rirs_interference tiene shape (N_fuentes_total, Mics, Samples)
# Tomamos solo la primera [0:1] manteniendo las dimensiones 3D
rir_interference_single = rirs_interference[0:1, :, :]

print(f"-> Procesando ruido blanco con RIR en posición: {setup_dictonary['interference_pos'][0]}")

# 3. Generar señales del arreglo (Target Voz + 1 Interferencia Ruido Blanco)
array_signals_wn = generate_array_signals(
    source_signal,          # Voz original
    white_noise_matrix,     # Ruido blanco
    rir_target,             # RIR del target original
    rir_interference_single # RIR única de interferencia
)

array_signals_wn = np.real(array_signals_wn).astype(np.float32)

# 4. Limpieza del Beamformer (RESET)
# Es vital limpiar el estado anterior para que la adaptación empiece de cero
bf.buffer[:] = 0         # Limpiar buffer de audio
bf.current_wa = None     # Reiniciar pesos adaptativos
bf.MU = 0.1             # Restaurar un MU estable para que adapte (ya que estaba en 0)

# 5. Procesar Bloque
output_wn = bf.process_block(array_signals_wn)

# 6. Guardar Resultados
save_wav('output_white_noise_adaptive.wav', fs, output_wn, folder='resultados_test')
save_wav('input_white_noise.wav', fs, array_signals_wn[0], folder='resultados_test')

print("--- PRUEBA RUIDO BLANCO FINALIZADA ---")