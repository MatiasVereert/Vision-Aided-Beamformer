import sys
import os
import json  # <--- [NUEVO] Necesario para guardar los metadatos
import numpy as np
from beamforming.kmvdr.system import LowRankAdaptive
from propagation.simulate_acoustics import SimAcoustic
from beamforming.array.microphone import Microphone
from utils.audio import save_wav

# --- CONFIGURACIÓN DE PARÁMETROS ---
FS = 48000
ALPHA = 0.96          
MIN_LOADING = 1e-4    
RANK_P = 2           
M1, M2 = 4, 3         # Estos definen la geometría Y el procesador
RT = .5               # Reverberación

def ensure_folder(base_path, p, m1, m2, alpha, RT, loading):
    """Crea una subcarpeta con nombre descriptivo de los parámetros."""
    folder_name = f"P={p}_M1={m1}_M2={m2}_Alpha={alpha}_RT={RT}_Load={loading}"
    full_path = os.path.join(base_path, folder_name)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    return full_path

# --- MAIN SCRIPT ---
if __name__ == "__main__":
    print("=== INTEGRATION TEST: IDEAL VS MIC MODEL COMPARISON ===")
    
    # 1. SETUP DE CARPETAS
    base_data_path = "tests/data"
    output_folder = ensure_folder(base_data_path, RANK_P, M1, M2, ALPHA, RT, MIN_LOADING)
    print(f"[IO] Los resultados se guardarán en: {output_folder}")
    
    if not os.path.exists(base_data_path): os.makedirs(base_data_path)
    
    # Definimos nombres para caché de datos y caché de metadatos
    cache_file_data = os.path.join(base_data_path, "room_simulation_cache.npy")
    cache_file_meta = os.path.join(base_data_path, "room_simulation_meta.json")

    # 2. GEOMETRÍA DEL ARRAY
    # [MODIFICADO] Vinculamos Mx y My a M1 y M2 para que el cambio sea consistente
    mic_spacing = 0.03
    Mx, My, Mz = M1, M2, 1  
    M = Mx * My * Mz 
    
    x = np.linspace(0, (Mx-1)*mic_spacing, Mx)
    y = np.linspace(0, (My-1)*mic_spacing, My)
    z = np.array([0.0])
    xv, yv, zv = np.meshgrid(x, y, z, indexing='xy') 
    mic_coords = np.column_stack([xv.flatten(), yv.flatten(), zv.flatten()])

    # Centrar el array en la sala
    array_center = np.array([1.25, 2.0, 1.25])
    mic_coords = mic_coords - np.mean(mic_coords, axis=0) + array_center
    
    # 3. POSICIONES DE FUENTES
    source_pos = array_center + np.array([0.9, 0.0, 0.1]) 
    interf_pos1 = array_center + np.array([0.0, 1.2, 0.0])
    interf_pos2 = array_center + np.array([-0.6, 0.6, 0.0])
    
    target_pos_flat = source_pos.flatten()

    # 4. CONFIGURACIÓN DE ESCENA ACÚSTICA
    acoustic_scene = SimAcoustic(mic_coords, array_mismatch=0.0, duration=8)
    room_dimensions = np.array([2.5, 4, 2.5])

    # Cargar audios
    source_path = "tools/data/signals/FA01_09.wav"
    int_path1 = "tools/data/signals/MC15_03.wav"
    int_path2 = "tools/data/signals/MF31_03.wav"

    acoustic_scene.set_source(source_path, gain=1, position=source_pos.reshape(1,3))
    acoustic_scene.set_interference(int_path1, gain=1, position=interf_pos1.reshape(1,3))
    acoustic_scene.set_interference(int_path2, gain=1, position=interf_pos2.reshape(1,3))

    # 5. MODELO DE MICRÓFONO
    mic_model = Microphone(model="MP34DT01-M", fs=FS)

    # =========================================================================
    # PARTE A: CAMPO REVERBERANTE (ROOM SIMULATION)
    # =========================================================================
    print("\n--- [A] PROCESAMIENTO CAMPO REVERBERANTE ---")
    
    # --- [NUEVO LOGICA DE CACHÉ INTELIGENTE] ---
    # Parámetros actuales que afectan la simulación física
    current_sim_params = {
        "M1": M1,
        "M2": M2,
        "RT": RT,
        "FS": FS,
        "mic_spacing": mic_spacing
    }

    use_cache = False
    
    # 1. Verificar si existen los archivos
    if os.path.exists(cache_file_data) and os.path.exists(cache_file_meta):
        try:
            # 2. Leer los metadatos guardados
            with open(cache_file_meta, 'r') as f:
                cached_params = json.load(f)
            
            # 3. Comparar con los actuales
            if cached_params == current_sim_params:
                print(f"[Cache] HIT: Parámetros (M1={M1}, M2={M2}, RT={RT}) coinciden. Cargando...")
                use_cache = True
            else:
                print(f"[Cache] MISS: Cambio detectado en parámetros. (Cache: {cached_params} vs Actual: {current_sim_params})")
        except Exception as e:
            print(f"[Cache] Error leyendo metadatos ({e}). Recalculando...")
            use_cache = False
    else:
        print("[Cache] MISS: No existe caché previo.")

    # A.1 Obtención o Cálculo
    if use_cache:
        room_input_ideal = np.load(cache_file_data)
    else:
        print("  -> Calculando simulación de sala (esto puede tardar)...")
        room_input_ideal = acoustic_scene.compute_room_ISB(room_dimensions, desire_RT=RT, iSIR_dB=0)
        
        # Guardar datos
        np.save(cache_file_data, room_input_ideal)
        
        # Guardar metadatos para la próxima vez
        with open(cache_file_meta, 'w') as f:
            json.dump(current_sim_params, f)
        print("  -> Simulación guardada en caché.")

    # Guardar Entrada Ideal
    save_wav("1_input_room_IDEAL.wav", FS, room_input_ideal[0], output_folder)

    # A.2 Procesamiento con Micrófonos IDEALES
    print("  -> Procesando: Micrófonos Ideales...")
    bf_ideal_room = LowRankAdaptive(mic_coords, FS, alpha=ALPHA)
    out_room_ideal = bf_ideal_room.block_process(
        input_signals=room_input_ideal,
        target_pos=target_pos_flat,
        M1=M1, M2=M2, P=RANK_P,
        mode="near_field",
        min_loading=MIN_LOADING
    )
    save_wav("2_output_room_IDEAL.wav", FS, out_room_ideal, output_folder)

    # A.3 Aplicar Modelo de Micrófono
    print("  -> Aplicando modelo de micrófono...")
    room_input_mic = mic_model.emulate(room_input_ideal, show_plots=False)

    # A.4 Procesamiento con Micrófonos REALES
    print("  -> Procesando: Micrófonos Reales (Modelados)...")
    bf_mic_room = LowRankAdaptive(mic_coords, FS, alpha=ALPHA)
    out_room_mic = bf_mic_room.block_process(
        input_signals=room_input_mic,
        target_pos=target_pos_flat,
        M1=M1, M2=M2, P=RANK_P,
        mode="near_field",
        min_loading=MIN_LOADING
    )
    save_wav("3_output_room_MIC.wav", FS, out_room_mic, output_folder)

    # =========================================================================
    # PARTE B: CAMPO LIBRE (ANECHOIC / FREE FIELD)
    # =========================================================================
    print("\n--- [B] PROCESAMIENTO CAMPO LIBRE ---")
    # ... (El resto del código sigue igual) ...
    
    # B.1 Generar Señal Base (Ideal)
    ff_input_ideal = acoustic_scene.free_field(iSIR_dB=0, normalize=True, mode="real")
    save_wav("4_input_freefield_IDEAL.wav", FS, ff_input_ideal[0], output_folder)

    # B.2 Procesamiento
    bf_ideal_ff = LowRankAdaptive(mic_coords, FS, alpha=ALPHA)
    out_ff_ideal = bf_ideal_ff.block_process(
        input_signals=ff_input_ideal,
        target_pos=target_pos_flat,
        M1=M1, M2=M2, P=RANK_P,
        mode= "near_field",
        min_loading= MIN_LOADING
    )
    save_wav("5_output_freefield_IDEAL.wav", FS, out_ff_ideal, output_folder)

    # B.3 Modelo Mic
    ff_input_mic = mic_model.emulate(ff_input_ideal, show_plots=False)

    # B.4 Procesamiento Real
    bf_mic_ff = LowRankAdaptive(mic_coords, FS, alpha=ALPHA)
    out_ff_mic = bf_mic_ff.block_process(
        input_signals=ff_input_mic,
        target_pos=target_pos_flat,
        M1=M1, M2=M2, P=RANK_P,
        mode="near_field",
        min_loading= MIN_LOADING
    )
    save_wav("6_output_freefield_MIC.wav", FS, out_ff_mic, output_folder)

    print(f"\n=== PROCESO TERMINADO ===")
    print(f"Archivos guardados en: {output_folder}")