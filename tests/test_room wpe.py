import sys
import os
import json
import numpy as np
from beamforming.kmvdr.system import LowRankAdaptive
from propagation.simulate_acoustics import SimAcoustic
from beamforming.array.microphone import Microphone
from utils.audio import save_wav
from beamforming.dereverberation.wpe import apply_wpe 
import matplotlib.pyplot as plt 
from beamforming.joint_source_extraction import OnlineWPExSRIVE_N1_Vectorized 

from beamforming.dereverberation.gwpe import batch_dereverb

def plot_scene_3d(room_dims, mics, target, interferences_list, save_path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Dibujar Micrófonos
    ax.scatter(mics[:,0], mics[:,1], mics[:,2], c='blue', marker='.', s=50, label='Mic Array', depthshade=False)

    # 2. Dibujar Target
    target = target.flatten()
    ax.scatter(target[0], target[1], target[2], c='green', marker='*', s=200, label='Target Source', edgecolors='black')
    ax.plot([target[0], target[0]], [target[1], target[1]], [0, target[2]], 'g--', alpha=0.3)

    # 3. Dibujar Interferencias
    for i, interf in enumerate(interferences_list):
        interf = interf.flatten()
        ax.scatter(interf[0], interf[1], interf[2], c='red', marker='X', s=100, label=f'Interference {i+1}')
        ax.plot([interf[0], interf[0]], [interf[1], interf[1]], [0, interf[2]], 'r--', alpha=0.3)

    # 4. Configurar la Sala
    ax.set_xlim([0, room_dims[0]])
    ax.set_ylim([0, room_dims[1]])
    ax.set_zlim([0, room_dims[2]])

    ax.set_xlabel('X [m] (Ancho)')
    ax.set_ylabel('Y [m] (Largo)')
    ax.set_zlabel('Z [m] (Alto)')
    ax.set_title(f'Geometría de Simulación\nRoom: {room_dims}m | RT60: {RT}s')
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1))

    try:
        ax.set_box_aspect((room_dims[0], room_dims[1], room_dims[2]))
    except AttributeError:
        pass 

    filename = os.path.join(save_path, "scene_setup_3d.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  -> Visualización guardada en: {filename}")

# --- CONFIGURACIÓN DE PARÁMETROS ---
FS = 48000
ALPHA = 0.99
MIN_LOADING = 1e-9    
RANK_P = 1           
M1, M2 = 3, 4         
RT = .6          

# Posición de la fuente RELATIVA al centro del array
SRC_REL_POS = [.4, .1, 0.1] 

def ensure_folder(base_path, p, m1, m2, alpha, RT, loading, src_pos):
    pos_str = f"{src_pos[0]}_{src_pos[1]}_{src_pos[2]}"
    folder_name = f"P={p}_M={m1}x{m2}_RT={RT}_Src={pos_str}_Alpha={alpha}"
    full_path = os.path.join(base_path, folder_name)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    return full_path

# --- MAIN SCRIPT ---
if __name__ == "__main__":
    print("=== INTEGRATION TEST: IDEAL VS MIC MODEL COMPARISON ===")
    
    # 1. SETUP DE CARPETAS
    base_data_path = "tests/data"
    output_folder = ensure_folder(base_data_path, RANK_P, M1, M2, ALPHA, RT, MIN_LOADING, SRC_REL_POS)
    print(f"[IO] Los resultados se guardarán en: {output_folder}")
    
    if not os.path.exists(base_data_path): os.makedirs(base_data_path)
    
    cache_file_data = os.path.join(base_data_path, "room_simulation_cache.npy")
    cache_file_meta = os.path.join(base_data_path, "room_simulation_meta.json")

    # 2. GEOMETRÍA DEL ARRAY
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
    source_pos = array_center + np.array(SRC_REL_POS)
    interf_pos1 = array_center + np.array([0.0, 1.2, 0.0])
    interf_pos2 = array_center + np.array([-0.6, 0.6, 0.0])
    target_pos_flat = source_pos.flatten()

    # 4. CONFIGURACIÓN DE ESCENA ACÚSTICA
    acoustic_scene = SimAcoustic(mic_coords, array_mismatch=0.0, duration=4, fs = FS)
    room_dimensions = np.array([2.5, 4, 2.5])

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
    
    current_sim_params = {
        "M1": M1, "M2": M2, "RT": RT, "FS": FS,
        "mic_spacing": mic_spacing, "source_rel_pos": SRC_REL_POS
    }

    use_cache = False
    if os.path.exists(cache_file_data) and os.path.exists(cache_file_meta):
        try:
            with open(cache_file_meta, 'r') as f:
                cached_params = json.load(f)
            if cached_params == current_sim_params:
                print(f"[Cache] HIT: Parámetros coinciden. Cargando...")
                use_cache = True
            else:
                print(f"[Cache] MISS: Cambio detectado en parámetros.")
        except Exception as e:
            print(f"[Cache] Error leyendo metadatos. Recalculando...")
            use_cache = False
    else:
        print("[Cache] MISS: No existe caché previo.")

    if use_cache:
        room_input_ideal = np.load(cache_file_data)
    else:
        print("  -> Calculando simulación de sala...")
        room_input_ideal = acoustic_scene.compute_room_ISB(room_dimensions, desire_RT=RT, iSIR_dB=0)    
        np.save(cache_file_data, room_input_ideal)
        with open(cache_file_meta, 'w') as f:
            json.dump(current_sim_params, f)
        print("  -> Simulación guardada en caché.")

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
    
    # B.1 Generar Señal Base
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

    # =========================================================================
    # PARTE C: DEREVERBERACIÓN (WPE) + BEAMFORMING
    # =========================================================================
    print("\n--- [C] PROCESAMIENTO CON DEREVERBERACIÓN (WPE) ---")

    # --- C.1 Caso Micrófonos Reales (WPE + Beamformer) ---
    print("  -> [MIC] Aplicando WPE a 'room_input_mic'...")
    room_input_wpe_mic = apply_wpe(room_input_mic, FS, taps=10, delay=3, iterations=3)
    save_wav("7a_input_room_WPE_MIC.wav", FS, room_input_wpe_mic[0], output_folder)

    print("  -> [MIC] Procesando señal WPE con Beamformer...")
    bf_wpe_mic = LowRankAdaptive(mic_coords, FS, alpha=ALPHA)
    out_room_wpe_mic = bf_wpe_mic.block_process(
        input_signals=room_input_wpe_mic,
        target_pos=target_pos_flat,
        M1=M1, M2=M2, P=RANK_P,
        mode="near_field",
        min_loading=MIN_LOADING
    )
    save_wav("8a_output_room_WPE_MIC.wav", FS, out_room_wpe_mic, output_folder)

    # --- C.2 Caso Micrófonos Ideales (WPE + Beamformer) ---
    print("  -> [IDEAL] Aplicando WPE a 'room_input_ideal'...")
    room_input_wpe_ideal = apply_wpe(room_input_ideal, FS, taps=10, delay=3, iterations=3)
    save_wav("7b_input_room_WPE_IDEAL.wav", FS, room_input_wpe_ideal[0], output_folder)

    print("  -> [IDEAL] Procesando señal WPE con Beamformer...")
    bf_wpe_ideal = LowRankAdaptive(mic_coords, FS, alpha=ALPHA)
    out_room_wpe_ideal = bf_wpe_ideal.block_process(
        input_signals=room_input_wpe_ideal,
        target_pos=target_pos_flat,
        M1=M1, M2=M2, P=RANK_P,
        mode="near_field",
        min_loading=MIN_LOADING
    )
    save_wav("8b_output_room_WPE_IDEAL.wav", FS, out_room_wpe_ideal, output_folder)

    # =========================================================================
    # VISUALIZACIÓN Y CIERRE
    # =========================================================================
    interferences = [interf_pos1, interf_pos2]
    plot_scene_3d(room_dimensions, mic_coords, source_pos, interferences, output_folder)

    print(f"\n=== PROCESO TERMINADO ===")
    print(f"Archivos guardados en: {output_folder}")