import sys
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt 

from propagation.simulate_acoustics import SimAcoustic
from beamforming.array.microphone import Microphone
from utils.audio import save_wav
from beamforming.dereverberation.wpe import apply_wpe 
from beamforming.dereverberation.gwpe import batch_dereverb
from beamforming.dereverberation.kwpe import dereverb_kawpe_mimo

def plot_scene_3d(room_dims, mics, target, interferences_list, save_path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Draw Microphones
    ax.scatter(mics[:,0], mics[:,1], mics[:,2], c='blue', marker='.', s=50, label='Mic Array', depthshade=False)

    # 2. Draw Target
    target = target.flatten()
    ax.scatter(target[0], target[1], target[2], c='green', marker='*', s=200, label='Target Source', edgecolors='black')
    ax.plot([target[0], target[0]], [target[1], target[1]], [0, target[2]], 'g--', alpha=0.3)

    # 3. Draw Interferences
    for i, interf in enumerate(interferences_list):
        interf = interf.flatten()
        ax.scatter(interf[0], interf[1], interf[2], c='red', marker='X', s=100, label=f'Interference {i+1}')
        ax.plot([interf[0], interf[0]], [interf[1], interf[1]], [0, interf[2]], 'r--', alpha=0.3)

    # 4. Configure Room Space
    ax.set_xlim([0, room_dims[0]])
    ax.set_ylim([0, room_dims[1]])
    ax.set_zlim([0, room_dims[2]])

    ax.set_xlabel('X [m] (Width)')
    ax.set_ylabel('Y [m] (Length)')
    ax.set_zlabel('Z [m] (Height)')
    ax.set_title(f'Simulation Geometry\nRoom: {room_dims}m | RT60: {RT}s')
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1))

    try:
        ax.set_box_aspect((room_dims[0], room_dims[1], room_dims[2]))
    except AttributeError:
        pass 

    filename = os.path.join(save_path, "scene_setup_3d.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  -> 3D Scene visualization saved at: {filename}")

def ensure_folder(base_path, m1, m2, RT, src_pos):
    pos_str = f"{src_pos[0]}_{src_pos[1]}_{src_pos[2]}"
    folder_name = f"M={m1}x{m2}_RT={RT}_Src={pos_str}_Dereverb_Only"
    full_path = os.path.join(base_path, folder_name)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    return full_path

def normalize_by_energy(signal, target_rms=0.05):
    """
    Normaliza la señal basándose en la energía (RMS) para igualar volúmenes.
    target_rms define el nivel de energía objetivo.
    """
    rms = np.sqrt(np.mean(signal**2))
    if rms == 0:
        return signal
    return signal * (target_rms / rms)

# --- PARAMETER CONFIGURATION ---
FS = 21000
M1, M2 = 3, 4        
RT = 0.6          

# GWPE Parameters
K_GWPE = 40

# Source position RELATIVE to array center
SRC_REL_POS = [0.4, 0.1, 0.1] 

# --- MAIN SCRIPT ---
if __name__ == "__main__":
    print("=== INTEGRATION TEST: WPE, GWPE & KWPE (MIC ONLY) ===")
    
    # Dictionary to store execution times
    execution_times = {}

    # 1. DIRECTORY SETUP
    base_data_path = "tests/data"
    output_folder = ensure_folder(base_data_path, M1, M2, RT, SRC_REL_POS)
    print(f"[IO] Results will be saved in: {output_folder}")
    
    if not os.path.exists(base_data_path): 
        os.makedirs(base_data_path)
    
    cache_file_data = os.path.join(base_data_path, "room_simulation_cache.npy")
    cache_file_meta = os.path.join(base_data_path, "room_simulation_meta.json")

    # 2. ARRAY GEOMETRY
    mic_spacing = 0.03
    Mx, My, Mz = M1, M2, 1                                                                                      
    M = Mx * My * Mz 
    
    x = np.linspace(0, (Mx-1)*mic_spacing, Mx)
    y = np.linspace(0, (My-1)*mic_spacing, My)
    z = np.array([0.0]) 
    xv, yv, zv = np.meshgrid(x, y, z, indexing='xy') 
    mic_coords = np.column_stack([xv.flatten(), yv.flatten(), zv.flatten()])

    # Center array in the room
    array_center = np.array([1.25, 2.0, 1.25])
    mic_coords = mic_coords - np.mean(mic_coords, axis=0) + array_center
    
    # 3. SOURCE POSITIONS
    source_pos = array_center + np.array(SRC_REL_POS)
    interf_pos1 = array_center + np.array([0.0, 1.2, 0.0])
    interf_pos2 = array_center + np.array([-0.6, 0.6, 0.0])

    # 4. ACOUSTIC SCENE CONFIGURATION
    acoustic_scene = SimAcoustic(mic_coords, array_mismatch=0.0, duration=8, fs = FS)
    room_dimensions = np.array([2.5, 4.0, 2.5])

    source_path = "tools/data/signals/FA01_09.wav"
    int_path1 = "tools/data/signals/MC15_03.wav"
    int_path2 = "tools/data/signals/MF31_03.wav"

    acoustic_scene.set_source(source_path, gain=1, position=source_pos.reshape(1,3))
    acoustic_scene.set_interference(int_path1, gain=1, position=interf_pos1.reshape(1,3))
    acoustic_scene.set_interference(int_path2, gain=1, position=interf_pos2.reshape(1,3))

    # 5. MICROPHONE MODEL
    mic_model = Microphone(model="MP34DT01-M", fs=FS)

    # =========================================================================
    # PART A: REVERBERANT FIELD (ROOM SIMULATION)
    # =========================================================================
    print("\n--- [A] REVERBERANT FIELD PROCESSING ---")
    
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
                print(f"[Cache] HIT: Parameters match. Loading cached data...")
                use_cache = True
            else:
                print(f"[Cache] MISS: Parameter change detected.")
        except Exception as e:
            print(f"[Cache] Error reading metadata. Recalculating...")
            use_cache = False
    else:
        print("[Cache] MISS: No previous cache found.")

    t0_sim = time.time()
    if use_cache:
        room_input_ideal = np.load(cache_file_data)
    else:
        print("  -> Computing room simulation...")
        room_input_ideal = acoustic_scene.compute_room_ISB(room_dimensions, desire_RT=RT, iSIR_dB=0)    
        np.save(cache_file_data, room_input_ideal)
        with open(cache_file_meta, 'w') as f:
            json.dump(current_sim_params, f)
        print("  -> Simulation saved to cache.")
    execution_times['Room Simulation'] = time.time() - t0_sim

    # A.3 Apply Microphone Model
    print("  -> Applying microphone emulation model...")
    room_input_mic = mic_model.emulate(room_input_ideal, show_plots=False)
    
    # Se normaliza la señal del micrófono antes de guardar
    mic_norm = normalize_by_energy(room_input_mic[0])
    save_wav("1_input_room_MIC.wav", FS, mic_norm, output_folder)

    # =========================================================================
    # PART B: DEREVERBERATION (KWPE)
    # =========================================================================
    print("\n--- [B] DEREVERBERATION PROCESSING (KWPE) ---")

    print("  -> [MIC] Applying KWPE to 'room_input_mic'...")
    t0 = time.time()
    room_input_kwpe_mic = dereverb_kawpe_mimo(room_input_mic, FS, K_GWPE )
    execution_times['KWPE_MIC'] = time.time() - t0
    
    kwpe_mic_norm = normalize_by_energy(room_input_kwpe_mic[0])
    save_wav("2_output_room_KWPE_MIC.wav", FS, kwpe_mic_norm, output_folder)

    # =========================================================================
    # PART C: DEREVERBERATION (WPE) 
    # =========================================================================
    print("\n--- [C] DEREVERBERATION PROCESSING (WPE) ---")

    print("  -> [MIC] Applying WPE to 'room_input_mic'...")
    t0 = time.time()
    room_input_wpe_mic = apply_wpe(room_input_mic, FS, taps=15, delay=3, iterations=3)
    execution_times['WPE_MIC'] = time.time() - t0
    
    wpe_mic_norm = normalize_by_energy(room_input_wpe_mic[0])
    save_wav("3_output_room_WPE_MIC.wav", FS, wpe_mic_norm, output_folder)

    # =========================================================================
    # PART D: DEREVERBERATION (GWPE)
    # =========================================================================
    print("\n--- [D] DEREVERBERATION PROCESSING (GWPE) ---")

    print("  -> [MIC] Applying GWPE to 'room_input_mic'...")
    t0 = time.time()
    room_input_gwpe_mic = batch_dereverb(room_input_mic, FS, K=K_GWPE)
    execution_times['GWPE_MIC'] = time.time() - t0
    
    gwpe_mic_norm = normalize_by_energy(room_input_gwpe_mic[0])
    save_wav("4_output_room_GWPE_MIC.wav", FS, gwpe_mic_norm, output_folder)

    # =========================================================================
    # VISUALIZATION & PERFORMANCE REPORT
    # =========================================================================
    interferences = [interf_pos1, interf_pos2]
    plot_scene_3d(room_dimensions, mic_coords, source_pos, interferences, output_folder)

    print("\n=== PERFORMANCE REPORT ===")
    for process_name, elapsed_time in execution_times.items():
        print(f"  -> {process_name:<18}: {elapsed_time:.4f} seconds")

    print(f"\n=== PROCESS COMPLETED ===")
    print(f"Files saved successfully in: {output_folder}")