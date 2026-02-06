import sys
import os
import numpy as np
from beamforming.kmvdr.system import LowRankAdaptive
from propagation.simulate_acoustics import SimAcoustic
from beamforming.array.microphone import Microphone  # <--- Nuevo Módulo
from utils.audio import save_wav

# --- MAIN SCRIPT ---
if __name__ == "__main__":
    print("=== INTEGRATION TEST: CACHED SIMULATION + MIC ERRORS + LOW-RANK BF ===")
    
    # 1. SETUP & GEOMETRY
    fs = 48000
    mic_spacing = 0.05
    folder_path = "tests/data"
    if not os.path.exists(folder_path): os.makedirs(folder_path)
    
    # Archivo de caché para la simulación acústica pesada
    cache_file = os.path.join(folder_path, "room_simulation_cache.npy")

    # --- Planar Array 4x4 (URA) for Kronecker Compatibility ---
    Mx, My, Mz = 3, 4, 1 
    M = Mx * My * Mz 
    M1, M2 = 3, 4 # Kronecker factors

    # Generate grid coordinates
    x = np.linspace(0, (Mx-1)*mic_spacing, Mx)
    y = np.linspace(0, (My-1)*mic_spacing, My)
    z = np.array([0.0])
    xv, yv, zv = np.meshgrid(x, y, z, indexing='xy') 
    
    mic_coords = np.column_stack([xv.flatten(), yv.flatten(), zv.flatten()])

    # Center array in the room
    array_center = np.array([1.25, 2.0, 1.25])
    mic_coords = mic_coords - np.mean(mic_coords, axis=0) + array_center
    
    print(f"[Setup] Planar Array Configured: {Mx}x{My}x{Mz} ({M} mics)")

    # 2. SOURCE POSITIONS
    source_pos = array_center + np.array([0.9, 0.0, 0.1]) 
    interf_pos1 = array_center + np.array([0.0, 1.2, 0.0])
    interf_pos2 = array_center + np.array([-0.6, 0.6, 0.0])

    source_pos = source_pos.reshape(1,3)
    interf_pos1 = interf_pos1.reshape(1,3)
    interf_pos2 = interf_pos2.reshape(1,3)

    # 3. ACOUSTIC SCENE SETUP
    # Nota: array_mismatch en SimAcoustic es posicional, el error de sensor se añade después con Microphone()
    acoustic_scene = SimAcoustic(mic_coords, array_mismatch=1e-3, duration=8)
    room_dimensions = np.array([2.5, 4, 2.5])

    # Load audio files
    source_path = "tools/data/signals/FA01_09.wav"
    int_path1 = "tools/data/signals/MC15_03.wav"
    int_path2 = "tools/data/signals/MF31_03.wav"

    print("[Sim] Loading sources...")
    acoustic_scene.set_source(source_path, gain=1, position=source_pos)
    acoustic_scene.set_interference(int_path1, gain=1, position=interf_pos1)
    acoustic_scene.set_interference(int_path2, gain=1, position=interf_pos2)

    # 4. BEAMFORMER & MICROPHONE INSTANTIATION
    bf = LowRankAdaptive(mic_coords, fs, alpha=0.96)
    target_pos_flat = source_pos.flatten()
    Rank_P = 1
    
    # Instanciamos el modelo de micrófono (Simula errores de hardware real)
    mic_model = Microphone(model="MP34DT01-M", fs=fs)

    # =========================================================================
    # PART A: ROOM SIMULATION (Reverberant) with CACHE
    # =========================================================================
    print("\n[Part A] Processing Room Simulation (RT60=1.0s)...")
    
    # --- ETAPA DE CACHÉ ---
    if os.path.exists(cache_file):
        print(f"[Cache] HIT: Cargando simulación previa desde {cache_file}...")
        room_input_raw = np.load(cache_file)
    else:
        print("[Cache] MISS: Computando simulación de sala (esto puede tardar)...")
        room_input_raw = acoustic_scene.compute_room_ISB(room_dimensions, desire_RT=.5, iSIR_dB=0)
        np.save(cache_file, room_input_raw)
        print(f"[Cache] Simulación guardada en {cache_file}")

    # --- ETAPA DE ERRORES DE SENSOR ---
    print("[Mic] Aplicando errores de sensores (Gain/Phase Mismatch + Ruido)...")
    # Pasamos la señal ideal de la sala por el modelo de micrófono imperfecto
    room_input_degraded = mic_model.emulate(room_input_raw, show_plots=False)
    
    save_wav("input_room_degraded.wav", fs, room_input_degraded[0], folder_path)

    # Process with Beamformer
    print("[BF] Ejecutando Beamformer Low-Rank...")
    output_room = bf.block_process(
        input_signals=room_input_degraded, 
        target_pos=target_pos_flat, 
        M1=M1, M2=M2, P=Rank_P,
        record_scene=True,
        mode="near_field",
        min_loading = 1e-4
    )

    # Save Output
    norm_room = output_room / (np.max(np.abs(output_room)) + 1e-9) * 0.9
    save_wav("output_room_cleaned.wav", fs, norm_room, folder_path)
    print(f"-> Saved: {folder_path}/output_room_cleaned.wav")

    # =========================================================================
    # PART B: FREE FIELD SIMULATION (Anechoic Reference)
    # =========================================================================
    print("\n[Part B] Processing Free Field Simulation (Anechoic)...")

    # Generate noisy anechoic input (Fast calculation, no cache needed usually)
    ff_input_raw = acoustic_scene.free_field(iSIR_dB=0, normalize=True, mode="real")
    
    # También aplicamos errores de micrófono aquí para ser justos en la comparación
    ff_input_degraded = mic_model.emulate(ff_input_raw, show_plots=False)
    
    save_wav("input_freefield_degraded.wav", fs, ff_input_degraded[0], folder_path)

    # Process with Beamformer
    output_ff = bf.block_process(
        input_signals=ff_input_degraded, 
        target_pos=target_pos_flat, 
        M1=M1, M2=M2, P=Rank_P,
        record_scene=True, 
        mode="near_field",
        min_loading = 1e-4
    )

    # Save Output
    norm_ff = output_ff / (np.max(np.abs(output_ff)) + 1e-9) * 0.9
    save_wav("output_freefield_cleaned.wav", fs, norm_ff, folder_path)
    print(f"-> Saved: {folder_path}/output_freefield_cleaned.wav")
    
    print("\n=== TEST COMPLETED SUCCESSFULLY ===")