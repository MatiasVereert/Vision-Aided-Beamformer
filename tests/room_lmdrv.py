import sys
import os
import numpy as np
from beamforming.kmvdr.system import LowRankAdaptive
from propagation.simulate_acoustics import SimAcoustic
from utils.audio import save_wav

# --- MAIN SCRIPT ---
if __name__ == "__main__":
    print("=== INTEGRATION TEST: SIMULATION + LOW-RANK BEAMFORMING ===")
    
    # 1. SETUP & GEOMETRY
    fs = 48000
    mic_spacing = 0.04 
    folder_path = "tests/data"
    if not os.path.exists(folder_path): os.makedirs(folder_path)

    # --- Planar Array 4x4 (URA) for Kronecker Compatibility ---
    Mx, My, Mz = 3, 4, 1 
    M = Mx * My * Mz 
    M1, M2 = 3, 4 # Kronecker factors

    # Generate grid coordinates
    x = np.linspace(0, (Mx-1)*mic_spacing, Mx)
    y = np.linspace(0, (My-1)*mic_spacing, My)
    z = np.array([0.0])
    xv, yv, zv = np.meshgrid(x, y, z, indexing='xy') 
    
    # Flatten using column-major order to align with Kronecker structure
    mic_coords = np.column_stack([xv.flatten(), yv.flatten(), zv.flatten()])

    # Center array in the room
    array_center = np.array([1.25, 2.0, 1.25])
    mic_coords = mic_coords - np.mean(mic_coords, axis=0) + array_center
    
    print(f"[Setup] planar Array Configured: {Mx}x{My}x{Mz} ({M} mics)")

    # 2. SOURCE POSITIONS
    # Target: Front (+X), slightly up (+Z) relative to array center
    source_pos = array_center + np.array([0.9, 0.0, 0.1]) 
    
    # Interference 1: Side (+Y)
    interf_pos1 = array_center + np.array([0.0, 1.2, 0.0])
    
    # Interference 2: Back-Diagonal (-X, +Y)
    interf_pos2 = array_center + np.array([-0.6, 0.6, 0.0])

    # Reshape for simulation input
    source_pos = source_pos.reshape(1,3)
    interf_pos1 = interf_pos1.reshape(1,3)
    interf_pos2 = interf_pos2.reshape(1,3)

    # 3. ACOUSTIC SCENE SETUP
    acoustic_scene = SimAcoustic(mic_coords, array_mismatch=1e-3, duration=4)
    room_dimensions = np.array([2.5, 4, 2.5])

    # Load audio files
    source_path = "tools/data/signals/FA01_09.wav"
    int_path1 = "tools/data/signals/MC15_03.wav"
    int_path2 = "tools/data/signals/MF31_03.wav"

    print("[Sim] Loading sources...")
    acoustic_scene.set_source(source_path, gain=1, position=source_pos)
    acoustic_scene.set_interference(int_path1, gain=1, position=interf_pos1)
    acoustic_scene.set_interference(int_path2, gain=1, position=interf_pos2)

    # 4. BEAMFORMER INSTANTIATION
    # Alpha=0.995 for stability in reverberant environments
    bf = LowRankAdaptive(mic_coords, fs, alpha=0.99)
    target_pos_flat = source_pos.flatten()
    Rank_P = 2

    # =========================================================================
    # PART A: ROOM SIMULATION (Reverberant)
    # =========================================================================
    print("\n[Part A] Processing Room Simulation (RT60=0.5s)...")
    
    # Generate noisy reverberant input
    room_input = acoustic_scene.compute_room_ISB(room_dimensions, desire_RT=1, iSIR_dB=5)
    save_wav("input_room_dirty.wav", fs, room_input[0], folder_path)

    # Process with Beamformer
    output_room = bf.block_process(
        input_signals=room_input, 
        target_pos=target_pos_flat, 
        M1=M1, M2=M2, P=Rank_P,
        record_scene=True,
        mode="near_field"
    )

    # Save Output
    norm_room = output_room / (np.max(np.abs(output_room)) + 1e-9) * 0.9
    save_wav("output_room_cleaned.wav", fs, norm_room, folder_path)
    print(f"-> Saved: {folder_path}/output_room_cleaned.wav")

    # =========================================================================
    # PART B: FREE FIELD SIMULATION (Anechoic Reference)
    # =========================================================================
    print("\n[Part B] Processing Free Field Simulation (Anechoic)...")

    # Generate noisy anechoic input
    ff_input = acoustic_scene.free_field(iSIR_dB=0, normalize=True, mode="real")
    save_wav("input_freefield_dirty.wav", fs, ff_input[0], folder_path)

    # Process with Beamformer (Using the CORRECT input signal)
    # Note: Using a fresh instance or resetting weights might be cleaner, 
    # but reusing 'bf' tests adaptation speed from room->freefield.
    output_ff = bf.block_process(
        input_signals=ff_input, # <--- FIXED: Now using ff_input
        target_pos=target_pos_flat, 
        M1=M1, M2=M2, P=Rank_P,
        record_scene=True, 
        mode="near_field"
    )

    # Save Output
    norm_ff = output_ff / (np.max(np.abs(output_ff)) + 1e-9) * 0.9
    save_wav("output_freefield_cleaned.wav", fs, norm_ff, folder_path)
    print(f"-> Saved: {folder_path}/output_freefield_cleaned.wav")
    
    print("\n=== TEST COMPLETED SUCCESSFULLY ===")