import os
import numpy as np
import scipy.signal as signal

# Ensure import paths strictly align with your localized package repository layout
from propagation.simulate_acoustics_v1 import SimAcoustic
from propagation.mird_loader import MirdDatasetProvider, generate_mird_linear_array
from beamforming.MVDR.base import apply_mvdr_stft_bridge  # Importing your existing STFT bridge core
from utils.audio import save_wav, normalize_signal
from dereverberation.nara_wrappers import process_wpe_online


def run_mird_mvdr_evaluation():
    print("\n==================================================")
    print("      FINAL VALIDATION: MIRD DATASET + MVDR       ")
    print("==================================================\n")

    # --- 1. CONFIGURATION & PATHS ---
    # Downsample target physical sampling rate to wideband speech processing standard
    FS = 16000
    sim_duration = 30
    target_isir_db = 0
    
    # Establish dynamic paths mapping structural evaluation roots safely
    root_mird_dir = os.path.abspath("data/rirs/mird")
    output_dir = os.path.abspath("tests/dataset_out/mird_mvdr_final")
    os.makedirs(output_dir, exist_ok=True)
    
    source_audio_path = r"data/audio/input/p002_emo_adoration_sentences.wav"
    interf_audio_path = r"data/audio/input/hairdryer_07_SH_MKH800.wav"

    # --- 2. INDEX DATASET PROVIDER ---
    print("[*] Bootstrapping physical multi-channel MIRD registry provider...")
    provider = MirdDatasetProvider(root_dir=root_mird_dir)
    
    available_t60s = provider.get_available_t60s()
    if not available_t60s:
        print("[!] Execution aborted: Verified database indices yield empty structures.")
        return
        
    # Evaluate targeted physical reverberation profile fallback logic safely
    chosen_t60_ms = 160 if 160 in available_t60s else available_t60s[0]
    chosen_t60 = chosen_t60_ms / 1000.0
    chosen_spacing = "4-4-4-8-4-4-4"
    target_radius = 1.0  # Radial distance mapping target source placement

    print(f"[*] Extracting physical scenario targeted at environment T60={chosen_t60_ms}ms...")

    # --- 3. EXTRACT PRECISE MEASUREMENT POSITIONS ---
    # Extract Target Source: Incidence matching forward broadside axis (0 degrees azimuth)
    _ = provider.load_rir(chosen_t60, chosen_spacing, target_radius, angle=0)
    rel_pos_target = provider.export_position(mode='cartesian')
    
    # Extract Interference Source: Lateral structural incidence profile (45 degrees azimuth)
    _ = provider.load_rir(chosen_t60, chosen_spacing, target_radius, angle=45)
    rel_pos_interf = provider.export_position(mode='cartesian')
    print(f"pos interf: {rel_pos_interf}")

    # --- 4. CONSTRUCT ABSOLUTE VIRTUAL SCENE ---
    # Define absolute boundary configurations bounding physical Bar-Ilan laboratory parameters
    room_dims = np.array([6.0, 6.0, 2.41])
    array_center = np.array([3.0, 3.0, 1.2])
    
    # Map absolute Cartesian array nodes leveraging theoretical ideal spatial geometries
    base_array = generate_mird_linear_array()
    translated_array = base_array + array_center

    print(f"array relative dim: {base_array}")
    print(f"source relative dim: {rel_pos_target}")
    
    # Translate relative database physical incidence coordinates to continuous absolute workspace vectors
    abs_pos_target = array_center + rel_pos_target.squeeze()
    abs_pos_interf = array_center + rel_pos_interf.squeeze()

    print(f"[*] Initializing virtual continuous scene environment...")
    print(f"    -> Target Source translated absolute coordinate: {abs_pos_target} m")
    print(f"    -> Interference Source translated absolute coordinate: {abs_pos_interf} m")

    scene = SimAcoustic(
        array_geometry=translated_array, 
        array_mismatch=0.0, 
        duration=sim_duration, 
        fs=FS
    )
    
    # Register audio arrays passing absolute spatial matrices matching target steering mappings
    scene.set_source(source_audio_path, gain=1.0, position=abs_pos_target.reshape(1, 3))
    scene.set_interference(interf_audio_path, gain=1.0, position=abs_pos_interf.reshape(1, 3))

    # --- 5. ASSEMBLE PHYSICAL MIRD RIRS ---
    print("\n[*] Assembling physical multichannel matrices overriding synthetic propagation kernels...")
    # Trigger spatial snapping translation algorithms natively applying high-fidelity polyphase resampling
    scene.import_rirs(
        dataset_provider=provider,
        target_t60=chosen_t60,
        array_center=array_center,
        spacing_cfg=chosen_spacing
    )

    # --- 6. EXECUTE MULTICHANNEL CONVOLUTION PIPELINE ---
    print("[*] Processing multichannel spatial array temporal convolutions...")
    scene.convolve_signals(t_early=0.050)
    
    print(f"[*] Blending degraded spatial channels matching physical evaluation iSIR={target_isir_db} dB...")
    scene_mix = scene.mix_and_normalize(iSIR_dB=target_isir_db)
    
    mic_signals = scene_mix["mic_signals"]
    vad_oracle = scene_mix["VAD"]

    # --- 7. APPLY RECURSIVE MVDR SPATIAL FILTERING ---
    print("\n[*] Triggering adaptive Recursive MVDR spatial beamforming algorithms...")
    # Pass physical array geometry and precise absolute target position to resolve accurate phase steering
    clean_output_time = apply_mvdr_stft_bridge(
        time_domain_input=mic_signals,
        vad_oracle = vad_oracle,
        mic_coords=    base_array ,
        source_pos_2d= rel_pos_target.reshape(1, 3),
        fs=FS,
        
    )

    # --- 8. EXPORT VALIDATED AUDIO ARTIFACTS ---
    print(f"\n[*] Exporting multi-channel evaluation tracks locally to: {output_dir}")
    
    # Save pure target anechoic reference track extracted safely from primary reference sensor node
    save_wav("1_MIRD_target_anechoic_ref.wav", FS, scene_mix["target_anechoic"][0], output_dir)
    
    # Save unenhanced degraded input physical sensor reference (Mic 0)
    save_wav("2_MIRD_degraded_input_mic0.wav", FS, normalize_signal(mic_signals[0]), output_dir)
    
    # Save final MVDR adaptive spatial reconstruction output normalized preventing numerical integer clipping
    save_wav("3_MIRD_MVDR_enhanced_output.wav", FS, normalize_signal(clean_output_time), output_dir)
    
    print("\n[+] Physical dataset evaluation framework validation successfully executed.")
    print("==================================================")
    # -------------------------------------------------------------------
    # PHASE 3: WPE DEREVERBERATION + RECURSIVE MVDR
    # -------------------------------------------------------------------
    print("\n--- PHASE 3: WPE + MVDR PIPELINE ---")
    print(" -> Applying Online WPE Dereverberation on Room Simulation...")
    
    wpe_output = process_wpe_online( mic_signals , delay = 3, taps = 7)
    
    save_wav("5_WPE_input_mix_mic0.wav", FS, wpe_output[0], output_dir)

    print(" -> Applying Recursive MVDR on Dereverberated Signals...")
    clean_output_time_wpe = apply_mvdr_stft_bridge(
        time_domain_input=wpe_output,
        vad_oracle = vad_oracle,
        mic_coords=    base_array ,
        source_pos_2d= rel_pos_target.reshape(1, 3),
        fs=FS,
        
    )
    save_wav("6_WPE_ROOM_output_final.wav", FS, normalize_signal(clean_output_time_wpe), output_dir)

    print("\n -> Pipeline completed successfully.") 

if __name__ == "__main__":
    run_mird_mvdr_evaluation()