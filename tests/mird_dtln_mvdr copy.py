import os
import numpy as np
import scipy.signal as signal
import tensorflow as tf

# Ensure import paths strictly align with your localized package repository layout
from propagation.simulate_acoustics_v1 import SimAcoustic
from propagation.mird_loader import MirdDatasetProvider, generate_mird_linear_array
from beamforming.mask.dtln_mvdr_v1 import apply_hybrid_pipeline
from utils.audio import save_wav, normalize_signal

# Import the DTLN real-time function from the provided file
from dnn_denoise.dtln_lite import apply_dtln_post_tflite_realtime

# IMPORT WPE WRAPPER
from dereverberation.nara_wrappers import process_wpe_online

def run_mird_mvdr_evaluation():
    print("\n==================================================")
    print("      FINAL VALIDATION: MIRD DATASET + WPE + MVDR + DTLN")
    print("==================================================\n")

    # --- 1. CONFIGURATION & PATHS ---
    FS = 16000
    sim_duration = 15
    target_isir_db = 0
    
    root_mird_dir = os.path.abspath("tools/data/rirs/mird")
    output_dir = os.path.abspath("tests/dataset_out/mird_mvdr_final")
    os.makedirs(output_dir, exist_ok=True)
    
    source_audio_path = r"tools\data\signals\p002_emo_adoration_sentences.wav"
    interf_audio_path = r"tools\data\signals\ruido_rosa_16k.wav"

    # --- 2. INDEX DATASET PROVIDER ---
    print("[*] Bootstrapping physical multi-channel MIRD registry provider...")
    provider = MirdDatasetProvider(root_dir=root_mird_dir)
    
    available_t60s = provider.get_available_t60s()
    if not available_t60s:
        print("[!] Execution aborted: Verified database indices yield empty structures.")
        return
        
    chosen_t60_ms = 610 if 610 in available_t60s else available_t60s[0]
    chosen_t60 = chosen_t60_ms / 1000.0
    chosen_spacing = "3-3-3-8-3-3-3"
    target_radius = 1.0

    print(f"[*] Extracting physical scenario targeted at environment T60={chosen_t60_ms}ms...")

    # --- 3. EXTRACT PRECISE MEASUREMENT POSITIONS ---
    _ = provider.load_rir(chosen_t60, chosen_spacing, target_radius, angle=0)
    rel_pos_target = provider.export_position(mode='cartesian')
    
    _ = provider.load_rir(chosen_t60, chosen_spacing, target_radius, angle=45)
    rel_pos_interf = provider.export_position(mode='cartesian')

    # --- 4. CONSTRUCT ABSOLUTE VIRTUAL SCENE ---
    room_dims = np.array([6.0, 6.0, 2.41])
    array_center = np.array([3.0, 3.0, 1.2])
    
    base_array = generate_mird_linear_array()
    translated_array = base_array + array_center
    
    abs_pos_target = array_center + rel_pos_target.squeeze()
    abs_pos_interf = array_center + rel_pos_interf.squeeze()

    print(f"[*] Initializing virtual continuous scene environment...")
    
    scene = SimAcoustic(
        array_geometry=translated_array, 
        array_mismatch=0.0, 
        duration=sim_duration, 
        fs=FS
    )
    
    scene.set_source(source_audio_path, gain=1.0, position=abs_pos_target.reshape(1, 3))
    scene.set_interference(interf_audio_path, gain=1.0, position=abs_pos_interf.reshape(1, 3))

    # --- 5. ASSEMBLE PHYSICAL MIRD RIRS ---
    print("\n[*] Assembling physical multichannel matrices overriding synthetic propagation kernels...")
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

    # --- 6.5. APLICAR DESREVERBERACIÓN ONLINE (WPE) ---
    print("\n[*] Applying Online WPE Dereverberation (Pre-processing branch)...")
    # Aplicamos WPE a la matriz multicanal. Mantenemos los parámetros por defecto del wrapper.
    wpe_signals = process_wpe_online(mic_signals, delay=1, taps =12)

    # --- 7. INITIALIZE DTLN TFLITE INTERPRETERS ---
    print("\n[*] Loading DTLN TF-Lite interpreters in memory...")
    model_1_path = r"tools\data\models\model_quant_1.tflite"
    model_2_path = r"tools\data\models\model_quant_2.tflite"
    
    interpreter_1 = tf.lite.Interpreter(model_path=model_1_path)
    interpreter_1.allocate_tensors()
    
    interpreter_2 = tf.lite.Interpreter(model_path=model_2_path)
    interpreter_2.allocate_tensors()

    # --- 8. PURE DTLN PROCESSING (Pre-MVDR Comparison) ---
    print("[*] Processing degraded Mic 0 with pure DTLN...")
    mic0_audio = mic_signals[0].copy()
    mic0_max = np.max(np.abs(mic0_audio))
    if mic0_max > 0: mic0_audio = mic0_audio / mic0_max
    dtln_only_output = apply_dtln_post_tflite_realtime(interpreter_1, interpreter_2, mic0_audio)

    print("[*] Processing WPE Mic 0 with pure DTLN...")
    wpe_mic0_audio = wpe_signals[0].copy()
    wpe_mic0_max = np.max(np.abs(wpe_mic0_audio))
    if wpe_mic0_max > 0: wpe_mic0_audio = wpe_mic0_audio / wpe_mic0_max
    wpe_dtln_only_output = apply_dtln_post_tflite_realtime(interpreter_1, interpreter_2, wpe_mic0_audio)


    # --- 9. APPLY DTLN MASK-BASED MVDR FILTERING ---
    print("\n[*] Triggering adaptive DTLN Mask-Based MVDR spatial beamforming algorithms...")
    mvdr_output_time, mask = apply_hybrid_pipeline(
        time_domain_input=mic_signals, fs=FS, model1_path=model_1_path, length_fft=512, hop_length_fft=128
    )

    print("[*] Triggering adaptive DTLN Mask-Based MVDR on WPE signals...")
    wpe_mvdr_output_time, wpe_mask = apply_hybrid_pipeline(
        time_domain_input=wpe_signals, fs=FS, model1_path=model_1_path, length_fft=512, hop_length_fft=128
    )

    # --- 10. APPLY DTLN POST-PROCESSING (MVDR + DTLN) ---
    print("\n[*] Processing MVDR output with DTLN (Post-processing)...")
    mvdr_audio = mvdr_output_time.copy()
    mvdr_max = np.max(np.abs(mvdr_audio))
    if mvdr_max > 0: mvdr_audio = mvdr_audio / mvdr_max
    mvdr_dtln_output = apply_dtln_post_tflite_realtime(interpreter_1, interpreter_2, mvdr_audio)

    print("[*] Processing WPE+MVDR output with DTLN (Post-processing)...")
    wpe_mvdr_audio = wpe_mvdr_output_time.copy()
    wpe_mvdr_max = np.max(np.abs(wpe_mvdr_audio))
    if wpe_mvdr_max > 0: wpe_mvdr_audio = wpe_mvdr_audio / wpe_mvdr_max
    wpe_mvdr_dtln_output = apply_dtln_post_tflite_realtime(interpreter_1, interpreter_2, wpe_mvdr_audio)


    # --- 11. EXPORT VALIDATED AUDIO ARTIFACTS ---
    print(f"\n[*] Exporting evaluation tracks locally to: {output_dir}")
    
    # Referencias y Mezcla Original (RAMA ESTANDAR)
    save_wav("1_MIRD_target_anechoic_ref.wav", FS, normalize_signal(scene_mix["target_anechoic"][0]), output_dir)
    save_wav("2_MIRD_degraded_input_mic0.wav", FS, normalize_signal(mic_signals[0]), output_dir)
    save_wav("3_MIRD_DTLN_only_output.wav", FS, normalize_signal(dtln_only_output), output_dir)
    save_wav("4_MIRD_MVDR_only_output.wav", FS, normalize_signal(mvdr_output_time), output_dir)
    save_wav("5_MIRD_MVDR_DTLN_output.wav", FS, normalize_signal(mvdr_dtln_output), output_dir)

    # Exportes de la RAMA WPE PRE-PROCESADA
    save_wav("6_MIRD_WPE_degraded_input_mic0.wav", FS, normalize_signal(wpe_signals[0]), output_dir)
    save_wav("7_MIRD_WPE_DTLN_only_output.wav", FS, normalize_signal(wpe_dtln_only_output), output_dir)
    save_wav("8_MIRD_WPE_MVDR_only_output.wav", FS, normalize_signal(wpe_mvdr_output_time), output_dir)
    save_wav("9_MIRD_WPE_MVDR_DTLN_output.wav", FS, normalize_signal(wpe_mvdr_dtln_output), output_dir)

    print("\n[+] Physical dataset evaluation framework validation successfully executed.")
    print("==================================================")


if __name__ == "__main__":
    run_mird_mvdr_evaluation()