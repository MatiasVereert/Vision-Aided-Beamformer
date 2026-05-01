from evaluation.full_benchmark_test_1 import run_grid_search
import os 
import numpy as np
from evaluation.bf_wrappers import (
    DS_Processor,
    MVDR_Recursive_Processor,
    KMVDR_Recursive_Processor,
    SDW_MWF_Processor,
    MPDR_Recursive_Processor,
    RTF_MVDR_Recursive_Processor,
    SPP_MVDR_Recursive_Processor
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

if __name__ == "__main__":
    
    # Define the dimensions for each RT60 scenario
    ROOM_PROFILES = {
        0.0: np.array([3.0, 4.0, 2.5]),
        0.3: np.array([3.0, 4.0, 2.5]),
        0.6: np.array([5.0, 6.0, 2.8]),
        0.9: np.array([6.0, 7.0, 3.0]) 
    }

    # Base configuration for the acoustic scene
    base_config = {
        'fs': 16000,
        'duration': 30,
        'd_min': 0.04,
        'd_max': 0.30,
        'radius_source': 0.6,    
        'radius_interf': 1.2,    
        'delta_ang_deg': 30.0,   
        'snr_db': 60.0,          
        'source_path': os.path.join(PROJECT_ROOT, "tools", "data", "signals", "p002_emo_adoration_sentences.wav"),
        'interf_paths': [
            os.path.join(PROJECT_ROOT, "tools", "data", "signals", "p017_rainbow_08_fast.wav"),
            os.path.join(PROJECT_ROOT, "tools", "data", "signals", "hairdryer_07_SH_MKH800.wav"),
            os.path.join(PROJECT_ROOT, "tools", "data", "signals", "p011_emo_anger_sentences.wav"),
            os.path.join(PROJECT_ROOT, "tools", "data", "signals", "rubber_band_02_SH_MKH800.wav")
        ],
        # Fixed WPE Parameters
        'wpe_taps': 7,
        'wpe_delay': 2,
        'wpe_alpha': 0.9999,
        'wpe_stft_size': 512,
        'wpe_stft_shift': 128
    }



    # --- EXPERIMENT B: Hardware Robustness ---
    param_grid_B = {
        'mismatch_gain': [0, 1, 3, 6],     
        'mismatch_phase': [0],    
        'isir_db': [ 0], 
        # Fixed variables for Experiment B
        'rt60': [0.6], 
        'M': [8],                
        'N_interferences': [1],      
        'use_wpe': [True],
        'mismatch_pos': [0.0]
    }

    # Instantiate all 7 algorithm wrappers
    processors_dict = {
        "MVDR_Oracle": MVDR_Recursive_Processor(min_loading=1e-6),
        "MVDR_SPP": SPP_MVDR_Recursive_Processor(min_loading=1e-6),
        "MPDR": MPDR_Recursive_Processor(beta=1e-3, min_loading=1e-6),
    }


    
    print("\n" + "="*60)
    print("LAUNCHING EXPERIMENT B: HARDWARE ROBUSTNESS")
    print("="*60)
    
    # Run Experiment B and save in a different subfolder to prevent overwriting
    df_B = run_grid_search(
        grid_params=param_grid_B, 
        room_profiles=ROOM_PROFILES, 
        processors=processors_dict, 
        scene_base_config=base_config, 
        output_dir="tests/dataset_out/Exp_C"
    )

    print("\n[SUCCESS] Both experimental batches finished successfully.")