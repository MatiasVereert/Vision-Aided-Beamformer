import os
import time
import itertools
import h5py
import pandas as pd
import numpy as np

from beamforming.array.geometry import generate_log_array_coords, generate_source_and_interferences
from beamforming.array.microphone import Microphone
from propagation.simulate_acoustics_v1 import SimAcoustic
from dereverberation.nara_wrappers import process_wpe_online


# Imported the real wrappers for the spatial filtering algorithms
from evaluation.bf_wrappers import (
    DS_Processor,
    MVDR_Recursive_Processor,
    KMVDR_Recursive_Processor,
    SDW_MWF_Processor,
    MPDR_Recursive_Processor
)


def save_extreme_case_h5(filepath, mic_signals, target_anechoic, weights, config, metrics, room_dims):
    """Saves heavy tensors to HDF5 only when a record is broken."""
    with h5py.File(filepath, 'w') as f:
        grp_meta = f.create_group("metadata")
        for key, value in config.items():
            grp_meta.attrs[key] = value
        grp_meta.attrs["room_dims"] = room_dims
        for key, value in metrics.items():
            grp_meta.attrs[f"metric_{key}"] = value

        grp_audio = f.create_group("audio")
        grp_audio.create_dataset("mic_signals", data=mic_signals, compression="gzip")
        grp_audio.create_dataset("target_anechoic", data=target_anechoic, compression="gzip")
        
        grp_weights = f.create_group("weights")
        grp_weights.create_dataset("beamformer_weights", data=weights, compression="gzip", compression_opts=4)


def run_grid_search(grid_params, room_profiles, processors, scene_base_config, output_dir="results/"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Generate combinations
    keys, values = zip(*grid_params.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # 2. STRATEGIC SORTING (Crucial for cascaded caching)
    # Order: Physics -> Mixture -> Hardware -> Pre-Processing
    experiments.sort(key=lambda x: (
        x['rt60'], x['M'], x['N_interferences'], x['mismatch_pos'], # Node 1 triggers
        x['isir_db'],                                               # Node 2 triggers
        x['mismatch_gain'], x['mismatch_phase'],                    # Node 3 triggers
        x['use_wpe']                                                # Node 4 triggers
    ))
    
    print(f"[*] Total experiments to run: {len(experiments)} per processor.")

    leaderboard = {
        proc_name: {"best_score": -float('inf'), "worst_score": float('inf')} 
        for proc_name in processors.keys()
    }
    all_metrics_results = []
    
    # --- CASCADING STATE VARIABLES ---
    current_rt60 = None
    current_M = None
    current_N_int = None
    current_mismatch_pos = None
    
    current_isir_db = None
    current_gain_mismatch = None
    current_phase_mismatch = None
    current_use_wpe = None
    
    # Persistent instances and data buffers
    acoustic_scene = None
    scene_data = None 
    mic_signals_degraded = None
    mic_signals_ready = None 
    
    mic_simulator = Microphone(fs=scene_base_config['fs'])

    start_total_time = time.time()

    # 4. Main Orchestrator Loop
    for i, exp in enumerate(experiments):
        print(f"\n--- Iteration {i+1}/{len(experiments)} | Config: {exp} ---")
        
        # --- STRICT CASCADING CACHE LOGIC ---
        recalc_physics = (exp['rt60'] != current_rt60 or 
                          exp['M'] != current_M or 
                          exp['N_interferences'] != current_N_int or 
                          exp['mismatch_pos'] != current_mismatch_pos)
        
        recalc_mixture = recalc_physics or (exp['isir_db'] != current_isir_db)
        
        recalc_hardware = recalc_mixture or (exp['mismatch_gain'] != current_gain_mismatch or 
                                             exp['mismatch_phase'] != current_phase_mismatch)
                                             
        recalc_wpe = recalc_hardware or (exp['use_wpe'] != current_use_wpe)
        
        
        # ---------------------------------------------------------
        # NODE 1: GEOMETRY AND PHYSICS (Heavy Computation)
        # ---------------------------------------------------------
        if recalc_physics:
            print(" -> [NODE 1] Physics changed. Re-computing RIRs and convolutions...")
            current_room_dims = room_profiles[exp['rt60']]
            room_center = current_room_dims / 2.0
            
            mic_coords = generate_log_array_coords(
                M=exp['M'], d_min=scene_base_config['d_min'], 
                d_max=scene_base_config['d_max'], room_dims=current_room_dims
            )
            
            # --- UPDATE CONFIG FOR WRAPPERS ---
            # Inject dynamic array geometry into the scene_base_config dictionary
            scene_base_config['mic_coords'] = mic_coords
            
            acoustic_scene = SimAcoustic(
                array_geometry=mic_coords, array_mismatch=exp['mismatch_pos'], 
                duration=scene_base_config['duration'], fs=scene_base_config['fs']
            )
            
            source_pos, interferences_pos = generate_source_and_interferences(
                N_interferences=exp['N_interferences'], radius_source=scene_base_config['radius_source'],
                radius_interf=scene_base_config['radius_interf'], delta_ang_deg=scene_base_config['delta_ang_deg'],
                array_center=room_center
            )
            
            # --- UPDATE CONFIG FOR WRAPPERS ---
            # Inject dynamic source position into the scene_base_config dictionary
            scene_base_config['source_pos'] = source_pos
            
            acoustic_scene.set_source(scene_base_config['source_path'], gain=1.0, position=source_pos.reshape(1,3))
            for idx in range(exp['N_interferences']):
                path_idx = idx % len(scene_base_config['interf_paths'])
                acoustic_scene.set_interference(
                    audio_path=scene_base_config['interf_paths'][path_idx], 
                    gain=1.0, position=interferences_pos[idx].reshape(1,3)
                )
            
            acoustic_scene.compute_rirs(room_dimensions=current_room_dims, desire_RT=exp['rt60'], ray_tracing=True)
            acoustic_scene.convolve_signals()
            
            current_rt60, current_M, current_N_int, current_mismatch_pos = exp['rt60'], exp['M'], exp['N_interferences'], exp['mismatch_pos']
            
        # ---------------------------------------------------------
        # NODE 2: ACOUSTIC MIXTURE (Lightweight Matrix Ops)
        # ---------------------------------------------------------
        if recalc_mixture:
            print(f" -> [NODE 2] Applying acoustic mixture (iSIR = {exp['isir_db']} dB)...")
            scene_data = acoustic_scene.mix_and_normalize(iSIR_dB=exp['isir_db'])
            current_isir_db = exp['isir_db']
            
            # --- UPDATE CONFIG FOR WRAPPERS ---
            # Inject the Oracle VAD mask into the scene_base_config dictionary
            scene_base_config['VAD'] = scene_data["VAD"]
            
        target_anechoic = scene_data["target_early"] + scene_data["target_late"]
        unprocessed_mic_signals = scene_data["mic_signals"]

        # ---------------------------------------------------------
        # NODE 3: HARDWARE EMULATION (Lightweight)
        # ---------------------------------------------------------
        if recalc_hardware:
            print(f" -> [NODE 3] Emulating hardware (Gain: {exp['mismatch_gain']}dB, Phase: {exp['mismatch_phase']}deg)...")
            mic_simulator.set_custom_errors(
                std_gain_dB=exp['mismatch_gain'], 
                std_phase_deg=exp['mismatch_phase'], 
                snr_dB=scene_base_config['snr_db']
            )
            mic_signals_degraded = mic_simulator.emulate(unprocessed_mic_signals)
            current_gain_mismatch, current_phase_mismatch = exp['mismatch_gain'], exp['mismatch_phase']

        # ---------------------------------------------------------
        # NODE 4: PRE-PROCESSING (WPE) (Heavy Computation)
        # ---------------------------------------------------------
        if recalc_wpe:
            if exp['use_wpe']:
                print(" -> [NODE 4] Applying heavy WPE pre-processing...")
                wpe_start = time.time()
                mic_signals_ready = process_wpe_online(
                    u=mic_signals_degraded,
                    taps=scene_base_config['wpe_taps'],
                    delay=scene_base_config['wpe_delay'],
                    alpha=scene_base_config['wpe_alpha'],
                    stft_size=scene_base_config['wpe_stft_size'],
                    stft_shift=scene_base_config['wpe_stft_shift']
                )
                print(f"    [WPE executed in {time.time() - wpe_start:.2f}s]")
            else:
                print(" -> [NODE 4] Bypassing WPE pre-processing...")
                mic_signals_ready = mic_signals_degraded.copy()
            
            current_use_wpe = exp['use_wpe']
        else:
            print(" -> [CACHE] Reusing previous WPE / Hardware state.")

        # ---------------------------------------------------------
        # NODE 5: SIGNAL PROCESSING AND EVALUATION
        # ---------------------------------------------------------
        base_pesq = 1.5 # MOCK Baseline

        for proc_name, processor in processors.items():
            print(f"   -> Processing with: {proc_name}")
            
            t0 = time.time()
            # Pass the READY signals (with or without WPE) to the algorithm wrapper
            y_processed, weights = processor.process(mic_signals_ready, scene_base_config)
            proc_time = time.time() - t0
            
            proc_pesq = 2.8 # MOCK Metric
            delta_pesq = proc_pesq - base_pesq
            
            row_data = {
                "processor": proc_name,
                "rt60": exp['rt60'],
                "M": exp['M'],
                "N_interferences": exp['N_interferences'], 
                "mismatch_pos": exp['mismatch_pos'],
                "isir_db": exp['isir_db'],
                "mismatch_gain": exp['mismatch_gain'],
                "mismatch_phase": exp['mismatch_phase'],
                "use_wpe": exp['use_wpe'],
                "exec_time_s": proc_time,
                "base_PESQ": base_pesq,
                "proc_PESQ": proc_pesq,
                "Delta_PESQ": delta_pesq
            }
            all_metrics_results.append(row_data)

            # --- TOP-K / BOTTOM-K CHECKPOINTING ---
            current_room_dims = room_profiles[exp['rt60']]
            if delta_pesq > leaderboard[proc_name]["best_score"]:
                leaderboard[proc_name]["best_score"] = delta_pesq
                filepath = os.path.join(output_dir, f"{proc_name}_BEST.h5")
                save_extreme_case_h5(filepath, mic_signals_ready, target_anechoic, weights, exp, row_data, current_room_dims)

            if delta_pesq < leaderboard[proc_name]["worst_score"]:
                leaderboard[proc_name]["worst_score"] = delta_pesq
                filepath = os.path.join(output_dir, f"{proc_name}_WORST.h5")
                save_extreme_case_h5(filepath, mic_signals_ready, target_anechoic, weights, exp, row_data, current_room_dims)

    # Finalization
    print(f"\n=== BATCH COMPLETED IN {(time.time() - start_total_time)/60:.2f} MINUTES ===")
    df_results = pd.DataFrame(all_metrics_results)
    parquet_path = os.path.join(output_dir, "benchmark_metrics.parquet")
    df_results.to_parquet(parquet_path, engine="pyarrow")
    
    return df_results


if __name__ == "__main__":
    
    ROOM_PROFILES = {
        0.0: np.array([3.0, 4.0, 2.5]),
        0.3: np.array([3.0, 4.0, 2.5]),
        0.6: np.array([5.0, 6.0, 2.8]),
        0.9: np.array([6.0, 7.0, 3.0]) 
    }

    base_config = {
        'fs': 16000,
        'duration': 30,
        'd_min': 0.04,
        'd_max': 0.20,
        'radius_source': 1.0,    
        'radius_interf': 1.5,    
        'delta_ang_deg': 30.0,   
        'snr_db': 60.0,          
        'source_path': "tools\data\signals\p002_emo_adoration_sentences.wav",
        'interf_paths': [
            "tools\data\signals\p017_rainbow_08_fast.wav",
            "tools\data\signals\hairdryer_07_SH_MKH800.wav",
            "tools\data\signals\p011_emo_anger_sentences.wav",
            "tools\data\signals\rubber_band_02_SH_MKH800.wav" 
        ],
        # Fixed WPE Parameters
        'wpe_taps': 7,
        'wpe_delay': 2,
        'wpe_alpha': 0.9999,
        'wpe_stft_size': 512,
        'wpe_stft_shift': 128
    }

    param_grid = {
        'rt60': list(ROOM_PROFILES.keys()), 
        'M': [4, 6, 8, 12],                
        'N_interferences': [1, 2, 3],      
        'mismatch_pos': [0.0, 1e-3],       
        'isir_db': [-5, 0, 5, 10],         
        'mismatch_gain': [0, 1, 3, 6],     
        'mismatch_phase': [0, 1, 3, 6],    
        'use_wpe': [False, True]           
    }

    # Instantiate all actual algorithm wrappers
    processors_dict = {
        "DS": DS_Processor(),
        "MPDR": MPDR_Recursive_Processor(beta=1e-3, min_loading=1e-6),
        "MVDR": MVDR_Recursive_Processor(min_loading=1e-6),
        "KMVDR": KMVDR_Recursive_Processor(target_P=2, beta=1e-3, min_loading=1e-6),
        "SDW_MWF": SDW_MWF_Processor(constrained=True)
    }

    df_final = run_grid_search(
        grid_params=param_grid, 
        room_profiles=ROOM_PROFILES, 
        processors=processors_dict, 
        scene_base_config=base_config, 
        output_dir="tests/dataset_out"
    )