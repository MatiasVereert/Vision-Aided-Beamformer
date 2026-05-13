import os
import time
import itertools
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm


import tensorflow as tf
from dnn_denoise.dtln_lite import apply_dtln_post_tflite_realtime

from evaluation.polar_plots import precompute_quantized_spatial_response, subsample_weights
from beamforming.array.geometry import generate_log_array_coords, generate_source_and_interferences
from beamforming.array.microphone import Microphone
from propagation.simulate_acoustics_v1 import SimAcoustic
from dereverberation.nara_wrappers import process_wpe_online
from evaluation.metrics import evaluate_full_pipeline

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

def save_extreme_case_to_master(master_path, proc_name, metric_name, case_type, processor_obj, 
                                mic_signals, y_processed, target_anechoic, weights, 
                                exp_config, row_data, scene_base_config, current_room_dims,
                                audio_dtln_alone=None, y_post_dtln=None):
    """
    Saves or updates the master H5 file. Includes audio streams for DTLN
    so the Dashboard can play the neural-enhanced versions.
    """
    fs = scene_base_config['fs']
    
    with h5py.File(master_path, 'a') as f:
        case_path = f"{proc_name}/{metric_name}/{case_type}"
        
        if case_path in f:
            del f[case_path]
        
        grp = f.create_group(case_path)
        
        # 1. METADATA & GEOMETRY
        grp_meta = grp.create_group("metadata")
        grp_meta.attrs["fs"] = fs
        for k, v in exp_config.items(): grp_meta.attrs[k] = v
            
        grp_geom = grp.create_group("geometry")
        grp_geom.create_dataset("mic_coords", data=scene_base_config['mic_coords'])
        grp_geom.create_dataset("source_pos", data=scene_base_config['source_pos'])
        grp_geom.create_dataset("room_dims", data=current_room_dims)
        
        # 2. AUDIO
        grp_audio = grp.create_group("audio")
        grp_audio.create_dataset("mic_signals", data=mic_signals, compression="gzip")
        grp_audio.create_dataset("target_anechoic", data=target_anechoic, compression="gzip")
        grp_audio.create_dataset(f"processed_{proc_name}", data=y_processed, compression="gzip")
        
        # Save neural enhanced streams if available
        if audio_dtln_alone is not None:
            grp_audio.create_dataset("processed_dtln_alone", data=audio_dtln_alone, compression="gzip")
        if y_post_dtln is not None:
            grp_audio.create_dataset(f"processed_{proc_name}_dtln", data=y_post_dtln, compression="gzip")
        
        # 3. METRICS
        grp_res = grp.create_group("metrics")
        for k, v in row_data.items():
            if any(m in k for m in ["PESQ", "STOI", "SDR", "SIR", "SAR", "CD"]):
                grp_res.attrs[k] = v if not np.isnan(v) else "NaN"
                    
        # 4. WEIGHTS & SPATIAL
        nfft = getattr(processor_obj, 'nfft', 1024)
        hop = nfft - getattr(processor_obj, 'noverlap', 768)
        
        w_sub, freqs_sub = subsample_weights(weights, np.fft.rfftfreq(nfft, 1/fs), fs, hop, target_fps=24)
        
        array_center = np.mean(scene_base_config['mic_coords'], axis=0)
        radius = np.linalg.norm(scene_base_config['source_pos'][0] - array_center)
        
        q_gain, points, max_db = precompute_quantized_spatial_response(
            weights=w_sub, freqs=freqs_sub, mic_pos=scene_base_config['mic_coords'],
            source_radius=radius, N_azimuth=90, min_dB=-30.0
        )

        grp_spat = grp.create_group(f"spatial_{proc_name}")
        grp_spat.create_dataset("quantized_gain", data=q_gain, compression="gzip")
        grp_spat.create_dataset("points", data=points)
        grp_spat.create_dataset("freqs", data=freqs_sub)
        grp_spat.create_dataset("max_db_per_frame", data=max_db)
        grp_spat.attrs.update({"min_dB": -30.0, "target_fps": 24})


def run_grid_search(grid_params, room_profiles, processors, scene_base_config, output_dir="results/", interpreter_1=None, interpreter_2=None):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Generate combinations
    keys, values = zip(*grid_params.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # 2. STRATEGIC SORTING (Crucial for cascaded caching)
    experiments.sort(key=lambda x: (
        x['rt60'], x['M'], x['N_interferences'], x['mismatch_pos'], # Node 1
        x['isir_db'],                                               # Node 2
        x['mismatch_gain'], x['mismatch_phase'],                    # Node 3
        x['use_wpe']                                                # Node 4
    ))
    
    tqdm.write(f"[*] Total experiments to run: {len(experiments)} per processor.")

    # Track metrics, adding the neural ones
    tracked_metrics = ["Delta_tot_PESQ", "Delta_tot_STOI", "Delta_tot_SDR", "Delta_tot_SIR", "Delta_tot_SAR", "Delta_tot_CD"]

    leaderboard = {
        proc: {
            m: {"best_val": -np.inf if m != "Delta_tot_CD" else np.inf, 
                "worst_val": np.inf if m != "Delta_tot_CD" else -np.inf}
            for m in tracked_metrics
        } for proc in processors.keys()
    }
    all_metrics_results = []

    master_h5 = os.path.join(output_dir, "benchmark_catalog.h5")
    
    # --- CASCADING STATE VARIABLES ---
    current_rt60, current_M, current_N_int, current_mismatch_pos = None, None, None, None
    current_isir_db, current_gain_mismatch, current_phase_mismatch, current_use_wpe = None, None, None, None
    
    acoustic_scene, scene_data, mic_signals_degraded, mic_signals_ready = None, None, None, None 
    baseline_metrics = {}
    wpe_metrics = {} 
    
    # New state variables for Node 4.5 (DTLN alone)
    audio_dtln_alone = None
    dtln_alone_metrics = {}

    mic_simulator = Microphone(fs=scene_base_config['fs'])
    start_total_time = time.time()
    eval_start_s = min(5.0, scene_base_config['duration'] * 0.3)

    # Flag to determine if DTLN is available
    use_dtln = interpreter_1 is not None and interpreter_2 is not None

    # 4. Main Orchestrator Loop
    for i, exp in enumerate(tqdm(experiments, desc="Running Benchmark", unit="exp")):
        tqdm.write(f"\n--- Iteration {i+1}/{len(experiments)} | Config: {exp} ---")
        
        # --- STRICT CASCADING CACHE LOGIC ---
        recalc_physics = (exp['rt60'] != current_rt60 or exp['M'] != current_M or 
                          exp['N_interferences'] != current_N_int or exp['mismatch_pos'] != current_mismatch_pos)
        recalc_mixture = recalc_physics or (exp['isir_db'] != current_isir_db)
        recalc_hardware = recalc_mixture or (exp['mismatch_gain'] != current_gain_mismatch or 
                                             exp['mismatch_phase'] != current_phase_mismatch)
        recalc_wpe = recalc_hardware or (exp['use_wpe'] != current_use_wpe)
        
        # ---------------------------------------------------------
        # NODE 1: GEOMETRY AND PHYSICS (Heavy Computation)
        # ---------------------------------------------------------
        if recalc_physics:
            tqdm.write(" -> [NODE 1] Physics changed. Re-computing RIRs...")
            current_room_dims = room_profiles[exp['rt60']]
            room_center = current_room_dims / 2.0
            
            mic_coords = generate_log_array_coords(
                M=exp['M'], d_min=scene_base_config['d_min'], 
                d_max=scene_base_config['d_max'], room_dims=current_room_dims
            )
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
        # NODE 2: ACOUSTIC MIXTURE
        # ---------------------------------------------------------
        if recalc_mixture:
            tqdm.write(f" -> [NODE 2] Applying acoustic mixture (iSIR = {exp['isir_db']} dB)...")
            scene_data = acoustic_scene.mix_and_normalize(iSIR_dB=exp['isir_db'])
            current_isir_db = exp['isir_db']
            scene_base_config['VAD'] = scene_data["VAD"]
            
        target_anechoic = scene_data["target_early"] + scene_data["target_late"]
        unprocessed_mic_signals = scene_data["mic_signals"]

        # ---------------------------------------------------------
        # NODE 3: HARDWARE EMULATION & BASELINE
        # ---------------------------------------------------------
        if recalc_hardware:
            tqdm.write(f" -> [NODE 3] Emulating hardware (Gain: {exp['mismatch_gain']}dB, Phase: {exp['mismatch_phase']}deg)...")
            mic_simulator.set_custom_errors(
                std_gain_dB=exp['mismatch_gain'], std_phase_deg=exp['mismatch_phase'], snr_dB=scene_base_config['snr_db']
            )
            mic_signals_degraded = mic_simulator.emulate(unprocessed_mic_signals)
            current_gain_mismatch, current_phase_mismatch = exp['mismatch_gain'], exp['mismatch_phase']

            tqdm.write(" -> Evaluating Baseline Metrics...")
            baseline_metrics = evaluate_full_pipeline(
                ref_sig=target_anechoic[0],
                deg_sig=mic_signals_degraded[0], 
                fs=scene_base_config['fs'],
                interf_sig=scene_data["interference_early"][0],
                compute_pesq=True,
                compute_cd=True,
                eval_start_s=eval_start_s,
                inspection_name=f"Baseline_Exp_{i}"
            )

        # ---------------------------------------------------------
        # NODE 4: WPE PRE-PROCESSING 
        # ---------------------------------------------------------
        if recalc_wpe:
            if exp['use_wpe']:
                tqdm.write(" -> [NODE 4] Applying heavy WPE pre-processing...")
                mic_signals_ready = process_wpe_online(
                    u=mic_signals_degraded, taps=scene_base_config['wpe_taps'], delay=scene_base_config['wpe_delay'],
                    alpha=scene_base_config['wpe_alpha'], stft_size=scene_base_config['wpe_stft_size'], stft_shift=scene_base_config['wpe_stft_shift']
                )
                
                tqdm.write(" -> Evaluating WPE Metrics...")
                wpe_metrics = evaluate_full_pipeline(
                    ref_sig=target_anechoic[0], deg_sig=mic_signals_ready[0], fs=scene_base_config['fs'],
                    interf_sig=scene_data["interference_early"][0], compute_pesq=True, compute_cd=True,
                    eval_start_s=eval_start_s, inspection_name=f"WPE_Exp_{i}"
                )
            else:
                tqdm.write(" -> [NODE 4] Bypassing WPE pre-processing...")
                mic_signals_ready = mic_signals_degraded.copy()
                wpe_metrics = baseline_metrics.copy() 
            
            current_use_wpe = exp['use_wpe']

            # ---------------------------------------------------------
            # NODE 4.5: SINGLE-MIC DTLN
            # ---------------------------------------------------------
            if use_dtln:
                tqdm.write(" -> [NODE 4.5] Applying DTLN to reference microphone...")
                audio_dtln_alone = apply_dtln_post_tflite_realtime(
                    interpreter_1=interpreter_1, interpreter_2=interpreter_2, 
                    audio_mono=mic_signals_ready[0]
                )
                dtln_alone_metrics = evaluate_full_pipeline(
                    ref_sig=target_anechoic[0], deg_sig=audio_dtln_alone, fs=scene_base_config['fs'],
                    interf_sig=scene_data["interference_early"][0], compute_pesq=True, compute_cd=True,
                    eval_start_s=eval_start_s, inspection_name=f"DTLN_Alone_Exp_{i}"
                )
        else:
            tqdm.write(" -> [CACHE] Reusing previous WPE & Single-Mic DTLN state.")
        
        # ---------------------------------------------------------
        # NODE 5: SIGNAL PROCESSING AND EVALUATION
        # ---------------------------------------------------------
        for proc_name, processor in processors.items():
            tqdm.write(f"   -> Processing with: {proc_name}...")
            
            t0 = time.time()
            y_processed, weights = processor.process(mic_signals_ready, scene_base_config)
            proc_time = time.time() - t0
            
            proc_metrics = evaluate_full_pipeline(
                ref_sig=target_anechoic[0], deg_sig=y_processed, fs=scene_base_config['fs'],
                interf_sig=scene_data["interference_early"][0], compute_pesq=True, compute_cd=True,
                eval_start_s=eval_start_s, inspection_name=f"Proc_{proc_name}_Exp_{i}"
            )
            
            # ---------------------------------------------------------
            # NODE 6: POST-BEAMFORMING DTLN
            # ---------------------------------------------------------
            y_post_dtln = None
            dtln_post_metrics = {}
            if use_dtln:
                tqdm.write(f"   -> [NODE 6] Applying DTLN post {proc_name}...")
                y_post_dtln = apply_dtln_post_tflite_realtime(
                    interpreter_1=interpreter_1, interpreter_2=interpreter_2, 
                    audio_mono=y_processed
                )
                dtln_post_metrics = evaluate_full_pipeline(
                    ref_sig=target_anechoic[0], deg_sig=y_post_dtln, fs=scene_base_config['fs'],
                    interf_sig=scene_data["interference_early"][0], compute_pesq=True, compute_cd=True,
                    eval_start_s=eval_start_s, inspection_name=f"DTLN_Post_{proc_name}_Exp_{i}"
                )

            # Compile Dataset Row
            row_data = {
                "processor": proc_name,
                "rt60": exp['rt60'], "M": exp['M'], "N_interferences": exp['N_interferences'], 
                "mismatch_pos": exp['mismatch_pos'], "isir_db": exp['isir_db'],
                "mismatch_gain": exp['mismatch_gain'], "mismatch_phase": exp['mismatch_phase'],
                "use_wpe": exp['use_wpe'], "exec_time_s": proc_time,
            }
            
            # Append absolute metrics
            row_data.update({f"base_{k}": v for k, v in baseline_metrics.items()})
            row_data.update({f"wpe_{k}": v for k, v in wpe_metrics.items()})
            row_data.update({f"proc_{k}": v for k, v in proc_metrics.items()})
            
            # Calculate and append standard 3-stage Deltas
            delta_tot_metrics = {f"Delta_tot_{k}": (proc_metrics.get(k, np.nan) - baseline_metrics.get(k, np.nan)) for k in proc_metrics.keys()}
            row_data.update({f"Delta_wpe_{k}": (wpe_metrics.get(k, np.nan) - baseline_metrics.get(k, np.nan)) for k in proc_metrics.keys()})
            row_data.update({f"Delta_bf_{k}": (proc_metrics.get(k, np.nan) - wpe_metrics.get(k, np.nan)) for k in proc_metrics.keys()})
            row_data.update(delta_tot_metrics)
            
            # Calculate and append DTLN Deltas if active
            if use_dtln:
                row_data.update({f"dtln_alone_{k}": v for k, v in dtln_alone_metrics.items()})
                row_data.update({f"dtln_post_{k}": v for k, v in dtln_post_metrics.items()})
                
                row_data.update({f"Delta_dtln_alone_{k}": (dtln_alone_metrics.get(k, np.nan) - wpe_metrics.get(k, np.nan)) for k in proc_metrics.keys()})
                row_data.update({f"Delta_dtln_post_{k}": (dtln_post_metrics.get(k, np.nan) - proc_metrics.get(k, np.nan)) for k in proc_metrics.keys()})
                row_data.update({f"Delta_tot_pipeline_{k}": (dtln_post_metrics.get(k, np.nan) - baseline_metrics.get(k, np.nan)) for k in proc_metrics.keys()})
            
            all_metrics_results.append(row_data)

           # --- TOP-K / BOTTOM-K CHECKPOINTING ---
            for m_name in tracked_metrics:
                current_val = delta_tot_metrics.get(m_name, np.nan)
                if np.isnan(current_val): continue
                
                is_best = current_val > leaderboard[proc_name][m_name]["best_val"] if m_name != "Delta_tot_CD" else current_val < leaderboard[proc_name][m_name]["best_val"]
                is_worst = current_val < leaderboard[proc_name][m_name]["worst_val"] if m_name != "Delta_tot_CD" else current_val > leaderboard[proc_name][m_name]["worst_val"]
                
                if is_best:
                    leaderboard[proc_name][m_name]["best_val"] = current_val
                    save_extreme_case_to_master(master_h5, proc_name, m_name, "best_case", 
                                                processor, mic_signals_ready, y_processed, 
                                                target_anechoic, weights, exp, row_data, 
                                                scene_base_config, current_room_dims,
                                                audio_dtln_alone, y_post_dtln)
                
                if is_worst:
                    leaderboard[proc_name][m_name]["worst_val"] = current_val
                    save_extreme_case_to_master(master_h5, proc_name, m_name, "worst_case", 
                                                processor, mic_signals_ready, y_processed, 
                                                target_anechoic, weights, exp, row_data, 
                                                scene_base_config, current_room_dims,
                                                audio_dtln_alone, y_post_dtln)

    tqdm.write(f"\n=== BATCH COMPLETED IN {(time.time() - start_total_time)/60:.2f} MINUTES ===")
    df_results = pd.DataFrame(all_metrics_results)
    parquet_path = os.path.join(output_dir, "benchmark_metrics.parquet")
    df_results.to_parquet(parquet_path, engine="pyarrow")
    
    return df_results


if __name__ == "__main__":
    
    # Initialize DTLN TF-Lite Interpreters
    # Assign the correct actual paths to your local .tflite files
    try:
        interpreter_1 = tf.lite.Interpreter(model_path="data/dnn_models/model_1.tflite")
        interpreter_1.allocate_tensors()
        
        interpreter_2 = tf.lite.Interpreter(model_path="data/dnn_models/model_2.tflite")
        interpreter_2.allocate_tensors()
        print("[*] DTLN TF-Lite interpreters successfully allocated.")
    except Exception as e:
        print(f"[!] Warning: Could not initialize DTLN models automatically. Running without neural enhancement. Details: {e}")
        interpreter_1, interpreter_2 = None, None

    ROOM_PROFILES = {
        0.3: np.array([3.0, 4.0, 2.5])
    }

    base_config = {
        'fs': 16000,
        'duration': 15, 
        'd_min': 0.02,
        'd_max': 0.30,
        'radius_source': .6,    
        'radius_interf': 1.2,    
        'delta_ang_deg': 30.0,   
        'snr_db': 60.0,          
        'source_path': "data/audio/input/FA01_09.wav",
        'interf_paths': [
            "data/audio/input/hairdryer_07_SH_MKH800.wav"
        ],
        'wpe_taps': 7,
        'wpe_delay': 3,
        'wpe_alpha': 0.9999,
        'wpe_stft_size': 512,
        'wpe_stft_shift': 128
    }

    param_grid = {
        'rt60': [0.3], 
        'M': [6],                
        'N_interferences': [1],      
        'mismatch_pos': [0.0],       
        'isir_db': [0],         
        'mismatch_gain': [0],     
        'mismatch_phase': [0],    
        'use_wpe': [ False] 
    }

    processors_dict = {
        "DS": DS_Processor(),
        "MVDR": MVDR_Recursive_Processor(min_loading=1e-6),
        "SPP-MVDR": SPP_MVDR_Recursive_Processor(min_loading=1e-6)
        

    }

    df_final = run_grid_search(
        grid_params=param_grid, 
        room_profiles=ROOM_PROFILES, 
        processors=processors_dict, 
        scene_base_config=base_config, 
        output_dir="tests/dataset_out/quick_test",
        interpreter_1=interpreter_1, 
        interpreter_2=interpreter_2  
    )
    
    print("\n[Preview] Quick Test Results:")
    
    # Updated column names to reflect the DTLN integration
    cols_to_show = [
        "processor", "use_wpe", 
        "Delta_tot_PESQ", "Delta_dtln_alone_PESQ", "Delta_dtln_post_PESQ", "Delta_tot_pipeline_PESQ"
    ]
    
    cols_exist = [c for c in cols_to_show if c in df_final.columns]
    print(df_final[cols_exist].to_string(index=False))