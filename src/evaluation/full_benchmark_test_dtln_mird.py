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
from beamforming.array.microphone import Microphone
from propagation.simulate_acoustics_v1 import SimAcoustic
from propagation.mird_loader import MirdDatasetProvider, generate_mird_linear_array
from dereverberation.nara_wrappers import process_wpe_online
from evaluation.metrics import evaluate_full_pipeline

from evaluation.bf_wrappers import (
    DS_Processor,
    MVDR_Recursive_Processor,
    KMVDR_Recursive_Processor,
    SDW_MWF_Processor,
    MPDR_Recursive_Processor,
    RTF_MVDR_Recursive_Processor,
    SPP_MVDR_Recursive_Processor,
    SPP_mono_MVDR_Recursive_Processor,
    DTLN_MB_MVDR_Processor,
    DTLN_RTF_MVDR_Processor
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

def save_extreme_case_to_master(master_path, proc_name, metric_name, case_type, processor_obj,
                                mic_signals, y_processed, target_reference, weights,
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
        for k, v in exp_config.items():
            # Convert lists/tuples to strings for HDF5 attribute compatibility
            grp_meta.attrs[k] = str(v) if isinstance(v, (list, tuple)) else v

        grp_geom = grp.create_group("geometry")
        grp_geom.create_dataset("mic_coords", data=scene_base_config['mic_coords'])
        grp_geom.create_dataset("source_pos", data=scene_base_config['source_pos'])
        grp_geom.create_dataset("room_dims", data=current_room_dims)

        # 2. AUDIO
        grp_audio = grp.create_group("audio")
        grp_audio.create_dataset("mic_signals", data=mic_signals, compression="gzip")
        grp_audio.create_dataset("target_reference", data=target_reference, compression="gzip")
        grp_audio.create_dataset(f"processed_{proc_name}", data=y_processed, compression="gzip")

        if audio_dtln_alone is not None:
            grp_audio.create_dataset("processed_dtln_alone", data=audio_dtln_alone, compression="gzip")
        if y_post_dtln is not None:
            grp_audio.create_dataset(f"processed_{proc_name}_dtln", data=y_post_dtln, compression="gzip")

        # 3. METRICS
        grp_res = grp.create_group("metrics")
        for k, v in row_data.items():
            if any(m in k for m in ["PESQ", "STOI", "SDR", "SIR", "SAR", "SINR", "CD"]):
                grp_res.attrs[k] = v if not np.isnan(v) else "NaN"

        # Infer the correct N_FFT directly from the weights shape (axis 0 is frequency)
        n_bins = weights.shape[0]
        nfft = (n_bins - 1) * 2

        # Retrieve hop length or estimate it
        hop = nfft - getattr(processor_obj, 'noverlap', nfft // 4)

        # Safely compute frequency axis and subsample
        freqs_axis = np.fft.rfftfreq(nfft, 1/fs)
        w_sub, freqs_sub = subsample_weights(weights, freqs_axis, fs, hop, target_fps=24)

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


def evaluate_all_references(refs_dict, deg_sig, fs, interf_early, interf_late, target_late, eval_start_s, prefix_name):
    """
    Helper function to iterate over all configured reference signals and
    compute metrics, appending the reference name to the metric keys.
    """
    combined_metrics = {}
    for ref_name, ref_sig in refs_dict.items():
        metrics = evaluate_full_pipeline(
            ref_sig=ref_sig,
            deg_sig=deg_sig,
            fs=fs,
            interf_early=interf_early,
            interf_late=interf_late,
            target_late=target_late,
            compute_pesq=True,
            compute_cd=True,
            eval_start_s=eval_start_s,
            inspection_name=f"{prefix_name}_{ref_name}"
        )
        for k, v in metrics.items():
            combined_metrics[f"{k}_{ref_name}"] = v
    return combined_metrics


def run_mird_grid_search(grid_params, dataset_provider, processors, scene_base_config, output_dir="results/", interpreter_1=None, interpreter_2=None):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Generate combinations
    keys, values = zip(*grid_params.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # 2. STRATEGIC SORTING (Crucial for cascaded caching adapted to MIRD)
    experiments.sort(key=lambda x: (
        x['rt60'], x['target_dist'], x['target_angle'], str(x['interf_configs']), # Node 1: Physical Environment
        x['isir_db'],                                                             # Node 2: Mixture
        x['mismatch_gain'], x['mismatch_phase'],                                  # Node 3: Hardware
        x['use_wpe'],                                                             # Node 4: Pre-processing
        x.get('error_angle_deg', 0.0), x.get('error_distance_m', 0.0)             # Node 5: Misinformation
    ))

    # --- DERIVE DYNAMIC t_early FROM WPE CONFIG ---
    t_early_dynamic = (scene_base_config['wpe_stft_shift'] * scene_base_config['wpe_delay']) / scene_base_config['fs']
    tqdm.write(f"[*] Derived Dynamic t_early: {t_early_dynamic*1000:.1f} ms based on WPE parameters.")
    tqdm.write(f"[*] Total experiments to run: {len(experiments)} per processor.")

    # Base tracked metrics.
    tracked_metrics = ["Delta_tot_PESQ", "Delta_tot_STOI", "Delta_tot_SDR", "Delta_tot_SIR", "Delta_tot_SAR", "Delta_tot_SINR", "Delta_tot_CD"]

    leaderboard = {
        proc: {
            m: {"best_val": -np.inf if m != "Delta_tot_CD" else np.inf,
                "worst_val": np.inf if m != "Delta_tot_CD" else -np.inf}
            for m in tracked_metrics
        } for proc in processors.keys()
    }
    all_metrics_results = []

    master_h5 = os.path.join(output_dir, "mird_benchmark_catalog.h5")

    # --- CASCADING STATE VARIABLES ---
    current_rt60, current_target_dist, current_target_angle, current_interf_configs = None, None, None, None
    current_isir_db, current_gain_mismatch, current_phase_mismatch, current_use_wpe = None, None, None, None

    acoustic_scene, scene_data, mic_signals_degraded, mic_signals_ready = None, None, None, None
    refs_dict = {}
    baseline_metrics = {}
    wpe_metrics = {}

    audio_dtln_alone = None
    dtln_alone_metrics = {}

    mic_simulator = Microphone(fs=scene_base_config['fs'])
    start_total_time = time.time()
    eval_start_s = min(5.0, scene_base_config['duration'] * 0.3)

    use_dtln = interpreter_1 is not None and interpreter_2 is not None

    # Dummy dimensions for the Bar-Ilan room to keep HDF5 inspector happy
    mird_room_dims = np.array([6.0, 6.0, 2.41])
    array_center = np.array(scene_base_config['array_center'])

    # 4. Main Orchestrator Loop
    for i, exp in enumerate(tqdm(experiments, desc="Running MIRD Benchmark", unit="exp")):
        tqdm.write(f"\n--- Iteration {i+1}/{len(experiments)} | Config: {exp} ---")

        recalc_physics = (exp['rt60'] != current_rt60 or
                          exp['target_dist'] != current_target_dist or
                          exp['target_angle'] != current_target_angle or
                          str(exp['interf_configs']) != str(current_interf_configs))
        recalc_mixture = recalc_physics or (exp['isir_db'] != current_isir_db)
        recalc_hardware = recalc_mixture or (exp['mismatch_gain'] != current_gain_mismatch or
                                             exp['mismatch_phase'] != current_phase_mismatch)
        recalc_wpe = recalc_hardware or (exp['use_wpe'] != current_use_wpe)

        # ---------------------------------------------------------
        # NODE 1: GEOMETRY AND PHYSICS (MIRD DATASET IMPORT)
        # ---------------------------------------------------------
        if recalc_physics:
            tqdm.write(" -> [NODE 1] Physical setup changed. Extracting MIRD RIRs...")

            # Setup array coordinates globally (fixed 8-channel linear for MIRD)
            base_array = generate_mird_linear_array()
            mic_coords = base_array + array_center
            scene_base_config['mic_coords'] = mic_coords

            # Reinitialize continuous space core
            acoustic_scene = SimAcoustic(
                array_geometry=mic_coords,
                array_mismatch=0.0, # Physical real arrays don't use simulated mismatch
                duration=scene_base_config['duration'],
                fs=scene_base_config['fs']
            )

            # Map Target Position
            _ = dataset_provider.load_rir(exp['rt60'], scene_base_config['mird_spacing'], exp['target_dist'], exp['target_angle'])
            rel_pos_target = dataset_provider.export_position('cartesian')
            abs_pos_target = array_center + rel_pos_target.squeeze()
            scene_base_config['source_pos'] = abs_pos_target.reshape(1,3)

            acoustic_scene.set_source(scene_base_config['source_path'], gain=1.0, position=abs_pos_target.reshape(1,3))

            # Map Interference Positions
            for idx, interf_cfg in enumerate(exp['interf_configs']):

                i_ang = interf_cfg[0]
                i_dist = interf_cfg[1]

                # Check if specific audio index is provided; otherwise, fallback to positional modulo logic
                if len(interf_cfg) >= 3:
                    audio_idx = interf_cfg[2]
                else:
                    audio_idx = idx % len(scene_base_config['interf_paths'])

                # Fetch RIR data for the specified spatial coordinates
                _ = dataset_provider.load_rir(exp['rt60'], scene_base_config['mird_spacing'], i_dist, i_ang)
                rel_pos_interf = dataset_provider.export_position('cartesian')
                abs_pos_interf = array_center + rel_pos_interf.squeeze()

                # Inject interference
                acoustic_scene.set_interference(
                    audio_path=scene_base_config['interf_paths'][audio_idx],
                    gain=1.0, position=abs_pos_interf.reshape(1,3)
                )
            # Inject actual measurement matrices
            acoustic_scene.import_rirs(
                dataset_provider=dataset_provider,
                target_t60=exp['rt60'],
                array_center=array_center,
                spacing_cfg=scene_base_config['mird_spacing']
            )

            acoustic_scene.convolve_signals(t_early=t_early_dynamic)

            # Update cache trackers
            current_rt60 = exp['rt60']
            current_target_dist = exp['target_dist']
            current_target_angle = exp['target_angle']
            current_interf_configs = exp['interf_configs']

        # ---------------------------------------------------------
        # NODE 2: ACOUSTIC MIXTURE & GROUND TRUTHS PREPARATION
        # ---------------------------------------------------------
        if recalc_mixture:
            tqdm.write(f" -> [NODE 2] Applying acoustic mixture (iSIR = {exp['isir_db']} dB)...")
            scene_data = acoustic_scene.mix_and_normalize(iSIR_dB=exp['isir_db'])
            current_isir_db = exp['isir_db']
            scene_base_config['VAD'] = scene_data["VAD"]

            refs_dict.clear()
            if 'anechoic' in scene_base_config['eval_references']:
                refs_dict['anechoic'] = scene_data["target_anechoic"][0]
            if 'early' in scene_base_config['eval_references']:
                refs_dict['early'] = scene_data["target_early"][0]
            if 'reverberant' in scene_base_config['eval_references']:
                refs_dict['reverberant'] = scene_data["target_early"][0] + scene_data["target_late"][0]

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

            tqdm.write(" -> Evaluating Baseline Metrics against all references...")
            baseline_metrics = evaluate_all_references(
                refs_dict=refs_dict, deg_sig=mic_signals_degraded[0], fs=scene_base_config['fs'],
                interf_early=scene_data["interference_early"][0],
                interf_late=scene_data["interference_late"][0],
                target_late=scene_data["target_late"][0],
                eval_start_s=eval_start_s, prefix_name=f"Baseline_Exp_{i}"
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

                tqdm.write(" -> Evaluating WPE Metrics against all references...")
                wpe_metrics = evaluate_all_references(
                    refs_dict=refs_dict, deg_sig=mic_signals_ready[0], fs=scene_base_config['fs'],
                    interf_early=scene_data["interference_early"][0],
                    interf_late=scene_data["interference_late"][0],
                    target_late=scene_data["target_late"][0],
                    eval_start_s=eval_start_s, prefix_name=f"WPE_Exp_{i}"
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
                dtln_alone_metrics = evaluate_all_references(
                    refs_dict=refs_dict, deg_sig=audio_dtln_alone, fs=scene_base_config['fs'],
                    interf_early=scene_data["interference_early"][0],
                    interf_late=scene_data["interference_late"][0],
                    target_late=scene_data["target_late"][0],
                    eval_start_s=eval_start_s, prefix_name=f"DTLN_Alone_Exp_{i}"
                )
        else:
            tqdm.write(" -> [CACHE] Reusing previous WPE & Single-Mic DTLN state.")

        # ---------------------------------------------------------
        # NODE 5: SIGNAL PROCESSING AND EVALUATION
        # ---------------------------------------------------------
        err_ang = exp.get('error_angle_deg', 0.0)
        err_dist = exp.get('error_distance_m', 0.0)

        array_center_val = np.mean(scene_base_config['mic_coords'], axis=0)
        true_src_pos = np.array(scene_base_config['source_pos']).flatten()

        rel_vec = true_src_pos - array_center_val
        r_xy = np.hypot(rel_vec[0], rel_vec[1])
        theta = np.arctan2(rel_vec[1], rel_vec[0])

        r_prime = max(0.01, r_xy + err_dist)
        theta_prime = theta + np.deg2rad(err_ang)

        assumed_pos_flat = np.array([
            array_center_val[0] + r_prime * np.cos(theta_prime),
            array_center_val[1] + r_prime * np.sin(theta_prime),
            true_src_pos[2]
        ])
        assumed_source_pos = assumed_pos_flat.reshape(np.array(scene_base_config['source_pos']).shape)

        proc_config = scene_base_config.copy()
        proc_config['source_pos'] = assumed_source_pos

        for proc_name, processor in processors.items():
            tqdm.write(f"   -> Processing with: {proc_name} (ErrAng: {err_ang}deg, ErrDist: {err_dist}m)...")

            t0 = time.time()
            y_processed, weights = processor.process(mic_signals_ready, proc_config)
            proc_time = time.time() - t0

            proc_metrics = evaluate_all_references(
                refs_dict=refs_dict, deg_sig=y_processed, fs=scene_base_config['fs'],
                interf_early=scene_data["interference_early"][0],
                interf_late=scene_data["interference_late"][0],
                target_late=scene_data["target_late"][0],
                eval_start_s=eval_start_s, prefix_name=f"Proc_{proc_name}_Exp_{i}"
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
                dtln_post_metrics = evaluate_all_references(
                    refs_dict=refs_dict, deg_sig=y_post_dtln, fs=scene_base_config['fs'],
                    interf_early=scene_data["interference_early"][0],
                    interf_late=scene_data["interference_late"][0],
                    target_late=scene_data["target_late"][0],
                    eval_start_s=eval_start_s, prefix_name=f"DTLN_Post_{proc_name}_Exp_{i}"
                )

            # Compile Dataset Row
            row_data = {
                "processor": proc_name,
                "rt60": exp['rt60'],
                "M": 8, # Fixed MIRD Array size
                "N_interferences": len(exp['interf_configs']),
                "mismatch_pos": 0.0, # Physical arrays inherently contain positioning reality, no injected random mismatch
                "isir_db": exp['isir_db'],
                "mismatch_gain": exp['mismatch_gain'],
                "mismatch_phase": exp['mismatch_phase'],
                "use_wpe": exp['use_wpe'],
                "error_angle_deg": err_ang,
                "error_distance_m": err_dist,
                "t_early_s": t_early_dynamic,
                "exec_time_s": proc_time,
                "target_dist": exp['target_dist'],
                "target_angle": exp['target_angle']
            }

            # Append absolute metrics for all references
            row_data.update({f"base_{k}": v for k, v in baseline_metrics.items()})
            row_data.update({f"wpe_{k}": v for k, v in wpe_metrics.items()})
            row_data.update({f"proc_{k}": v for k, v in proc_metrics.items()})

            # Calculate and append standard Deltas for ALL references
            delta_tot_metrics = {}
            for ref_name in scene_base_config['eval_references']:
                core_metrics = [k.replace(f"_{ref_name}", "") for k in proc_metrics.keys() if k.endswith(f"_{ref_name}")]
                for metric in core_metrics:
                    key = f"{metric}_{ref_name}"
                    base_val = baseline_metrics.get(key, np.nan)
                    wpe_val = wpe_metrics.get(key, np.nan)
                    proc_val = proc_metrics.get(key, np.nan)

                    row_data[f"Delta_tot_{key}"] = proc_val - base_val
                    row_data[f"Delta_wpe_{key}"] = wpe_val - base_val
                    row_data[f"Delta_bf_{key}"] = proc_val - wpe_val
                    delta_tot_metrics[f"Delta_tot_{key}"] = proc_val - base_val

                    if use_dtln:
                        dtln_alone_val = dtln_alone_metrics.get(key, np.nan)
                        dtln_post_val = dtln_post_metrics.get(key, np.nan)

                        row_data[f"dtln_alone_{key}"] = dtln_alone_val
                        row_data[f"dtln_post_{key}"] = dtln_post_val

                        row_data[f"Delta_dtln_alone_{key}"] = dtln_alone_val - wpe_val
                        row_data[f"Delta_dtln_post_{key}"] = dtln_post_val - proc_val
                        row_data[f"Delta_tot_pipeline_{key}"] = dtln_post_val - base_val

            all_metrics_results.append(row_data)

            # --- TOP-K / BOTTOM-K CHECKPOINTING (Strictly anchored to 'early') ---
            for m_name in tracked_metrics:
                eval_key = f"{m_name}_early"
                current_val = delta_tot_metrics.get(eval_key, np.nan)

                if np.isnan(current_val): continue

                is_best = current_val > leaderboard[proc_name][m_name]["best_val"] if m_name != "Delta_tot_CD" else current_val < leaderboard[proc_name][m_name]["best_val"]
                is_worst = current_val < leaderboard[proc_name][m_name]["worst_val"] if m_name != "Delta_tot_CD" else current_val > leaderboard[proc_name][m_name]["worst_val"]

                if is_best or is_worst:
                    target_ref_audio = refs_dict.get('early')

                    if is_best:
                        leaderboard[proc_name][m_name]["best_val"] = current_val
                        save_extreme_case_to_master(master_h5, proc_name, m_name, "best_case",
                                                    processor, mic_signals_ready, y_processed,
                                                    target_ref_audio, weights, exp, row_data,
                                                    scene_base_config, mird_room_dims,
                                                    audio_dtln_alone, y_post_dtln)

                    if is_worst:
                        leaderboard[proc_name][m_name]["worst_val"] = current_val
                        save_extreme_case_to_master(master_h5, proc_name, m_name, "worst_case",
                                                    processor, mic_signals_ready, y_processed,
                                                    target_ref_audio, weights, exp, row_data,
                                                    scene_base_config, mird_room_dims,
                                                    audio_dtln_alone, y_post_dtln)

    tqdm.write(f"\n=== BATCH COMPLETED IN {(time.time() - start_total_time)/60:.2f} MINUTES ===")
    df_results = pd.DataFrame(all_metrics_results)

    # Cast list columns to string for parquet compatibility
    for col in df_results.columns:
        if df_results[col].apply(lambda x: isinstance(x, (list, tuple))).any():
            df_results[col] = df_results[col].astype(str)

    parquet_path = os.path.join(output_dir, "mird_benchmark_metrics.parquet")
    df_results.to_parquet(parquet_path, engine="pyarrow")

    return df_results


if __name__ == "__main__":

    # Initialize TF-Lite interpreters
    try:
        interpreter_1 = tf.lite.Interpreter(model_path="/home/matias/Documents/Tesis/Vision-Aided-Beamformer/src/dnn_denoise/models/model_quant_1.tflite")
        interpreter_1.allocate_tensors()

        interpreter_2 = tf.lite.Interpreter(model_path="/home/matias/Documents/Tesis/Vision-Aided-Beamformer/src/dnn_denoise/models/model_quant_2.tflite")
        interpreter_2.allocate_tensors()
        print("[*] DTLN TF-Lite interpreters successfully allocated.")
    except Exception as e:
        print(f"[!] Warning: Could not initialize DTLN models automatically. Running without neural enhancement. Details: {e}")
        interpreter_1, interpreter_2 = None, None

    # Bootstrapping the MIRD Provider
    root_mird_dir = os.path.abspath("/home/matias/Documents/Tesis/Vision-Aided-Beamformer/tools/data/rirs/mird")
    print("[*] Initializing automated MIRD dataset provider...")
    provider = MirdDatasetProvider(root_dir=root_mird_dir)

    base_config = {
        'fs': 16000,
        'duration': 15,
        't_early': 0.050,  # (50 ms)
        'array_center': [3.0, 3.0, 1.2], # Virtual translation anchor for SimAcoustic
        'mird_spacing': "3-3-3-8-3-3-3", # Target linear array spacing in the dataset

        'snr_db': 60.0,
        'source_path': r"/home/matias/Documents/Tesis/Vision-Aided-Beamformer/tools/data/signals/p002_emo_adoration_sentences.wav",
        'interf_paths': [
            r"/home/matias/Documents/Tesis/Vision-Aided-Beamformer/tools/data/signals/all minor ab oz.wav"
        ],

        'wpe_taps': 7,
        'wpe_delay': 3,
        'wpe_alpha': 0.9999,
        'wpe_stft_size': 512,
        'wpe_stft_shift': 128,

        'stft_window': 512,
        'stft_overlap': 384,

        'eval_references': ['anechoic', 'early', 'reverberant'],
        'dtln_model_path': r"/home/matias/Documents/Tesis/Vision-Aided-Beamformer/src/dnn_denoise/models/model_quant_1.tflite",

    }

    # Physical parameters adapted to MIRD boundaries
    param_grid = {
        'rt60': [ 0.610],
        'target_angle': [0],
        'target_dist': [1.0],

        # Interferences configuration: list of (angle, distance) tuples per experiment
        # Ex: [[(45, 1.0)]] means 1 interference at 45 degrees, 1.0m
        # Ex: [[(45, 1.0), (90, 2.0)]] means 2 interferences
        'interf_configs': [
            [(45, 1.0)],
        ],

        'isir_db': [0],
        'mismatch_gain': [3],
        'mismatch_phase': [5],
        'use_wpe': [False],
        'error_angle_deg': [0.0],
        'error_distance_m': [0.0]
    }

    processors_dict = {
        "DS": DS_Processor(),
        "DTLN-MVDR": DTLN_MB_MVDR_Processor(),
        "DTLN-RTF-MVDR": DTLN_RTF_MVDR_Processor(),
        "MVDR-Recursive": MVDR_Recursive_Processor()
    }

    df_final = run_mird_grid_search(
        grid_params=param_grid,
        dataset_provider=provider,
        processors=processors_dict,
        scene_base_config=base_config,
        output_dir="tests/dataset_out/mird_benchmark_test",
        interpreter_1=interpreter_1,
        interpreter_2=interpreter_2
    )

    print("\n[Preview] MIRD Benchmark Test Results:")

    cols_to_show = [
        "processor", "rt60", "target_angle", "interf_configs", "use_wpe",
        "error_angle_deg", "Delta_tot_PESQ_early", "Delta_tot_SIR_early"
    ]

    cols_exist = [c for c in cols_to_show if c in df_final.columns]
    print(df_final[cols_exist].to_string(index=False))