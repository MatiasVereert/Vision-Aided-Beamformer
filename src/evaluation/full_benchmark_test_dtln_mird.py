import os
import time
import itertools
import hashlib
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
from dereverberation.nara_wrappers import process_wpe_online, process_wpe_online_with_components
from dereverberation.nara_wrappers_fixed import process_wpe_online_fixed, FixedPointConfig
from evaluation.metrics import evaluate_full_pipeline

from evaluation.bf_wrappers import (
    DS,
    MVDR_Recursive,
    KMVDR_Recursive,
    SDW_MWF,
    MPDR_Recursive,
    NM_MVDR,
    DTLN_MB_MVDR_SOUDEN_BAN,
    DTLN_MB_MVDR_SOUDEN_SLOW,
    NM_MVDR_PF,
    ORACLE_MB_MVDR_SOUDEN,
    SOUDEN_ORACLE_SCM

)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))


def compute_scene_seed(exp, scene_base_config):
    """
    Deriva una semilla DETERMINISTA y ESTABLE para la emulacion de hardware a
    partir UNICAMENTE de la fisica de la escena: rt60, geometria target/interf,
    iSIR e identidad de los audios. Deliberadamente NO incluye mismatch_gain ni
    mismatch_phase, de modo que todas las celdas del barrido de mismatch de una
    misma escena comparten la misma semilla -> el patron base de mismatch y el
    ruido termico son identicos y las celdas difieren solo por la escala del error
    (heatmap suave/monotono en la Fig. 10).

    Se usa hashlib (no hash() de Python, que no es estable entre procesos por
    PYTHONHASHSEED) para que la misma escena de siempre la misma semilla entre
    corridas -> reproducibilidad.
    """
    physics_key = "|".join(str(x) for x in [
        exp['rt60'],
        exp['target_dist'],
        exp['target_angle'],
        exp['interf_configs'],
        exp['isir_db'],
        exp.get('source_path', scene_base_config.get('source_path', '')),
        scene_base_config.get('interf_paths', ''),
    ])
    digest = hashlib.sha256(physics_key.encode('utf-8')).digest()
    # 64 bits bastan y son comodos para SeedSequence de numpy.
    return int.from_bytes(digest[:8], 'big')

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
        # Absolute interference positions (N_interferences, 3), if any were placed.
        interferences_pos = scene_base_config.get('interferences_pos')
        if interferences_pos is not None and len(interferences_pos) > 0:
            grp_geom.create_dataset("interferences_pos", data=np.asarray(interferences_pos))

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

    # --- BACKWARD-COMPAT: promote wpe_taps/wpe_delay to grid axes ---
    # Callers that do not sweep these (e.g. notebooks A/B/C) still pass them as
    # scalars in scene_base_config. Inject them as single-value lists so the
    # itertools.product below keeps working and every experiment dict carries
    # 'wpe_taps'/'wpe_delay' regardless of the caller.
    grid_params = dict(grid_params)  # do not mutate the caller's dict
    if 'wpe_taps' not in grid_params:
        grid_params['wpe_taps'] = [scene_base_config['wpe_taps']]
    if 'wpe_delay' not in grid_params:
        grid_params['wpe_delay'] = [scene_base_config['wpe_delay']]
    # source_path is now a grid axis too (target-speaker variety). Callers that do
    # not sweep it (E0, notebooks A/B/C) keep passing the scalar in scene_base_config;
    # promote it to a single-value list so every experiment dict carries it.
    if 'source_path' not in grid_params:
        grid_params['source_path'] = [scene_base_config['source_path']]

    # 1. Generate combinations
    keys, values = zip(*grid_params.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # 2. STRATEGIC SORTING (Crucial for cascaded caching adapted to MIRD)
    experiments.sort(key=lambda x: (
        x['rt60'], x['target_dist'], x['target_angle'], str(x['interf_configs']), x['source_path'], # Node 1: Physical Environment
        x['isir_db'],                                                             # Node 2: Mixture
        x['mismatch_gain'], x['mismatch_phase'],                                  # Node 3: Hardware
        x['use_wpe'], x['wpe_taps'], x['wpe_delay'],                              # Node 4: Pre-processing (WPE; wpe_delay no longer feeds t_early)
        x.get('error_angle_deg', 0.0), x.get('error_distance_m', 0.0)             # Node 5: Misinformation
    ))

    # --- FIXED t_early (constant across the whole grid) ---
    # t_early defines the "early" reference window of the metrics. It is a FIXED
    # acoustic quantity read from scene_base_config['t_early'] and is deliberately
    # DECOUPLED from wpe_delay: when sweeping wpe_delay (e.g. E0) every delay must
    # be scored against the SAME early/late reference, otherwise the Delta metrics
    # are biased by a moving target and delays are not comparable (correction A1).
    t_early_dynamic = scene_base_config['t_early']
    tqdm.write(f"[*] t_early FIXED at {t_early_dynamic*1000:.1f} ms (decoupled from wpe_delay).")
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
    current_wpe_taps, current_wpe_delay = None, None
    current_source_path = None

    acoustic_scene, scene_data, mic_signals_degraded, mic_signals_ready = None, None, None, None
    # Componentes target-solo / (interf+ruido)-solo POST-HW (dominio de Node 3), a
    # partir de los cuales se derivan las refs del oracle POST-WPE en Node 4.
    hw_oracle_target, hw_oracle_noise = None, None
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

        # t_early is FIXED (read once above); it drives the early/late split in
        # convolve_signals (Node 1) and the metric reference window. It does NOT
        # depend on wpe_delay, so sweeping the delay keeps the SAME reference.
        t_early_dynamic = scene_base_config['t_early']

        # wpe_delay no longer feeds t_early, so a delay-only change does NOT touch
        # the physics/mixture; it is handled downstream by recalc_wpe (Node 4).
        recalc_physics = (exp['rt60'] != current_rt60 or
                          exp['target_dist'] != current_target_dist or
                          exp['target_angle'] != current_target_angle or
                          str(exp['interf_configs']) != str(current_interf_configs) or
                          exp['source_path'] != current_source_path)
        recalc_mixture = recalc_physics or (exp['isir_db'] != current_isir_db)
        recalc_hardware = recalc_mixture or (exp['mismatch_gain'] != current_gain_mismatch or
                                             exp['mismatch_phase'] != current_phase_mismatch)
        recalc_wpe = recalc_hardware or (exp['use_wpe'] != current_use_wpe or
                                         exp['wpe_taps'] != current_wpe_taps or
                                         exp['wpe_delay'] != current_wpe_delay)

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

            # Target speaker for this experiment (now a grid axis). Persist it into
            # scene_base_config so compute_scene_seed() derives a distinct, stable
            # seed per speaker.
            scene_base_config['source_path'] = exp['source_path']

            # Map Target Position
            _ = dataset_provider.load_rir(exp['rt60'], scene_base_config['mird_spacing'], exp['target_dist'], exp['target_angle'])
            rel_pos_target = dataset_provider.export_position('cartesian')
            abs_pos_target = array_center + rel_pos_target.squeeze()
            scene_base_config['source_pos'] = abs_pos_target.reshape(1,3)

            acoustic_scene.set_source(exp['source_path'], gain=1.0, position=abs_pos_target.reshape(1,3))

            # Map Interference Positions
            interferences_pos = []
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
                interferences_pos.append(abs_pos_interf)

                # Inject interference
                acoustic_scene.set_interference(
                    audio_path=scene_base_config['interf_paths'][audio_idx],
                    gain=1.0, position=abs_pos_interf.reshape(1,3)
                )
            # Persist absolute interference positions (N_interferences, 3) so they can be
            # saved to H5 and rendered by the dashboard.
            scene_base_config['interferences_pos'] = np.asarray(interferences_pos) if interferences_pos else np.zeros((0, 3))
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
            current_source_path = exp['source_path']

        # ---------------------------------------------------------
        # NODE 2: ACOUSTIC MIXTURE & GROUND TRUTHS PREPARATION
        # ---------------------------------------------------------
        if recalc_mixture:
            tqdm.write(f" -> [NODE 2] Applying acoustic mixture (iSIR = {exp['isir_db']} dB)...")
            scene_data = acoustic_scene.mix_and_normalize(iSIR_dB=exp['isir_db'])
            current_isir_db = exp['isir_db']
            scene_base_config['VAD'] = scene_data["VAD"]

            # NOTA: las referencias del ORACLE (oracle_target/oracle_noise) YA NO se
            # fijan aca. Deben vivir en el MISMO dominio que la senal que se filtra
            # (mic_signals_ready = post HW mismatch + WPE), no en el dominio limpio
            # pre-front-end. Se construyen en Node 3 (HW) + Node 4 (WPE). Ver el
            # bloque "ORACLE REFERENCES" mas abajo.

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
            # Semilla derivada SOLO de la fisica de la escena (no del mismatch): fija
            # el patron base de mismatch/ruido para que el barrido de gain/phase varie
            # solo la escala del error. Al no depender del mismatch, set_seed es no-op
            # cuando solo cambian gain/phase -> se preservan los patrones cacheados.
            mic_simulator.set_seed(compute_scene_seed(exp, scene_base_config))
            mic_simulator.set_custom_errors(
                std_gain_dB=exp['mismatch_gain'], std_phase_deg=exp['mismatch_phase'], snr_dB=scene_base_config['snr_db']
            )
            mic_signals_degraded = mic_simulator.emulate(unprocessed_mic_signals)
            current_gain_mismatch, current_phase_mismatch = exp['mismatch_gain'], exp['mismatch_phase']

            # --- COMPONENTES POST-HW PARA EL ORACLE (dominio de la observacion) ---
            # emulate() = mismatch(mezcla) + ruido_termico. _apply_mismatch es LINEAL
            # (ganancia por-mic + fase via FFT) y su patron ya quedo fijado por la
            # semilla dentro de emulate(), asi que reaplicarlo al target-solo es
            # consistente con lo aplicado a la mezcla:
            #   hw_target = mismatch(target_limpio)
            #   hw_noise  = mic_signals_degraded - hw_target
            #              = mismatch(interf_limpio) + ruido_termico
            # => hw_target + hw_noise == mic_signals_degraded (exacto). El ruido
            # termico, aditivo, cae ENTERO en el ruido (donde corresponde). Con
            # mismatch=0 -> mismatch() es identidad -> hw_target = target_limpio.
            target_clean = scene_data["target_early"] + scene_data["target_late"]
            hw_oracle_target = mic_simulator._apply_mismatch(target_clean)
            hw_oracle_noise = mic_signals_degraded - hw_oracle_target

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
            wpe_kw = dict(
                taps=exp['wpe_taps'], delay=exp['wpe_delay'],
                alpha=scene_base_config['wpe_alpha'],
                stft_size=scene_base_config['wpe_stft_size'],
                stft_shift=scene_base_config['wpe_stft_shift'],
            )
            if exp['use_wpe']:
                fixed_bits = scene_base_config.get('wpe_fixed_bits', None)
                if fixed_bits is None:
                    tqdm.write(" -> [NODE 4] Applying WPE pre-processing (float)...")
                    # Un solo pase de WPE que devuelve la mezcla dereverberada Y las
                    # componentes target/ruido filtradas con el MISMO G (estimado de la
                    # mezcla). Al ser lineal dado G: WPE(target)+WPE(ruido)==WPE(mezcla).
                    # z_u es IDENTICO (bit a bit) a process_wpe_online(mezcla): el
                    # filtrado de componentes no toca el estado del filtro de la mezcla.
                    mic_signals_ready, (oracle_target, oracle_noise) = process_wpe_online_with_components(
                        u=mic_signals_degraded,
                        components=[hw_oracle_target, hw_oracle_noise], **wpe_kw
                    )
                else:
                    tqdm.write(f" -> [NODE 4] Applying WPE pre-processing (FIXED-POINT {fixed_bits}-bit, FPGA emulation)...")
                    fp_cfg = FixedPointConfig.wordlength(
                        fixed_bits, rounding=scene_base_config.get('wpe_fixed_round', 'nearest')
                    )
                    mic_signals_ready, fp_stats = process_wpe_online_fixed(
                        u=mic_signals_degraded, taps=exp['wpe_taps'], delay=exp['wpe_delay'],
                        alpha=scene_base_config['wpe_alpha'], stft_size=scene_base_config['wpe_stft_size'],
                        stft_shift=scene_base_config['wpe_stft_shift'], fp_cfg=fp_cfg, return_stats=True
                    )
                    tqdm.write(f"    [FP-STATS] overflow={fp_stats.overflow} max|P|={fp_stats.max_absP:.2e} "
                               f"max|G|={fp_stats.max_absG:.2e} diverged={fp_stats.diverged}")
                    # OPCION A (aproximacion documentada): el front-end fixed-point es NO
                    # lineal, asi que no admite descomposicion exacta. Las refs del oracle
                    # se computan con la descomposicion FLOAT (exacta) sobre la MISMA
                    # mezcla degradada. A alta word-length fixed~=float, asi que las refs
                    # son una aproximacion cercana del target/ruido dentro de la
                    # observacion fixed; el desajuste esta acotado por el error de
                    # cuantizacion (crece a pocos bits -> interpretar el oracle con
                    # cuidado en ese regimen).
                    tqdm.write("    [NODE 4] Oracle refs via FLOAT decomposition (Opcion A): "
                               "aprox. del dominio fixed-point acotada por la cuantizacion.")
                    _, (oracle_target, oracle_noise) = process_wpe_online_with_components(
                        u=mic_signals_degraded,
                        components=[hw_oracle_target, hw_oracle_noise], **wpe_kw
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
                # Sin WPE: las refs del oracle son las componentes POST-HW (mismo
                # dominio que mic_signals_ready = mezcla degradada sin WPE).
                oracle_target, oracle_noise = hw_oracle_target, hw_oracle_noise

            # --- ORACLE REFERENCES (dominio consistente con mic_signals_ready) ---
            # Cualquier procesador oracle (SOUDEN_ORACLE_SCM, ORACLE_MB_MVDR_SOUDEN)
            # consume estas dos senales; ahora viven en el MISMO dominio (HW+WPE) que
            # la senal que filtran -> las SCM / mascaras ideales quedan consistentes.
            scene_base_config['oracle_target'] = oracle_target
            scene_base_config['oracle_noise'] = oracle_noise

            current_use_wpe = exp['use_wpe']
            current_wpe_taps = exp['wpe_taps']
            current_wpe_delay = exp['wpe_delay']  # tracked here now that t_early is fixed

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
                "wpe_taps": exp['wpe_taps'],
                "wpe_delay": exp['wpe_delay'],
                "wpe_bits": scene_base_config.get('wpe_fixed_bits', None),
                "error_angle_deg": err_ang,
                "error_distance_m": err_dist,
                "t_early_s": t_early_dynamic,
                "exec_time_s": proc_time,
                "target_dist": exp['target_dist'],
                "target_angle": exp['target_angle'],
                "source": os.path.basename(exp['source_path'])
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

    df_results.to_csv(os.path.join(output_dir, "mird_benchmark_metrics.csv"), index=False)

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
            r"/home/matias/Documents/Tesis/Vision-Aided-Beamformer/tools/data/signals/techno_gated commune.wav"
        ],

        'wpe_taps': 7,
        'wpe_delay': 3,
        'wpe_alpha': 0.9999,
        'wpe_stft_size': 512,
        'wpe_stft_shift': 128,

        # --- FPGA fixed-point emulation of WPE ---
        # None => float (original). Set to 16/18/24/32 to run the causal
        # Online-WPE RLS in fixed-point (URAM-storage precision emulation).
        # Change this value and re-run to sweep word length against MIRD metrics.
        'wpe_fixed_bits': None,   # <-- set to 24 (safe) / 20 / 18 to emulate fixed-point FPGA WPE
        'wpe_fixed_round': 'nearest',   # 'nearest' | 'floor'

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

        'isir_db': [3],
        'mismatch_gain': [0],
        'mismatch_phase': [0],
        'use_wpe': [True],

        # WPE hyper-parameter sweep (NEW grid axes).
        # If omitted, run_mird_grid_search falls back to the scalar values in
        # base_config (wpe_taps / wpe_delay), preserving old A/B/C behaviour.
        'wpe_taps': [3, 7, 10],
        'wpe_delay': [1, 2, 3],

        'error_angle_deg': [0.0],
        'error_distance_m': [0.0]
    }

    processors_dict = {
        "NM-MVDR_alpha_1_ref" : NM_MVDR(min_loading =1e-6, alpha = 1),
        "NM-MVDR_alpha_0.99_ref" : NM_MVDR(min_loading =1e-6, alpha = 0.99),
        # Cota superior agnostica al modelo: misma cadena Souden pero con mascara ideal.
        # SOFT (sharpen_exp=1.0, IRM continua) y HARD-EDGE (sharpen_exp=4.0, == **4 del DTLN).
        "Oracle-MVDR_alpha_1" : ORACLE_MB_MVDR_SOUDEN(min_loading =1e-6, alpha = 1, sharpen_exp=1.0),
        "Oracle-MVDR_alpha_0.99" : ORACLE_MB_MVDR_SOUDEN(min_loading =1e-6, alpha = 0.99, sharpen_exp=1.0),
        "Oracle-MVDR_hard_alpha_1" : ORACLE_MB_MVDR_SOUDEN(min_loading =1e-6, alpha = 1, sharpen_exp=4.0),
        "Oracle-MVDR_hard_alpha_0.99" : ORACLE_MB_MVDR_SOUDEN(min_loading =1e-6, alpha = 0.99, sharpen_exp=4.0),
        "Slow"  : DTLN_MB_MVDR_SOUDEN_SLOW(),
        "NM-MVDR_PF" : NM_MVDR_PF(smooth=0.33, min_loading=1e-6),
    }


    df_final = run_mird_grid_search(
        grid_params=param_grid,
        dataset_provider=provider,
        processors=processors_dict,
        scene_base_config=base_config,
        output_dir="tests/dataset_out/with_ds",
        interpreter_1=interpreter_1,
        interpreter_2=interpreter_2
    )

    print("\n[Preview] MIRD Benchmark Test Results:")

    cols_to_show = [
        "processor", "rt60", "target_angle", "interf_configs", "use_wpe",
        "wpe_taps", "wpe_delay",
        "error_angle_deg", "Delta_tot_PESQ_early", "Delta_tot_SIR_early"
    ]

    cols_exist = [c for c in cols_to_show if c in df_final.columns]
    print(df_final[cols_exist].to_string(index=False))