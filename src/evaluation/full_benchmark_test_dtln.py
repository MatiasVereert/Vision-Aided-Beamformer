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
from beamforming.array.geometry import (generate_log_array_coords, generate_source_and_interferences,
                                         generate_array_coords, place_spherical, max_distance_in_room)
from beamforming.array.microphone import Microphone
from propagation.simulate_acoustics_v1 import SimAcoustic
from propagation.mird_loader import generate_mird_linear_array, generate_mird_linear_array_from_spacing
from utils.geometry import spherical_to_cartesian
from dereverberation.nara_wrappers import process_wpe_online_with_components
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
    partir UNICAMENTE de la fisica de la escena SIMULADA: rt60, geometria del
    array (M), numero/posicion de interferencias, mismatch posicional, iSIR e
    identidad de los audios. Deliberadamente NO incluye mismatch_gain ni
    mismatch_phase, de modo que todas las celdas del barrido de mismatch de una
    misma escena comparten la misma semilla -> el patron base de mismatch y el
    ruido termico son identicos y las celdas difieren solo por la escala del error.

    Se usa hashlib (no hash() de Python, que no es estable entre procesos por
    PYTHONHASHSEED) para que la misma escena de siempre la misma semilla entre
    corridas -> reproducibilidad.
    """
    physics_key = "|".join(str(x) for x in [
        scene_base_config.get('geometry_mode', 'log'),
        exp.get('rt60'),
        # Ejes fisicos del modo 'log' (array log-espaciado)
        exp.get('M'),
        exp.get('N_interferences'),
        exp.get('mismatch_pos'),
        # Ejes fisicos del modo 'topology' (topologia 2D inscripta en circulo).
        # Deben entrar en la clave para que (a) la geometria aleatoria sea estable
        # y distinta por topologia/diametro, y (b) dos topologias de una misma
        # escena no compartan el patron base de mismatch de HW.
        exp.get('topology'),
        exp.get('diameter'),
        exp.get('interf_scenario'),
        exp.get('source_dist'),
        # Ejes fisicos del modo 'mird_linear' (replica geometrica de MIRD)
        exp.get('target_angle'),
        exp.get('target_dist'),
        exp.get('interf_configs'),
        exp.get('isir_db'),
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


def run_grid_search(grid_params, room_profiles, processors, scene_base_config, output_dir="results/", interpreter_1=None, interpreter_2=None, save_catalog=True, apply_dtln_post=True):
    os.makedirs(output_dir, exist_ok=True)

    # geometry_mode selecciona como se arma la escena en Node 1:
    #   'log'         = array log-espaciado (default historico) con ejes de grilla
    #                   M / N_interferences / mismatch_pos, posiciones via
    #                   generate_source_and_interferences (broadside + delta_ang).
    #   'mird_linear' = REPLICA GEOMETRICA de MIRD: array lineal 4-4-4-8-4-4-4 en
    #                   array_center, fuente/interferencias colocadas por
    #                   target_angle/target_dist/interf_configs (misma convencion
    #                   que el modulo full_benchmark_test_dtln_mird), pero con RIRs
    #                   SIMULADAS por ISM (compute_rirs) en una sala shoebox. Esto
    #                   permite validar ISM vs MIRD medido bajo geometria identica.
    geom_mode = scene_base_config.get('geometry_mode', 'log')

    # source_path como eje de grilla opcional (alineado con el modulo MIRD). Si el
    # caller no lo barre, se promueve al escalar de base_config para que cada
    # experimento lo lleve y compute_scene_seed derive una semilla estable.
    grid_params = dict(grid_params)  # no mutar el dict del caller
    if 'source_path' not in grid_params:
        grid_params['source_path'] = [scene_base_config['source_path']]

    # 1. Generate combinations
    keys, values = zip(*grid_params.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # 2. STRATEGIC SORTING (Crucial for cascaded caching)
    if geom_mode == 'mird_linear':
        experiments.sort(key=lambda x: (
            x['rt60'], x.get('target_dist', 0.0), x.get('target_angle', 0.0),
            str(x.get('interf_configs')), str(x.get('source_path', '')),      # Node 1
            x['isir_db'],                                                     # Node 2
            x['mismatch_gain'], x['mismatch_phase'],                          # Node 3
            x['use_wpe'],                                                     # Node 4
            x.get('error_angle_deg', 0.0), x.get('error_distance_m', 0.0)     # Node 5
        ))
    elif geom_mode == 'topology':
        experiments.sort(key=lambda x: (
            x['rt60'], str(x.get('topology', '')), x['M'], x.get('diameter', 0.0),
            str(x.get('interf_scenario', '')), x.get('source_dist', 0.0), x['mismatch_pos'],  # Node 1
            x['isir_db'],                                                     # Node 2
            x['mismatch_gain'], x['mismatch_phase'],                          # Node 3
            x['use_wpe'],                                                     # Node 4
            x.get('error_angle_deg', 0.0), x.get('error_distance_m', 0.0)     # Node 5
        ))
    else:
        experiments.sort(key=lambda x: (
            x['rt60'], x['M'], x['N_interferences'], x['mismatch_pos'], # Node 1
            x['isir_db'],                                               # Node 2
            x['mismatch_gain'], x['mismatch_phase'],                    # Node 3
            x['use_wpe'],                                               # Node 4
            x.get('error_angle_deg', 0.0), x.get('error_distance_m', 0.0) # Node 5
        ))

    # --- FIXED t_early (constant across the whole grid) ---
    # t_early defines the "early" reference window of the metrics. It is a FIXED
    # acoustic quantity read from scene_base_config['t_early'] and is deliberately
    # DECOUPLED from wpe_delay: every experiment must be scored against the SAME
    # early/late reference, otherwise the Delta metrics are biased by a moving
    # target and cells are not comparable (correction A1, alineado con el modulo MIRD).
    t_early_dynamic = scene_base_config['t_early']
    tqdm.write(f"[*] t_early FIXED at {t_early_dynamic*1000:.1f} ms (decoupled from wpe_delay).")
    tqdm.write(f"[*] Total experiments to run: {len(experiments)} per processor.")

    # Base tracked metrics. Leaderboard will specifically track the '_early' variations.
    tracked_metrics = ["Delta_tot_PESQ", "Delta_tot_STOI", "Delta_tot_SDR", "Delta_tot_SIR", "Delta_tot_SAR", "Delta_tot_SINR", "Delta_tot_CD"]

    leaderboard = {
        proc: {
            m: {"best_val": -np.inf if m != "Delta_tot_CD" else np.inf,
                "worst_val": np.inf if m != "Delta_tot_CD" else -np.inf}
            for m in tracked_metrics
        } for proc in processors.keys()
    }
    all_metrics_results = []

    master_h5 = os.path.join(output_dir, "ism_benchmark_catalog.h5")

    # --- CASCADING STATE VARIABLES ---
    current_rt60, current_M, current_N_int, current_mismatch_pos = None, None, None, None
    current_isir_db, current_gain_mismatch, current_phase_mismatch, current_use_wpe = None, None, None, None
    # Estado extra para geometry_mode='mird_linear' (ejes fisicos tipo MIRD).
    current_target_angle, current_target_dist, current_interf_configs, current_source_path = None, None, None, None
    # Estado extra para geometry_mode='topology' (topologia 2D inscripta en circulo).
    current_topology, current_diameter = None, None
    current_interf_scenario, current_source_dist = None, None

    acoustic_scene, scene_data, mic_signals_degraded, mic_signals_ready = None, None, None, None
    # Componentes target-solo / (interf+ruido)-solo POST-HW (dominio de Node 3), a
    # partir de los cuales se derivan las refs del oracle en Node 4.
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

    # 4. Main Orchestrator Loop
    for i, exp in enumerate(tqdm(experiments, desc="Running Benchmark", unit="exp")):
        tqdm.write(f"\n--- Iteration {i+1}/{len(experiments)} | Config: {exp} ---")

        if geom_mode == 'mird_linear':
            recalc_physics = (exp['rt60'] != current_rt60 or
                              exp.get('target_angle') != current_target_angle or
                              exp.get('target_dist') != current_target_dist or
                              str(exp.get('interf_configs')) != str(current_interf_configs) or
                              exp.get('source_path') != current_source_path)
        elif geom_mode == 'topology':
            recalc_physics = (exp['rt60'] != current_rt60 or
                              exp.get('topology') != current_topology or
                              exp['M'] != current_M or
                              exp.get('diameter') != current_diameter or
                              str(exp.get('interf_scenario')) != str(current_interf_scenario) or
                              exp.get('source_dist') != current_source_dist or
                              exp['mismatch_pos'] != current_mismatch_pos)
        else:
            recalc_physics = (exp['rt60'] != current_rt60 or exp['M'] != current_M or
                              exp['N_interferences'] != current_N_int or exp['mismatch_pos'] != current_mismatch_pos)
        recalc_mixture = recalc_physics or (exp['isir_db'] != current_isir_db)
        recalc_hardware = recalc_mixture or (exp['mismatch_gain'] != current_gain_mismatch or
                                             exp['mismatch_phase'] != current_phase_mismatch)
        recalc_wpe = recalc_hardware or (exp['use_wpe'] != current_use_wpe)

        # ---------------------------------------------------------
        # NODE 1: GEOMETRY AND PHYSICS
        # ---------------------------------------------------------
        if recalc_physics and geom_mode == 'mird_linear':
            # ---- REPLICA GEOMETRICA DE MIRD, RIRs SIMULADAS POR ISM ----
            # Mismo array lineal (4-4-4-8-4-4-4) y misma colocacion por
            # angulo/distancia que full_benchmark_test_dtln_mird, pero las RIRs
            # se COMPUTAN con el modelo ISM calibrado (compute_rirs) en una
            # shoebox de dimensiones scene_base_config['room_dims'] (MIRD: 6x6x2.4).
            tqdm.write(" -> [NODE 1|MIRD-replica] Building linear array + simulating RIRs (ISM)...")
            current_room_dims = np.array(scene_base_config['room_dims'], dtype=float)
            array_center = np.array(scene_base_config['array_center'], dtype=float)

            # Array lineal construido desde el spacing (barrer las 3 config MIRD:
            # 3-3-3-8-3-3-3 / 4-4-4-8-4-4-4 / 8-8-8-8-8-8-8). Default 4-4-4.
            mird_spacing = scene_base_config.get('mird_spacing', '4-4-4-8-4-4-4')
            mic_coords = generate_mird_linear_array_from_spacing(mird_spacing) + array_center
            scene_base_config['mic_coords'] = mic_coords

            acoustic_scene = SimAcoustic(
                array_geometry=mic_coords, array_mismatch=0.0,  # array real, sin mismatch posicional inyectado
                duration=scene_base_config['duration'], fs=scene_base_config['fs']
            )

            # Target por (angulo, distancia) con la MISMA convencion que MIRD:
            # spherical_to_cartesian(r, az=deg2rad(angle), inc=pi/2) -> (r cos, r sin, 0)
            # con el array a lo largo de Y => angle 0 = broadside (+X).
            scene_base_config['source_path'] = exp['source_path']
            rel_target = spherical_to_cartesian(
                np.array([exp['target_dist']], dtype=float),
                np.array([np.deg2rad(exp['target_angle'])], dtype=float),
                np.array([np.pi / 2.0], dtype=float),
            ).squeeze()
            source_pos = (array_center + rel_target).reshape(1, 3)
            scene_base_config['source_pos'] = source_pos
            acoustic_scene.set_source(exp['source_path'], gain=1.0, position=source_pos)

            interferences_pos = []
            for idx, interf_cfg in enumerate(exp['interf_configs']):
                i_ang = interf_cfg[0]
                i_dist = interf_cfg[1]
                if len(interf_cfg) >= 3:
                    audio_idx = interf_cfg[2]
                else:
                    audio_idx = idx % len(scene_base_config['interf_paths'])

                rel_i = spherical_to_cartesian(
                    np.array([i_dist], dtype=float),
                    np.array([np.deg2rad(i_ang)], dtype=float),
                    np.array([np.pi / 2.0], dtype=float),
                ).squeeze()
                abs_i = array_center + rel_i
                interferences_pos.append(abs_i)
                acoustic_scene.set_interference(
                    audio_path=scene_base_config['interf_paths'][audio_idx],
                    gain=1.0, position=abs_i.reshape(1, 3)
                )
            scene_base_config['interferences_pos'] = (np.asarray(interferences_pos)
                                                      if interferences_pos else np.zeros((0, 3)))

            acoustic_scene.compute_rirs(room_dimensions=current_room_dims, desire_RT=exp['rt60'],
                                        ray_tracing=scene_base_config.get('ray_tracing', False))
            acoustic_scene.convolve_signals(t_early=t_early_dynamic)

            current_rt60 = exp['rt60']
            current_target_angle = exp['target_angle']
            current_target_dist = exp['target_dist']
            current_interf_configs = exp['interf_configs']
            current_source_path = exp['source_path']

        elif recalc_physics and geom_mode == 'topology':
            # ---- TOPOLOGIA 2D INSCRIPTA EN CIRCULO, RIRs SIMULADAS POR ISM ----
            # La geometria del array se genera con generate_array_coords segun
            # exp['topology'] (circular/grid/spiral/concentric/random), inscripta en
            # un circulo de exp['diameter']. El array se ubica DESCENTRADO (mira a
            # +Y) en array_center_map[rt60] para que fuente/interferencias se abran
            # hacia adentro de la sala y quepan las distancias grandes.
            #
            # Fuente e interferencias se colocan en 3D por (azimut, elevacion,
            # distancia slant) via place_spherical:
            #   - source: azimut/elevacion de config, distancia = exp['source_dist'].
            #   - interferencias: lista de specs del escenario
            #     scene_base_config['interf_scenarios'][exp['interf_scenario']],
            #     cada una (azimut, elevacion, distancia). La ELEVACION != 0 saca a
            #     la interferencia del plano del arreglo (estresor principal).
            # Las distancias se recortan con max_distance_in_room para no salir de
            # la sala (margen de pared configurable).
            tqdm.write(f" -> [NODE 1|topology] Building '{exp['topology']}' array "
                       f"(M={exp['M']}, D={exp['diameter']} m) | room RT={exp['rt60']}s | "
                       f"interf='{exp['interf_scenario']}' | src_dist={exp['source_dist']}m; simulating RIRs (ISM)...")
            current_room_dims = np.asarray(room_profiles[exp['rt60']], dtype=float)

            # Centro del array: mapa por rt60 (descentrado); fallback al centro de sala.
            ac_map = scene_base_config.get('array_center_map', {})
            array_center = np.asarray(ac_map.get(exp['rt60'], current_room_dims / 2.0), dtype=float)
            wall_margin = scene_base_config.get('wall_margin', 0.3)

            # La topologia 'random' recibe una semilla ESTABLE derivada de la fisica
            # de la escena: el mismo arreglo aleatorio se reproduce entre corridas y
            # es reutilizado IDENTICO por todos los procesadores (Node 1 se cachea y
            # el loop de procesadores esta anidado adentro). topology_kwargs permite
            # pasar opciones por-topologia (p.ej. n_turns, inner_ratio).
            topo_kwargs = dict(scene_base_config.get('topology_kwargs', {}))
            if exp['topology'] == 'random':
                topo_kwargs.setdefault('seed', compute_scene_seed(exp, scene_base_config))
            mic_coords = generate_array_coords(
                topology=exp['topology'], M=exp['M'], diameter=exp['diameter'], **topo_kwargs
            ) + array_center
            scene_base_config['mic_coords'] = mic_coords

            acoustic_scene = SimAcoustic(
                array_geometry=mic_coords, array_mismatch=exp['mismatch_pos'],
                duration=scene_base_config['duration'], fs=scene_base_config['fs']
            )

            # --- FUENTE (target) por (azimut, elevacion, distancia slant) ---
            # Por defecto se toman los angulos globales de config y la distancia del
            # eje del grid (exp['source_dist']). Si el config trae 'source_specs' (un
            # dict {interf_scenario -> (az, el, dist)}, usado en el modo de escenas
            # aleatorias), la spec por-escena OVERRIDE los tres valores. Backward-
            # compatible: sin 'source_specs' el comportamiento es identico al previo.
            s_az = scene_base_config.get('source_azimuth_deg', 0.0)
            s_el = scene_base_config.get('source_elevation_deg', 0.0)
            s_dist_req = float(exp['source_dist'])
            _src_specs = scene_base_config.get('source_specs')
            if _src_specs is not None and exp['interf_scenario'] in _src_specs:
                s_az, s_el, s_dist_req = (float(v) for v in _src_specs[exp['interf_scenario']])
            s_dist = min(s_dist_req, max_distance_in_room(s_az, s_el, array_center, current_room_dims, wall_margin))
            if s_dist < s_dist_req - 1e-6:
                tqdm.write(f"    [!] source dist recortada {s_dist_req:.2f}->{s_dist:.2f} m (pared).")
            source_pos = place_spherical(s_az, s_el, s_dist, array_center).reshape(1, 3)
            scene_base_config['source_pos'] = source_pos
            acoustic_scene.set_source(scene_base_config['source_path'], gain=1.0, position=source_pos)

            # --- INTERFERENCIAS 3D segun el escenario seleccionado ---
            specs = scene_base_config['interf_scenarios'][exp['interf_scenario']]
            interferences_pos = []
            for idx, (i_az, i_el, i_dist_req) in enumerate(specs):
                i_dist = min(float(i_dist_req),
                             max_distance_in_room(i_az, i_el, array_center, current_room_dims, wall_margin))
                if i_dist < float(i_dist_req) - 1e-6:
                    tqdm.write(f"    [!] interf {idx} dist recortada {float(i_dist_req):.2f}->{i_dist:.2f} m (pared).")
                pos = place_spherical(i_az, i_el, i_dist, array_center)
                interferences_pos.append(pos)
                path_idx = idx % len(scene_base_config['interf_paths'])
                acoustic_scene.set_interference(
                    audio_path=scene_base_config['interf_paths'][path_idx],
                    gain=1.0, position=pos.reshape(1, 3)
                )
            scene_base_config['interferences_pos'] = (np.asarray(interferences_pos)
                                                      if interferences_pos else np.zeros((0, 3)))

            acoustic_scene.compute_rirs(room_dimensions=current_room_dims, desire_RT=exp['rt60'],
                                        ray_tracing=scene_base_config.get('ray_tracing', True))
            acoustic_scene.convolve_signals(t_early=t_early_dynamic)
            current_rt60, current_M, current_mismatch_pos = exp['rt60'], exp['M'], exp['mismatch_pos']
            current_N_int = len(specs)
            current_topology, current_diameter = exp['topology'], exp['diameter']
            current_interf_scenario, current_source_dist = exp['interf_scenario'], exp['source_dist']

        elif recalc_physics:
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
            # Persist absolute interference positions so they can be saved to H5 and
            # rendered by the dashboard (shape (N_interferences, 3)).
            scene_base_config['interferences_pos'] = interferences_pos

            acoustic_scene.set_source(scene_base_config['source_path'], gain=1.0, position=source_pos.reshape(1,3))
            for idx in range(exp['N_interferences']):
                path_idx = idx % len(scene_base_config['interf_paths'])
                acoustic_scene.set_interference(
                    audio_path=scene_base_config['interf_paths'][path_idx],
                    gain=1.0, position=interferences_pos[idx].reshape(1,3)
                )

            acoustic_scene.compute_rirs(room_dimensions=current_room_dims, desire_RT=exp['rt60'],
                                        ray_tracing=scene_base_config.get('ray_tracing', True))
            acoustic_scene.convolve_signals(t_early=t_early_dynamic)
            current_rt60, current_M, current_N_int, current_mismatch_pos = exp['rt60'], exp['M'], exp['N_interferences'], exp['mismatch_pos']

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
            # (mic_signals_ready = post HW mismatch [+ WPE]), no en el dominio limpio
            # pre-front-end. Se construyen en Node 3 (HW) + Node 4 (WPE). Ver el
            # bloque "ORACLE REFERENCES" mas abajo (alineado con el modulo MIRD).

            # Prepare multiple ground truth references based on configuration
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
            # => hw_target + hw_noise == mic_signals_degraded (exacto). Con
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
            if exp['use_wpe']:
                tqdm.write(" -> [NODE 4] Applying heavy WPE pre-processing (float)...")
                # Un solo pase de WPE que devuelve la mezcla dereverberada Y las
                # componentes target/ruido filtradas con el MISMO G (estimado de la
                # mezcla). Al ser lineal dado G: WPE(target)+WPE(ruido)==WPE(mezcla).
                # z_u es IDENTICO (bit a bit) a process_wpe_online(mezcla).
                mic_signals_ready, (oracle_target, oracle_noise) = process_wpe_online_with_components(
                    u=mic_signals_degraded,
                    components=[hw_oracle_target, hw_oracle_noise],
                    taps=scene_base_config['wpe_taps'], delay=scene_base_config['wpe_delay'],
                    alpha=scene_base_config['wpe_alpha'], stft_size=scene_base_config['wpe_stft_size'],
                    stft_shift=scene_base_config['wpe_stft_shift'],
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
            # consume estas dos senales; ahora viven en el MISMO dominio (HW [+WPE]) que
            # la senal que filtran -> las SCM / mascaras ideales quedan consistentes.
            scene_base_config['oracle_target'] = oracle_target
            scene_base_config['oracle_noise'] = oracle_noise

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

        array_center = np.mean(scene_base_config['mic_coords'], axis=0)
        true_src_pos = np.array(scene_base_config['source_pos']).flatten()

        rel_vec = true_src_pos - array_center
        r_xy = np.hypot(rel_vec[0], rel_vec[1])
        theta = np.arctan2(rel_vec[1], rel_vec[0])

        r_prime = max(0.01, r_xy + err_dist)
        theta_prime = theta + np.deg2rad(err_ang)

        assumed_pos_flat = np.array([
            array_center[0] + r_prime * np.cos(theta_prime),
            array_center[1] + r_prime * np.sin(theta_prime),
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
            # DTLN-completo como post-filtro (2do nucleo). apply_dtln_post=False lo
            # saltea SIN afectar el baseline DTLN-mono (Node 4.5), que sigue vivo.
            if use_dtln and apply_dtln_post:
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
            if geom_mode == 'mird_linear':
                _M = mic_signals_ready.shape[0]
                _n_int = len(exp['interf_configs'])
            elif geom_mode == 'topology':
                _M = exp['M']
                _n_int = int(np.asarray(scene_base_config.get('interferences_pos', np.zeros((0, 3)))).shape[0])
            else:
                _M = exp['M']
                _n_int = exp['N_interferences']
            row_data = {
                "processor": proc_name,
                "rt60": exp['rt60'], "M": _M, "N_interferences": _n_int,
                "mismatch_pos": exp.get('mismatch_pos', 0.0), "isir_db": exp['isir_db'],
                "mismatch_gain": exp['mismatch_gain'], "mismatch_phase": exp['mismatch_phase'],
                "use_wpe": exp['use_wpe'],
                "wpe_taps": scene_base_config['wpe_taps'],
                "wpe_delay": scene_base_config['wpe_delay'],
                "error_angle_deg": err_ang, "error_distance_m": err_dist,
                "t_early_s": t_early_dynamic, "exec_time_s": proc_time,
                # Ejes geometricos tipo MIRD (None en modo 'log' clasico).
                "target_dist": exp.get('target_dist'),
                "target_angle": exp.get('target_angle'),
                # Ejes del modo 'topology' (None en los demas modos).
                "topology": exp.get('topology'),
                "diameter": exp.get('diameter'),
                "interf_scenario": exp.get('interf_scenario'),
                "source_dist": exp.get('source_dist'),
                "source": os.path.basename(scene_base_config['source_path']),
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
            # Skipped when save_catalog=False (bulk metric runs): the per-case polar
            # spatial-response computation (precompute_quantized_spatial_response) is
            # expensive and only feeds the dashboard, not the metrics parquet/csv.
            for m_name in (tracked_metrics if save_catalog else []):
                # We specifically pull the metric evaluating against the 'early' signal
                eval_key = f"{m_name}_early"
                current_val = delta_tot_metrics.get(eval_key, np.nan)

                # Failsafe: If 'early' wasn't computed because it was removed from config, we skip leaderboard checks.
                if np.isnan(current_val): continue

                is_best = current_val > leaderboard[proc_name][m_name]["best_val"] if m_name != "Delta_tot_CD" else current_val < leaderboard[proc_name][m_name]["best_val"]
                is_worst = current_val < leaderboard[proc_name][m_name]["worst_val"] if m_name != "Delta_tot_CD" else current_val > leaderboard[proc_name][m_name]["worst_val"]

                if is_best or is_worst:
                    # We save using the strictly requested 'early' ground truth audio to keep data lightweight
                    target_ref_audio = refs_dict.get('early')

                    if is_best:
                        leaderboard[proc_name][m_name]["best_val"] = current_val
                        save_extreme_case_to_master(master_h5, proc_name, m_name, "best_case",
                                                    processor, mic_signals_ready, y_processed,
                                                    target_ref_audio, weights, exp, row_data,
                                                    scene_base_config, current_room_dims,
                                                    audio_dtln_alone, y_post_dtln)

                    if is_worst:
                        leaderboard[proc_name][m_name]["worst_val"] = current_val
                        save_extreme_case_to_master(master_h5, proc_name, m_name, "worst_case",
                                                    processor, mic_signals_ready, y_processed,
                                                    target_ref_audio, weights, exp, row_data,
                                                    scene_base_config, current_room_dims,
                                                    audio_dtln_alone, y_post_dtln)

    tqdm.write(f"\n=== BATCH COMPLETED IN {(time.time() - start_total_time)/60:.2f} MINUTES ===")
    df_results = pd.DataFrame(all_metrics_results)

    # Cast list columns to string for parquet compatibility
    for col in df_results.columns:
        if df_results[col].apply(lambda x: isinstance(x, (list, tuple))).any():
            df_results[col] = df_results[col].astype(str)

    parquet_path = os.path.join(output_dir, "ism_benchmark_metrics.parquet")
    df_results.to_parquet(parquet_path, engine="pyarrow")

    df_results.to_csv(os.path.join(output_dir, "ism_benchmark_metrics.csv"), index=False)

    return df_results


if __name__ == "__main__":

    try:
        interpreter_1 = tf.lite.Interpreter(model_path="/home/matias/Documents/Tesis/Vision-Aided-Beamformer/src/dnn_denoise/models/model_quant_1.tflite")
        interpreter_1.allocate_tensors()

        interpreter_2 = tf.lite.Interpreter(model_path="/home/matias/Documents/Tesis/Vision-Aided-Beamformer/src/dnn_denoise/models/model_quant_2.tflite")
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
        't_early': 0.050,  # (50 ms) referencia early/late FIJA, desacoplada del WPE
        'd_min': 0.02,
        'd_max': 0.30,
        # DIAGNOSTICO: matchear la escena MIRD (fuente/interf a 1 m, separacion 45 deg)
        'radius_source': 1.0,
        'radius_interf': 1.0,
        'delta_ang_deg': 45.0,
        # Generacion de RIRs SIMULADAS: True = hibrido ISM+RayTracing (estocastico);
        # False = ISM puro (deterministico, mas rapido, sin cola difusa de rayos).
        'ray_tracing': True,
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

        'stft_window': 512,
        'stft_overlap': 384,

        'dtln_model_path': r"/home/matias/Documents/Tesis/Vision-Aided-Beamformer/src/dnn_denoise/models/model_quant_1.tflite",


        # --- NEW SETTING: List of references to compute metrics against ---
        'eval_references': ['anechoic', 'early', 'reverberant']
    }

    param_grid = {
        # --- Ejes propios de la generacion de RIRs SIMULADAS (no MIRD) ---
        'rt60': [0.3],
        'M': [12],
        'N_interferences': [1],
        'mismatch_pos': [0.0],

        # --- Ejes compartidos, alineados con full_benchmark_test_dtln_mird ---
        'isir_db': [3],
        'mismatch_gain': [0],
        'mismatch_phase': [0],
        'use_wpe': [False],
        'error_angle_deg': [0.0],
        'error_distance_m': [0.0]
    }

    processors_dict = {
        #"NM-MVDR_alpha_1_ref" : NM_MVDR(min_loading =1e-6, alpha = 1),
        "NM-MVDR_alpha_0.99_ref" : NM_MVDR(min_loading =1e-6, alpha = 0.99),
        # Cota superior agnostica al modelo: misma cadena Souden pero con mascara ideal.
        # SOFT (sharpen_exp=1.0, IRM continua) y HARD-EDGE (sharpen_exp=4.0, == **4 del DTLN).
        #"Oracle-MVDR_alpha_1" : ORACLE_MB_MVDR_SOUDEN(min_loading =1e-6, alpha = 1, sharpen_exp=1.0),
        "Oracle-MVDR_alpha_0.99" : ORACLE_MB_MVDR_SOUDEN(min_loading =1e-6, alpha = 0.99, sharpen_exp=1.0),
        #"Oracle-MVDR_hard_alpha_1" : ORACLE_MB_MVDR_SOUDEN(min_loading =1e-6, alpha = 1, sharpen_exp=4.0),
        #"Oracle-MVDR_hard_alpha_0.99" : ORACLE_MB_MVDR_SOUDEN(min_loading =1e-6, alpha = 0.99, sharpen_exp=4.0),
        #"Slow"  : DTLN_MB_MVDR_SOUDEN_SLOW(),
        "NM-MVDR_PF" : NM_MVDR_PF(smooth=0.33, min_loading=1e-6),
    }

    df_final = run_grid_search(
        grid_params=param_grid,
        room_profiles=ROOM_PROFILES,
        processors=processors_dict,
        scene_base_config=base_config,
        output_dir="tests/dataset_out/ism_benchmark_test",
        interpreter_1=interpreter_1,
        interpreter_2=interpreter_2
    )

    print("\n[Preview] Quick Test Results:")

    cols_to_show = [
        "processor", "use_wpe", "error_angle_deg", "error_distance_m",
        "Delta_tot_PESQ_early", "Delta_tot_SIR_early", "Delta_tot_PESQ_anechoic"
    ]

    cols_exist = [c for c in cols_to_show if c in df_final.columns]
    print(df_final[cols_exist].to_string(index=False))