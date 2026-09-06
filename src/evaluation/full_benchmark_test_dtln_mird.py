import os
import time
import itertools
import hashlib
from dataclasses import replace
import h5py
import pandas as pd
import numpy as np

import tensorflow as tf
from dnn_denoise.dtln_lite import apply_dtln_post_tflite_realtime

from evaluation.polar_plots import precompute_quantized_spatial_response, subsample_weights
from beamforming.array.microphone import Microphone
from beamforming.array.geometry import select_reference_mic
from propagation.simulate_acoustics_v1 import SimAcoustic
from propagation.mird_loader import MirdDatasetProvider, generate_mird_linear_array, generate_mird_linear_array_from_spacing
from dereverberation.nara_wrappers import (
    process_wpe_online, process_wpe_online_with_components,
    process_wpe_block_online_with_components, block_wpe_warmup,
)
from dereverberation.nara_wrappers_fixed import (
    process_wpe_online_fixed, FixedPointConfig, Fx, process_wpe_block_online_fixed,
)
from dereverberation.qrd_wpe_fixed import process_qrd_wpe_online_fixed, QRDFixedPointConfig
from evaluation.metrics import evaluate_full_pipeline
from evaluation.bench_ui import BenchmarkUI, compact_config, varying_keys

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
    SOUDEN_ORACLE_SCM,
    DTLN_MB_MVDR_SOUDEN_BAN_alphaless,
    NM_MVDR_DSM_BLIND,
    NM_MVDR_DSM_FB

    

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


def run_mird_grid_search(grid_params, dataset_provider, processors, scene_base_config, output_dir="results/", interpreter_1=None, interpreter_2=None, save_catalog=True, apply_dtln_post=True,
                         show_progress=True, quiet_console=True,
                         save_weights_dir=None, save_weights_cells=None,
                         save_weights_fps=16):
    """
    Barrido MIRD. `show_progress`/`quiet_console` SOLO afectan la consola:
      show_progress=False -> sin panel de barras (log plano, util en CI/logs).
      quiet_console=False -> deja pasar los prints internos de procesadores y
                             metricas (util para depurar una etapa puntual).
    Ninguno de los dos cambia las llamadas, el orden de ejecucion ni el DataFrame
    devuelto.

    GUARDADO DE PESOS (opcional)
    ----------------------------
    save_weights_dir : carpeta donde volcar los pesos w(k,t,m) de cada
        (celda, procesador) como .npz. None (default) = no se guarda nada, o sea
        el comportamiento historico.
    save_weights_cells : lista de INDICES de experimento cuyos pesos guardar.
        None = TODOS. Conviene acotarlo: los pesos son (K, T, M) complejos, y una
        celda de 15 s con K=257, M=8 pesa ~60 MB en complex128. Aca se guardan
        submuestreados en tiempo y en complex64 (ver abajo), pero aun asi son
        varios MB por celda y procesador.
    save_weights_fps : cuadros por segundo del submuestreo temporal (default 16).
        Los pesos de un filtro recursivo con alpha=0.99 cambian con una constante
        de tiempo de ~100 frames (~0.8 s a hop=128/16 kHz), asi que 16 fps ya
        sobre-muestrea la dinamica real. Se conservan TODOS los bins de frecuencia
        (a diferencia de `subsample_weights`, que ademas colapsa a tercios de
        octava y serviria solo para los polares).
    """
    os.makedirs(output_dir, exist_ok=True)
    if save_weights_dir is not None:
        os.makedirs(save_weights_dir, exist_ok=True)

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
    # wpe_method selecciona el back-end del WPE: 'online' = RLS recursivo por-frame
    # (default, comportamiento previo), 'block' = block-online con Cholesky por
    # ventana trailing (Opcion B). Es un EJE de grilla mas (no reemplaza online):
    # se pueden correr ambos en la misma corrida para compararlos.
    if 'wpe_method' not in grid_params:
        grid_params['wpe_method'] = [scene_base_config.get('wpe_method', 'online')]
    # Sub-parametros del block-online (Opcion B) como ejes de grilla barreables:
    # wpe_block_L (largo de la ventana trailing) y wpe_block_shift (refresco de G).
    # Solo afectan a las celdas con wpe_method='block'; en las online se colapsan al
    # default y se deduplican mas abajo (para no repetir corridas online identicas).
    if 'wpe_block_L' not in grid_params:
        grid_params['wpe_block_L'] = [scene_base_config.get('wpe_block_L', 512)]
    if 'wpe_block_shift' not in grid_params:
        grid_params['wpe_block_shift'] = [scene_base_config.get('wpe_block_shift', 64)]
    # source_path is now a grid axis too (target-speaker variety). Callers that do
    # not sweep it (E0, notebooks A/B/C) keep passing the scalar in scene_base_config;
    # promote it to a single-value list so every experiment dict carries it.
    if 'source_path' not in grid_params:
        grid_params['source_path'] = [scene_base_config['source_path']]

    # 1. Generate combinations
    keys, values = zip(*grid_params.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # 1b. DEDUP de sub-parametros condicionales del block. wpe_block_L/shift solo
    # tienen sentido si wpe_method='block'; en las celdas online el producto
    # cartesiano genera duplicados que solo difieren en esos campos irrelevantes.
    # Se colapsan al default y se deduplica -> no se re-corren celdas online iguales.
    _blk_L_def = scene_base_config.get('wpe_block_L', 512)
    _blk_S_def = scene_base_config.get('wpe_block_shift', 64)
    _seen, _dedup = set(), []
    for e in experiments:
        if e.get('wpe_method', 'online') != 'block':
            e['wpe_block_L'] = _blk_L_def
            e['wpe_block_shift'] = _blk_S_def
        key = tuple(sorted((k, str(v)) for k, v in e.items()))
        if key not in _seen:
            _seen.add(key)
            _dedup.append(e)
    experiments = _dedup

    # 2. STRATEGIC SORTING (Crucial for cascaded caching adapted to MIRD)
    experiments.sort(key=lambda x: (
        x['rt60'], x['target_dist'], x['target_angle'], str(x['interf_configs']), x['source_path'], # Node 1: Physical Environment
        x['isir_db'],                                                             # Node 2: Mixture
        x['mismatch_gain'], x['mismatch_phase'],                                  # Node 3: Hardware
        x['use_wpe'], x['wpe_method'], x['wpe_taps'], x['wpe_delay'],             # Node 4: Pre-processing (WPE; wpe_delay no longer feeds t_early)
        x['wpe_block_L'], x['wpe_block_shift'],                                   # Node 4: block sub-params
        x.get('error_angle_deg', 0.0), x.get('error_distance_m', 0.0)             # Node 5: Misinformation
    ))

    # --- FIXED t_early (constant across the whole grid) ---
    # t_early defines the "early" reference window of the metrics. It is a FIXED
    # acoustic quantity read from scene_base_config['t_early'] and is deliberately
    # DECOUPLED from wpe_delay: when sweeping wpe_delay (e.g. E0) every delay must
    # be scored against the SAME early/late reference, otherwise the Delta metrics
    # are biased by a moving target and delays are not comparable (correction A1).
    t_early_dynamic = scene_base_config['t_early']
    print(f"[*] t_early FIXED at {t_early_dynamic*1000:.1f} ms (decoupled from wpe_delay).")
    print(f"[*] Total experiments to run: {len(experiments)} per processor "
          f"({len(processors)} processors -> {len(experiments) * len(processors)} rows).")

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
    current_wpe_taps, current_wpe_delay, current_wpe_method = None, None, None
    current_wpe_block_L, current_wpe_block_shift = None, None
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
    # Segundos iniciales que NO entran en ninguna metrica. El default historico
    # (min(5, 0.3*duracion)) esconde todo transitorio de convergencia de los
    # procesadores adaptativos: para medirlo hay que poner 'eval_start_s': 0.0 en
    # la config de escena. Ojo que con use_wpe=True eso mete tambien el arranque
    # en frio del WPE dentro de la ventana.
    eval_start_s = scene_base_config.get('eval_start_s')
    if eval_start_s is None:
        eval_start_s = min(5.0, scene_base_config['duration'] * 0.3)
    eval_start_s = float(eval_start_s)

    use_dtln = interpreter_1 is not None and interpreter_2 is not None

    # Dummy dimensions for the Bar-Ilan room to keep HDF5 inspector happy
    mird_room_dims = np.array([6.0, 6.0, 2.41])
    array_center = np.array(scene_base_config['array_center'])

    # --- PANEL DE CONSOLA (3 lineas fijas: total / prueba / etapa) ---
    ui = BenchmarkUI(len(experiments), desc="MIRD Benchmark", unit="exp",
                     quiet=quiet_console, enabled=show_progress)
    # La etiqueta de cada prueba muestra solo los ejes que realmente se barren.
    label_keys = varying_keys(experiments)

    # 4. Main Orchestrator Loop
    for i, exp in enumerate(experiments):

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
                                         exp.get('wpe_method', 'online') != current_wpe_method or
                                         exp['wpe_taps'] != current_wpe_taps or
                                         exp['wpe_delay'] != current_wpe_delay or
                                         exp.get('wpe_block_L') != current_wpe_block_L or
                                         exp.get('wpe_block_shift') != current_wpe_block_shift)

        # Presupuesto de etapas de ESTA prueba = 100% de la barra inferior. Se
        # deriva de los flags de cache (los nodos reusados no cuentan) + el trabajo
        # por procesador. Es solo para la escala de la barra; si aparecen mas
        # etapas de las previstas, BenchmarkUI estira el total.
        n_steps = ((1 if recalc_physics else 0) + (1 if recalc_mixture else 0) +
                   (2 if recalc_hardware else 0))
        if recalc_wpe:
            n_steps += 2 if exp['use_wpe'] else 1
            if use_dtln:
                n_steps += 2
        n_steps += len(processors) * (2 + (2 if (use_dtln and apply_dtln_post) else 0))
        ui.begin_experiment(i, compact_config(exp, keep=label_keys), steps=n_steps)

        # ---------------------------------------------------------
        # NODE 1: GEOMETRY AND PHYSICS (MIRD DATASET IMPORT)
        # ---------------------------------------------------------
        if recalc_physics:
            ui.stage(f"[NODE 1] RIRs MIRD (rt60={exp['rt60']}, dist={exp['target_dist']} m, "
                     f"ang={exp['target_angle']} deg)")
            with ui.quiet():
                # Setup array coordinates from the spacing being loaded (8-channel linear
                # for MIRD). IMPORTANTE: la geometria debe coincidir con el spacing de las
                # RIRs cargadas (mird_spacing), si no los steering vectors de los
                # beamformers no corresponden al array real. Antes estaba fijo en 4-4-4.
                base_array = generate_mird_linear_array_from_spacing(scene_base_config['mird_spacing'])
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
        # MICROFONO DE REFERENCIA (canal de escucha de TODO el pipeline)
        # ---------------------------------------------------------
        # Un beamformer NO devuelve "la voz limpia" en abstracto: devuelve la voz TAL
        # COMO LLEGA a un microfono de referencia. La familia Souden lo hace explicito
        # (proyecta con el vector one-hot u_ref) y los geometricos tambien, via la RTF
        # normalizada a ese canal (w^H d = 1 reconstruye d_ref). Si las metricas se
        # calculan contra OTRO canal, la diferencia de camino acustico entre los dos
        # micros se contabiliza como distorsion del procesador: medido sobre MIRD, eso
        # costaba ~3 dB de SI-SDR y ~2 dB de SAR a la familia mask-based (que proyectaba
        # al M//2) frente a las metricas, que salian del canal 0.
        #
        # Aca se elige UN canal y se usa para TODO: las referencias early/anechoic/
        # reverberant, el baseline, las componentes de ruido de BSS-Eval, la entrada del
        # DTLN mono y el 'ref_mic_idx' que leen todos los wrappers de procesador.
        #
        # scene_base_config['ref_mic_mode']:
        #   None / ausente -> M//2, la referencia historica de la familia Souden (para
        #                     el arreglo lineal de 8 micros del MIRD, el canal 4).
        #   'first'        -> canal 0.
        #   'centroid'     -> mic mas cercano al centro geometrico (select_reference_mic);
        #                     util al comparar topologias, donde M//2 cae en un lugar
        #                     arbitrario segun como cada generador enumera sus micros.
        #   int            -> indice fijo.
        _M_arr = len(scene_base_config['mic_coords'])
        ref_mic_mode = scene_base_config.get('ref_mic_mode')
        if ref_mic_mode is None:
            ref_ch = _M_arr // 2
        elif ref_mic_mode == 'centroid':
            ref_ch = select_reference_mic(scene_base_config['mic_coords'])
        elif ref_mic_mode == 'first':
            ref_ch = 0
        else:
            ref_ch = int(ref_mic_mode)
        if not (0 <= ref_ch < _M_arr):
            raise ValueError(f"ref_mic_mode={ref_mic_mode!r} -> canal {ref_ch} fuera de rango (M={_M_arr}).")
        if scene_base_config.get('ref_mic_idx') != ref_ch:
            ui.log(f" [*] [REF-MIC] canal {ref_ch} de {_M_arr} "
                   f"(metricas + proyeccion de TODOS los beamformers).")
        # Se propaga a los procesadores: los wrappers leen esta clave.
        scene_base_config['ref_mic_idx'] = ref_ch

        # ---------------------------------------------------------
        # NODE 2: ACOUSTIC MIXTURE & GROUND TRUTHS PREPARATION
        # ---------------------------------------------------------
        if recalc_mixture:
            ui.stage(f"[NODE 2] mezcla acustica (iSIR = {exp['isir_db']} dB)")
            with ui.quiet():
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
                refs_dict['anechoic'] = scene_data["target_anechoic"][ref_ch]
            if 'early' in scene_base_config['eval_references']:
                refs_dict['early'] = scene_data["target_early"][ref_ch]
            if 'reverberant' in scene_base_config['eval_references']:
                refs_dict['reverberant'] = scene_data["target_early"][ref_ch] + scene_data["target_late"][ref_ch]

        unprocessed_mic_signals = scene_data["mic_signals"]

        # ---------------------------------------------------------
        # NODE 3: HARDWARE EMULATION & BASELINE
        # ---------------------------------------------------------
        if recalc_hardware:
            ui.stage(f"[NODE 3] emulacion de hardware (gain {exp['mismatch_gain']} dB, "
                     f"fase {exp['mismatch_phase']} deg)")
            # Semilla derivada SOLO de la fisica de la escena (no del mismatch): fija
            # el patron base de mismatch/ruido para que el barrido de gain/phase varie
            # solo la escala del error. Al no depender del mismatch, set_seed es no-op
            # cuando solo cambian gain/phase -> se preservan los patrones cacheados.
            mic_simulator.set_seed(compute_scene_seed(exp, scene_base_config))
            with ui.quiet():
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

            ui.stage("[NODE 3] metricas del baseline (todas las referencias)")
            with ui.quiet():
                baseline_metrics = evaluate_all_references(
                    refs_dict=refs_dict, deg_sig=mic_signals_degraded[ref_ch], fs=scene_base_config['fs'],
                    interf_early=scene_data["interference_early"][ref_ch],
                    interf_late=scene_data["interference_late"][ref_ch],
                    target_late=scene_data["target_late"][ref_ch],
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
                wpe_method = exp.get('wpe_method', 'online')
                fixed_bits = scene_base_config.get('wpe_fixed_bits', None)
                if wpe_method == 'block':
                    # ---- OPCION B: BLOCK-ONLINE con Cholesky (float) ----
                    # Estima G resolviendo las ecuaciones normales sobre una ventana
                    # trailing de L frames pasados, refresca cada block_shift, y aplica
                    # G causal congelado. La descomposicion del oracle es EXACTA (el
                    # block es lineal dado G): usa el _with_components analogo al online.
                    # L y block_shift vienen del experimento (ejes de grilla
                    # barreables); iters/reg/solver siguen siendo escalares de config.
                    _L = exp.get('wpe_block_L', scene_base_config.get('wpe_block_L', 512))
                    _bs = exp.get('wpe_block_shift', scene_base_config.get('wpe_block_shift', 64))
                    _iters = scene_base_config.get('wpe_block_iters', 3)
                    _reg = scene_base_config.get('wpe_block_reg', 1e-6)
                    _stft = dict(stft_size=scene_base_config['wpe_stft_size'],
                                 stft_shift=scene_base_config['wpe_stft_shift'])
                    if fixed_bits is None:
                        block_kw = dict(
                            L=_L, block_shift=_bs, iterations=_iters, reg=_reg,
                            solver=scene_base_config.get('wpe_block_solver', 'cholesky'),
                            mode=scene_base_config.get('wpe_block_mode', 'resolve'),
                            warm_start=scene_base_config.get('wpe_block_warm_start', False),
                        )
                        ui.stage(f"[NODE 4] WPE BLOCK-online (float) taps={exp['wpe_taps']} "
                                 f"delay={exp['wpe_delay']} L={_L} shift={_bs}")
                        with ui.quiet():
                            mic_signals_ready, (oracle_target, oracle_noise) = process_wpe_block_online_with_components(
                                u=mic_signals_degraded,
                                components=[hw_oracle_target, hw_oracle_noise],
                                taps=exp['wpe_taps'], delay=exp['wpe_delay'], **_stft, **block_kw)
                    else:
                        # ---- BLOCK FIXED-POINT: cuantiza buffer(in)/filtro(g)/salida(pred).
                        # g/p necesitan headroom (G del block ~1-2 con reg alto; sube reg
                        # p/ acotar G -> usar wpe_block_reg~1e-2 en el fixed). window_bits
                        # opcional baja SOLO el buffer (idea on-chip para L grande).
                        ibits = scene_base_config.get('wpe_block_int_bits', {'in': 1, 'pred': 1, 'g': 6, 'p': 6})
                        fp_cfg = FixedPointConfig.wordlength(fixed_bits, int_bits=ibits)
                        # BLOCK-FLOAT del buffer (exponente compartido entre mics): idea HLS
                        # para igualar float+L512 con menos memoria. Tiene prioridad sobre wb.
                        bf_mant = scene_base_config.get('wpe_block_window_mant', None)
                        wb = scene_base_config.get('wpe_block_window_bits', None)
                        if bf_mant is not None:
                            fp_cfg = replace(fp_cfg, buffer_blockfloat=True,
                                             buffer_mant_bits=bf_mant,
                                             buffer_exp_bits=scene_base_config.get('wpe_block_window_exp', None))
                            buf_desc = f"bf-mant{bf_mant}"
                        elif wb is not None:
                            fmts = dict(fp_cfg.formats)
                            fmts['in'] = Fx(wb, max(0, wb - 1 - ibits['in']))
                            fp_cfg = replace(fp_cfg, formats=fmts)
                            buf_desc = f"{wb}b"
                        else:
                            buf_desc = f"{fixed_bits}b"
                        # BLOCK-FLOAT del filtro G (storage persistente aparte del buffer).
                        g_mant = scene_base_config.get('wpe_block_g_mant', None)
                        if g_mant is not None:
                            fp_cfg = replace(fp_cfg, g_blockfloat=True, g_mant_bits=g_mant,
                                             g_exp_bits=scene_base_config.get('wpe_block_g_exp', None))
                            g_desc = f"bf-mant{g_mant}"
                        else:
                            g_desc = f"{fixed_bits}b"
                        # DATAPATH del SOLVE (Cholesky interno) en punto fijo.
                        sb = scene_base_config.get('wpe_block_solve_bits', None)
                        smeth = scene_base_config.get('wpe_block_solve_method', 'cholesky')
                        if sb is not None:
                            si = scene_base_config.get('wpe_block_solve_int', 12)
                            fp_cfg = replace(fp_cfg, solve_fx=Fx(sb, max(0, sb - 1 - si)),
                                             solve_method=smeth)
                            s_desc = f"{smeth}/{sb}b(int{si})"
                        else:
                            fp_cfg = replace(fp_cfg, solve_method=smeth)
                            s_desc = f"{smeth}/float"
                        ui.stage(f"[NODE 4] WPE BLOCK-online FIXED {fixed_bits}b "
                                 f"(buffer={buf_desc}, G={g_desc}, solve={s_desc}, reg={_reg:.0e})")
                        with ui.quiet():
                            mic_signals_ready, fp_stats = process_wpe_block_online_fixed(
                                u=mic_signals_degraded, taps=exp['wpe_taps'], delay=exp['wpe_delay'],
                                L=_L, block_shift=_bs, iterations=_iters, reg=_reg,
                                fp_cfg=fp_cfg, return_stats=True, **_stft)
                        # Las FP-STATS solo se persisten si hay algo que mirar (overflow o
                        # divergencia); si no, quedan en la linea de estado y no ensucian.
                        _fp_msg = (f"[FP-STATS block] overflow={fp_stats.overflow} "
                                   f"max|G|={fp_stats.max_absG:.2e} diverged={fp_stats.diverged}")
                        if fp_stats.overflow or fp_stats.diverged:
                            ui.log(f" [!] exp {i+1}/{len(experiments)} {_fp_msg}")
                        ui.stage(f"[NODE 4] refs oracle (descomposicion float) · {_fp_msg}",
                                 advance=False)
                        # Oracle refs por descomposicion FLOAT (Opcion A, como el online fixed).
                        with ui.quiet():
                            _, (oracle_target, oracle_noise) = process_wpe_block_online_with_components(
                                u=mic_signals_degraded,
                                components=[hw_oracle_target, hw_oracle_noise],
                                taps=exp['wpe_taps'], delay=exp['wpe_delay'], reg=_reg,
                                L=_L, block_shift=_bs, iterations=_iters, **_stft)
                    block_kw = dict(L=_L, block_shift=_bs)  # para el warmup check de abajo
                    # La ventana de metricas (eval_start_s) debe empezar DESPUES del
                    # warmup del block (primera ventana llena de L frames), o el
                    # transitorio en frio contamina las metricas.
                    _, warm_s = block_wpe_warmup(exp['wpe_taps'], exp['wpe_delay'],
                                                 block_kw['L'], block_kw['block_shift'],
                                                 scene_base_config['wpe_stft_shift'])
                    warm_t = warm_s / scene_base_config['fs']
                    if warm_t > eval_start_s:
                        ui.log(f" [!] exp {i+1}/{len(experiments)}: warmup del block {warm_t:.2f}s > "
                               f"eval_start {eval_start_s:.2f}s: subi 'duration'/eval_start o baja "
                               f"wpe_block_L para no medir el arranque en frio.")
                elif fixed_bits is None:
                    ui.stage(f"[NODE 4] WPE online (float) taps={exp['wpe_taps']} delay={exp['wpe_delay']}")
                    # Un solo pase de WPE que devuelve la mezcla dereverberada Y las
                    # componentes target/ruido filtradas con el MISMO G (estimado de la
                    # mezcla). Al ser lineal dado G: WPE(target)+WPE(ruido)==WPE(mezcla).
                    # z_u es IDENTICO (bit a bit) a process_wpe_online(mezcla): el
                    # filtrado de componentes no toca el estado del filtro de la mezcla.
                    with ui.quiet():
                        mic_signals_ready, (oracle_target, oracle_noise) = process_wpe_online_with_components(
                            u=mic_signals_degraded,
                            components=[hw_oracle_target, hw_oracle_noise], **wpe_kw
                        )
                else:
                    # Back-end selector: 'cov' = covariance-form RLS (default, needs
                    # 24 bit); 'qrd' = inverse-QRD square-root RLS (stable ~16 bit).
                    # Both share the same drop-in signature and fixed-point stats.
                    wpe_backend = scene_base_config.get('wpe_backend', 'cov')
                    ui.stage(f"[NODE 4] WPE online FIXED-POINT {fixed_bits}b "
                             f"(backend={wpe_backend.upper()}, emulacion FPGA)")
                    if wpe_backend == 'qrd':
                        fp_cfg = QRDFixedPointConfig.wordlength(
                            fixed_bits, rounding=scene_base_config.get('wpe_fixed_round', 'nearest')
                        )
                        with ui.quiet():
                            mic_signals_ready, fp_stats = process_qrd_wpe_online_fixed(
                                u=mic_signals_degraded, taps=exp['wpe_taps'], delay=exp['wpe_delay'],
                                alpha=scene_base_config['wpe_alpha'], stft_size=scene_base_config['wpe_stft_size'],
                                stft_shift=scene_base_config['wpe_stft_shift'], fp_cfg=fp_cfg, return_stats=True,
                                n_iter=scene_base_config.get('wpe_n_iter', 1),
                                refine_floor=scene_base_config.get('wpe_refine_floor', 0.0),
                            )
                    else:
                        fp_cfg = FixedPointConfig.wordlength(
                            fixed_bits, rounding=scene_base_config.get('wpe_fixed_round', 'nearest')
                        )
                        with ui.quiet():
                            mic_signals_ready, fp_stats = process_wpe_online_fixed(
                                u=mic_signals_degraded, taps=exp['wpe_taps'], delay=exp['wpe_delay'],
                                alpha=scene_base_config['wpe_alpha'], stft_size=scene_base_config['wpe_stft_size'],
                                stft_shift=scene_base_config['wpe_stft_shift'], fp_cfg=fp_cfg, return_stats=True,
                                n_iter=scene_base_config.get('wpe_n_iter', 1),
                                refine_floor=scene_base_config.get('wpe_refine_floor', 0.0),
                            )
                    # NOTE: for the QRD backend max|P| below is really max|L| (the
                    # stored triangular factor); FxStats reuses the same field.
                    _fp_msg = (f"[FP-STATS] overflow={fp_stats.overflow} max|P/L|={fp_stats.max_absP:.2e} "
                               f"max|G|={fp_stats.max_absG:.2e} diverged={fp_stats.diverged}")
                    if fp_stats.overflow or fp_stats.diverged:
                        ui.log(f" [!] exp {i+1}/{len(experiments)} {_fp_msg}")
                    # OPCION A (aproximacion documentada): el front-end fixed-point es NO
                    # lineal, asi que no admite descomposicion exacta. Las refs del oracle
                    # se computan con la descomposicion FLOAT (exacta) sobre la MISMA
                    # mezcla degradada. A alta word-length fixed~=float, asi que las refs
                    # son una aproximacion cercana del target/ruido dentro de la
                    # observacion fixed; el desajuste esta acotado por el error de
                    # cuantizacion (crece a pocos bits -> interpretar el oracle con
                    # cuidado en ese regimen).
                    ui.stage(f"[NODE 4] refs oracle (descomposicion float, Opcion A) · {_fp_msg}",
                             advance=False)
                    with ui.quiet():
                        _, (oracle_target, oracle_noise) = process_wpe_online_with_components(
                            u=mic_signals_degraded,
                            components=[hw_oracle_target, hw_oracle_noise], **wpe_kw
                        )

                ui.stage("[NODE 4] metricas del WPE (todas las referencias)")
                with ui.quiet():
                    wpe_metrics = evaluate_all_references(
                        refs_dict=refs_dict, deg_sig=mic_signals_ready[ref_ch], fs=scene_base_config['fs'],
                        interf_early=scene_data["interference_early"][ref_ch],
                        interf_late=scene_data["interference_late"][ref_ch],
                        target_late=scene_data["target_late"][ref_ch],
                        eval_start_s=eval_start_s, prefix_name=f"WPE_Exp_{i}"
                    )
            else:
                ui.stage("[NODE 4] WPE desactivado (bypass)")
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
            current_wpe_method = exp.get('wpe_method', 'online')
            current_wpe_taps = exp['wpe_taps']
            current_wpe_delay = exp['wpe_delay']  # tracked here now that t_early is fixed
            current_wpe_block_L = exp.get('wpe_block_L')
            current_wpe_block_shift = exp.get('wpe_block_shift')

            # ---------------------------------------------------------
            # NODE 4.5: SINGLE-MIC DTLN
            # ---------------------------------------------------------
            if use_dtln:
                ui.stage("[NODE 4.5] DTLN mono sobre el mic de referencia")
                with ui.quiet():
                    audio_dtln_alone = apply_dtln_post_tflite_realtime(
                        interpreter_1=interpreter_1, interpreter_2=interpreter_2,
                        audio_mono=mic_signals_ready[ref_ch]
                    )
                ui.stage("[NODE 4.5] metricas DTLN mono")
                with ui.quiet():
                    dtln_alone_metrics = evaluate_all_references(
                        refs_dict=refs_dict, deg_sig=audio_dtln_alone, fs=scene_base_config['fs'],
                        interf_early=scene_data["interference_early"][ref_ch],
                        interf_late=scene_data["interference_late"][ref_ch],
                        target_late=scene_data["target_late"][ref_ch],
                        eval_start_s=eval_start_s, prefix_name=f"DTLN_Alone_Exp_{i}"
                    )
        else:
            ui.stage("[CACHE] reusando WPE + DTLN mono de la prueba anterior", advance=False)

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

        for p_idx, (proc_name, processor) in enumerate(processors.items()):
            ui.stage(f"[BF {p_idx+1}/{len(processors)}] {proc_name} "
                     f"(ErrAng {err_ang}deg, ErrDist {err_dist}m)")

            t0 = time.time()
            with ui.quiet():
                y_processed, weights = processor.process(mic_signals_ready, proc_config)
            proc_time = time.time() - t0

            # --- volcado opcional de los pesos del beamformer -----------------
            # Se guardan ANTES de cualquier post-filtro de salida: los pesos
            # describen SOLO la etapa espacial (una ganancia espectral posterior
            # no esta en w). Submuestreo temporal + complex64 para que el tamano
            # sea manejable; todos los bins de frecuencia se conservan.
            if (save_weights_dir is not None and weights is not None
                    and (save_weights_cells is None or i in save_weights_cells)):
                _hop = (scene_base_config['stft_window']
                        - scene_base_config['stft_overlap'])
                _T = weights.shape[1]
                _dur = _T * _hop / scene_base_config['fs']
                _nt = max(1, int(_dur * save_weights_fps))
                _ti = np.round(np.linspace(0, _T - 1, _nt)).astype(int)
                _freqs = np.fft.rfftfreq(scene_base_config['stft_window'],
                                         1.0 / scene_base_config['fs'])
                _tag = (f"exp{i:03d}_{proc_name}_rt{exp['rt60']:g}"
                        f"_isir{exp['isir_db']:g}")
                np.savez_compressed(
                    os.path.join(save_weights_dir, f"{_tag}.npz"),
                    weights=weights[:, _ti, :].astype(np.complex64),
                    frame_idx=_ti, freqs=_freqs, hop=_hop,
                    fs=scene_base_config['fs'], ref_mic_idx=ref_ch,
                    processor=proc_name, rt60=exp['rt60'], isir_db=exp['isir_db'],
                    interf_configs=str(exp['interf_configs']),
                    mic_coords=scene_base_config['mic_coords'])

            ui.stage(f"[BF {p_idx+1}/{len(processors)}] {proc_name}: metricas ({proc_time:.1f}s de BF)")
            with ui.quiet():
                proc_metrics = evaluate_all_references(
                    refs_dict=refs_dict, deg_sig=y_processed, fs=scene_base_config['fs'],
                    interf_early=scene_data["interference_early"][ref_ch],
                    interf_late=scene_data["interference_late"][ref_ch],
                    target_late=scene_data["target_late"][ref_ch],
                    eval_start_s=eval_start_s, prefix_name=f"Proc_{proc_name}_Exp_{i}"
                )

            # ---------------------------------------------------------
            # NODE 6: POST-BEAMFORMING DTLN
            # ---------------------------------------------------------
            y_post_dtln = None
            dtln_post_metrics = {}
            # DTLN-completo como post-filtro (2do nucleo). Descartado del sistema
            # (demasiadas cascadas). apply_dtln_post=False lo saltea SIN afectar el
            # baseline DTLN-mono (Node 4.5), que sigue vivo para la comparacion de P4.
            if use_dtln and apply_dtln_post:
                ui.stage(f"[NODE 6] DTLN post {proc_name}")
                with ui.quiet():
                    y_post_dtln = apply_dtln_post_tflite_realtime(
                        interpreter_1=interpreter_1, interpreter_2=interpreter_2,
                        audio_mono=y_processed
                    )
                ui.stage(f"[NODE 6] metricas DTLN post {proc_name}")
                with ui.quiet():
                    dtln_post_metrics = evaluate_all_references(
                        refs_dict=refs_dict, deg_sig=y_post_dtln, fs=scene_base_config['fs'],
                        interf_early=scene_data["interference_early"][ref_ch],
                        interf_late=scene_data["interference_late"][ref_ch],
                        target_late=scene_data["target_late"][ref_ch],
                        eval_start_s=eval_start_s, prefix_name=f"DTLN_Post_{proc_name}_Exp_{i}"
                    )

            # Compile Dataset Row
            row_data = {
                "processor": proc_name,
                "rt60": exp['rt60'],
                "M": 8, # Fixed MIRD Array size
                "ref_mic_idx": ref_ch,   # canal de referencia (metricas + proyeccion)
                "N_interferences": len(exp['interf_configs']),
                # Geometria de las interferencias (angulo, distancia). Sin esta
                # columna, dos escenas que solo difieren en DONDE esta el
                # interferente quedan indistinguibles en el CSV/parquet.
                "interf_configs": str(exp['interf_configs']),
                "mismatch_pos": 0.0, # Physical arrays inherently contain positioning reality, no injected random mismatch
                "isir_db": exp['isir_db'],
                "mismatch_gain": exp['mismatch_gain'],
                "mismatch_phase": exp['mismatch_phase'],
                "use_wpe": exp['use_wpe'],
                "wpe_method": exp.get('wpe_method', 'online'),
                "wpe_taps": exp['wpe_taps'],
                "wpe_delay": exp['wpe_delay'],
                "wpe_bits": scene_base_config.get('wpe_fixed_bits', None),
                # Sub-parametros del block (desde el experimento; None si online).
                "wpe_block_L": (exp.get('wpe_block_L')
                                if exp.get('wpe_method', 'online') == 'block' else None),
                "wpe_block_shift": (exp.get('wpe_block_shift')
                                    if exp.get('wpe_method', 'online') == 'block' else None),
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
            # Skipped when save_catalog=False (bulk metric runs): the per-case polar
            # spatial-response computation (precompute_quantized_spatial_response) is
            # expensive and only feeds the dashboard, not the metrics parquet/csv.
            for m_name in (tracked_metrics if save_catalog else []):
                eval_key = f"{m_name}_early"
                current_val = delta_tot_metrics.get(eval_key, np.nan)

                if np.isnan(current_val): continue

                is_best = current_val > leaderboard[proc_name][m_name]["best_val"] if m_name != "Delta_tot_CD" else current_val < leaderboard[proc_name][m_name]["best_val"]
                is_worst = current_val < leaderboard[proc_name][m_name]["worst_val"] if m_name != "Delta_tot_CD" else current_val > leaderboard[proc_name][m_name]["worst_val"]

                if is_best or is_worst:
                    target_ref_audio = refs_dict.get('early')

                    if is_best:
                        leaderboard[proc_name][m_name]["best_val"] = current_val
                        ui.stage(f"{proc_name}: catalogo H5 · best {m_name} ({current_val:+.2f})",
                                 advance=False)
                        with ui.quiet():
                            save_extreme_case_to_master(master_h5, proc_name, m_name, "best_case",
                                                        processor, mic_signals_ready, y_processed,
                                                        target_ref_audio, weights, exp, row_data,
                                                        scene_base_config, mird_room_dims,
                                                        audio_dtln_alone, y_post_dtln)

                    if is_worst:
                        leaderboard[proc_name][m_name]["worst_val"] = current_val
                        ui.stage(f"{proc_name}: catalogo H5 · worst {m_name} ({current_val:+.2f})",
                                 advance=False)
                        with ui.quiet():
                            save_extreme_case_to_master(master_h5, proc_name, m_name, "worst_case",
                                                        processor, mic_signals_ready, y_processed,
                                                        target_ref_audio, weights, exp, row_data,
                                                        scene_base_config, mird_room_dims,
                                                        audio_dtln_alone, y_post_dtln)

        ui.end_experiment()

    ui.close()
    print(f"\n=== BATCH COMPLETED IN {(time.time() - start_total_time)/60:.2f} MINUTES ===")
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
        # RLS back-end for the fixed-point path:
        #   'cov' = covariance form R^-1 (default; hard cliff, needs 24 bit)
        #   'qrd' = inverse-QRD square-root form (PSD-by-construction, stable ~16 bit)
        'wpe_backend': 'cov',

        # --- Block-online (Opcion B) sub-parametros (solo se usan si wpe_method='block') ---
        # Estos son escalares (constantes en toda la corrida). Para barrerlos habria que
        # promoverlos a ejes de grilla como wpe_taps/wpe_delay. OJO: el warmup del block
        # (~L frames = wpe_block_L*shift muestras) debe caer DENTRO de eval_start_s
        # (=min(5s, 0.3*duration)); con L=512 son ~4.2s < 4.5s -> ok con duration>=15.
        'wpe_block_L': 512,          # ventana trailing de estadistica [frames]
        'wpe_block_shift':2,       # refresco de G [frames]
        'wpe_block_iters': 2,        # iteraciones tipo-offline por re-solve (solo mode='resolve')
        'wpe_block_reg': 1e-6,       # carga diagonal relativa (Cholesky)
        'wpe_block_solver': 'cholesky',   # 'cholesky'/'lu'=solve LAPACK rapido; 'cholesky_explicit'=fidelidad FPGA
        # mode: 'resolve' (exacto, respeta wpe_block_iters) | 'sliding' (MUCHO mas
        # rapido para block_shift chico; equivale a iters=1 con potencia global).
        # Recomendado 'sliding' para barridos de block_shift chico.
        'wpe_block_mode': 'resolve',

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

        'isir_db': [-5,0,10],
        'mismatch_gain': [0],
        'mismatch_phase': [0],
        'use_wpe': [  False],

        # Back-end del WPE como eje de grilla: corre 'online' (RLS) y 'block'
        # (Opcion B, Cholesky) en la MISMA corrida para compararlos. Si se omite,
        # cae al escalar de base_config ('wpe_method', default 'online').
        'wpe_method': ['online'],

        # Sub-parametros del block como ejes barreables (solo afectan celdas 'block';
        # en 'online' se colapsan al default de base_config y se deduplican). Ej. para
        # barrer el largo de ventana y buscar el minimo que entra en memoria interna:
        #   'wpe_block_L': [512, 256, 128, 96, 64],
        #   'wpe_block_shift': [20],
        # Si se omiten, caen a los escalares de base_config (wpe_block_L/shift).

        # WPE hyper-parameter sweep (NEW grid axes).
        # If omitted, run_mird_grid_search falls back to the scalar values in
        # base_config (wpe_taps / wpe_delay), preserving old A/B/C behaviour.
        'wpe_taps': [7],
        'wpe_delay': [ 2],

        'error_angle_deg': [0.0],
        'error_distance_m': [0.0]
    }

    processors_dict = {
       # "DS" :DS(),
        "NM_MVDR" : NM_MVDR(min_loading =1e-9, alpha =0.99),

        "NM_MVDR_DSM_FB_4" : NM_MVDR_DSM_FB(mode="fb", win_type='rect', synth='hann',
               sharpen_exp=4.0, smooth=0.5, alpha=0.99),

        "NM_MVDR_DSM_FB_8" : NM_MVDR_DSM_FB(mode="fb", win_type='rect', synth='hann',
               sharpen_exp=8.0, smooth=0.5, alpha=0.99),
        "NM_MVDR_PF_05" :NM_MVDR_PF(min_loading =1e-9, alpha =0.99, smooth = 0.5),
        "NM_MVDR_DSM_BLIND_PF_05": NM_MVDR_DSM_BLIND(win_type='rect', synth='hann', sharpen_exp=8, causal=True, smooth = 0.5, alpha = 0.99),




    } 


    df_final = run_mird_grid_search(
        grid_params=param_grid,
        dataset_provider=provider,
        processors=processors_dict,
        scene_base_config=base_config,
        output_dir="tests/dataset_out/retroo",
        interpreter_1=interpreter_1,
        interpreter_2=interpreter_2
    )

    print("\n[Preview] MIRD Benchmark Test Results:")

    cols_to_show = [
        "processor", "rt60", "target_angle", "interf_configs", "use_wpe",
        "isir_db", "wpe_taps", "wpe_delay", "Delta_tot_PESQ_early", "Delta_tot_SIR_early"
    ]

    cols_exist = [c for c in cols_to_show if c in df_final.columns]
    print(df_final[cols_exist].to_string(index=False))