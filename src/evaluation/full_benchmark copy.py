import os
import time
import itertools
import h5py
import pandas as pd
import numpy as np
from beamforming.array.geometry import generate_log_array_coords, generate_source_and_interferences

# Importamos tu clase (ajusta la ruta según tu estructura)
from propagation.simulate_acoustics_v1 import SimAcoustic

def save_extreme_case_h5(filepath, mic_signals, target_anechoic, weights, config, metrics, room_dims):
    """Guarda los tensores pesados en HDF5 solo cuando se rompe un récord."""
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
    
    # 1. Generar combinaciones
    keys, values = zip(*grid_params.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # 2. ORDENAR estratégicamente. 
    # Orden de re-cálculo: rt60 > M > N_interferences > mismatch > isir_db
    experiments.sort(key=lambda x: (x['rt60'], x['M'], x['N_interferences'], x['mismatch'], x['isir_db']))
    
    print(f"[*] Total de experimentos a ejecutar: {len(experiments)} por procesador.")

    # 3. Inicializar Leaderboard
    leaderboard = {
        proc_name: {"best_score": -float('inf'), "worst_score": float('inf')} 
        for proc_name in processors.keys()
    }
    
    all_metrics_results = []
    
    # Variables de estado para evitar recalcular RIRs innecesariamente
    current_rt60 = None
    current_M = None
    current_N_int = None
    current_mismatch = None
    acoustic_scene = None

    start_total_time = time.time()

    # 4. Bucle Principal del Orquestador
    for i, exp in enumerate(experiments):
        print(f"\n--- Iteración {i+1}/{len(experiments)} | Config: {exp} ---")
        
        # --- NODOS 1 y 2: Geometría y Física ---
        # Si cambia sala, micrófonos, CANTIDAD DE INTERFERENCIAS, o error posicional -> Recalcular
        if (exp['rt60'] != current_rt60 or 
            exp['M'] != current_M or 
            exp['N_interferences'] != current_N_int or 
            exp['mismatch'] != current_mismatch):
            
            print(" -> [RAMA 1] Cambio de Geometría/Escena detectado. Reconstruyendo...")
            current_room_dims = room_profiles[exp['rt60']]
            room_center = current_room_dims / 2.0
            
            # Fabrica 1: Arreglo de Micrófonos
            mic_coords = generate_log_array_coords(
                M=exp['M'],
                d_min=scene_base_config['d_min'],
                d_max=scene_base_config['d_max'],
                room_dims=current_room_dims
            )
            
            # Instanciar el Simulador
            acoustic_scene = SimAcoustic(
                array_geometry=mic_coords, 
                array_mismatch=exp['mismatch'], 
                duration=scene_base_config['duration'], 
                fs=scene_base_config['fs']
            )
            
            # Fabrica 2: Posiciones de Fuentes
            source_pos, interferences_pos = generate_source_and_interferences(
                N_interferences=exp['N_interferences'],
                radius_source=scene_base_config['radius_source'],
                radius_interf=scene_base_config['radius_interf'],
                delta_ang_deg=scene_base_config['delta_ang_deg'],
                array_center=room_center
            )
            
            # Cargar Target Source
            acoustic_scene.set_source(scene_base_config['source_path'], gain=1.0, position=source_pos.reshape(1,3))
            
            # Cargar Interferencias (Mapeando posiciones a la lista de paths fijos)
            for idx in range(exp['N_interferences']):
                # Usamos módulo (%) para que, si pides más interferencias que audios disponibles, 
                # vuelva a usar los de la lista (aunque lo ideal es tener suficientes audios)
                path_idx = idx % len(scene_base_config['interf_paths'])
                interf_path = scene_base_config['interf_paths'][path_idx]
                
                acoustic_scene.set_interference(
                    audio_path=interf_path, 
                    gain=1.0, 
                    position=interferences_pos[idx].reshape(1,3)
                )
            
            # Computar Física (RIRs y Convolución)
            acoustic_scene.compute_rirs(room_dimensions=current_room_dims, desire_RT=exp['rt60'], ray_tracing=True)
            acoustic_scene.convolve_signals()
            
            # Actualizar variables de estado
            current_rt60 = exp['rt60']
            current_M = exp['M']
            current_N_int = exp['N_interferences']
            current_mismatch = exp['mismatch']
            
        else:
            current_room_dims = room_profiles[exp['rt60']]
            print(" -> [CACHÉ] Reutilizando RIRs y señales convolucionadas.")

        # --- NODO 3: Mezcla y Normalización ---
        scene_data = acoustic_scene.mix_and_normalize(iSIR_dB=exp['isir_db'])
        
        mic_signals = scene_data["mic_signals"]
        target_anechoic = scene_data["target_early"] + scene_data["target_late"]
        
        base_pesq = 1.5 # MOCK Baseline

        # --- NODO 4: Procesamiento de Señales ---
        for proc_name, processor in processors.items():
            print(f"   -> Procesando con: {proc_name}")
            
            t0 = time.time()
            y_processed, weights = processor.process(mic_signals, scene_base_config)
            proc_time = time.time() - t0
            
            proc_pesq = 2.8 # MOCK Metric
            delta_pesq = proc_pesq - base_pesq
            
            row_data = {
                "processor": proc_name,
                "rt60": exp['rt60'],
                "M": exp['M'],
                "N_interferences": exp['N_interferences'], # Nueva variable guardada
                "isir_db": exp['isir_db'],
                "mismatch": exp['mismatch'],
                "exec_time_s": proc_time,
                "base_PESQ": base_pesq,
                "proc_PESQ": proc_pesq,
                "Delta_PESQ": delta_pesq
            }
            all_metrics_results.append(row_data)

            # Lógica Extremos
            if delta_pesq > leaderboard[proc_name]["best_score"]:
                leaderboard[proc_name]["best_score"] = delta_pesq
                filepath = os.path.join(output_dir, f"{proc_name}_BEST.h5")
                save_extreme_case_h5(filepath, mic_signals, target_anechoic, weights, exp, row_data, current_room_dims)

            if delta_pesq < leaderboard[proc_name]["worst_score"]:
                leaderboard[proc_name]["worst_score"] = delta_pesq
                filepath = os.path.join(output_dir, f"{proc_name}_WORST.h5")
                save_extreme_case_h5(filepath, mic_signals, target_anechoic, weights, exp, row_data, current_room_dims)

    # 5. Finalización
    print(f"\n=== BARRIDO COMPLETADO EN {(time.time() - start_total_time)/60:.2f} MINUTOS ===")
    df_results = pd.DataFrame(all_metrics_results)
    
    parquet_path = os.path.join(output_dir, "benchmark_metrics.parquet")
    df_results.to_parquet(parquet_path, engine="pyarrow")
    
    return df_results


if __name__ == "__main__":
    
    # 1. PERFILES DE SALA
    ROOM_PROFILES = {
        0.0: np.array([3.0, 4.0, 2.5]),
        0.3: np.array([3.0, 4.0, 2.5]),
        0.6: np.array([5.0, 6.0, 2.8]),
        0.9: np.array([6.0, 7.0, 3.0]) 
    }

    # 2. CONFIGURACIÓN ESTÁTICA
    base_config = {
        'fs': 16000,
        'duration': 5,
        'd_min': 0.04,
        'd_max': 0.20,
        # Parámetros para la función fábrica de fuentes:
        'radius_source': 1.0,    # Distancia de la voz objetivo al centro (m)
        'radius_interf': 1.5,    # Distancia de las interferencias al centro (m)
        'delta_ang_deg': 30.0,   # Ángulo de separación base entre interferencias
        'source_path': "tools/data/signals/FA01_09.wav",
        # Lista fija de audios de interferencia
        'interf_paths': [
            "tools/data/signals/MC15_03.wav",
            "tools/data/signals/MF31_03.wav",
            "tools/data/signals/FC22_05.wav",
            "tools/data/signals/noise_babble.wav" 
        ]
    }

    # 3. EL ESPACIO DE BÚSQUEDA (El Grid)
    param_grid = {
        'rt60': list(ROOM_PROFILES.keys()), 
        'M': [4, 6, 8, 12],                
        'N_interferences': [1, 2, 3],      # Nueva variable añadida al iterador
        'isir_db': [-5, 0, 5, 10],         
        'mismatch': [0.0, 1e-3, 5e-3]      
    }

    class MockProcessor:
        def process(self, *args): return np.zeros(10), np.zeros((10,10))
    processors_dict = {"Mock_MPDR": MockProcessor()}

    df_final = run_grid_search(
        grid_params=param_grid, 
        room_profiles=ROOM_PROFILES, 
        processors=processors_dict, 
        scene_base_config=base_config, 
        output_dir="tests/dataset_out"
    )