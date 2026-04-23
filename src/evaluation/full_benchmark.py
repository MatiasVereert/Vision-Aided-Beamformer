import os
import h5py
import time
import numpy as np
import pandas as pd
import scipy.signal as sig
import hashlib
import pickle
import noisereduce as nr


# Import the Numba function and steering vector helper from your module
from beamforming.MPDRxWPE.MPDRxWPE import MPDRxWPE_numba
from beamforming.MPDRxWPE.MPDRxWPE import MPDRxWPE_numba_scaled
from beamforming.signal_model import compute_rtf_steering_vector
from evaluation.metrics import evaluate_full_pipeline
from propagation.simulate_acoustics_v1 import SimAcoustic

from evaluation.polar_plots import precompute_quantized_spatial_response, subsample_weights
from beamforming.MWF.onlineMWF import online_mwf_numba




class MPDR_WPE_Scaled_Exp_Processor:
    """
    Object-oriented wrapper for the MPDR-WPE bilinear framework.
    Implements Scaled Diagonal Loading (Trace-based) and an 
    Exponential Forgetting Factor to accelerate initial convergence.
    """
    def __init__(self, T_init=62, L=20, Delta=4, alpha_steady=0.994, 
                 alpha_init=0.90, tau=20.0, beta=1e-2, p_min=1e-10, 
                 nperseg=1024, noverlap=768):
        # Initialization and algorithm parameters
        self.T_init = T_init
        self.L = L
        self.Delta = Delta
        self.alpha_steady = alpha_steady
        self.alpha_init = alpha_init
        self.tau = tau
        self.beta = beta
        self.p_min = p_min
        
        # STFT parameters
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nperseg

    def process(self, mic_signals: np.ndarray, scene_config: dict) -> tuple:
        # Extract simulation context
        fs = scene_config['fs']
        source_pos = scene_config['source_pos'].reshape(1, 3)
        mic_coords = scene_config['mic_coords']
        
        # Transform Time to Frequency domain (STFT)
        freqs, times, X_stft = sig.stft(
            mic_signals, fs=fs, window='hamming', 
            nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft
        )
        
        # Transpose array to match Numba function expectations (K, T, M) 
        # and force contiguous memory layout for maximum speed
        X_stft_mpdr = np.transpose(X_stft, (1, 2, 0))
        X_stft_mpdr = np.ascontiguousarray(X_stft_mpdr, dtype=np.complex128)
        
        # Compute Near-Field Steering Vector for the target source
        sv = compute_rtf_steering_vector(
            freqs, source_pos, mic_coords, 
            ref_mic_idx=0, mode="near_field", squeeze=True
        )
        
        # Execute core beamforming algorithm with exponential alpha
        # NOTE: Ensure MPDRxWPE_numba_scaled is imported at the top of evaluate.py
        X_hat_stft, weights = MPDRxWPE_numba_scaled(
            X_stft_mpdr, 
            sv, 
            T_init=self.T_init,
            alpha_steady=self.alpha_steady,
            alpha_init=self.alpha_init,
            tau=self.tau,
            L=self.L, 
            Delta=self.Delta, 
            beta=self.beta,
            p_min=self.p_min,
            save_weights=True
        )
        
        # Transform Frequency back to Time domain (ISTFT)
        _, y_time = sig.istft(
            X_hat_stft, fs=fs, window='hamming', 
            nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft
        )
        
        return y_time, weights

class MPDR_WPE_Scaled_Exp_Processor:
    """
    Object-oriented wrapper for the MPDR-WPE bilinear framework.
    Implements Scaled Diagonal Loading (Trace-based) and an 
    Exponential Forgetting Factor to accelerate initial convergence.
    """
    def __init__(self, T_init=62, L=20, Delta=4, alpha_steady=0.994, 
                 alpha_init=0.90, tau=20.0, beta=1e-2, p_min=1e-10, 
                 nperseg=1024, noverlap=768):
        # Initialization and algorithm parameters
        self.T_init = T_init
        self.L = L
        self.Delta = Delta
        self.alpha_steady = alpha_steady
        self.alpha_init = alpha_init
        self.tau = tau
        self.beta = beta
        self.p_min = p_min
        
        # STFT parameters
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nperseg

    def process(self, mic_signals: np.ndarray, scene_config: dict) -> tuple:
        # Extract simulation context
        fs = scene_config['fs']
        source_pos = scene_config['source_pos'].reshape(1, 3)
        mic_coords = scene_config['mic_coords']
        
        # Transform Time to Frequency domain (STFT)
        freqs, times, X_stft = sig.stft(
            mic_signals, fs=fs, window='hamming', 
            nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft
        )
        
        # Transpose array to match Numba function expectations (K, T, M) 
        # and force contiguous memory layout for maximum speed
        X_stft_mpdr = np.transpose(X_stft, (1, 2, 0))
        X_stft_mpdr = np.ascontiguousarray(X_stft_mpdr, dtype=np.complex128)
        
        # Compute Near-Field Steering Vector for the target source
        sv = compute_rtf_steering_vector(
            freqs, source_pos, mic_coords, 
            ref_mic_idx=0, mode="near_field", squeeze=True
        )
        
        # Execute core beamforming algorithm with exponential alpha
        # NOTE: Ensure MPDRxWPE_numba_scaled is imported at the top of evaluate.py
        X_hat_stft, weights = MPDRxWPE_numba_scaled(
            X_stft_mpdr, 
            sv, 
            T_init=self.T_init,
            alpha_steady=self.alpha_steady,
            alpha_init=self.alpha_init,
            tau=self.tau,
            L=self.L, 
            Delta=self.Delta, 
            beta=self.beta,
            p_min=self.p_min,
            save_weights=True
        )
        
        # Transform Frequency back to Time domain (ISTFT)
        _, y_time = sig.istft(
            X_hat_stft, fs=fs, window='hamming', 
            nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft
        )
        
        return y_time, weights

class MPDR_WPE_Scaled_NR_Processor:
    """
    Object-oriented wrapper for the MPDR-WPE bilinear framework.
    Includes Scaled Diagonal Loading and a post-filtering stage using
    the noisereduce library to tackle residual diffuse reverberation.
    """
    def __init__(self, T_init=62, L=20, Delta=4, alpha=0.994, beta=1e-2, 
                 p_min=1e-10, nperseg=1024, noverlap=768, nr_prop_decrease=0.8):
        # MPDR-WPE scaled parameters
        self.T_init = T_init
        self.L = L
        self.Delta = Delta
        self.alpha = alpha
        self.beta = beta
        self.p_min = p_min
        
        # STFT parameters
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nperseg
        
        # Post-filter parameters
        self.nr_prop_decrease = nr_prop_decrease

    def process(self, mic_signals: np.ndarray, scene_config: dict) -> tuple:
        # 1. Extract simulation context
        fs = scene_config['fs']
        source_pos = scene_config['source_pos'].reshape(1, 3)
        mic_coords = scene_config['mic_coords']
        
        # 2. Time to Frequency domain (STFT)
        freqs, times, X_stft = sig.stft(
            mic_signals, fs=fs, window='hamming', 
            nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft
        )
        
        # 3. Transpose to match Numba function expectations (K, T, M)
        X_stft_mpdr = np.transpose(X_stft, (1, 2, 0))
        X_stft_mpdr = np.ascontiguousarray(X_stft_mpdr, dtype=np.complex128)
        
        # 4. Compute Steering Vector for the target source
        sv = compute_rtf_steering_vector(
            freqs, source_pos, mic_coords, 
            ref_mic_idx=0, mode="near_field", squeeze=True
        )
        
        # 5. Execute core bilinear algorithm
        # En MPDR_WPE_Scaled_NR_Processor.process
        X_hat_stft, weights = MPDRxWPE_numba_scaled(
            X_stft_mpdr, 
            sv, 
            T_init=self.T_init,
            alpha_steady=self.alpha,  # CAMBIO AQUÍ: de alpha a alpha_steady
            alpha_init=self.alpha,    # Agregado para cumplir con la firma
            tau=20.0,                 # Agregado (o pasalo desde el init)
            L=self.L, 
            Delta=self.Delta, 
            beta=self.beta,
            p_min=self.p_min,
            save_weights=True
        )
                
        # 6. Frequency to Time domain (ISTFT)
        _, y_time_intermediate = sig.istft(
            X_hat_stft, fs=fs, window='hamming', 
            nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft
        )
        
        # 7. Post-filtering stage (Non-linear diffuse noise reduction)
        # We apply the spectral gating on the 1D output of the beamformer
        y_time_final = nr.reduce_noise(
            y=y_time_intermediate, 
            sr=fs, 
            prop_decrease=self.nr_prop_decrease,
            stationary=False # False allows tracking dynamic residual reverb
        )
        
        return y_time_final, weights

class MPDR_WPE_Scaled_Processor:
    """
    Object-oriented wrapper for the MPDR-WPE bilinear framework.
    Implements Scaled Diagonal Loading (Trace-based) and Energy Initialization.
    """
    def __init__(self, T_init=62, L=20, Delta=4, alpha=0.994, beta=1e-2, p_min=1e-10, nperseg=1024, noverlap=768):
        self.T_init = T_init  # Number of frames to estimate initial energy (e.g., 62 for ~1 sec at 16kHz)
        self.L = L
        self.Delta = Delta
        self.alpha = alpha
        self.beta = beta      # Trace scaling factor for diagonal loading
        self.p_min = p_min    # Minimum energy floor
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nperseg

    def process(self, mic_signals: np.ndarray, scene_config: dict) -> tuple:
        # Extract simulation context
        fs = scene_config['fs']
        source_pos = scene_config['source_pos'].reshape(1, 3)
        mic_coords = scene_config['mic_coords']
        
        # Time to Frequency domain (STFT)
        freqs, times, X_stft = sig.stft(
            mic_signals, fs=fs, window='hamming', 
            nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft
        )
        
        # Transpose to match Numba function expectations (K, T, M) and force contiguous memory
        X_stft_mpdr = np.transpose(X_stft, (1, 2, 0))
        X_stft_mpdr = np.ascontiguousarray(X_stft_mpdr, dtype=np.complex128)
        
        # Compute Steering Vector for the target source
        sv = compute_rtf_steering_vector(
            freqs, source_pos, mic_coords, 
            ref_mic_idx=0, mode="near_field", squeeze=True
        )
        
        # NOTE: Make sure you import MPDRxWPE_numba_scaled at the top of your file
        # Execute core beamforming algorithm with trace-scaled diagonal loading
        # En MPDR_WPE_Scaled_NR_Processor.process
        X_hat_stft, weights = MPDRxWPE_numba_scaled(
            X_stft_mpdr, 
            sv, 
            T_init=self.T_init,
            alpha_steady=self.alpha,  # CAMBIO AQUÍ: de alpha a alpha_steady
            alpha_init=self.alpha,    # Agregado para cumplir con la firma
            tau=20.0,                 # Agregado (o pasalo desde el init)
            L=self.L, 
            Delta=self.Delta, 
            beta=self.beta,
            p_min=self.p_min,
            save_weights=True
        )
        
        # Frequency to Time domain (ISTFT)
        _, y_time = sig.istft(
            X_hat_stft, fs=fs, window='hamming', 
            nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft
        )
        
        return y_time, weights


class MWF_Processor:
    """
    Object-oriented wrapper for the Online Multichannel Wiener Filter (MWF).
    """
    def __init__(self, alpha=0.95, diag_load = 1e-3, nperseg=1024, noverlap=768):
        self.alpha = alpha
        self.diag_load = diag_load
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nperseg

    def process(self, mic_signals: np.ndarray, scene_config: dict) -> np.ndarray:
        # 1. Extract simulation context
        fs = scene_config['fs']
        source_pos = scene_config['source_pos'].reshape(1, 3)
        mic_coords = scene_config['mic_coords']
        
        # 2. Time to Frequency domain (STFT)
        freqs, times, X_stft = sig.stft(
            mic_signals, fs=fs, window='hamming', 
            nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft
        )
        
        # 3. Transpose to match Numba function expectations (K, T, M)
        X_stft_mwf = np.transpose(X_stft, (1, 2, 0))
        X_stft_mwf = np.ascontiguousarray(X_stft_mwf, dtype=np.complex128)
        
        # 4. Compute Steering Vector for the target source
        sv = compute_rtf_steering_vector(
            freqs, source_pos, mic_coords, 
            ref_mic_idx=0, mode="near_field", squeeze=True
        )
        
        # 5. Execute core MWF algorithm
        X_hat_stft, weights = online_mwf_numba(
            X_stft_mwf, sv, alpha=self.alpha, 
        )
        
        # 6. Frequency to Time domain (ISTFT)
        _, y_time = sig.istft(
            X_hat_stft, fs=fs, window='hamming', 
            nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft
        )
        
        return y_time, weights

    
class DS_Processor:
    """
    Standard Delay-and-Sum beamformer for baseline comparison.
    Produces a completely static spatial filter.
    """
    def __init__(self, nperseg=1024, noverlap=768):
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nperseg

    def process(self, mic_signals: np.ndarray, scene_config: dict) -> np.ndarray:
        fs = scene_config['fs']
        source_pos = scene_config['source_pos'].reshape(1, 3)
        mic_coords = scene_config['mic_coords']
        
        # STFT
        freqs, times, X_stft = sig.stft(
            mic_signals, fs=fs, window='hamming', 
            nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft
        )
        
        X_stft_ds = np.transpose(X_stft, (1, 2, 0))
        K, T, M = X_stft_ds.shape
        
        # Compute exact steering vector
        sv = compute_rtf_steering_vector(
            freqs, source_pos, mic_coords, 
            ref_mic_idx=0, mode="near_field", squeeze=True
        )
        
        # Initialize output tensors
        weights = np.zeros((K, T, M), dtype=np.complex128)
        X_hat_stft = np.zeros((K, T), dtype=np.complex128)
        
        # Apply static Delay-and-Sum weights
        for k in range(K):
            # The static weight is simply the steering vector divided by M
            w_ds = sv[k] / M
            for t in range(T):
                weights[k, t, :] = w_ds
                # Complex conjugate dot product
                X_hat_stft[k, t] = np.vdot(w_ds, X_stft_ds[k, t, :])
                
        # ISTFT
        _, y_time = sig.istft(
            X_hat_stft, fs=fs, window='hamming', 
            nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft
        )
        
        return y_time, weights

    
class MPDR_WPE_Processor:
    """
    Object-oriented wrapper to interface the time-domain benchmark 
    with the frequency-domain Numba MPDR-WPE implementation.
    """
    def __init__(self, L=12, Delta=2, alpha=0.994, nperseg=1024, noverlap=768, diag_load=1e-6):
        self.L = L
        self.Delta = Delta
        self.alpha = alpha
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nperseg
        self.diag_loading = diag_load
        
    def process(self, mic_signals: np.ndarray, scene_config: dict) -> np.ndarray:
        # 1. Extract simulation context
        fs = scene_config['fs']
        source_pos = scene_config['source_pos'].reshape(1, 3)
        mic_coords = scene_config['mic_coords']
        
        # 2. Time to Frequency domain (STFT)
        freqs, times, X_stft = sig.stft(
            mic_signals, fs=fs, window='hamming', 
            nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft
        )
        
        # 3. Transpose to match Numba function expectations (K, T, M)
        X_stft_mpdr = np.transpose(X_stft, (1, 2, 0))
        X_stft_mpdr = np.ascontiguousarray(X_stft_mpdr, dtype=np.complex128)
        
        # 4. Compute Steering Vector for the target source
        sv = compute_rtf_steering_vector(
            freqs, source_pos, mic_coords, 
            ref_mic_idx=0, mode="near_field", squeeze=True
        )
        
        # 5. Execute core beamforming algorithm
        X_hat_stft, weights = MPDRxWPE_numba(
            X_stft_mpdr, sv, alpha=self.alpha, L=self.L, Delta=self.Delta, save_weights = True, epsilon= self.diag_loading
        )
        
        # 6. Frequency to Time domain (ISTFT)
        _, y_time = sig.istft(
            X_hat_stft, fs=fs, window='hamming', 
            nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft
        )
        
        # Return a 1D array for the metric evaluator
        return y_time, weights
import os
import time
import itertools
import h5py
import pandas as pd
import numpy as np

# Importamos tu clase (ajusta la ruta según tu estructura)
from propagation.simulate_acoustics_v1 import SimAcoustic
# from evaluate import evaluate_full_pipeline, MPDR_WPE_Processor ...

def save_extreme_case_h5(filepath, mic_signals, target_anechoic, weights, config, metrics):
    """
    Guarda los tensores pesados en HDF5 solo cuando se rompe un récord (Mejor/Peor).
    """
    with h5py.File(filepath, 'w') as f:
        # Metadatos
        grp_meta = f.create_group("metadata")
        for key, value in config.items():
            grp_meta.attrs[key] = value
            
        for key, value in metrics.items():
            grp_meta.attrs[f"metric_{key}"] = value

        # Audios
        grp_audio = f.create_group("audio")
        grp_audio.create_dataset("mic_signals", data=mic_signals, compression="gzip")
        grp_audio.create_dataset("target_anechoic", data=target_anechoic, compression="gzip")
        
        # Pesos
        grp_weights = f.create_group("weights")
        grp_weights.create_dataset("beamformer_weights", data=weights, compression="gzip", compression_opts=4)

def run_grid_search(grid_params, processors, scene_base_config, output_dir="results/"):
    """
    Orquestador inteligente. Ejecuta todas las combinaciones evitando recalcular RIRs
    si la geometría o el RT60 no han cambiado respecto a la iteración anterior.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Generar todas las combinaciones usando itertools
    keys, values = zip(*grid_params.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # 2. ORDENAR los experimentos estratégicamente.
    # Queremos que los parámetros pesados cambien lo menos posible en el bucle.
    # Orden de peso computacional: mismatch (requiere instanciar clase) > rt60 (RIRs) > isir_db (Mezcla ligera)
    experiments.sort(key=lambda x: (x['mismatch'], x['rt60'], x['isir_db']))
    
    print(f"[*] Total de experimentos a ejecutar: {len(experiments)} por cada procesador.")

    # 3. Inicializar el Leaderboard (Rastreador de mejores y peores casos)
    # Usaremos Delta_PESQ como métrica de decisión, pero puedes cambiarla.
    leaderboard = {
        proc_name: {"best_score": -float('inf'), "worst_score": float('inf')} 
        for proc_name in processors.keys()
    }
    
    all_metrics_results = []
    
    # Variables de estado para evitar recalcular
    current_mismatch = None
    current_rt60 = None
    acoustic_scene = None

    start_total_time = time.time()

    # 4. Bucle Principal del Orquestador
    for i, exp in enumerate(experiments):
        print(f"\n--- Iteración {i+1}/{len(experiments)} | Config: {exp} ---")
        
        # --- NODO 1: Geometría e Inicialización (Nivel de clase) ---
        if exp['mismatch'] != current_mismatch:
            print(" -> [RAMA 1] Cambio de geometría detectado. Instanciando SimAcoustic...")
            acoustic_scene = SimAcoustic(
                array_geometry=scene_base_config['mic_coords'], 
                array_mismatch=exp['mismatch'], 
                duration=scene_base_config['duration'], 
                fs=scene_base_config['fs']
            )
            # Cargar audios
            acoustic_scene.set_source(scene_base_config['source_path'], gain=1.0, position=scene_base_config['source_pos'])
            for interf in scene_base_config['interferences']:
                acoustic_scene.set_interference(interf['path'], gain=1.0, position=interf['pos'])
            
            current_mismatch = exp['mismatch']
            current_rt60 = None # Forzamos recálculo de RIRs si cambió la geometría

        # --- NODO 2: Física Acústica y Convolución (Cálculo Pesado) ---
        if exp['rt60'] != current_rt60:
            print(" -> [RAMA 2] Cambio de RT60 detectado. Calculando RIRs y Convolucionando...")
            acoustic_scene.compute_rirs(
                room_dimensions=scene_base_config['room_dims'], 
                desire_RT=exp['rt60'], 
                ray_tracing=True
            )
            acoustic_scene.convolve_signals()
            current_rt60 = exp['rt60']
        else:
            print(" -> [CACHÉ] Reutilizando RIRs y señales convolucionadas de la iteración anterior.")

        # --- NODO 3: Mezcla y Normalización (Cálculo Ligero e instantáneo) ---
        # Esto se ejecuta SIEMPRE, pero es puramente multiplicativo gracias a tu refactorización
        scene_data = acoustic_scene.mix_and_normalize(iSIR_dB=exp['isir_db'])
        
        mic_signals = scene_data["mic_signals"]
        target_anechoic = scene_data["target_early"] + scene_data["target_late"] # O target_anechoic puro si prefieres
        
        # Evaluar línea base (Micrófono de referencia sin procesar)
        # base_metrics = evaluate_full_pipeline(target_anechoic[0], mic_signals[0], ...)
        base_pesq = 1.5 # MOCK: Asume que aquí llamas a tu función de PESQ para el baseline

        # --- NODO 4: Procesamiento de Señales ---
        for proc_name, processor in processors.items():
            print(f"   -> Procesando con: {proc_name}")
            
            t0 = time.time()
            y_processed, weights = processor.process(mic_signals, scene_base_config)
            proc_time = time.time() - t0
            
            # Evaluar salida
            # proc_metrics = evaluate_full_pipeline(target_anechoic[0], y_processed, ...)
            proc_pesq = 2.8 # MOCK: Resultado de tu evaluación real
            delta_pesq = proc_pesq - base_pesq
            
            # Guardar resultados tabulares
            row_data = {
                "processor": proc_name,
                "rt60": exp['rt60'],
                "isir_db": exp['isir_db'],
                "mismatch": exp['mismatch'],
                "exec_time_s": proc_time,
                "base_PESQ": base_pesq,
                "proc_PESQ": proc_pesq,
                "Delta_PESQ": delta_pesq
            }
            all_metrics_results.append(row_data)

            # --- LÓGICA DE TOP-K / BOTTOM-K (Checkpointing Condicional) ---
            if delta_pesq > leaderboard[proc_name]["best_score"]:
                leaderboard[proc_name]["best_score"] = delta_pesq
                print(f"   [!] NUEVO MEJOR para {proc_name} (Delta PESQ: {delta_pesq:.3f}). Escribiendo H5...")
                filepath = os.path.join(output_dir, f"{proc_name}_BEST.h5")
                save_extreme_case_h5(filepath, mic_signals, target_anechoic, weights, exp, row_data)

            if delta_pesq < leaderboard[proc_name]["worst_score"]:
                leaderboard[proc_name]["worst_score"] = delta_pesq
                print(f"   [!] NUEVO PEOR para {proc_name} (Delta PESQ: {delta_pesq:.3f}). Escribiendo H5...")
                filepath = os.path.join(output_dir, f"{proc_name}_WORST.h5")
                save_extreme_case_h5(filepath, mic_signals, target_anechoic, weights, exp, row_data)

    # 5. Finalización y guardado tabular
    print(f"\n=== BARRIDO COMPLETADO EN {(time.time() - start_total_time)/60:.2f} MINUTOS ===")
    df_results = pd.DataFrame(all_metrics_results)
    
    # Exportar a Parquet (Recomendado para análisis estadístico)
    parquet_path = os.path.join(output_dir, "benchmark_metrics.parquet")
    df_results.to_parquet(parquet_path, engine="pyarrow")
    print(f"[*] Dataset estadístico guardado en: {parquet_path}")
    
    return df_results

# ==========================================
# CÓMO LLAMAR AL ORQUESTADOR
# ==========================================
if __name__ == "__main__":
    
    # Configuración estática (no cambia en el grid)
    base_config = {
        'fs': 16000,
        'duration': 5,
        'room_dims': np.array([5.0, 4.0, 2.5]),
        'mic_coords': np.array([[2.5, 2.0, 1.0], [2.54, 2.0, 1.0]]), # Ejemplo
        'source_path': "tools/data/signals/FA01_09.wav",
        'source_pos': np.array([[1.0, 2.0, 1.0]]),
        'interferences': [
            {'path': "tools/data/signals/MC15_03.wav", 'pos': np.array([[4.0, 1.0, 1.0]])}
        ]
    }

    # TU ESPACIO DE BÚSQUEDA (El Grid)
    param_grid = {
        'rt60': [0.2, 0.4, 0.6],           # Tiempos de reverberación
        'isir_db': [-5, 0, 5, 10],         # Relación Señal/Interferencia
        'mismatch': [0.0, 1e-3, 5e-3]      # Errores de posicionamiento del sensor
    }


    processors_dict = {
        


        "MPDR_WPE_Scaled": MPDR_WPE_Scaled_Processor(
            T_init=62,       # Approx 1 second for 16kHz, 1024 window, 768 overlap
            L=20, 
            Delta=4, 
            alpha=0.994, 
            beta=1e-2,       # Ajusta este parámetro (1e-2 o 1e-3 suelen ser buenos puntos de partida)
            p_min=1e-6,
            nperseg=1024, 
            noverlap=768
        ),


        #"MWF_Test": MWF_Processor(alpha=0.95, diag_load=1e-3, nperseg=1024, noverlap=768)
    }
    
    # MOCK PARA QUE CORRA AHORA (Reemplazar con dict real arriba)
    class MockProcessor:
        def process(self, *args): return np.zeros(10), np.zeros((10,10))
    processors_dict = {"Mock_MPDR": MockProcessor()}

    # Ejecutar!
    df_final = run_grid_search(param_grid, processors_dict, base_config, output_dir="tests/dataset_out")
    print("\nVista previa del dataset:\n", df_final.head())