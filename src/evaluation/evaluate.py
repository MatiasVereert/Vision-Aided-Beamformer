import os
import h5py
import time
import numpy as np
import pandas as pd
import scipy.signal as sig

# Import the Numba function and steering vector helper from your module
from beamforming.MPDRxWPE.MPDRxWPE import MPDRxWPE_numba
from beamforming.signal_model import compute_rtf_steering_vector
from evaluation.metrics import evaluate_full_pipeline
from propagation.simulate_acoustics import SimAcoustic

from evaluation.polar_plots import precompute_quantized_spatial_response, subsample_weights


class DS_Processor:
    """
    Modified Delay-and-Sum beamformer for DASHBOARD TESTING.
    Produces a dynamic spatial filter with a strictly forced rotating null
    to visually verify that the dashboard renders time-varying changes.
    """
    def __init__(self, nperseg=1024, noverlap=768):
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nperseg

    def process(self, mic_signals: np.ndarray, scene_config: dict) -> np.ndarray:
        fs = scene_config['fs']
        source_pos = scene_config['source_pos'].reshape(1, 3)
        mic_coords = scene_config['mic_coords']
        M_mics = mic_coords.shape[0]
        
        # 1. STFT Conversion
        freqs, times, X_stft = sig.stft(
            mic_signals, fs=fs, window='hamming', 
            nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft
        )
        
        X_stft_ds = np.transpose(X_stft, (1, 2, 0))
        K, T, M = X_stft_ds.shape
        
        # 2. Compute exact steering vector for the static target
        sv_target = compute_rtf_steering_vector(
            freqs, source_pos, mic_coords, 
            ref_mic_idx=0, mode="near_field", squeeze=True
        ) 
        
        # 3. Generate rotating null positions spanning a full circle over T frames
        array_center = np.mean(mic_coords, axis=0)
        angles = np.linspace(0, 2 * np.pi, T)
        r_null = 1.5 
        
        null_positions = np.zeros((T, 3))
        null_positions[:, 0] = array_center[0] + r_null * np.cos(angles)
        null_positions[:, 1] = array_center[1] + r_null * np.sin(angles)
        null_positions[:, 2] = array_center[2] 
        
        # 4. Compute steering vectors for the rotating null in a single batch
        # This returns a tensor of shape (K, T, M)
        sv_null = compute_rtf_steering_vector(
            freqs, null_positions, mic_coords,
            ref_mic_idx=0, mode="near_field", squeeze=True
        ) 
        
        weights = np.zeros((K, T, M), dtype=np.complex128)
        X_hat_stft = np.zeros((K, T), dtype=np.complex128)
        
        # 5. Apply orthogonal projection to force the rotating null
        for k in range(K):
            v_targ = sv_target[k] 
            for t in range(T):
                v_n = sv_null[k, t] 
                
                # Gram-Schmidt orthogonalization to enforce a strict null
                # w = v_target - v_null * (<v_null, v_target> / <v_null, v_null>)
                overlap = np.vdot(v_n, v_targ) / np.vdot(v_n, v_n)
                w_t = v_targ - (v_n * overlap)
                
                # Scale down by M to maintain overall magnitude similar to standard DS
                w_t = w_t / M_mics
                
                weights[k, t, :] = w_t
                X_hat_stft[k, t] = np.vdot(w_t, X_stft_ds[k, t, :])
                
        # 6. ISTFT Reconstruction
        _, y_time = sig.istft(
            X_hat_stft, fs=fs, window='hamming', 
            nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft
        )
        
        return y_time, weights


"""
class DS_Processor:

    Standard Delay-and-Sum beamformer for baseline comparison.
    Produces a completely static spatial filter.
    
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
    """
class MPDR_WPE_Processor:
    """
    Object-oriented wrapper to interface the time-domain benchmark 
    with the frequency-domain Numba MPDR-WPE implementation.
    """
    def __init__(self, L=12, Delta=2, alpha=0.994, nperseg=1024, noverlap=768):
        self.L = L
        self.Delta = Delta
        self.alpha = alpha
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
        X_stft_mpdr = np.transpose(X_stft, (1, 2, 0))
        X_stft_mpdr = np.ascontiguousarray(X_stft_mpdr, dtype=np.complex128)
        
        # 4. Compute Steering Vector for the target source
        sv = compute_rtf_steering_vector(
            freqs, source_pos, mic_coords, 
            ref_mic_idx=0, mode="near_field", squeeze=True
        )
        
        # 5. Execute core beamforming algorithm
        X_hat_stft, weights = MPDRxWPE_numba(
            X_stft_mpdr, sv, alpha=self.alpha, L=self.L, Delta=self.Delta, save_weights = True
        )
        
        # 6. Frequency to Time domain (ISTFT)
        _, y_time = sig.istft(
            X_hat_stft, fs=fs, window='hamming', 
            nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft
        )
        
        # Return a 1D array for the metric evaluator
        return y_time, weights

# Make sure to import the optimized functions at the top of your evaluate.py
# from polar_plots import subsample_weights, precompute_quantized_spatial_response

def run_benchmark_scenario(scenario_id: str, 
                           scene_config: dict, 
                           processors: dict, 
                           output_dir: str = "results/") -> list:
    """
    Master benchmarking pipeline.
    Simulates the acoustic scene, evaluates processors, calculates metrics,
    and precomputes lightweight 3D spatial responses for interactive visualization.
    """
    start_time = time.time()
    fs = scene_config.get('fs', 16000)
    
    print(f"--- Starting Benchmark: {scenario_id} ---")
    
    # ==========================================
    # 1. SIMULATION STAGE
    # ==========================================
    print(" -> Simulating acoustic scene...")
    acoustic_scene = SimAcoustic(
        array_geometry=scene_config['mic_coords'], 
        array_mismatch=scene_config.get('mismatch', 1e-12), 
        duration=scene_config['duration'], 
        fs=fs
    )

    acoustic_scene.set_source(scene_config['source_path'], gain=1.0, position=scene_config['source_pos'].reshape(1,3))
    
    for interf in scene_config['interferences']:
        acoustic_scene.set_interference(interf['path'], gain=1.0, position=interf['pos'].reshape(1,3))

    scene_data = acoustic_scene.get_eval_scene(
        room_dimensions=scene_config['room_dims'], 
        desire_RT=scene_config['rt60'], 
        iSIR_dB=scene_config['sir_db'],
        mode="real" 
    )

    mic_signals = scene_data["mic_signals"]
    target_early = scene_data["target_early"][0] 
    interf_early = scene_data["interference_early"][0]
    
    print(" -> Evaluating baseline metrics...")
    baseline_metrics = evaluate_full_pipeline(
        target_early, 
        mic_signals[0], 
        fs, 
        interf_sig=interf_early,
        compute_pesq=True, 
        compute_cd=True
    )
    
    os.makedirs(output_dir, exist_ok=True)
    h5_filepath = os.path.join(output_dir, f"{scenario_id}.h5")
    results_summary = []

    with h5py.File(h5_filepath, 'w') as f:
        # Save shared simulation data
        grp_audio = f.create_group("audio")
        grp_audio.create_dataset("mic_signals", data=mic_signals, compression="gzip")
        grp_audio.create_dataset("target_early", data=target_early, compression="gzip")
        
        grp_meta = f.create_group("metadata")
        grp_meta.attrs["fs"] = fs
        grp_meta.attrs["rt60"] = scene_config['rt60']
        grp_meta.attrs["sir_db"] = scene_config['sir_db']

        grp_geometry = f.create_group("geometry")
        grp_geometry.create_dataset("source_pos", data=scene_config['source_pos'])

        interf_array = np.array([interf['pos'] for interf in scene_config['interferences']])
        grp_geometry.create_dataset("interferences_pos", data=interf_array)
        grp_geometry.create_dataset("mic_coords", data=scene_config['mic_coords'])
        grp_geometry.create_dataset("room_dims", data=scene_config['room_dims'])

        grp_weights = f.create_group("weights")
        grp_results = f.create_group("metrics")
        
        for metric, val in baseline_metrics.items():
            grp_results.attrs[f"baseline_{metric}"] = val
        
        # ==========================================
        # 2. MULTI-PROCESSOR EVALUATION STAGE
        # ==========================================
        for proc_name, processor_obj in processors.items():
            print(f" -> Running processor: {proc_name}...")
            
            proc_start = time.time()
            y_processed, weights_rec = processor_obj.process(mic_signals, scene_config)
            proc_time = time.time() - proc_start
            
            print(f" -> Evaluating metrics for: {proc_name}...")
            proc_metrics = evaluate_full_pipeline(
                target_early, 
                y_processed, 
                fs, 
                interf_sig=interf_early,
                compute_pesq=True, 
                compute_cd=True
            )
            
            grp_audio.create_dataset(f"processed_{proc_name}", data=y_processed, compression="gzip")
            grp_weights.create_dataset(
                f"processed_{proc_name}", 
                data=weights_rec, 
                compression="gzip",
                compression_opts=4
            )

            # --- PRECOMPUTE SPATIAL RESPONSE WITH DOWNSAMPLING ---
            print(f" -> Sub-sampling and precomputing spatial response for {proc_name}...")
            
            # Extract STFT configuration from processor (fallback to common defaults)
            nfft = getattr(processor_obj, 'nfft', 1024)
            nperseg = getattr(processor_obj, 'nperseg', 1024)
            noverlap = getattr(processor_obj, 'noverlap', 768)
            hop_length = nperseg - noverlap
            
            # Calculate full frequency bins
            freqs_full = np.fft.rfftfreq(nfft, d=1.0/fs)
            
            # 1. Sub-sample weights tensor (Frequencies to 1/3 Octave, Time to 24 FPS)
            target_fps = 24
            w_sub, freqs_sub = subsample_weights(
                weights=weights_rec, 
                freqs=freqs_full, 
                fs=fs, 
                hop_length=hop_length, 
                target_fps=target_fps
            )
            
            # 2. Calculate physical distance for the evaluation sphere
            array_center = np.mean(scene_config['mic_coords'], axis=0)
            source_radius = np.linalg.norm(scene_config['source_pos'][0] - array_center)
            
            # 3. Run heavily optimized Numba core
            n_azimuth = 90
            min_db_threshold = -30.0
            
            q_gain, points, max_db_per_frame = precompute_quantized_spatial_response(
                weights=w_sub,
                freqs=freqs_sub,
                mic_pos=scene_config['mic_coords'],
                source_radius=source_radius,
                N_azimuth=n_azimuth,
                min_dB=min_db_threshold
            )

            # 4. Save lightweight data to HDF5
            grp_spatial = f.create_group(f"spatial_{proc_name}")
            grp_spatial.create_dataset("quantized_gain", data=q_gain, compression="gzip", compression_opts=4)
            grp_spatial.create_dataset("points", data=points, compression="gzip")
            grp_spatial.create_dataset("freqs", data=freqs_sub)
            grp_spatial.create_dataset("max_db_per_frame", data=max_db_per_frame)
            
            # Save relevant metadata for the Streamlit dashboard
            grp_spatial.attrs["min_dB"] = min_db_threshold
            grp_spatial.attrs["N_azimuth"] = n_azimuth
            grp_spatial.attrs["target_fps"] = target_fps
            
            # --- METRICS DELTA CALCULATION ---
            delta_metrics = {}
            for metric, val in proc_metrics.items():
                baseline_val = baseline_metrics.get(metric, np.nan)
                
                if baseline_val is not None and not np.isnan(baseline_val) and val is not None and not np.isnan(val):
                    delta_val = val - baseline_val
                else:
                    delta_val = np.nan
                    
                delta_metrics[f"Delta_{metric}"] = delta_val
                grp_results.attrs[f"{proc_name}_{metric}"] = val
                grp_results.attrs[f"{proc_name}_Delta_{metric}"] = delta_val
                
            results_summary.append({
                "scenario": scenario_id,
                "processor": proc_name,
                "exec_time_s": proc_time,
                **proc_metrics,
                **delta_metrics
            })
            
    total_time = time.time() - start_time
    print(f"--- Benchmark {scenario_id} Finished in {total_time:.2f}s ---")
    
    return results_summary

if __name__ == "__main__":
    print("=== INITIALIZING PIPELINE TEST ===")
    
    mic_spacing = 0.04 
    M1, M2 = 3, 3  
    M = M1 * M2
    
    x = np.linspace(0, (M2-1)*mic_spacing, M2)
    y = np.linspace(0, (M1-1)*mic_spacing, M1)
    xv, yv = np.meshgrid(x, y, indexing='xy') 
    
    array_center = np.array([1.25, 2.0, 1.25])
    mic_coords = np.column_stack([xv.flatten(), yv.flatten(), np.zeros(M)])
    mic_coords = mic_coords - np.mean(mic_coords, axis=0) + array_center

    scene_config = {
        'fs': 16000,
        'duration': 15,
        'room_dims': np.array([4.0, 5.0, 2.5]),
        'rt60': 0.5,
        'sir_db': 0,
        'mismatch': 1e-12,
        'mic_coords': mic_coords,
        'source_path': "tools/data/signals/FA01_09.wav",
        'source_pos': array_center + np.array([1.0, 1.0, 0.0]),
        'interferences': [
            {
                'path': "tools/data/signals/MC15_03.wav",
                'pos': array_center + np.array([-1.0, 1.0, 0.0])
            },
            {
                'path': "tools/data/signals/MF31_03.wav",
                'pos': array_center + np.array([0.0, -1.0, 0.5])
            }
        ]
    }

    # Use the Object-Oriented wrapper instead of the raw function
# Run both the adaptive and the static baseline processors
    processors_dict = {
        "Delay_and_Sum": DS_Processor(nperseg=1024, noverlap=768),
        "MPDR_WPE_Test": MPDR_WPE_Processor(L=25, Delta=2, nperseg=1024, noverlap=768)
    }

    output_directory = "tests/data/benchmark_results"
    
    summary = run_benchmark_scenario(
        scenario_id="test_scene_001",
        scene_config=scene_config,
        processors=processors_dict,
        output_dir=output_directory
    )

    print("\n=== PIPELINE EXECUTION COMPLETE ===")
    df_results = pd.DataFrame(summary)
    print(df_results.to_string(index=False))