import os
import h5py
import time
import numpy as np
import pandas as pd
import scipy.signal as sig
import hashlib
import pickle
ç

# Import the Numba function and steering vector helper from your module
from beamforming.MPDRxWPE.MPDRxWPE import MPDRxWPE_numba

from beamforming.MPDRxWPE.MPDRxWPE import MPDRxWPE_numba_scaled
from beamforming.signal_model import compute_rtf_steering_vector
from evaluation.metrics import evaluate_full_pipeline
from propagation.simulate_acoustics import SimAcoustic

from evaluation.polar_plots import precompute_quantized_spatial_response, subsample_weights
from beamforming.MWF.onlineMWF import online_mwf_numba

import numpy as np
from beamforming.MVDR.base import MVDR_recursive
class MVDR:
    """
    Object-oriented wrapper for the recursive MVDR beamformer.
    It retrieves the oracle VAD mask from the scene configuration.
    """
    def __init__(self, nperseg=1024, noverlap=768, min_loading=1e-6):
        # Store STFT parameters
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nperseg
        self.min_loading = min_loading

    def process(self, mic_signals: np.ndarray, scene_config: dict) -> tuple:
        # 1. Extract simulation context and VAD
        fs = scene_config['fs']
        source_pos = scene_config['source_pos'].reshape(1, 3)
        mic_coords = scene_config['mic_coords']
        vad = scene_config.get('vad', None)
        
        # Saftey check to ensure VAD was successfully passed
        if vad is None:
            raise ValueError("MVDR requires a valid VAD array in scene_config['vad'].")

        # 2. Transform Time to Frequency domain (STFT)
        freqs, times, X_stft = sig.stft(
            mic_signals, fs=fs, window='hamming', 
            nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft
        )
        
        # 3. Transpose to match the expected shape (K, T, M) for the recursive MVDR
        X_stft_mvdr = np.transpose(X_stft, (1, 2, 0))
        X_stft_mvdr = np.ascontiguousarray(X_stft_mvdr, dtype=np.complex128)
        
        # 4. Execute the core recursive MVDR function
        # Note: Depending on your MVDR_recursive implementation, you might need 
        # to adjust if it returns only the output or also the weights.
        Y_stft, weights = MVDR_recursive(
            X_stft=X_stft_mvdr, 
            vad=vad, 
            fs=fs, 
            array_geometry=mic_coords, 
            source_pos=source_pos, 
            length_fft=self.nperseg, 
            hop_length_fft=self.nperseg - self.noverlap, 
            min_loading=self.min_loading,
            save_weights = True
        )
        
        # 5. Transform Frequency back to Time domain (ISTFT)
        _, y_time = sig.istft(
            Y_stft, fs=fs, window='hamming', 
            nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft
        )
        
        return y_time, weights

    
class MPDR_WPE:
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
    # 1. SIMULATION STAGE (WITH CACHING)
    # ==========================================
    # Create a stable hash of the configuration to use as a fingerprint
    config_bytes = pickle.dumps(scene_config)
    config_hash = hashlib.md5(config_bytes).hexdigest()
    
    # Define cache directory and specific file for this configuration
    cache_dir = os.path.join(output_dir, ".sim_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"scene_{config_hash}.pkl")

    if os.path.exists(cache_file):
        print(f" -> Loading CACHED acoustic scene (Hash: {config_hash[:8]})...")
        with open(cache_file, 'rb') as f:
            scene_data = pickle.load(f)
    else:
        print(" -> Simulating NEW acoustic scene...")
        acoustic_scene = SimAcoustic(
            array_geometry=scene_config['mic_coords'], 
            array_mismatch=scene_config.get('mismatch', 1e-3), 
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
            mode="ideal" 
        )
        
        # Save the simulated data to cache for future runs
        print(" -> Saving acoustic scene to cache...")
        with open(cache_file, 'wb') as f:
            pickle.dump(scene_data, f)

    # Extract signals from the data dictionary (whether cached or freshly generated)
    mic_signals = scene_data["mic_signals"]
    target_early = scene_data["target_early"][0] 
    target_late = scene_data["target_late"][0] 
    target_anechoic  =scene_data["target_anechoic"][0] 

    #HARDCODE
    target_anechoic =  target_early + target_late

    # --- NEW: Extract VAD and inject it into the scene_config dictionary ---
    # This allows any processor requiring VAD (like MVDR) to access it 
    # without changing the generic .process() signature.
    vad = scene_data.get("VAD", None)
    scene_config['vad'] = vad
    
    # Sum all early interferences at the reference microphone (index 0)
    # to compute the true spatial SIR against the combined interference field.
    interferences = scene_data["interference_early"]
    if isinstance(interferences, list) or interferences.ndim == 3:
        interf_early = np.sum([interf[0] for interf in interferences], axis=0)
    else:
        interf_early = interferences[0]
    
    
    print(" -> Evaluating baseline metrics...")
    baseline_metrics = evaluate_full_pipeline(
        target_anechoic, 
        mic_signals[0], 
        fs, 
        interf_sig=interf_early,
        compute_pesq=True, 
        compute_cd=True,
        inspection_name = "Baseline_Mic0"
    )
    
    os.makedirs(output_dir, exist_ok=True)
    h5_filepath = os.path.join(output_dir, f"{scenario_id}.h5")
    results_summary = []

    with h5py.File(h5_filepath, 'w') as f:
        # Save shared simulation data
        grp_audio = f.create_group("audio")
        grp_audio.create_dataset("mic_signals", data=mic_signals, compression="gzip")
        grp_audio.create_dataset("target_anechoic", data=target_anechoic, compression="gzip")
        
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
                target_anechoic, 
                y_processed, 
                fs, 
                interf_sig=interf_early,
                compute_pesq=True, 
                compute_cd=True,
                inspection_name=f"Processed_{proc_name}"
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
    
    # Broadband Concentric Circular Array (12 microphones)
    # Designed to balance high-frequency spatial aliasing (up to 8 kHz) 
    # and low-frequency array aperture (down to 100 Hz)
    M = 12
    array_center = np.array([1.25, 2.0, 1.25])
    
    # Ring configuration: [Center, Inner, Middle, Outer]
    # Inner radius (2.1 cm) respects the spatial Nyquist limit for 8 kHz (d <= lambda/2)
    # Outer radius (15 cm) maximizes aperture for low-frequency directivity
    radii = [0.0, 0.021, 0.06, 0.15] 
    mics_per_ring = [1, 3, 4, 4]
    
    coords = []
    for r, n_mics in zip(radii, mics_per_ring):
        if r == 0.0:
            coords.append([0.0, 0.0, 0.0])
        else:
            # Stagger angles between rings to optimize 2D spatial sampling
            angle_offset = np.pi / n_mics if r > 0.021 else 0.0
            angles = np.linspace(0, 2 * np.pi, n_mics, endpoint=False) + angle_offset
            for angle in angles:
                coords.append([r * np.cos(angle), r * np.sin(angle), 0.0])
                
    mic_coords = np.array(coords)
    
    # Shift the generated geometry to the requested room position
    mic_coords = mic_coords + array_center

    scene_config = {
        'fs': 16000,
        'duration': 15,
        'room_dims': np.array([4.0, 5.0, 2.5]),
        'rt60': 0.1,
        'sir_db': 0,
        'mismatch': 1e-20,
        'mic_coords': mic_coords,
        'source_path': "tools/data/signals/FA01_09.wav",
        'source_pos': array_center + np.array([1.0, 0, 0.0]),
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
        
        #"Delay_and_Sum": DS(nperseg=1024, noverlap=768),
        "MPDR_WPE_Test": MPDR_WPE(
            L=20, 
            Delta=4, 
            alpha=0.994, 
            nperseg=1024, 
            noverlap=768,
            diag_load=1e-14)
        ,

        "MVDR_Oracle_VAD": MVDR(
            nperseg=1024, 
            noverlap=768, 
            min_loading=1e-6
        ),
    }
        


        #"MWF_Test": MWF(alpha=0.95, diag_load=1e-3, nperseg=1024, noverlap=768)
    


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
