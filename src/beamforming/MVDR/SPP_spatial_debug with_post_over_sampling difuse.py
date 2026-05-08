import os
import numpy as np 
import scipy.signal as signal
import torch
import torchaudio

# Assuming the imports work correctly in your local environment
from beamforming.signal_model import compute_rtf_steering_vector
from utils.audio import normalize_signal, save_wav
from propagation.simulate_acoustics import SimAcoustic
from dereverberation.nara_wrappers import process_wpe_online

# Import standard DeepFilterNet enhance function
from df.enhance import enhance, init_df

def apply_deepfilter_post_resampled(model, df_state, audio_mono, fs_in=16000, blend_alpha=0.90):
    """
    Applies DeepFilterNet as a post-processing stage.
    Dynamically upsamples the input to the network's native sample rate, 
    processes the signal, and downsamples it back to the original input rate.
    
    Args:
        model: Loaded DeepFilterNet model.
        df_state: DeepFilterNet state configuration.
        audio_mono: 1D numpy array of the audio signal (e.g., MVDR output).
        fs_in: Sampling frequency of the input audio array.
        blend_alpha: Wet/Dry mix ratio (0.0 = only MVDR, 1.0 = Max DeepFilterNet).
        
    Returns:
        processed_np: 1D numpy array at the original fs_in sample rate.
    """
    # Get the native sample rate required by the DeepFilterNet model
    fs_net = df_state.sr() 
    
    # Ensure input is a float32 PyTorch tensor with shape [Channels, Time]
    input_tensor = torch.tensor(audio_mono, dtype=torch.float32).unsqueeze(0)
    
    # 1. Upsample from original rate to network's native rate
    if fs_in != fs_net:
        resampler_up = torchaudio.transforms.Resample(orig_freq=fs_in, new_freq=fs_net)
        audio_net_fs = resampler_up(input_tensor)
    else:
        audio_net_fs = input_tensor

    # 2. Process with DeepFilterNet
    with torch.no_grad():
        enhanced_tensor = enhance(model, df_state, audio_net_fs)
        
    # 3. Downsample back to the original input rate
    if fs_in != fs_net:
        resampler_down = torchaudio.transforms.Resample(orig_freq=fs_net, new_freq=fs_in)
        enhanced_tensor_out = resampler_down(enhanced_tensor)
    else:
        enhanced_tensor_out = enhanced_tensor

    # 4. Extract the raw NumPy arrays for blending
    processed_np = enhanced_tensor_out.squeeze(0).numpy()
    dry_np = audio_mono  # The original input is already at fs_in
    
    # Match lengths in case resampling introduced a 1-sample difference due to rounding
    min_length = min(processed_np.shape[0], dry_np.shape[0])
    processed_np = processed_np[:min_length]
    dry_np = dry_np[:min_length]

    # 5. Apply Wet/Dry blending at the original sample rate
    if blend_alpha < 1.0:
        processed_np = (blend_alpha * processed_np) + ((1.0 - blend_alpha) * dry_np)

    return processed_np

import numpy as np

import numpy as np

def SPP_MVDR_recursive(X_stft, fs, array_geometry, source_pos, beta=1e-3, min_loading=1e-6, save_weights=False, c=343.0):
    """
    SPP-guided MVDR Beamformer enriched with a Diffuse Field Covariance Model 
    and a Coherence-Based Prior for enhanced reverberation suppression.
    """
    # Forgetting factor for covariance matrix smoothing
    lamda = 0.99
    K, T, M = X_stft.shape  
    frecs = np.linspace(0, fs/2, K)

    # --- ENHANCEMENT 1: DIFFUSE FIELD COVARIANCE MODEL ---
    # Calculate pairwise distances between all microphones [M, M]
    diff = array_geometry[:, np.newaxis, :] - array_geometry[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    
    # Calculate Diffuse Field Spatial Coherence Matrix (sinc function)
    # np.sinc(x) computes sin(pi*x)/(pi*x). We need sin(2*pi*f*d/c)/(2*pi*f*d/c), 
    # so we pass x = 2 * f * d / c. Adding a tiny epsilon to frecs to avoid 0/0 edge cases
    frecs_safe = np.maximum(frecs, 1e-6)
    x = 2.0 * frecs_safe[:, np.newaxis, np.newaxis] * distances[np.newaxis, :, :] / c
    R_diffuse = np.sinc(x) # Shape: [K, M, M]
    
    # Pre-extract diffuse coherence for the primary pair (mic 0 and 1) for the SPP Prior
    if M > 1:
        C_diff_01 = R_diffuse[:, 0, 1]

    # Get steering vectors, expected shape (K, M)
    sv = compute_rtf_steering_vector(frecs, source_pos, array_geometry, ref_mic_idx=0, mode="near_field", squeeze=True)
    
    # Initialize output complex STFT matrices
    Y_stft = np.zeros((K, T), dtype=np.complex128)
    Y_spp_stft = np.zeros((K, T), dtype=np.complex128) 
    
    # Initialize noise covariance matrix R_nn using the Diffuse Field model
    R_nn = R_diffuse.astype(np.complex128) * 1e-6
    
    # Initialize noisy signal covariance matrix R_xx using the steering vector
    phi_s = 1e-3 
    R_ss_init = phi_s * np.einsum("fm,fn->fmn", sv, sv.conj())
    R_xx = R_ss_init + R_nn

    # CORRECTED: The diagonal load MUST be an Identity matrix to push eigenvalues up (Sensor noise)
    initial_diag_load = np.tile(np.eye(M, dtype=np.complex128) * min_loading, (K, 1, 1))
    R_nn_inv = np.linalg.inv(R_nn + initial_diag_load)

    # SPP Hyperparameters
    gamma_th = 8  
    spp_slope = 5 

    weights_rec = np.zeros((K, T, M), dtype=np.complex128)

    for m in range(T):
        X_frame = X_stft[:, m, :]

        # --- 1. Calculate A Posteriori Spatial SNR (gamma) ---
        R_nn_inv_x = np.einsum("fmn,fn->fm", R_nn_inv, X_frame)
        num_complex = np.einsum("fm,fm->f", sv.conj(), R_nn_inv_x)
        num = np.abs(num_complex)**2
        
        R_nn_inv_v = np.einsum("fmn,fn->fm", R_nn_inv, sv)
        den_complex = np.einsum("fm,fm->f", sv.conj(), R_nn_inv_v)
        den = np.real(den_complex) 
        
        gamma = num / (den + 1e-10)

        # --- ENHANCEMENT 2: COHERENCE-BASED PRIOR SPP ---
        if M > 1:
            # Calculate instantaneous real coherence between Mic 0 and Mic 1
            cross_power = np.real(X_frame[:, 0] * X_frame[:, 1].conj())
            auto_power = np.abs(X_frame[:, 0]) * np.abs(X_frame[:, 1]) + 1e-10
            C_inst = cross_power / auto_power
            
            # Distance from diffuse field: if instantaneous coherence is close to diffuse 
            # coherence, delta is near 0 (highly reverberant). If it's larger, it's directive.
            delta = np.maximum(C_inst - C_diff_01, 0.0)
            
            # Map distance to a prior probability (0.1 to 1.0)
            prior = 1.0 - np.exp(-3.0 * delta)
            prior = np.clip(prior, 0.1, 1.0)
        else:
            prior = 1.0

        # Map Gamma to Spatial Presence Probability (SPP)
        P_base = 1.0 / (1.0 + np.exp(-spp_slope * (gamma - gamma_th)))
        
        # Modulate the base SPP with the Coherence Prior
        P = P_base * prior
        P = np.clip(P, 0.05, 0.95)
        P_expand = P[:, np.newaxis, np.newaxis]

        Y_spp_stft[:, m] = P * X_frame[:, 0]

        # --- 3. Update Covariance Matrices ---
        R_instant = np.einsum("fm,fn->fmn", X_frame, X_frame.conj())
        
        R_xx = lamda * R_xx + (1 - lamda) * P_expand * R_instant
        R_nn = lamda * R_nn + (1 - lamda) * (1 - P_expand) * R_instant

        # --- 4. Compute MVDR Weights with Diagonal Loading ---
        tr_R = np.real(np.trace(R_nn, axis1=1, axis2=2))
        adaptive_load = beta * (tr_R[:, None, None] / M)
        
        loading = np.maximum(adaptive_load, min_loading)
        
        # CORRECTED: Diagonal loading MUST use np.eye to prevent singular matrices
        R_nn_stable = R_nn + np.eye(M)[None, :, :] * loading
        R_nn_inv = np.linalg.inv(R_nn_stable)

        weights_nom = np.einsum("fmn,fn->fm", R_nn_inv, sv)
        weights_den = np.einsum("fm,fm->f", sv.conj(), weights_nom)
        
        weights = weights_nom / (weights_den[:, np.newaxis] + 1e-10)
        weights_rec[:, m, :] = weights
        
        # --- 5. Apply Filter ---
        Y_stft[:, m] = np.einsum("fm,fm->f", weights.conj(), X_frame)

    if save_weights:
        return Y_stft, Y_spp_stft, weights_rec
    else:
        return Y_stft, Y_spp_stft
    

def apply_mvdr_stft_bridge(time_domain_input, vad_oracle, mic_coords, source_pos_2d, fs, length_fft=512, hop_length_fft=256):
    """
    Helper function to wrap the STFT -> MVDR -> ISTFT process.
    """
    freqs, times, Zxx = signal.stft(
        time_domain_input, 
        fs=fs, 
        nperseg=length_fft, 
        noverlap=length_fft - hop_length_fft
    )
    
    X_stft = np.transpose(Zxx, (1, 2, 0))
    vad_padded = np.pad(vad_oracle, (0, length_fft + hop_length_fft), mode='constant')

    Y_stft, Y_spp_stft = SPP_MVDR_recursive(
        X_stft=X_stft, 
        fs=fs, 
        array_geometry=mic_coords, 
        source_pos=source_pos_2d, 
    )
    
    _, y_time = signal.istft(
        Y_stft, 
        fs=fs, 
        nperseg=length_fft, 
        noverlap=length_fft - hop_length_fft
    )

    _, y_spp_time = signal.istft(
        Y_spp_stft, 
        fs=fs, 
        nperseg=length_fft, 
        noverlap=length_fft - hop_length_fft
    )
    
    original_length = time_domain_input.shape[1]
    return y_time[:original_length], y_spp_time[:original_length]


if __name__ == "__main__":
    # We set the global processing rate for the simulation and beamformer to 16kHz
    FS = 16000
    M1, M2 = 8, 1          
    M = M1 * M2
    speed_of_sound = 343.0 
    iSIR_dB = 0
    
    print("=== INTEGRATION TEST: 16k PIPELINE WITH 48k DEEPFILTERNET POST-PROCESSING ===")
    
    # Initialize DeepFilterNet globally. 
    # Calling init_df() without arguments loads the latest DeepFilterNet3 model by default.
    print(" -> [Neural Network] Loading global DeepFilterNet3 model...")
    df_model, df_state, _ = init_df() 
    
    output_folder = "tests/data/spp_mvdr_dfnet_online_output"
    os.makedirs(output_folder, exist_ok=True)
    
    max_length = 0.30
    if M > 1:
        base = 2.0
        indices = np.arange(M)
        x_norm = (base**indices - 1) / (base**(M - 1) - 1)
        x = x_norm * max_length
    else:
        x = np.array([0.0])

    mic_coords = np.column_stack([x, np.zeros(M), np.zeros(M)])
    array_center = np.array([1.25, 2.0, 1.25])
    mic_coords = mic_coords - np.mean(mic_coords, axis=0) + array_center
    
    r = 1.0 
    ang_target = np.deg2rad(130)
    ang_interf = np.deg2rad(50)
    ang_interf2 = np.deg2rad(-50)
    
    source_pos = array_center + np.array([r * np.cos(ang_target), r * np.sin(ang_target), 0.0])
    interf_pos1 = array_center + np.array([r * np.cos(ang_interf), r * np.sin(ang_interf), 0.0])
    interf_pos2 = array_center + np.array([r * np.cos(ang_interf2), r * np.sin(ang_interf2), 0.0])
    source_pos_2d = source_pos.reshape(1, 3)

    print(" -> Initializing 16kHz acoustic scene...")
    acoustic_scene = SimAcoustic(mic_coords, array_mismatch=0.0, duration=15, fs=FS)
    acoustic_scene.set_source(r"tools\data\signals\p002_emo_adoration_sentences.wav", gain=1, position=source_pos_2d)
    acoustic_scene.set_interference(r"tools\data\signals\hairdryer_07_SH_MKH800.wav", gain=1, position=interf_pos1.reshape(1,3))
    
    # -------------------------------------------------------------------
    # PHASE 1: FREE FIELD SIMULATION (Anechoic)
    # -------------------------------------------------------------------
    cache_ff_path = os.path.join(output_folder, "cache_free_field_16k.npz")
    
    if os.path.exists(cache_ff_path):
        print("\n--- PHASE 1: LOADING FREE FIELD SIMULATION FROM CACHE ---")
        cache_data = np.load(cache_ff_path)
        free_field_input = cache_data['input']
        vad_oracle_ff = cache_data['vad']
    else:
        print("\n--- PHASE 1: COMPUTING 16kHz FREE FIELD SIMULATION ---")
        free_field_input, vad_oracle_ff = acoustic_scene.free_field(iSIR_dB=iSIR_dB, normalize=True, mode="ideal", VAD=True)
        np.savez(cache_ff_path, input=free_field_input, vad=vad_oracle_ff)
        
    print(" -> Applying SPP-guided MVDR (at 16kHz)...")
    output_ff_mvdr, output_ff_spp = apply_mvdr_stft_bridge(free_field_input, vad_oracle_ff, mic_coords, source_pos_2d, FS)
    
    print(" -> Applying DeepFilterNet (Upsamples for processing, returns at input FS)...")
    output_ff_dfnet = apply_deepfilter_post_resampled(df_model, df_state, output_ff_mvdr, FS)
    
    # Output saves - Everything is strictly maintained at FS
    save_wav("1A_FF_input_mic0_16k.wav", FS, free_field_input[0], output_folder)
    save_wav("1C_FF_MVDR_output_16k.wav", FS, normalize_signal(output_ff_mvdr), output_folder)
    save_wav("1D_FF_DeepFilter_Post_16k.wav", FS, normalize_signal(output_ff_dfnet), output_folder)

    # -------------------------------------------------------------------
    # PHASE 2: ROOM SIMULATION (Reverberant)
    # -------------------------------------------------------------------
    cache_room_path = os.path.join(output_folder, "cache_room_16k.npz")
    
    if os.path.exists(cache_room_path):
        print("\n--- PHASE 2: LOADING ROOM SIMULATION FROM CACHE ---")
        cache_data = np.load(cache_room_path)
        room_input = cache_data['input']
        vad_oracle_room = cache_data['vad']
    else:
        print("\n--- PHASE 2: COMPUTING 16kHz ROOM SIMULATION ---")
        room_dimensions = np.array([4.0, 5.0, 2.5])
        room_sim_dic = acoustic_scene.get_eval_scene(
            room_dimensions=room_dimensions, desire_RT=0.5, iSIR_dB=iSIR_dB, mode="ideal"
        )
        room_input = room_sim_dic["mic_signals"]
        vad_oracle_room = room_sim_dic["VAD"]
        np.savez(cache_room_path, input=room_input, vad=vad_oracle_room)
    
    print(" -> Applying SPP-guided MVDR (at 16kHz)...")
    output_rm_mvdr, output_rm_spp = apply_mvdr_stft_bridge(room_input, vad_oracle_room, mic_coords, source_pos_2d, FS)
    
    print(" -> Applying DeepFilterNet (Upsamples for processing, returns at input FS)...")
    output_ff_dfnet = apply_deepfilter_post_resampled(df_model, df_state, output_ff_mvdr, FS)
    
    # Output saves - Everything is strictly maintained at FS
    save_wav("1A_FF_input_mic0_16k.wav", FS, free_field_input[0], output_folder)
    save_wav("1C_FF_MVDR_output_16k.wav", FS, normalize_signal(output_ff_mvdr), output_folder)
    save_wav("1D_FF_DeepFilter_Post_16k.wav", FS, normalize_signal(output_ff_dfnet), output_folder)

    print("\n -> Pipeline completed successfully.")

    # -------------------------------------------------------------------
    # PHASE 3: WPE + MVDR + DFNET PIPELINE
    # -------------------------------------------------------------------
    print("\n--- PHASE 3: WPE + MVDR + DFNET PIPELINE ---")
    print(" -> Applying Online WPE Dereverberation (at 16kHz)...")
    wpe_output = process_wpe_online(room_input, delay=3, taps=7, stft_size=1024, stft_shift=128)
    
    print(" -> Applying SPP-guided MVDR (at 16kHz)...")
    mvdr_wpe, _ = apply_mvdr_stft_bridge(wpe_output, vad_oracle_room, mic_coords, source_pos_2d, FS)
    
    print(" -> Applying DeepFilterNet3 (Resampled internally)...")
    dfnet_wpe = apply_deepfilter_post_resampled(df_model, df_state, mvdr_wpe, FS)
    
    # Save final results
    save_wav("3_WPE_16k.wav", FS, normalize_signal(wpe_output[0]), output_folder)
    save_wav("3_WPE_MVDR_16k.wav", FS, normalize_signal(mvdr_wpe), output_folder)
    save_wav("3_WPE_MVDR_DFNet_16k.wav", FS, normalize_signal(dfnet_wpe), output_folder)

    print("\n -> Full integrated pipeline completed successfully.")