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

# Import DeepFilterNet
from df.enhance import enhance, init_df

def apply_deepfilter_post(model, df_state, audio_mono, fs, blend_alpha=0.95):
    """
    Applies DeepFilterNet as a post-processing stage with tunable cancellation via Wet/Dry blending.
    
    Args:
        model: Loaded DeepFilterNet model.
        df_state: DeepFilterNet state configuration.
        audio_mono: 1D numpy array of the audio signal (e.g., MVDR output).
        fs: Sampling frequency of the input audio.
        blend_alpha: Wet/Dry mix ratio (0.0 = only MVDR, 1.0 = Max DeepFilterNet suppression).
    """
    # Convert input to PyTorch tensor format required by DFNet [Channels, Time]
    input_tensor = torch.tensor(audio_mono, dtype=torch.float32).unsqueeze(0)

    # Resample if the simulation sample rate differs from the model's native rate
    if fs != df_state.sr():
        input_tensor = torchaudio.functional.resample(
            input_tensor, orig_freq=fs, new_freq=df_state.sr()
        )

    # Process without calculating gradients to save memory
    with torch.no_grad():
        # Standard DeepFilterNet enhance call (without atten_lim)
        enhanced_tensor = enhance(model, df_state, input_tensor)

    # Resample the enhanced audio back to the original sample rate
    if fs != df_state.sr():
        enhanced_tensor = torchaudio.functional.resample(
            enhanced_tensor, orig_freq=df_state.sr(), new_freq=fs
        )

    # Extract the raw NumPy array
    processed_audio = enhanced_tensor.squeeze(0).numpy()

    # Apply Wet/Dry blending to restore naturalness
    if blend_alpha < 1.0:
        # Mix the network output (Wet) with the original input (Dry)
        processed_audio = (blend_alpha * processed_audio) + ((1.0 - blend_alpha) * audio_mono)

    return processed_audio

def SPP_MVDR_recursive(X_stft, fs, array_geometry, source_pos, beta=1e-3, min_loading=1e-6, save_weights=False):
    # Forgetting factor for covariance matrix smoothing
    lamda = 0.99
    K, T, M = X_stft.shape  
    frecs = np.linspace(0, fs/2, K)

    # Get steering vectors, expected shape (K, M)
    sv = compute_rtf_steering_vector(frecs, source_pos, array_geometry, ref_mic_idx=0, mode="near_field", squeeze=True)
    
    # Initialize output complex STFT matrices
    Y_stft = np.zeros((K, T), dtype=np.complex128)
    Y_spp_stft = np.zeros((K, T), dtype=np.complex128) 
    
    # Initialize noise covariance matrix R_nn
    R_nn = np.tile(np.eye(M, dtype=np.complex128) * 1e-6, (K, 1, 1))
    
    # Initialize noisy signal covariance matrix R_xx using the steering vector
    phi_s = 1e-3 
    R_ss_init = phi_s * np.einsum("fm,fn->fmn", sv, sv.conj())
    R_xx = R_ss_init + R_nn

    # Initialize the inverse of R_nn for the first frame's SPP calculation
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

        # --- 2. Map Gamma to Spatial Presence Probability (SPP) ---
        P = 1.0 / (1.0 + np.exp(-spp_slope * (gamma - gamma_th)))
        P = np.clip(P, 0.05, 0.95)
        P_expand = P[:, np.newaxis, np.newaxis]

        Y_spp_stft[:, m] = P * X_frame[:, 0]

        # --- 3. Update Covariance Matrices ---
        R_instant = np.einsum("fm,fn->fmn", X_frame, X_frame.conj())
        
        R_xx = lamda * R_xx + (1 - lamda) * P_expand * R_instant
        R_nn = lamda * R_nn + (1 - lamda) * (1 - P_expand) * R_instant

        # --- 4. Compute MVDR Weights with Dynamic Diagonal Loading ---
        tr_R = np.real(np.trace(R_nn, axis1=1, axis2=2))
        adaptive_load = beta * (tr_R[:, None, None] / M)
        
        loading = np.maximum(adaptive_load, min_loading)
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
    FS = 48000
    M1, M2 = 8, 1          
    M = M1 * M2
    speed_of_sound = 343.0 
    iSIR_dB = 0
    
    print("=== INTEGRATION TEST: PIPELINE WITH DEEPFILTERNET POST-PROCESSING ===")
    
    # Initialize DeepFilterNet globally once to save memory and time
    print(" -> [Neural Network] Loading global DeepFilterNet model...")
    df_model, df_state, _ = init_df() 
    
    output_folder = "tests/data/spp_mvdr_dfnet_pipeline_output"
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

    print(" -> Initializing acoustic scene...")
    acoustic_scene = SimAcoustic(mic_coords, array_mismatch=0.0, duration=15, fs=FS)
    acoustic_scene.set_source(r"tools\data\signals\p002_emo_adoration_sentences.wav", gain=1, position=source_pos_2d)
    #acoustic_scene.set_interference(r"tools\data\signals\p011_emo_anger_sentences.wav", gain=1, position=interf_pos1.reshape(1,3))
    acoustic_scene.set_interference(r"tools\data\signals\hairdryer_07_SH_MKH800.wav", gain=1, position=interf_pos1.reshape(1,3))
    
    # -------------------------------------------------------------------
    # PHASE 1: FREE FIELD SIMULATION (Anechoic)
    # -------------------------------------------------------------------
    cache_ff_path = os.path.join(output_folder, "cache_free_field.npz")
    
    if os.path.exists(cache_ff_path):
        print("\n--- PHASE 1: LOADING FREE FIELD SIMULATION FROM CACHE ---")
        cache_data = np.load(cache_ff_path)
        free_field_input = cache_data['input']
        vad_oracle_ff = cache_data['vad']
    else:
        print("\n--- PHASE 1: COMPUTING FREE FIELD SIMULATION ---")
        free_field_input, vad_oracle_ff = acoustic_scene.free_field(iSIR_dB=iSIR_dB, normalize=True, mode="ideal", VAD=True)
        np.savez(cache_ff_path, input=free_field_input, vad=vad_oracle_ff)
        
    print(" -> Applying SPP-guided MVDR...")
    output_ff_mvdr, output_ff_spp = apply_mvdr_stft_bridge(free_field_input, vad_oracle_ff, mic_coords, source_pos_2d, FS)
    
    print(" -> Applying DeepFilterNet Post-Processing...")
    output_ff_dfnet = apply_deepfilter_post(df_model, df_state, output_ff_mvdr, FS)
    
    print(" -> Applying DeepFilterNet on Raw Input (No Beamformer)...")
    output_ff_dfnet_only = apply_deepfilter_post(df_model, df_state, free_field_input[0], FS)
    
    # Output saves
    save_wav("1A_FF_input_mic0.wav", FS, free_field_input[0], output_folder)
    save_wav("1B_FF_SPP_mask.wav", FS, normalize_signal(output_ff_spp), output_folder)
    save_wav("1C_FF_MVDR_output.wav", FS, normalize_signal(output_ff_mvdr), output_folder)
    save_wav("1D_FF_DeepFilter_Post_BF.wav", FS, normalize_signal(output_ff_dfnet), output_folder)
    save_wav("1E_FF_DeepFilter_Only.wav", FS, normalize_signal(output_ff_dfnet_only), output_folder)

    # -------------------------------------------------------------------
    # PHASE 2: ROOM SIMULATION (Reverberant)
    # -------------------------------------------------------------------
    cache_room_path = os.path.join(output_folder, "cache_room.npz")
    
    if os.path.exists(cache_room_path):
        print("\n--- PHASE 2: LOADING ROOM SIMULATION FROM CACHE ---")
        cache_data = np.load(cache_room_path)
        room_input = cache_data['input']
        vad_oracle_room = cache_data['vad']
    else:
        print("\n--- PHASE 2: COMPUTING ROOM SIMULATION ---")
        room_dimensions = np.array([4.0, 5.0, 2.5])
        room_sim_dic = acoustic_scene.get_eval_scene(
            room_dimensions=room_dimensions, desire_RT=0.5, iSIR_dB=iSIR_dB, mode="ideal"
        )
        room_input = room_sim_dic["mic_signals"]
        vad_oracle_room = room_sim_dic["VAD"]
        np.savez(cache_room_path, input=room_input, vad=vad_oracle_room)
    
    print(" -> Applying SPP-guided MVDR...")
    output_rm_mvdr, output_rm_spp = apply_mvdr_stft_bridge(room_input, vad_oracle_room, mic_coords, source_pos_2d, FS)
    
    print(" -> Applying DeepFilterNet Post-Processing...")
    output_rm_dfnet = apply_deepfilter_post(df_model, df_state, output_rm_mvdr, FS)
    
    print(" -> Applying DeepFilterNet on Raw Input (No Beamformer)...")
    output_rm_dfnet_only = apply_deepfilter_post(df_model, df_state, room_input[0], FS)
    
    # Output saves
    save_wav("2A_ROOM_input_mic0.wav", FS, room_input[0], output_folder)
    save_wav("2B_ROOM_SPP_mask.wav", FS, normalize_signal(output_rm_spp), output_folder)
    save_wav("2C_ROOM_MVDR_output.wav", FS, normalize_signal(output_rm_mvdr), output_folder)
    save_wav("2D_ROOM_DeepFilter_Post_BF.wav", FS, normalize_signal(output_rm_dfnet), output_folder)
    save_wav("2E_ROOM_DeepFilter_Only.wav", FS, normalize_signal(output_rm_dfnet_only), output_folder)

    # -------------------------------------------------------------------
    # PHASE 3: WPE DEREVERBERATION + MVDR + DEEPFILTERNET
    # -------------------------------------------------------------------
    print("\n--- PHASE 3: WPE + MVDR + DFNET PIPELINE ---")
    print(" -> Applying Online WPE Dereverberation...")
    wpe_output = process_wpe_online(room_input, delay=3, taps=7, stft_size=1024, stft_shift=128)
    
    print(" -> Applying SPP-guided MVDR...")
    output_wpe_mvdr, output_wpe_spp = apply_mvdr_stft_bridge(wpe_output, vad_oracle_room, mic_coords, source_pos_2d, FS)
    
    print(" -> Applying DeepFilterNet Post-Processing...")
    output_wpe_dfnet = apply_deepfilter_post(df_model, df_state, output_wpe_mvdr, FS)
    
    print(" -> Applying DeepFilterNet on Raw WPE Input (No Beamformer)...")
    output_wpe_dfnet_only = apply_deepfilter_post(df_model, df_state, wpe_output[0], FS)
    
    # Output saves
    save_wav("3A_WPE_ROOM_input_mic0.wav", FS, wpe_output[0], output_folder)
    save_wav("3B_WPE_ROOM_SPP_mask.wav", FS, normalize_signal(output_wpe_spp), output_folder)
    save_wav("3C_WPE_ROOM_MVDR_output.wav", FS, normalize_signal(output_wpe_mvdr), output_folder)
    save_wav("3D_WPE_ROOM_DeepFilter_Post_BF.wav", FS, normalize_signal(output_wpe_dfnet), output_folder)
    save_wav("3E_WPE_ROOM_DeepFilter_Only.wav", FS, normalize_signal(output_wpe_dfnet_only), output_folder)

    print("\n -> Pipeline completed successfully.")