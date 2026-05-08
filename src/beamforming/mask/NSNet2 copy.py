import os
import numpy as np
import scipy.signal as signal
import onnxruntime as ort

# Assuming these are available from your local environment modules
from beamforming.signal_model import compute_rtf_steering_vector
from beamforming.MWF.SP_SDW_MWF_base import process_wpe_online
from propagation.simulate_acoustics import SimAcoustic
from utils.audio import save_wav, normalize_signal
def Neural_MVDR_recursive(X_stft, fs, array_geometry, source_pos, ort_session, beta=1e-3, min_loading=1e-6, save_weights=False):
    """
    Recursive MVDR using a Neural Network (e.g., NSNet2) for mask estimation
    instead of the statistical Spatial Presence Probability (SPP).
    """
    # Forgetting factor for covariance matrix smoothing
    lamda = 0.99
    K, T, M = X_stft.shape  
    frecs = np.linspace(0, fs/2, K)

    # Get steering vectors, expected shape (K, M)
    sv = compute_rtf_steering_vector(frecs, source_pos, array_geometry, ref_mic_idx=0, mode="near_field", squeeze=True)
    
    # Initialize output complex STFT matrices
    Y_stft = np.zeros((K, T), dtype=np.complex128)
    Y_stft_masked = np.zeros((K, T), dtype=np.complex128)
    
    # Initialize noise covariance matrix R_nn
    R_nn = np.tile(np.eye(M, dtype=np.complex128) * 1e-6, (K, 1, 1))
    
    # Initialize noisy signal covariance matrix R_xx using the steering vector
    phi_s = 1e-3 
    R_ss_init = phi_s * np.einsum("fm,fn->fmn", sv, sv.conj())
    R_xx = R_ss_init + R_nn

    # RTF Estimation Hyperparameters
    alpha_rtf = 0.98         
    spp_threshold_rtf = 0.8  

    weights_rec = np.zeros((K, T, M), dtype=np.complex128)

    onnx_input_name = ort_session.get_inputs()[0].name

    # --- 1. Offline Neural Mask Estimation (Sequence Level) ---
    # Extract magnitude spectrum for the reference mic across ALL frames
    mag_ref_full = np.abs(X_stft[:, :, 0])
    
    # Calculate Log-Power Spectrum (LPS) expected by NSNet2
    # Transpose to match ONNX expected shape: (Time, Freq)
    lps_features = np.log10(mag_ref_full.T ** 2 + 1e-8).astype(np.float32)
    
    # Add batch dimension to create shape (1, T, K)
    onnx_input_full = np.expand_dims(lps_features, axis=0)
    
    # Run the model on the entire sequence at once to preserve GRU/RNN states
    neural_mask_full = ort_session.run(None, {onnx_input_name: onnx_input_full})[0]
    
    # Remove batch dimension to get shape (T, K)
    neural_mask_full = np.squeeze(neural_mask_full)

    for m in range(T):
        # Extract the current frame across all frequencies, shape (K, M)
        X_frame = X_stft[:, m, :]

        # Extract probability P for the current frame and clip
        P = np.clip(neural_mask_full[m, :], 0.05, 0.95)
        P_expand = P[:, np.newaxis, np.newaxis]

        # --- 1.5 Generate Output directly from the Neural Mask ---
        Y_stft_masked[:, m] = P * X_frame[:, 0]

        # --- 2. Update Covariance Matrices ---
        R_instant = np.einsum("fm,fn->fmn", X_frame, X_frame.conj())
        R_xx = lamda * R_xx + (1 - lamda) * P_expand * R_instant
        R_nn = lamda * R_nn + (1 - lamda) * (1 - P_expand) * R_instant

        # --- 3. Estimate and Update RTF Dynamically ---
        R_ss = R_xx - R_nn
        ref_mic = 0
        R_ss_ref_col = R_ss[:, :, ref_mic]
        R_ss_ref_diag = np.maximum(np.real(R_ss[:, ref_mic, ref_mic]), 1e-12)
        rtf_inst = R_ss_ref_col / R_ss_ref_diag[:, np.newaxis]
        
        for f in range(K):
            if P[f] > spp_threshold_rtf:
                sv[f, :] = alpha_rtf * sv[f, :] + (1 - alpha_rtf) * rtf_inst[f, :]
                
        sv = sv / (np.linalg.norm(sv, axis=1, keepdims=True) + 1e-10)

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
        return Y_stft, Y_stft_masked, weights_rec
    else:
        return Y_stft, Y_stft_masked


def apply_mvdr_stft_bridge(time_domain_input, vad_oracle, mic_coords, source_pos_2d, fs, ort_session, length_fft=320, hop_length_fft=160):
    """
    Helper function to wrap the STFT -> Neural MVDR -> ISTFT process.
    """
    # Compute STFT
    freqs, times, Zxx = signal.stft(
        time_domain_input, 
        fs=fs, 
        nperseg=length_fft, 
        noverlap=length_fft - hop_length_fft
    )
    
    # Transpose Zxx from (M, K, T) to (K, T, M)
    X_stft = np.transpose(Zxx, (1, 2, 0))

    # Execute the Recursive Neural MVDR
    Y_stft, Y_stft_masked = Neural_MVDR_recursive(
        X_stft=X_stft, 
        fs=fs, 
        array_geometry=mic_coords, 
        source_pos=source_pos_2d, 
        ort_session=ort_session
    )
    
    # Compute Inverse STFT for the MVDR Output
    _, y_time = signal.istft(
        Y_stft, 
        fs=fs, 
        nperseg=length_fft, 
        noverlap=length_fft - hop_length_fft
    )

    # Compute Inverse STFT for the Neural Mask purely
    _, y_masked_time = signal.istft(
        Y_stft_masked, 
        fs=fs, 
        nperseg=length_fft, 
        noverlap=length_fft - hop_length_fft
    )
    
    # Truncate to original length
    original_length = time_domain_input.shape[1]
    return y_time[:original_length], y_masked_time[:original_length]


if __name__ == "__main__":
    # Basic simulation parameters
    FS = 16000
    M1, M2 = 12, 1          
    M = M1 * M2
    speed_of_sound = 343.0 
    iSIR_dB = 0
    
    # Load the ONNX model (NSNet2 causal baseline)
    onnx_model_path = r"tools\data\redes\nsnet2-20ms-baseline.onnx"
    print(f" -> Loading Neural Mask Model from {onnx_model_path}...")
    try:
        ort_session = ort.InferenceSession(onnx_model_path)
    except Exception as e:
        print(f"Error loading ONNX model. Please download the NSNet2 ONNX file and place it at the specified path.\n{e}")
        exit()
    
    print("=== INTEGRATION TEST: PIPELINE (FREE-FIELD, ROOM, WPE+ROOM) ===")
    
    output_folder = "tests/data/Neural_MVDR_output"
    os.makedirs(output_folder, exist_ok=True)
    
    # Create logarithmic spacing for the microphone array
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
    
    source_pos = array_center + np.array([r * np.cos(ang_target), r * np.sin(ang_target), 0.0])
    interf_pos1 = array_center + np.array([r * np.cos(ang_interf), r * np.sin(ang_interf), 0.0])
    source_pos_2d = source_pos.reshape(1, 3)

    print(" -> Initializing acoustic scene...")
    acoustic_scene = SimAcoustic(mic_coords, array_mismatch=0.0, duration=40, fs=FS)
    acoustic_scene.set_source(r"tools\data\signals\p002_emo_adoration_sentences.wav", gain=1, position=source_pos_2d)
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
        
    save_wav("1_FF_input_mix_mic0.wav", FS, free_field_input[0], output_folder)
    
    print(" -> Applying Neural MVDR...")
    output_ff, masked_ff = apply_mvdr_stft_bridge(free_field_input, vad_oracle_ff, mic_coords, source_pos_2d, FS, ort_session)
    
    save_wav("2a_FF_output_final.wav", FS, normalize_signal(output_ff), output_folder)
    save_wav("2b_FF_masked_only.wav", FS, normalize_signal(masked_ff), output_folder)

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
    
    save_wav("3_ROOM_input_mix_mic0.wav", FS, room_input[0], output_folder)

    print(" -> Applying Neural MVDR (Without WPE)...")
    output_rm, masked_rm = apply_mvdr_stft_bridge(room_input, vad_oracle_room, mic_coords, source_pos_2d, FS, ort_session)
    
    save_wav("4a_ROOM_output_final.wav", FS, normalize_signal(output_rm), output_folder)
    save_wav("4b_ROOM_masked_only.wav", FS, normalize_signal(masked_rm), output_folder)

    # -------------------------------------------------------------------
    # PHASE 3: WPE DEREVERBERATION + NEURAL MVDR
    # -------------------------------------------------------------------
    print("\n--- PHASE 3: WPE + NEURAL MVDR PIPELINE ---")
    print(" -> Applying Online WPE Dereverberation on Room Simulation...")
    
    wpe_output = process_wpe_online(room_input, delay=3)
    
    # Variance normalization
    rms_room = np.sqrt(np.mean(room_input**2, axis=1, keepdims=True))
    rms_wpe = np.sqrt(np.mean(wpe_output**2, axis=1, keepdims=True))
    wpe_output = wpe_output * (rms_room / (rms_wpe + 1e-10))
    
    save_wav("5_WPE_input_mix_mic0.wav", FS, wpe_output[0], output_folder)

    print(" -> Applying Neural MVDR on Dereverberated Signals...")
    output_wpe, masked_wpe = apply_mvdr_stft_bridge(wpe_output, vad_oracle_room, mic_coords, source_pos_2d, FS, ort_session)
    
    save_wav("6a_WPE_ROOM_output_final.wav", FS, normalize_signal(output_wpe), output_folder)
    save_wav("6b_WPE_ROOM_masked_only.wav", FS, normalize_signal(masked_wpe), output_folder)

    print("\n -> Pipeline completed successfully.")