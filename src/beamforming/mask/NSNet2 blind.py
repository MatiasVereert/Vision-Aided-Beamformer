import os
import numpy as np
import scipy.signal as signal
import onnxruntime as ort

# SimAcoustic is still needed to create the test audio, but NOT for the MVDR
from propagation.simulate_acoustics import SimAcoustic
from beamforming.MWF.SP_SDW_MWF_base import process_wpe_online
from utils.audio import save_wav, normalize_signal

def Neural_MVDR_Blind(X_stft, fs, ort_session, beta=1e-3, min_loading=1e-6, save_weights=False):
    """
    Completely Blind Neural MVDR.
    Estimates the Relative Transfer Function (RTF) purely from the data 
    via Covariance Subtraction, without geometric priors.
    """
    lamda = 0.99
    K, T, M = X_stft.shape  
    
    Y_stft = np.zeros((K, T), dtype=np.complex128)
    
    # Initialize noise covariance matrix R_nn
    R_nn = np.tile(np.eye(M, dtype=np.complex128) * 1e-6, (K, 1, 1))
    
    # Blind RTF Initialization: Start assuming the target is perfectly aligned 
    # with the reference microphone (cold start pass-through)
    ref_mic = 0
    rtf_blind = np.zeros((K, M), dtype=np.complex128)
    rtf_blind[:, ref_mic] = 1.0  # The reference mic RTF is exactly 1 + 0j
    
    # Initialize noisy signal covariance matrix R_xx
    phi_s = 1e-3 
    R_ss_init = phi_s * np.einsum("fm,fn->fmn", rtf_blind, rtf_blind.conj())
    R_xx = R_ss_init + R_nn

    # RTF Estimation Hyperparameters
    alpha_rtf = 0.98         # Smoothing factor for RTF recursive update
    spp_threshold_rtf = 0.8  # Minimum Neural Probability required to update the RTF

    weights_rec = np.zeros((K, T, M), dtype=np.complex128)
    
    # Get ONNX input name dynamically (assuming single input model like NSNet2)
    onnx_input_name = ort_session.get_inputs()[0].name

    for m in range(T):
        X_frame = X_stft[:, m, :]

        # --- 1. Neural Mask Estimation ---
        mag_ref = np.abs(X_frame[:, ref_mic])
        onnx_input = mag_ref.astype(np.float32).reshape(1, 1, -1)
        neural_mask = ort_session.run(None, {onnx_input_name: onnx_input})[0]
        
        P = np.clip(neural_mask.flatten(), 0.05, 0.95)
        P_expand = P[:, np.newaxis, np.newaxis]

        # --- 2. Update Covariance Matrices ---
        R_instant = np.einsum("fm,fn->fmn", X_frame, X_frame.conj())
        
        R_xx = lamda * R_xx + (1 - lamda) * P_expand * R_instant
        R_nn = lamda * R_nn + (1 - lamda) * (1 - P_expand) * R_instant

        # --- 3. BLIND RTF Estimation via Covariance Subtraction ---
        R_ss = R_xx - R_nn
        
        # Extract the reference column
        R_ss_ref_col = R_ss[:, :, ref_mic]
        
        # Extract the reference diagonal element (safeguard against negative variance)
        R_ss_ref_diag = np.maximum(np.real(R_ss[:, ref_mic, ref_mic]), 1e-12)
        
        # Compute instantaneous RTF 
        rtf_inst = R_ss_ref_col / R_ss_ref_diag[:, np.newaxis]
        
        # Update the RTF recursively ONLY when neural network detects speech strongly
        for f in range(K):
            if P[f] > spp_threshold_rtf:
                rtf_blind[f, :] = alpha_rtf * rtf_blind[f, :] + (1 - alpha_rtf) * rtf_inst[f, :]
                
        # CRITICAL DIFFERENCE: We DO NOT L2-normalize rtf_blind here.
        # Keeping rtf_blind[ref_mic] = 1 preserves the distortionless constraint.

        # --- 4. Compute MVDR Weights ---
        tr_R = np.real(np.trace(R_nn, axis1=1, axis2=2))
        adaptive_load = beta * (tr_R[:, None, None] / M)
        loading = np.maximum(adaptive_load, min_loading)
        
        R_nn_stable = R_nn + np.eye(M)[None, :, :] * loading
        R_nn_inv = np.linalg.inv(R_nn_stable)

        # Numerator: R_nn_inv * h (using the blind RTF vector instead of geometric SV)
        weights_nom = np.einsum("fmn,fn->fm", R_nn_inv, rtf_blind)
        
        # Denominator: h^H * numerator
        weights_den = np.einsum("fm,fm->f", rtf_blind.conj(), weights_nom)
        
        weights = weights_nom / (weights_den[:, np.newaxis] + 1e-10)
        
        if save_weights:
            weights_rec[:, m, :] = weights
        
        # --- 5. Apply Filter ---
        Y_stft[:, m] = np.einsum("fm,fm->f", weights.conj(), X_frame)

    if save_weights:
        return Y_stft, weights_rec
    else:
        return Y_stft


def apply_blind_mvdr_bridge(time_domain_input, fs, ort_session, length_fft=320, hop_length_fft=160):
    """
    Bridge function for the Blind Neural MVDR.
    Notice the complete removal of vad_oracle, mic_coords, and source_pos.
    """
    freqs, times, Zxx = signal.stft(
        time_domain_input, 
        fs=fs, 
        nperseg=length_fft, 
        noverlap=length_fft - hop_length_fft
    )
    X_stft = np.transpose(Zxx, (1, 2, 0))

    # Execute the Blind Recursive MVDR
    Y_stft = Neural_MVDR_Blind(
        X_stft=X_stft, 
        fs=fs, 
        ort_session=ort_session
    )
    
    _, y_time = signal.istft(
        Y_stft, 
        fs=fs, 
        nperseg=length_fft, 
        noverlap=length_fft - hop_length_fft
    )
    
    original_length = time_domain_input.shape[1]
    return y_time[:original_length]


if __name__ == "__main__":
    FS = 16000
    M1, M2 = 12, 1          
    M = M1 * M2
    iSIR_dB = 0
    
    onnx_model_path = r"tools\data\redes\nsnet2-20ms-baseline.onnx"
    try:
        ort_session = ort.InferenceSession(onnx_model_path)
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        exit()
    
    output_folder = "tests/data/Blind_MVDR_output"
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

    cache_room_path = os.path.join(output_folder, "cache_room.npz")
    if os.path.exists(cache_room_path):
        print("\n--- LOADING ROOM SIMULATION FROM CACHE ---")
        room_input = np.load(cache_room_path)['input']
    else:
        print("\n--- COMPUTING ROOM SIMULATION ---")
        room_sim_dic = acoustic_scene.get_eval_scene(
            room_dimensions=np.array([4.0, 5.0, 2.5]), desire_RT=0.5, iSIR_dB=iSIR_dB, mode="ideal"
        )
        room_input = room_sim_dic["mic_signals"]
        np.savez(cache_room_path, input=room_input)
    
    save_wav("1_ROOM_input_mix_mic0.wav", FS, room_input[0], output_folder)

    # -------------------------------------------------------------------
    # WPE DEREVERBERATION + BLIND NEURAL MVDR
    # -------------------------------------------------------------------
    print("\n--- WPE + BLIND NEURAL MVDR PIPELINE ---")
    print(" -> Applying Online WPE...")
    wpe_output = process_wpe_online(room_input, delay=3)
    
    rms_room = np.sqrt(np.mean(room_input**2, axis=1, keepdims=True))
    rms_wpe = np.sqrt(np.mean(wpe_output**2, axis=1, keepdims=True))
    wpe_output = wpe_output * (rms_room / (rms_wpe + 1e-10))
    
    save_wav("2_WPE_output_mic0.wav", FS, wpe_output[0], output_folder)

    print(" -> Applying Completely Blind Neural MVDR...")
    # Notice how we only pass the audio, the fs, and the neural model. No geometry!
    output_blind = apply_blind_mvdr_bridge(wpe_output, FS, ort_session)
    
    save_wav("3_BLIND_MVDR_output_final.wav", FS, normalize_signal(output_blind), output_folder)

    print("\n -> Pipeline completed successfully.")