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

import numpy as np
import tensorflow as tf
# Note: On the Kria KV260, you will likely replace the above with:
# import tflite_runtime.interpreter as tflite
def apply_dtln_post_tflite_realtime(interpreter_1, interpreter_2, audio_mono, blend_alpha=1):
    """
    Applies the Float32 DTLN TF-Lite models using a strict real-time 
    frame-by-frame loop. Handles LSTM hidden states properly to prevent 
    signal degradation. Compensates for algorithmic delay during blending.
    """
    # 1. Normalize input to prevent LSTM saturation
    max_val = np.max(np.abs(audio_mono))
    if max_val > 0.0:
        # Scale to peak at 0.9 to give the network optimal headroom
        audio_mono = audio_mono * (0.9 / max_val)

    audio_mono = np.asarray(audio_mono, dtype=np.float32)
    
    block_len = 512
    block_shift = 128
    
    out_audio = np.zeros_like(audio_mono)
    in_buffer = np.zeros((block_len), dtype=np.float32)
    out_buffer = np.zeros((block_len), dtype=np.float32)
    
    # Get input/output indices 
    input_details_1 = interpreter_1.get_input_details()
    output_details_1 = interpreter_1.get_output_details()
    
    input_details_2 = interpreter_2.get_input_details()
    output_details_2 = interpreter_2.get_output_details()

    # INITIALIZE LSTM STATES
    states_1 = np.zeros(input_details_1[1]['shape'], dtype=np.float32)
    states_2 = np.zeros(input_details_2[1]['shape'], dtype=np.float32)

    num_blocks = (len(audio_mono) - (block_len - block_shift)) // block_shift
    
    for idx in range(num_blocks):
        # Shift buffer and load data directly
        in_buffer[:-block_shift] = in_buffer[block_shift:]
        start_idx = idx * block_shift
        in_buffer[-block_shift:] = audio_mono[start_idx : start_idx + block_shift]
        
        # Compute FFT, magnitude and phase
        in_block_fft = np.fft.rfft(in_buffer)
        in_mag = np.abs(in_block_fft)
        in_phase = np.angle(in_block_fft)
        
        # Reshape magnitude
        in_mag = np.reshape(in_mag, (1, 1, -1)).astype(np.float32)
        
        # Feed magnitude AND previous states to Model 1
        interpreter_1.set_tensor(input_details_1[1]['index'], states_1)
        interpreter_1.set_tensor(input_details_1[0]['index'], in_mag)
        interpreter_1.invoke()
        
        # Extract mask and NEW states
        out_mask = interpreter_1.get_tensor(output_details_1[0]['index'])
        states_1 = interpreter_1.get_tensor(output_details_1[1]['index']) 
        
        # Reconstruct complex FFT and apply IFFT
        estimated_complex = in_mag * out_mask * np.exp(1j * in_phase)
        estimated_block = np.fft.irfft(estimated_complex, n=block_len)
        estimated_block = np.reshape(estimated_block, (1, 1, -1)).astype(np.float32)
        
        # Feed time-domain block AND previous states to Model 2
        interpreter_2.set_tensor(input_details_2[1]['index'], states_2)
        interpreter_2.set_tensor(input_details_2[0]['index'], estimated_block)
        interpreter_2.invoke()
        
        # Extract final audio block and NEW states
        out_block = interpreter_2.get_tensor(output_details_2[0]['index'])
        states_2 = interpreter_2.get_tensor(output_details_2[1]['index'])
        
        # Overlap-add
        out_buffer[:-block_shift] = out_buffer[block_shift:]
        out_buffer[-block_shift:] = np.zeros((block_shift))
        out_buffer += np.squeeze(out_block)
        
        out_audio[start_idx : start_idx + block_shift] = out_buffer[:block_shift]

    # 2. Apply Wet/Dry blending with precise delay compensation
    if blend_alpha < 1.0:
        delay = block_len - block_shift # 384 samples algorithmic delay
        
        # Shift the dry audio to align perfectly with the processed output
        audio_mono_delayed = np.zeros_like(audio_mono)
        audio_mono_delayed[delay:] = audio_mono[:-delay]
        
        out_audio = (blend_alpha * out_audio) + ((1.0 - blend_alpha) * audio_mono_delayed)

    return out_audio

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
    # We set the global processing rate for the simulation and beamformer to 16kHz
    FS = 16000
    M1, M2 = 8, 1          
    M = M1 * M2
    speed_of_sound = 343.0 
    iSIR_dB = 0
    
    print("=== INTEGRATION TEST: 16k PIPELINE WITH DTLN TFLITE POST-PROCESSING ===")
    
    # Initialize BOTH DTLN TF-Lite Interpreters
    print(" -> [Neural Network] Loading global quantized DTLN TF-Lite models...")
    tflite_model_1_path = r"tools\data\redes\model_1.tflite" 
    tflite_model_2_path = r"tools\data\redes\model_2.tflite" 
    
    try:
        # Load Model 1
        interpreter_1 = tf.lite.Interpreter(model_path=tflite_model_1_path)
        interpreter_1.allocate_tensors()
        
        # Load Model 2
        interpreter_2 = tf.lite.Interpreter(model_path=tflite_model_2_path)
        interpreter_2.allocate_tensors()
    except Exception as e:
        print(" -> [Error] Failed to load TFLite models. Make sure both model_quant_1 and model_quant_2 are in the folder.")
        raise e
    
    output_folder = "tests/data/spp_mvdr_dtln_tflite_output"
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
    
    print(" -> Applying DTLN TF-Lite Post-Processing...")
    output_ff_dtln_only = apply_dtln_post_tflite_realtime(interpreter_1, interpreter_2, free_field_input[0])
    
    output_ff_dtln = apply_dtln_post_tflite_realtime(interpreter_1, interpreter_2, output_ff_mvdr)
    
    # Output saves
    save_wav("1A_FF_output_dtln_only_16k.wav", FS, output_ff_dtln_only, output_folder)
    save_wav("1B_FF_input_mic0_16k.wav", FS, free_field_input[0], output_folder)

    save_wav("1C_FF_MVDR_output_16k.wav", FS, normalize_signal(output_ff_mvdr), output_folder)
    save_wav("1D_FF_DTLN_Post_16k.wav", FS, normalize_signal(output_ff_dtln), output_folder)

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
    
    print(" -> Applying DTLN TF-Lite Post-Processing...")
    output_rm_dtln_only = apply_dtln_post_tflite_realtime(interpreter_1, interpreter_2, room_input[0])
    output_rm_dtln = apply_dtln_post_tflite_realtime(interpreter_1, interpreter_2, output_rm_mvdr)

    # Output saves
    save_wav("2A_RM_input_mic0_16k.wav", FS, room_input[0], output_folder)
    save_wav("2B_RM_output_dtln_only_16k.wav", FS, output_rm_dtln_only, output_folder)
    save_wav("2C_RM_MVDR_output_16k.wav", FS, normalize_signal(output_rm_mvdr), output_folder)
    save_wav("2D_RM_DTLN_Post_16k.wav", FS, normalize_signal(output_rm_dtln), output_folder)

    # -------------------------------------------------------------------
    # PHASE 3: WPE + MVDR + DTLN PIPELINE
    # -------------------------------------------------------------------
    print("\n--- PHASE 3: WPE + MVDR + DTLN PIPELINE ---")
    print(" -> Applying Online WPE Dereverberation (at 16kHz)...")
    wpe_output = process_wpe_online(room_input, delay=3, taps=7, stft_size=1024, stft_shift=128)
    
    print(" -> Applying SPP-guided MVDR (at 16kHz)...")
    mvdr_wpe, _ = apply_mvdr_stft_bridge(wpe_output, vad_oracle_room, mic_coords, source_pos_2d, FS)
    
    print(" -> Applying DTLN TF-Lite Post-Processing...")
    dtln_wpe = apply_dtln_post_tflite_realtime(interpreter_1, interpreter_2, mvdr_wpe)
    
    # Save final results
    save_wav("3_WPE_16k.wav", FS, normalize_signal(wpe_output[0]), output_folder)
    save_wav("3_WPE_MVDR_16k.wav", FS, normalize_signal(mvdr_wpe), output_folder)
    save_wav("3_WPE_MVDR_DTLN_16k.wav", FS, normalize_signal(dtln_wpe), output_folder)

    print("\n -> Full integrated pipeline completed successfully.")