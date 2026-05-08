import numpy as np
import scipy.signal as signal
import torch
import torchaudio
import os

# Import DeepFilterNet instead of ConvTasNet
from df.enhance import enhance, init_df
from df.utils import download_file

from propagation.simulate_acoustics import SimAcoustic
from utils.audio import save_wav, normalize_signal
from dereverberation.nara_wrappers import process_wpe_online
from beamforming.signal_model import compute_rtf_steering_vector

def extract_neural_mask_dfnet(mic_0_audio, fs, length_fft=512, hop_length_fft=256):
    """
    Passes Mic 0 audio through DeepFilterNet and generates the speech enhancement mask.
    """
    print(" -> [Neural Network] Loading DeepFilterNet model...")
    # Initialize the default pre-trained DeepFilterNet model
    model, df_state, _ = init_df() 
    
    # PyTorch and DFNet expect tensors of shape [Channels, Time]
    input_tensor = torch.tensor(mic_0_audio, dtype=torch.float32).unsqueeze(0)

    # DeepFilterNet natively operates at a specific sample rate (usually 48kHz). 
    # We must resample the input if it does not match.
    if fs != df_state.sr():
        input_tensor = torchaudio.functional.resample(
            input_tensor, orig_freq=fs, new_freq=df_state.sr()
        )

    print(" -> [Neural Network] Processing audio (Inference)...")
    with torch.no_grad():
        enhanced_tensor = enhance(model, df_state, input_tensor)

    # Resample the enhanced audio back to the original simulation sample rate
    if fs != df_state.sr():
        enhanced_tensor = torchaudio.functional.resample(
            enhanced_tensor, orig_freq=df_state.sr(), new_freq=fs
        )

    # Extract the raw enhanced audio numpy array
    audio_clean = enhanced_tensor.squeeze(0).numpy()

    # Compute STFTs to calculate the ideal ratio mask
    print(" -> [Neural Network] Computing mask in the frequency domain...")
    _, _, Zxx_mix = signal.stft(mic_0_audio, fs=fs, nperseg=length_fft, noverlap=length_fft - hop_length_fft)
    _, _, Zxx_clean = signal.stft(audio_clean, fs=fs, nperseg=length_fft, noverlap=length_fft - hop_length_fft)

    mag_mix = np.abs(Zxx_mix)
    mag_clean = np.abs(Zxx_clean)
    
    # Calculate ratio mask for the target source and apply clipping
    M_target = mag_clean / (mag_mix + 1e-8)
    M_target = np.clip(M_target, 0.05, 0.95)

    return M_target, audio_clean

def Neural_MVDR_recursive(X_stft, M_target, fs, array_geometry, source_pos, beta=1e-3, min_loading=1e-6):
    """
    Exact MVDR algorithm, but guided by the neural mask instead of analytical SPP.
    """
    lamda = 0.99
    K, T, M = X_stft.shape  
    frecs = np.linspace(0, fs/2, K)

    sv = compute_rtf_steering_vector(frecs, source_pos, array_geometry, ref_mic_idx=0, mode="near_field", squeeze=True)
    
    Y_stft = np.zeros((K, T), dtype=np.complex128)
    R_nn = np.tile(np.eye(M, dtype=np.complex128) * 1e-6, (K, 1, 1))
    
    phi_s = 1e-3 
    R_ss_init = phi_s * np.einsum("fm,fn->fmn", sv, sv.conj())
    R_xx = R_ss_init + R_nn

    initial_diag_load = np.tile(np.eye(M, dtype=np.complex128) * min_loading, (K, 1, 1))
    R_nn_inv = np.linalg.inv(R_nn + initial_diag_load)

    for m in range(T):
        X_frame = X_stft[:, m, :]

        # Replace SPP calculation with the neural mask value for this frame
        P_expand = M_target[:, m, np.newaxis, np.newaxis]

        # Update Covariance Matrices
        R_instant = np.einsum("fm,fn->fmn", X_frame, X_frame.conj())
        R_xx = lamda * R_xx + (1 - lamda) * P_expand * R_instant
        R_nn = lamda * R_nn + (1 - lamda) * (1 - P_expand) * R_instant

        # Compute dynamic diagonal loading
        tr_R = np.real(np.trace(R_nn, axis1=1, axis2=2))
        adaptive_load = beta * (tr_R[:, None, None] / M)
        loading = np.maximum(adaptive_load, min_loading)
        R_nn_stable = R_nn + np.eye(M)[None, :, :] * loading
        R_nn_inv = np.linalg.inv(R_nn_stable)

        # Calculate MVDR weights
        weights_nom = np.einsum("fmn,fn->fm", R_nn_inv, sv)
        weights_den = np.einsum("fm,fm->f", sv.conj(), weights_nom)
        weights = weights_nom / (weights_den[:, np.newaxis] + 1e-10)
        
        # Apply weights
        Y_stft[:, m] = np.einsum("fm,fm->f", weights.conj(), X_frame)

    return Y_stft

def apply_neural_mvdr_bridge(time_domain_input, mic_coords, source_pos_2d, fs, length_fft=512, hop_length_fft=256):
    """
    The complete pipeline: WAV -> DeepFilterNet Mask -> MVDR -> WAV.
    The spatial judge has been removed since DFNet inherently solves the permutation problem.
    """
    mic_0_audio = time_domain_input[0, :]
    
    # 1. Extract the target mask and raw enhanced audio directly from DeepFilterNet
    M_target, raw_clean = extract_neural_mask_dfnet(mic_0_audio, fs, length_fft, hop_length_fft)
    
    # 2. Compute STFT of all channels
    freqs, times, Zxx = signal.stft(
        time_domain_input, fs=fs, nperseg=length_fft, noverlap=length_fft - hop_length_fft
    )
    X_stft = np.transpose(Zxx, (1, 2, 0)) # Shape (K, T, M)
    
    # Adjust temporal dimensions for possible STFT truncation
    min_T = min(M_target.shape[1], X_stft.shape[1])
    M_target = M_target[:, :min_T]
    X_stft = X_stft[:, :min_T, :]

    # 3. Apply MVDR using the DeepFilterNet mask
    print(" -> [MVDR] Applying Spatial Beamforming...")
    Y_stft = Neural_MVDR_recursive(X_stft, M_target, fs, mic_coords, source_pos_2d)
    
    # 4. Inverse STFT to get the spatialized clean audio
    _, y_time = signal.istft(Y_stft, fs=fs, nperseg=length_fft, noverlap=length_fft - hop_length_fft)
    
    original_length = time_domain_input.shape[1]
    
    # Return both the MVDR output and the raw DFNet output for comparison
    return y_time[:original_length], raw_clean[:original_length]

if __name__ == "__main__":
    FS = 16000
    M1, M2 = 8, 1          
    M = M1 * M2
    speed_of_sound = 343.0 

    iSIR_dB = 0 
    
    print("=== INTEGRATION TEST: DFNet PIPELINE (FREE-FIELD, ROOM, WPE+ROOM) ===")
    
    output_folder = "tests/data/mvdr_deepfilternet_output"
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
    acoustic_scene = SimAcoustic(mic_coords, array_mismatch=0.0, duration=15, fs=FS)
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

    print(" -> Applying Recursive MVDR (Guided by DeepFilterNet)...")
    output_ff, raw_net_ff = apply_neural_mvdr_bridge(free_field_input, mic_coords, source_pos_2d, FS)
    
    save_wav("2_FF_output_final.wav", FS, normalize_signal(output_ff), output_folder)
    save_wav("2_FF_RAW_NETWORK_mic0.wav", FS, normalize_signal(raw_net_ff), output_folder)

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

    print(" -> Applying Recursive MVDR (Guided by DeepFilterNet)...")
    output_rm, raw_net_rm = apply_neural_mvdr_bridge(room_input, mic_coords, source_pos_2d, FS)
    
    save_wav("4_ROOM_output_final.wav", FS, normalize_signal(output_rm), output_folder)
    save_wav("4_ROOM_RAW_NETWORK_mic0.wav", FS, normalize_signal(raw_net_rm), output_folder)

    # -------------------------------------------------------------------
    # PHASE 3: WPE + MVDR PIPELINE
    # -------------------------------------------------------------------
    print("\n--- PHASE 3: WPE + MVDR PIPELINE ---")
    print(" -> Applying Online WPE Dereverberation on Room Simulation...")
    
    wpe_output = process_wpe_online(room_input, delay=1, taps=12)
    
    save_wav("5_WPE_input_mix_mic0.wav", FS, wpe_output[0], output_folder)

    print(" -> Applying Recursive MVDR on Dereverberated Signals...")
    output_wpe, raw_net_wpe = apply_neural_mvdr_bridge(wpe_output, mic_coords, source_pos_2d, FS)
    
    save_wav("6_WPE_ROOM_RAW_NETWORK_mic0.wav", FS, normalize_signal(raw_net_wpe), output_folder)
    save_wav("6_WPE_ROOM_output_final.wav", FS, normalize_signal(output_wpe), output_folder)

    print("\n -> Pipeline completed successfully.")