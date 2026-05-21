import os
import numpy as np
import scipy.signal as signal

# Imports for data simulation (from your base structure)
from propagation.simulate_acoustics import SimAcoustic
from utils.audio import save_wav, normalize_signal

# Imports for neural mask estimation (AkojimaSLP repo)
from mask.maskestimator import model, shaper, feature

# =====================================================================
# 1. ONLINE MVDR BEAMFORMER (MASK-BASED)
# =====================================================================
def MVDR_recursive_mask_based(X_stft, mask_s, mask_n, min_loading=1e-5, lamda=0.99):
    K, T, M = X_stft.shape  
    
    Y_stft = np.zeros((K, T), dtype=np.complex128)
    
    Phi_NN = np.tile(np.eye(M, dtype=np.complex128) * 1e-6, (K, 1, 1))
    Phi_XX = np.tile(np.eye(M, dtype=np.complex128) * 1e-6, (K, 1, 1))
    
    # Memoria para el vector direccional
    d_prev = np.ones((K, M), dtype=np.complex128) 
    d_prev[:, 0] = 1.0 # Referencia en mic 0
    
    for m in range(T):
        print(f"\rProccesing grame {m} of {T}",end="")
        X_frame = X_stft[:, m, :]
        m_s_frame = mask_s[:, m, np.newaxis, np.newaxis]
        m_n_frame = mask_n[:, m, np.newaxis, np.newaxis]

        R_instant = np.einsum("fm,fn->fmn", X_frame, X_frame.conj())

        Phi_XX = lamda * Phi_XX + (1 - lamda) * (m_s_frame * R_instant)
        Phi_NN = lamda * Phi_NN + (1 - lamda) * (m_n_frame * R_instant)

        # Calculamos los autovectores
        eigenvalues, eigenvectors = np.linalg.eigh(Phi_XX)
        d = eigenvectors[:, :, -1]  
        
        # Normalización con respecto al micrófono de referencia.
        # Agregamos epsilon complejo para evitar división por cero o cambios de fase violentos
        d = d / (d[:, 0:1] + 1e-15 + 1j*1e-15)

        # Lógica de estabilización: 
        # Si la máscara de voz es muy baja en esta frecuencia, el RTF estimado es ruido.
        # Nos quedamos con el RTF del frame anterior para esas frecuencias.
        mask_s_1d = mask_s[:, m]
        update_mask = mask_s_1d > 0.1 # Umbral de confianza
        d = np.where(update_mask[:, np.newaxis], d, d_prev)
        d_prev = d.copy()

        tr_Phi = np.real(np.trace(Phi_NN, axis1=1, axis2=2))
        adaptive_load = min_loading * (tr_Phi / M)
        loading_matrix = np.eye(M)[np.newaxis, :, :] * np.maximum(adaptive_load, 1e-6)[:, np.newaxis, np.newaxis]
        
        Phi_NN_stable = Phi_NN + loading_matrix
        Phi_NN_inv = np.linalg.inv(Phi_NN_stable)

        weights_nom = np.einsum("fmn,fn->fm", Phi_NN_inv, d)
        weights_den = np.einsum("fm,fm->f", d.conj(), weights_nom)
        
        # El denominador $d^H \Phi_{NN}^{-1} d$ siempre es real. 
        # Forzamos np.real para evitar residuos imaginarios numéricos.
        weights = weights_nom / (np.real(weights_den[:, np.newaxis]) + 1e-10)

        Y_stft[:, m] = np.einsum("fm,fm->f", weights.conj(), X_frame)

    return Y_stft
# =====================================================================
# 2. OFFLINE MASK ESTIMATION WRAPPER
# =====================================================================
def get_offline_masks(time_domain_input, fs, weight_path, length_fft=512, hop_length_fft=256):
    """
    Processes the entire signal offline to extract median-pooled neural masks.
    """
    M, samples = time_domain_input.shape
    
    # Initialize AkojimaSLP's parameters
    truncate_grad = 7
    number_of_stack = 1
    number_of_skip_frame = 0
    
    mask_estimator_generator = model.NeuralMaskEstimation(
        truncate_grad, number_of_stack, 0.1, length_fft // 2 + 1, recurrent_init=0.00001
    )
    mask_model = mask_estimator_generator.get_model(is_stateful=True, is_show_detail=False, is_adapt=False)
    mask_model = mask_estimator_generator.load_weight_param(mask_model, weight_path)
    
    data_shaper = shaper.Shape_data(0, 0, truncate_grad, number_of_skip_frame)
    feature_extractor = feature.Feature(fs, length_fft, hop_length_fft)
    
    n_masks_list = []
    sp_masks_list = []
    
    for ch in range(M):
        speech_channel = time_domain_input[ch, :]
        
        # Feature extraction
        noisy_spectrogram = feature_extractor.get_feature(speech_channel)
        noisy_spectrogram = np.flipud(noisy_spectrogram)
        noisy_spectrogram = feature_extractor.apply_cmvn(noisy_spectrogram)
        
        features = np.array(data_shaper.convert_for_predict(noisy_spectrogram))
        
        # Iteramos sobre las capas y reiniciamos los estados de las recurrentes (LSTM)
        for layer in mask_model.layers:
            if hasattr(layer, 'reset_states'):
                layer.reset_states()
        
        padding_feature, original_batch_size = data_shaper.get_padding_features(features)
        
        # Predict masks
        sp_mask, n_mask = mask_model.predict(padding_feature, batch_size=5000)

        # Predict masks
        sp_mask, n_mask = mask_model.predict(padding_feature, batch_size=5000)
        
        # --- AGREGAR ESTO PARA DEBUG ---
        print(f"\n[DEBUG CANAL {ch}]")
        print(f"  Features Entrada - min: {np.min(features):.4f}, max: {np.max(features):.4f}, media: {np.mean(features):.4f}")
        print(f"  Mascara Voz (NN) - min: {np.min(sp_mask):.4f}, max: {np.max(sp_mask):.4f}, media: {np.mean(sp_mask):.4f}")
        print(f"  Mascara Ruido    - min: {np.min(n_mask):.4f}, max: {np.max(n_mask):.4f}, media: {np.mean(n_mask):.4f}")
        # -------------------------------
        
        # Slice back to original size and transpose to (K, T)
        sp_masks_list.append(sp_mask[:original_batch_size, :].T)
        n_masks_list.append(n_mask[:original_batch_size, :].T)



    # Pool masks across all microphones using median operation
    sp_mask_condensed = np.median(np.stack(sp_masks_list, axis=-1), axis=-1)
    n_mask_condensed = np.median(np.stack(n_masks_list, axis=-1), axis=-1)
    
    # --- PARCHE DE CONTRASTE (apply_range_norm) ---
    # Estiramos los valores para forzar que el mínimo sea 0 y el máximo sea 1
    def stretch_mask(m):
        return (m - np.min(m)) / (np.max(m) - np.min(m) + 1e-12)

    sp_mask_condensed = stretch_mask(sp_mask_condensed)
    n_mask_condensed = stretch_mask(n_mask_condensed)
    
    # Opcional: Elevar al cuadrado o al cubo "afila" la máscara, 
    # aplastando la incertidumbre hacia 0 y dejando solo las certezas cerca de 1.
    sp_mask_condensed = sp_mask_condensed ** 2
    n_mask_condensed = n_mask_condensed ** 2
    # ----------------------------------------------
    
    return sp_mask_condensed, n_mask_condensed
    



# =====================================================================
# 3. PIPELINE INTEGRATION
# =====================================================================
def apply_hybrid_pipeline(time_domain_input, fs, weight_path, length_fft=1024, hop_length_fft=256):
    
    print(" -> Computing offline neural masks...")
    mask_s, mask_n = get_offline_masks(time_domain_input, fs, weight_path, length_fft, hop_length_fft)
    
    print(" -> Computing STFT of the mixture...")
    freqs, times, Zxx = signal.stft(
        time_domain_input, fs=fs, nperseg=length_fft, noverlap=length_fft - hop_length_fft
    )
    X_stft = np.transpose(Zxx, (1, 2, 0)) # Transpose to (K, T, M)
    
    # Ensure time dimensions match between STFT and neural masks
    min_frames = min(X_stft.shape[1], mask_s.shape[1])
    X_stft = X_stft[:, :min_frames, :]
    mask_s = mask_s[:, :min_frames]
    mask_n = mask_n[:, :min_frames]

    # ==============================================================
    # DEBUG DUMP: Señal de referencia enmascarada
    # ==============================================================
    print(" -> [DEBUG] Generating single-channel masked dump...")
    X_ref = X_stft[:, :, 0] # Tomamos el micrófono 0 (K, T)
    Y_debug_stft = X_ref * mask_s # Multiplicamos por la máscara de voz
    
    _, y_debug_time = signal.istft(
        Y_debug_stft, fs=fs, nperseg=length_fft, noverlap=length_fft - hop_length_fft
    )
    
    # Lo guardamos directamente en la carpeta de tests
    save_wav("DEBUG_mask_only_mic0.wav", fs, normalize_signal(y_debug_time), "tests/data/hybrid_mvdr_output")
    # ==============================================================

    print(" -> Running mask-based recursive MVDR...")
    Y_stft = MVDR_recursive_mask_based(X_stft, mask_s, mask_n)

    print(" -> Reconstructing time-domain signal...")
    _, y_time = signal.istft(
        Y_stft, fs=fs, nperseg=length_fft, noverlap=length_fft - hop_length_fft
    )
    
    return y_time,  y_debug_time
import matplotlib.pyplot as plt

def plot_results_comparison(input_signal, output_signal, mask, fs, length_fft, hop_length_fft):
    """
    Plots the spectrograms of input, output, and the estimated mask for comparison.
    """
    # Use a style for better visibility
    plt.style.use('seaborn-v0_8-muted')
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    cmap_choice = 'viridis' # Uniform colormap
    
    # Common parameters for spectrograms
    spec_params = dict(
        Fs=fs, 
        NFFT=length_fft, 
        noverlap=length_fft - hop_length_fft, 
        cmap=cmap_choice
    )

    # 1. Input Spectrogram (Mic 0)
    print(" -> Plotting Input Spectrogram...")
    axes[0].specgram(input_signal[0], **spec_params)
    axes[0].set_title("Input Signal Spectrogram (Mic 0 - Mixture)")
    axes[0].set_ylabel("Frequency [Hz]")

    # 2. Output Spectrogram (MVDR Enhanced)
    print(" -> Plotting Output Spectrogram...")
    axes[1].specgram(output_signal, **spec_params)
    axes[1].set_title("Output Signal Spectrogram (Hybrid MVDR Enhanced)")
    axes[1].set_ylabel("Frequency [Hz]")

    # 3. Neural Mask (Speech Mask)
    # The mask is already in (K, T) format, we use imshow
    print(" -> Plotting Neural Mask...")
    # Extent is needed to align axes: [xmin, xmax, ymin, ymax]
    duration = len(output_signal) / fs
    im = axes[2].imshow(
        mask, 
        aspect='auto', 
        origin='lower', 
        extent=[0, duration, 0, fs / 2],
        cmap=cmap_choice
    )
    axes[2].set_title("Neural Speech Mask ($M_s$)")
    axes[2].set_ylabel("Frequency [Hz]")
    axes[2].set_xlabel("Time [s]")

    # Add a colorbar for the mask to show confidence levels
    fig.colorbar(im, ax=axes[2], label="Mask Intensity")

    plt.tight_layout()
    plt.show()

# --- Integration at the end of your __main__ block ---
# Add this after "output_rm = apply_hybrid_pipeline(...)"
# We need to recover the mask from the pipeline to plot it
# (Note: You might need to return mask_s from apply_hybrid_pipeline to use it here)
# plot_results_comparison(room_input, output_rm, mask_s, FS, 1024, 256)
# =====================================================================
# 4. MAIN SCRIPT (SIMULATION & TESTING)
# =====================================================================
if __name__ == "__main__":
    
    FS = 16000
    M1, M2 = 12, 1          
    M = M1 * M2
    iSIR_dB = 0
    # The path must point to the prefix of the checkpoint files
# Usamos una cadena 'r' (raw) para que Windows no tenga problemas con las barras diagonales
    WEIGHT_PATH = r'C:\Users\Matias\Documents\Tesis\Vision-Aided-Beamformer\src\mask\model\194sequence_false_e1.ckpt'
     # Update with your correct path
    
    print("=== HYBRID TEST: MASK-BASED ONLINE MVDR ===")
    
    output_folder = "tests/data/hybrid_mvdr_output"
    os.makedirs(output_folder, exist_ok=True)
    
    # Array geometry (logarithmic spacing)
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
    
    # Positions
    r = 1.0 
    ang_target = np.deg2rad(130)
    ang_interf = np.deg2rad(50)
    
    source_pos = array_center + np.array([r * np.cos(ang_target), r * np.sin(ang_target), 0.0])
    interf_pos1 = array_center + np.array([r * np.cos(ang_interf), r * np.sin(ang_interf), 0.0])
    source_pos_2d = source_pos.reshape(1, 3)

    print(" -> Initializing acoustic scene...")
    acoustic_scene = SimAcoustic(mic_coords, array_mismatch=0.0, duration=20, fs=FS)
    acoustic_scene.set_source(r"tools\data\signals\p002_emo_adoration_sentences.wav", gain=1, position=source_pos_2d)
    acoustic_scene.set_interference(r"tools\data\signals\hairdryer_07_SH_MKH800.wav", gain=1, position=interf_pos1.reshape(1,3))
    
    # =================================================================
    # PHASE: ROOM SIMULATION (Reverberant)
    # =================================================================
    cache_room_path = os.path.join(output_folder, "cache_room.npz")
    
    if os.path.exists(cache_room_path):
        print("\n--- LOADING ROOM SIMULATION FROM CACHE ---")
        cache_data = np.load(cache_room_path)
        room_input = cache_data['input']
    else:
        print("\n--- COMPUTING ROOM SIMULATION ---")
        room_dimensions = np.array([4.0, 5.0, 2.5])
        room_sim_dic = acoustic_scene.get_eval_scene(
            room_dimensions=room_dimensions, desire_RT=0.5, iSIR_dB=iSIR_dB, mode="ideal"
        )
        room_input = room_sim_dic["mic_signals"]
        np.savez(cache_room_path, input=room_input)
    
    save_wav("1_ROOM_input_mix_mic0.wav", FS, room_input[0], output_folder)

    # Execute Hybrid Pipeline
    print("\n--- APPLYING HYBRID MVDR (OFFLINE MASK + ONLINE FILTER) ---")
    output_rm, mask_rm = apply_hybrid_pipeline(room_input, FS, WEIGHT_PATH, length_fft=1024, hop_length_fft=256)
    
    save_wav("2_ROOM_hybrid_output_final.wav", FS, normalize_signal(output_rm), output_folder)

    print("\n -> Processing complete. Check output folder.")

    plot_results_comparison(room_input, output_rm, mask_rm, FS, 1024, 256)

