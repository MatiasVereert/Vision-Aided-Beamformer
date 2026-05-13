import os
import numpy as np 
import scipy.signal as signal

# Assuming the imports work correctly in your local environment
from beamforming.signal_model import compute_rtf_steering_vector
from utils.audio import normalize_signal, save_wav
from propagation.simulate_acoustics import SimAcoustic
from dereverberation.nara_wrappers import process_wpe_online

# Import standard DeepFilterNet enhance function
import tensorflow as tf
from dnn_denoise.dtln_lite import apply_dtln_post_tflite_realtime

import numpy as np

def compute_robust_rtf_steering_vector(R_xx_voice, ref_mic_idx=0):
    """
    Data-driven Relative Transfer Function (RTF) estimation using the Principal 
    Eigenvector method. Extracts the empirical propagation vector directly from the 
    spatial correlation matrix, fully mitigating physical capsule mismatch, geometric 
    uncertainties, and array shadowing effects without aggressive diagonal loading.
    """
    K, M, _ = R_xx_voice.shape
    v_empirical = np.zeros((K, M), dtype=np.complex128)
    
    for f in range(K):
        # Extract the spatial covariance matrix for the current frequency bin
        R_f = R_xx_voice[f]
        
        # Perform Eigenvalue Decomposition to find the dominant spatial subspace
        eigenvalues, eigenvectors = np.linalg.eigh(R_f)
        
        # The principal eigenvector corresponds to the dominant acoustic source path
        dominant_vector = eigenvectors[:, -1]
        
        # Normalize strictly relative to the designated reference microphone channel
        v_empirical[f, :] = dominant_vector / (dominant_vector[ref_mic_idx] + 1e-15)
        
    return v_empirical




def SPP_MVDR_recursive(X_stft, fs, array_geometry, source_pos, interpreter_1, interpreter_2, 
                       beta=1e-3, min_loading=1e-6, save_weights=False, save_mask=True):
    """
    MVDR Recursivo donde la máscara de Probabilidad de Presencia de Voz (SPP) 
    ha sido reemplazada por una estimación de máscara basada en las magnitudes 
    espectrales de la salida del modelo DTLN cuantizado.
    """
    # Factor de olvido para el suavizado de la matriz de covarianza
    lamda = 0.99
    K, T, M = X_stft.shape  
    frecs = np.linspace(0, fs/2, K)

    # Obtener vectores de dirección (Steering Vectors), forma esperada (K, M)
    sv = compute_rtf_steering_vector(frecs, source_pos, array_geometry, ref_mic_idx=3, mode="near_field", squeeze=True)
    
    # Inicializar matrices STFT complejas de salida
    Y_stft = np.zeros((K, T), dtype=np.complex128)
    Y_spp_stft = np.zeros((K, T), dtype=np.complex128) # Matriz para la salida filtrada por la máscara DTLN
    
    # =========================================================================
    # --- ETAPA DTLN OFFLINE (Reemplazo conceptual de SPP) ---
    # =========================================================================
    # 1. Reconstruir la señal temporal ruidosa del canal central (micrófono 0)
    # Asumimos los parámetros estándar de puente (nperseg=512, noverlap=256)
    _, x_mic0_time = signal.istft(X_stft[:, :, 0], fs=fs, nperseg=512, noverlap=256)
    
    # 2. Procesar la señal completa offline a través de los intérpretes DTLN TFLite
    # (El audio de entrada debe ser float32 plano)
    x_clean_time = apply_dtln_post_tflite_realtime(interpreter_1, interpreter_2, x_mic0_time.astype(np.float32))
    
    # 3. Llevar la estimación limpia de vuelta al dominio STFT para alinearla con X_stft
    _, _, Zxx_clean = signal.stft(x_clean_time, fs=fs, nperseg=512, noverlap=256)
    
    # Ajustar dimensiones en caso de discrepancias mínimas por el padding de ISTFT/STFT
    T_clean = min(T, Zxx_clean.shape[1])
    X_clean_stft = np.zeros((K, T), dtype=np.complex128)
    X_clean_stft[:, :T_clean] = Zxx_clean[:, :T_clean]
    
    # 4. Derivar la Máscara de Ganancia Espectral (Wiener-like mask)
    # M = |X_clean| / |X_noisy|. Limitamos a [0, 1] para que actúe como una pseudo-probabilidad.
    mag_noisy = np.abs(X_stft[:, :, 0])
    mag_clean = np.abs(X_clean_stft)
    
    # Evitar división por cero
    dtln_mask = mag_clean / (mag_noisy + 1e-10)
    dtln_mask = np.clip(dtln_mask, 0.0, 1.0)
    # =========================================================================

    # Inicializar matriz de covarianza de ruido R_nn
    R_nn = np.tile(np.eye(M, dtype=np.complex128) * 1e-6, (K, 1, 1))
    
    # Inicializar matriz de covarianza de señal ruidosa R_xx usando el steering vector
    phi_s = 1e-3 
    R_ss_init = phi_s * np.einsum("fm,fn->fmn", sv, sv.conj())
    R_xx = R_ss_init + R_nn

    # Inicializar la inversa de R_nn para la primera trama
    initial_diag_load = np.tile(np.eye(M, dtype=np.complex128) * min_loading, (K, 1, 1))
    R_nn_inv = np.linalg.inv(R_nn + initial_diag_load)

    weights_rec = np.zeros((K, T, M), dtype=np.complex128)

    for m in range(T):
        X_frame = X_stft[:, m, :]

        # --- 1. Extraer la máscara DTLN precalculada para la trama actual ---
        # Reemplaza el cálculo a posteriori de la SNR espacial (gamma) y su mapeo sigmoidal
        P = dtln_mask[:, m]
        
        # Expandir dimensiones para operaciones matriciales
        P_expand = P[:, np.newaxis, np.newaxis]
        
        # Guardar la salida enmascarada para propósitos de depuración/visualización
        Y_spp_stft[:, m] = P * X_frame[:, 0]

        # --- 2. Actualizar Matrices de Covarianza ---
        R_instant = np.einsum("fm,fn->fmn", X_frame, X_frame.conj())

        # Actualizar R_xx ponderada por la "probabilidad" (máscara DTLN) de presencia de voz
        R_xx = lamda * R_xx + (1 - lamda) * P_expand * R_instant
        
        # Actualizar R_nn ponderada por la ausencia de voz (1 - máscara DTLN)
        R_nn = lamda * R_nn + (1 - lamda) * (1 - P_expand) * R_instant

        # --- 3. Calcular Pesos MVDR con Carga Diagonal Dinámica ---
        tr_R = np.real(np.trace(R_nn, axis1=1, axis2=2))
        adaptive_load = beta * (tr_R[:, None, None] / M)
        
        loading = np.maximum(adaptive_load, min_loading)
        R_nn_stable = R_nn + np.eye(M)[None, :, :] * loading

        # Invertir de forma segura
        R_nn_inv = np.linalg.inv(R_nn_stable)

        # Numerador: R_nn_inv * v
        weights_nom = np.einsum("fmn,fn->fm", R_nn_inv, sv)
        
        # Denominador: v^H * numerador
        weights_den = np.einsum("fm,fm->f", sv.conj(), weights_nom)
        
        # Pesos finales
        weights = weights_nom / (weights_den[:, np.newaxis] + 1e-10)
        weights_rec[:, m, :] = weights
        
        # --- 4. Aplicar Filtro Espacial ---
        Y_stft[:, m] = np.einsum("fm,fm->f", weights.conj(), X_frame)

    if save_weights:
        if save_mask:
            return Y_stft, Y_spp_stft, weights_rec
        else:
            return Y_stft, weights_rec
    else:
        if save_mask:
            return Y_stft, Y_spp_stft
        else: 
            return Y_stft

def apply_mvdr_stft_bridge(time_domain_input, vad_oracle, mic_coords, source_pos_2d, fs, 
                           interpreter_1, interpreter_2, length_fft=512, hop_length_fft=256):
    """
    Función puente actualizada para recibir y propagar los intérpretes TFLite al beamformer.
    """
    freqs, times, Zxx = signal.stft(
        time_domain_input, 
        fs=fs, 
        nperseg=length_fft, 
        noverlap=length_fft - hop_length_fft
    )
    
    X_stft = np.transpose(Zxx, (1, 2, 0))
    # vad_padded = np.pad(vad_oracle, (0, length_fft + hop_length_fft), mode='constant')

    # Pasamos los intérpretes explícitamente a la función recursiva
    Y_stft, Y_spp_stft = SPP_MVDR_recursive(
        X_stft=X_stft, 
        fs=fs, 
        array_geometry=mic_coords, 
        source_pos=source_pos_2d,
        interpreter_1=interpreter_1,
        interpreter_2=interpreter_2
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
    tflite_model_1_path = r"data/dnn_models/model_1.tflite" 
    tflite_model_2_path = r"data/dnn_models/model_2.tflite" 
    
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
    acoustic_scene.set_source(r"data/audio/input/p002_emo_adoration_sentences.wav", gain=1, position=source_pos_2d)
    acoustic_scene.set_interference(r"data/audio/input/hairdryer_07_SH_MKH800.wav", gain=1, position=interf_pos1.reshape(1,3))
    
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
    output_ff_mvdr, output_ff_spp = apply_mvdr_stft_bridge(free_field_input, vad_oracle_ff, mic_coords, source_pos_2d, FS,
                                                                   interpreter_1=interpreter_1,  # Passing live object instance instead of string path
                                                                    interpreter_2=interpreter_2)
    
    print(" -> Applying DTLN TF-Lite Post-Processing...")
    output_ff_dtln_only = apply_dtln_post_tflite_realtime(interpreter_1, interpreter_2, free_field_input[0])
    
    output_ff_dtln = apply_dtln_post_tflite_realtime(interpreter_1, interpreter_2, output_ff_mvdr)
    
    # Output saves
    save_wav("1A_FF_output_dtln_only_16k.wav", FS, output_ff_dtln_only, output_folder)
    save_wav("1B_FF_input_mic0_16k.wav", FS, free_field_input[0], output_folder)

    save_wav("1C_FF_MVDR_output_16k.wav", FS, normalize_signal(output_ff_mvdr), output_folder)
    save_wav("1D_FF_DTLN_Post_16k.wav", FS, normalize_signal(output_ff_dtln), output_folder)
    save_wav("1E_mask.wav", FS, normalize_signal(output_ff_spp), output_folder)

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
    output_rm_mvdr, output_rm_spp = apply_mvdr_stft_bridge(room_input, vad_oracle_room, mic_coords, source_pos_2d, FS,
                                                                   interpreter_1=interpreter_1,  # Passing live object instance instead of string path
                                                                        interpreter_2=interpreter_2)
    
    print(" -> Applying DTLN TF-Lite Post-Processing...")
    output_rm_dtln_only = apply_dtln_post_tflite_realtime(interpreter_1, interpreter_2, room_input[0])
    output_rm_dtln = apply_dtln_post_tflite_realtime(interpreter_1, interpreter_2, output_rm_mvdr)

    # Output saves
    save_wav("2A_RM_input_mic0_16k.wav", FS, room_input[0], output_folder)
    save_wav("2B_RM_output_dtln_only_16k.wav", FS, output_rm_dtln_only, output_folder)
    save_wav("2C_RM_MVDR_output_16k.wav", FS, normalize_signal(output_rm_mvdr), output_folder)
    save_wav("2D_RM_DTLN_Post_16k.wav", FS, normalize_signal(output_rm_dtln), output_folder)
    save_wav("2E_RM_Mask.wav", FS, normalize_signal(output_rm_spp), output_folder)
    # -------------------------------------------------------------------
    # PHASE 3: WPE + MVDR + DTLN PIPELINE
    # -------------------------------------------------------------------
    print("\n--- PHASE 3: WPE + MVDR + DTLN PIPELINE ---")
    print(" -> Applying Online WPE Dereverberation (at 16kHz)...")
    wpe_output = process_wpe_online(room_input, delay=3, taps=7, stft_size=1024, stft_shift=128)
    
    print(" -> Applying SPP-guided   MVDR (at 16kHz)...")
    mvdr_wpe, _ = apply_mvdr_stft_bridge(wpe_output, vad_oracle_room, mic_coords, source_pos_2d, FS,
                                                    interpreter_1=interpreter_1,  # Passing live object instance instead of string path
                                                        interpreter_2=interpreter_2)
    
    print(" -> Applying DTLN TF-Lite Post-Processing...")
    dtln_wpe = apply_dtln_post_tflite_realtime(interpreter_1, interpreter_2, mvdr_wpe)
    
    # Save final results
    save_wav("3_WPE_16k.wav", FS, normalize_signal(wpe_output[0]), output_folder)
    save_wav("3_WPE_MVDR_16k.wav", FS, normalize_signal(mvdr_wpe), output_folder)
    save_wav("3_WPE_MVDR_DTLN_16k.wav", FS, normalize_signal(dtln_wpe), output_folder)

    print("\n -> Full integrated pipeline completed successfully.")