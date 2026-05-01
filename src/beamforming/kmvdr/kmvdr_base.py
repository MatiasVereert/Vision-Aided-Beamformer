import numpy as np
from scipy import signal
from scipy.io import wavfile
import os
import warnings
C_SOUND = 343.0
from beamforming.signal_model import steering_vector

import numpy as np
import scipy.signal as signal


# Note: Assuming this import is available in your environment just like in base.py
from beamforming.signal_model import compute_rtf_steering_vector

def KMVDR_recursive(X_stft, vad, fs, array_geometry, source_pos, M1, M2, P=2, 
                    alpha=0.95, ALS_iterations=2, beta=1e-3, min_loading=1e-6, 
                    length_fft=512, hop_length_fft=256, save_weights=False):
    """
    Functional implementation of the Kronecker MVDR beamformer.
    It expects X_stft in the shape (Frequency_bins, Time_frames, Mics).
    """
    K, T, M = X_stft.shape
    
    if M != M1 * M2:
        raise ValueError(f"Total number of mics M ({M}) must equal M1 ({M1}) * M2 ({M2})")

    # Get steering vectors, expected shape (K, M)
    frecs = np.linspace(0, fs / 2, K)
    sv = compute_rtf_steering_vector(frecs, source_pos, array_geometry, 
                                     ref_mic_idx=0, mode="near_field", squeeze=True)
    sv = np.nan_to_num(sv)

    # Initialize output complex STFT matrix and weights storage
    Y_stft = np.zeros((K, T), dtype=np.complex128)
    weights_rec = np.zeros((K, T, M), dtype=np.complex128)

    # Initialize state variables
    R_cov = np.tile(np.eye(M, dtype=np.complex128) * 1e-5, (K, 1, 1))
    h1 = np.random.randn(K, M1, P) + 1j * np.random.randn(K, M1, P)
    h2 = np.random.randn(K, M2, P) + 1j * np.random.randn(K, M2, P)
    
    # Initialize adaptive loading factor
    current_delta = np.ones((K, 1, 1)) * 0.1
    I_M = np.eye(M)[None, :, :]
    
    # WNG Control loop parameters
    target_wng_lin = 10**(-6.0 / 10)
    step_up = 1.05
    step_down = 0.98

    for m in range(T):
        # Extract the current frame across all frequencies
        X_frame = X_stft[:, m, :]       # Shape: (K, M)
        x_t = X_frame[:, :, None]       # Shape: (K, M, 1)

        # Define VAD frame state
        vad_frame = vad[m * hop_length_fft : length_fft + m * hop_length_fft]
        vad_status = np.mean(vad_frame) > 0.1

        # 1. Update covariance matrix R ONLY when target speech is absent
        if not vad_status:
            update_term = np.matmul(x_t, x_t.conj().transpose(0, 2, 1))
            R_cov = alpha * R_cov + (1 - alpha) * update_term

        # 2. Apply Adaptive Loading
        tr_R = np.real(np.trace(R_cov, axis1=1, axis2=2))
        adaptive_loading = current_delta * (tr_R[:, None, None] / M)
        loading = np.maximum(adaptive_loading, min_loading)
        R_loaded = R_cov + I_M * loading

        # 3. ALS Optimization
        h1_curr, h2_curr = h1, h2
        
        for _ in range(ALS_iterations):
            # Step h1
            H2 = np.einsum('ab, fcp -> facbp', np.eye(M1), h2_curr).reshape(K, M, M1 * P)
            Phi_y2 = H2.conj().transpose(0, 2, 1) @ R_loaded @ H2
            d_2 = H2.conj().transpose(0, 2, 1) @ sv[:, :, None]
            
            h1_flat = np.linalg.pinv(Phi_y2, rcond=1e-5) @ d_2
            den = d_2.conj().transpose(0, 2, 1) @ h1_flat
            h1_flat = h1_flat / (den + 1e-12)
            h1_curr = h1_flat.reshape(K, M1, P)

            # Step h2
            H1_raw = np.einsum('fap, cd -> facpd', h1_curr, np.eye(M2)).transpose(0, 1, 2, 4, 3) 
            H1 = H1_raw.reshape(K, M, M2 * P)
            Phi_y1 = H1.conj().transpose(0, 2, 1) @ R_loaded @ H1
            d_1 = H1.conj().transpose(0, 2, 1) @ sv[:, :, None]
            
            h2_flat = np.linalg.pinv(Phi_y1, rcond=1e-5) @ d_1
            den = d_1.conj().transpose(0, 2, 1) @ h2_flat
            h2_flat = h2_flat / (den + 1e-12)
            h2_curr = h2_flat.reshape(K, M2, P)

        # Update subfilters state
        h1, h2 = h1_curr, h2_curr

        # Calculate final combined filter
        h_total = np.einsum('fap, fbp -> fab', h1, h2).reshape(K, M)

        # Save weights if requested
        if save_weights:
            weights_rec[:, m, :] = h_total

        # 4. Feedback Loop for robustness (WNG)
        w_norm2 = np.sum(np.abs(h_total)**2, axis=1)[:, None, None]
        current_wng = 1.0 / (w_norm2 + 1e-12)
        
        factor = np.where(current_wng < target_wng_lin, step_up, step_down)
        current_delta *= factor
        current_delta = np.clip(current_delta, 1e-6, 1.0)

        # 5. Apply weights to the current observation
        Y_stft[:, m] = (h_total.conj()[:, None, :] @ x_t)[:, 0, 0]

    if save_weights:
        return Y_stft, weights_rec
    else:
        return Y_stft

def apply_kmvdr_stft_bridge(time_domain_input, vad_oracle, mic_coords, source_pos_2d, fs, 
                            M1, M2, P=2, length_fft=512, hop_length_fft=256):
    """
    Helper function to wrap the STFT -> KMVDR -> ISTFT process.
    """
    # Compute STFT: output shape (Mics, Freqs, Frames)
    freqs, times, Zxx = signal.stft(
        time_domain_input, 
        fs=fs, 
        nperseg=length_fft, 
        noverlap=length_fft - hop_length_fft
    )
    
    # Transpose Zxx from (M, K, T) to (K, T, M) to match KMVDR_recursive expectations
    X_stft = np.transpose(Zxx, (1, 2, 0))
    
    # Pad VAD to avoid index out of bounds during the last STFT frames
    vad_padded = np.pad(vad_oracle, (0, length_fft + hop_length_fft), mode='constant')

    # Execute the Recursive KMVDR
    # We pass M1, M2 and P which are specific to the Kronecker decomposition
    Y_stft = KMVDR_recursive(
        X_stft=X_stft, 
        vad=vad_padded, 
        fs=fs, 
        array_geometry=mic_coords, 
        source_pos=source_pos_2d,
        M1=M1,
        M2=M2,
        P=P,
        length_fft=length_fft, 
        hop_length_fft=hop_length_fft
    )
    
    # Compute Inverse STFT
    _, y_time = signal.istft(
        Y_stft, 
        fs=fs, 
        nperseg=length_fft, 
        noverlap=length_fft - hop_length_fft
    )
    
    # Truncate to original length to ensure consistency
    original_length = time_domain_input.shape[1]
    return y_time[:original_length]



from propagation.simulate_acoustics import SimAcoustic
from utils.audio import save_wav, normalize_signal

if __name__ == "__main__":
    
    import os
    import numpy as np
    from propagation.simulate_acoustics import SimAcoustic
    from utils.audio import save_wav
    from beamforming.MWF.SP_SDW_MWF_base import process_wpe_online

    # Basic simulation parameters identical to base.py
    FS = 16000
    M1, M2 = 12, 1          
    M = M1 * M2
    
    # According to theory, P must be <= min(M1, M2). Since M2=1, P must be 1.
    P = 1 
    
    speed_of_sound = 343.0 
    iSIR_dB = 0
    
    print(f"=== INTEGRATION TEST: KMVDR PIPELINE (M={M}, M1={M1}, M2={M2}, P={P}) ===")
    
    # Change output folder to avoid overwriting the original MVDR baseline
    output_folder = "tests/data/kmvdr_integration_output"
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
    acoustic_scene.set_source("tools/data/signals/FA01_09.wav", gain=1, position=source_pos_2d)
    acoustic_scene.set_interference("tools/data/signals/MC15_03.wav", gain=1, position=interf_pos1.reshape(1,3))

    # Helper function to normalize signals before saving
    def normalize_signal(sig):
        max_abs = np.max(np.abs(sig))
        if max_abs > 0:
            return sig * (0.99 / max_abs)
        return sig

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
        # Save to cache
        np.savez(cache_ff_path, input=free_field_input, vad=vad_oracle_ff)
        
    save_wav("1_FF_input_mix_mic0.wav", FS, free_field_input[0], output_folder)
    
    print(" -> Applying Recursive KMVDR...")
    output_ff = apply_kmvdr_stft_bridge(free_field_input, vad_oracle_ff, mic_coords, source_pos_2d, FS, M1=M1, M2=M2, P=P)
    
    save_wav("2_FF_output_final.wav", FS, normalize_signal(output_ff), output_folder)

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
        # Save to cache
        np.savez(cache_room_path, input=room_input, vad=vad_oracle_room)
    
    save_wav("3_ROOM_input_mix_mic0.wav", FS, room_input[0], output_folder)

    print(" -> Applying Recursive KMVDR (Without WPE)...")
    output_rm = apply_kmvdr_stft_bridge(room_input, vad_oracle_room, mic_coords, source_pos_2d, FS, M1=M1, M2=M2, P=P)
    
    save_wav("4_ROOM_output_final.wav", FS, normalize_signal(output_rm), output_folder)

    # -------------------------------------------------------------------
    # PHASE 3: WPE DEREVERBERATION + RECURSIVE KMVDR
    # -------------------------------------------------------------------
    print("\n--- PHASE 3: WPE + KMVDR PIPELINE ---")
    print(" -> Applying Online WPE Dereverberation on Room Simulation...")
    
    wpe_output = process_wpe_online(room_input)
    
    save_wav("5_WPE_input_mix_mic0.wav", FS, wpe_output[0], output_folder)

    print(" -> Applying Recursive KMVDR on Dereverberated Signals...")
    output_wpe = apply_kmvdr_stft_bridge(wpe_output, vad_oracle_room, mic_coords, source_pos_2d, FS, M1=M1, M2=M2, P=P)
    
    save_wav("6_WPE_ROOM_output_final.wav", FS, normalize_signal(output_wpe), output_folder)

    print("\n -> Pipeline completed successfully.")