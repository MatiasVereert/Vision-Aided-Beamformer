import numpy as np 
import scipy.signal as signal
import matplotlib.pyplot as plt

# Assuming the import works correctly in your local environment
from beamforming.signal_model import compute_rtf_steering_vector
import numpy as np 
import scipy.signal as signal
import matplotlib.pyplot as plt
from utils.audio import  normalize_signal


# Assuming the import works correctly in your local environment
from beamforming.signal_model import compute_rtf_steering_vector

import numpy as np
import numpy as np
import numpy as np

import numpy as np
import numpy as np

class RobustOnlineCGMM:
    def __init__(self, K_freqs, M_mics, sv_oracle):
        self.K = K_freqs
        self.M = M_mics
        self.num_classes = 2
        self.sv_oracle = sv_oracle
        
        self.R = np.zeros((self.K, self.num_classes, self.M, self.M), dtype=np.complex128)
        self.invR = np.zeros_like(self.R)
        self.alpha = np.ones((self.K, self.num_classes)) / self.num_classes
        self.Eta = np.zeros(self.K) 
        self.Lambda = np.zeros((self.K, self.num_classes)) 

    def process_chunk(self, X_chunk, em_iterations=3):
        K, T, M = X_chunk.shape
        gamma = np.zeros((K, self.num_classes, T))
        Phi = np.zeros((K, self.num_classes, T), dtype=np.float64)
        
        # --- CRITICAL FIX: DYNAMIC INITIALIZATION FOR SPEAKER SEPARATION ---
        # If this is the very first chunk (Eta is 0), initialize matrices dynamically
        if np.sum(self.Eta) == 0:
            for f in range(self.K):
                # 1. Target Class: Initialized using the known geometry
                v = self.sv_oracle[f:f+1, :].conj().T  
                self.R[f, 0] = v @ v.conj().T + 1e-6 * np.eye(self.M)
                
                # 2. Interference Class: Initialized using empirical data covariance
                # This captures the reverberation and the competing directional speaker
                R_empirical = np.einsum('tm,tn->mn', X_chunk[f], X_chunk[f].conj()) / T
                
                # Subtract the oracle target steering vector from the empirical mixture
                # to strongly bias Class 1 towards the "other" speaker
                R_interf = R_empirical - (0.5 * self.R[f, 0])
                
                # Ensure the matrix is mathematically valid (Hermitian and semi-positive definite)
                R_interf = 0.5 * (R_interf + R_interf.conj().T)
                vals, vecs = np.linalg.eigh(R_interf)
                vals = np.maximum(vals, 1e-6) # Floor negative eigenvalues
                self.R[f, 1] = (vecs * vals) @ vecs.conj().T
                
                self.invR[f, 0] = np.linalg.inv(self.R[f, 0])
                self.invR[f, 1] = np.linalg.inv(self.R[f, 1])
        # -----------------------------------------------------------------

        for itr in range(em_iterations):
            # E-STEP
            for c in range(self.num_classes):
                self.R[:, c] = 0.5 * (self.R[:, c] + self.R[:, c].conj().transpose(0, 2, 1))
                X_invR = np.einsum('kmn,ktn->ktm', self.invR[:, c], X_chunk)
                quad_form = np.real(np.einsum('ktm,ktm->kt', X_chunk.conj(), X_invR))
                Phi[:, c, :] = quad_form / M
                
                det_R = np.abs(np.linalg.det(self.R[:, c] + 1e-8 * np.eye(M)))
                log_prob = -M * np.log(Phi[:, c, :] * np.pi + 1e-10) - np.log(det_R[:, np.newaxis] + 1e-10) - M
                gamma[:, c, :] = log_prob + np.log(self.alpha[:, c, np.newaxis] + 1e-10)
                
            max_log = np.max(gamma, axis=1, keepdims=True)
            gamma_lin = np.exp(gamma - max_log)
            gamma = gamma_lin / (np.sum(gamma_lin, axis=1, keepdims=True) + 1e-10)
            post_sum = np.sum(gamma, axis=2)

            # M-STEP (MAP Update)
            lambda_next = self.Lambda + post_sum
            tmpConst = (self.Eta + M + 1) / 2
            num_w = self.Lambda + tmpConst[:, np.newaxis]
            den_w = lambda_next + tmpConst[:, np.newaxis]
            
            for c in range(self.num_classes):
                weight = gamma[:, c, :] / (Phi[:, c, :] + 1e-10)
                R_new = np.einsum('ktm,ktn->kmn', X_chunk * weight[:, :, np.newaxis], X_chunk.conj())
                
                # Update matrices using MAP logic
                prior_part = self.R[:, c] * (num_w[:, c, np.newaxis, np.newaxis] / den_w[:, c, np.newaxis, np.newaxis])
                new_part = R_new * (1.0 / den_w[:, c, np.newaxis, np.newaxis])
                self.R[:, c] = prior_part + new_part
                self.invR[:, c] = np.linalg.inv(self.R[:, c] + 1e-7 * np.eye(M))
            
            self.alpha = post_sum / T

        # Aligment/Permutation check
        for f in range(K):
            score0 = np.abs(self.sv_oracle[f].conj() @ self.R[f, 0] @ self.sv_oracle[f])
            score1 = np.abs(self.sv_oracle[f].conj() @ self.R[f, 1] @ self.sv_oracle[f])
            if score1 > score0:
                self.R[f, 0], self.R[f, 1] = self.R[f, 1].copy(), self.R[f, 0].copy()
                self.invR[f, 0], self.invR[f, 1] = self.invR[f, 1].copy(), self.invR[f, 0].copy()
                gamma[f, 0], gamma[f, 1] = gamma[f, 1].copy(), gamma[f, 0].copy()

        self.Eta += T
        self.Lambda += post_sum
        return gamma[:, 0, :]

def SPP_MVDR_recursive(X_stft, fs, array_geometry, source_pos, save_weights=False, chunk_size=20):
    K, T, M = X_stft.shape  
    frecs = np.linspace(0, fs/2, K)
    sv = compute_rtf_steering_vector(frecs, source_pos, array_geometry, ref_mic_idx=0, mode="near_field", squeeze=True)
    
    engine = RobustOnlineCGMM(K_freqs=K, M_mics=M, sv_oracle=sv)
    Y_stft = np.zeros((K, T), dtype=np.complex128)
    weights_rec = np.zeros((K, T, M), dtype=np.complex128)
    num_chunks = int(np.ceil(T / chunk_size))

    for c in range(num_chunks):
        start, end = c * chunk_size, min((c + 1) * chunk_size, T)
        X_chunk = X_stft[:, start:end, :]
        
        # 1. Update CGMM and get the probabilities
        P_chunk = engine.process_chunk(X_chunk)

        # 2. Use the matrices estimated by CGMM directly
        # R_target = engine.R[:, 0]
        # R_noise = engine.R[:, 1]
        
        for m_local in range(X_chunk.shape[1]):
            m_idx = start + m_local
            
            # Extract robust RTF from the target matrix of the CGMM
            # We use subtraction for extra robustness as per your preference
            R_ss = engine.R[:, 0] - engine.R[:, 1]
            sv_robust = np.zeros_like(sv)
            for f in range(K):
                # Ensure matrix is valid for EVD
                mat = 0.5 * (R_ss[f] + R_ss[f].conj().T)
                vals, vecs = np.linalg.eigh(mat)
                v_dom = vecs[:, -1]
                sv_robust[f] = v_dom / (v_dom[0] + 1e-12)

            # 3. Compute MVDR using the Noise matrix from CGMM
            Rn_stable = engine.R[:, 1] + 1e-6 * np.eye(M)
            Rn_inv = np.linalg.inv(Rn_stable)
            
            num = np.einsum("fmn,fn->fm", Rn_inv, sv_robust)
            den = np.einsum("fm,fm->f", sv_robust.conj(), num)
            weights = num / (den[:, np.newaxis] + 1e-12)
            
            weights_rec[:, m_idx, :] = weights
            Y_stft[:, m_idx] = np.einsum("fm,fm->f", weights.conj(), X_chunk[:, m_local, :])

    return (Y_stft, weights_rec) if save_weights else Y_stft

import scipy.signal as signal

# Assuming these are available from your local environment modules
from beamforming.signal_model import compute_rtf_steering_vector
    
import os
from propagation.simulate_acoustics import SimAcoustic
from utils.audio import save_wav
import os
import numpy as np
import scipy.signal as signal

# Assuming these are available from your local environment modules
from beamforming.signal_model import compute_rtf_steering_vector
# from simulation_module import SimAcoustic 
# from utils import save_wav, normalize_signal 
# from your_mvdr_module import SPP_SPP_MVDR_recursive 
# from your_wpe_module import process_wpe_online

def apply_mvdr_stft_bridge(time_domain_input, vad_oracle, mic_coords, source_pos_2d, fs, length_fft=512, hop_length_fft=256):
    """
    Helper function to wrap the STFT -> MVDR -> ISTFT process.
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
    
    # Pad VAD to avoid index out of bounds during the last STFT frames
    vad_padded = np.pad(vad_oracle, (0, length_fft + hop_length_fft), mode='constant')

    # Execute the Recursive MVDR
    Y_stft = SPP_MVDR_recursive(
        X_stft=X_stft, 
        fs=fs, 
        array_geometry=mic_coords, 
        source_pos=source_pos_2d, 

    )
    
    # Compute Inverse STFT
    _, y_time = signal.istft(
        Y_stft, 
        fs=fs, 
        nperseg=length_fft, 
        noverlap=length_fft - hop_length_fft
    )
    
    # Truncate to original length
    original_length = time_domain_input.shape[1]
    return y_time[:original_length]



from beamforming.MWF.SP_SDW_MWF_base import process_wpe_online




if __name__ == "__main__":
    # Basic simulation parameters
    FS = 16000
    M1, M2 = 12, 1          
    M = M1 * M2
    speed_of_sound = 343.0 

    iSIR_dB = 0
    
    print("=== INTEGRATION TEST: PIPELINE (FREE-FIELD, ROOM, WPE+ROOM) ===")
    
    output_folder = "tests/data/mvdr_CGMM_output"
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
    
    print(" -> Applying Recursive MVDR...")
    output_ff = apply_mvdr_stft_bridge(free_field_input, vad_oracle_ff, mic_coords, source_pos_2d, FS)
    
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

    print(" -> Applying Recursive MVDR (Without WPE)...")
    output_rm = apply_mvdr_stft_bridge(room_input, vad_oracle_room, mic_coords, source_pos_2d, FS)
    
    save_wav("4_ROOM_output_final.wav", FS, normalize_signal(output_rm), output_folder)

    # -------------------------------------------------------------------
    # PHASE 3: WPE DEREVERBERATION + RECURSIVE MVDR
    # -------------------------------------------------------------------
    print("\n--- PHASE 3: WPE + MVDR PIPELINE ---")
    print(" -> Applying Online WPE Dereverberation on Room Simulation...")
    
    wpe_output = process_wpe_online(room_input)
    
    save_wav("5_WPE_input_mix_mic0.wav", FS, wpe_output[0], output_folder)

    print(" -> Applying Recursive MVDR on Dereverberated Signals...")
    output_wpe = apply_mvdr_stft_bridge(wpe_output, vad_oracle_room, mic_coords, source_pos_2d, FS)
    
    save_wav("6_WPE_ROOM_output_final.wav", FS, normalize_signal(output_wpe), output_folder)

    print("\n -> Pipeline completed successfully.")