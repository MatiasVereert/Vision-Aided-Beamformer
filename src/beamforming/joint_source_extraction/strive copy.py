import os
import sys
import numpy as np
from beamforming.kmvdr.system import LowRankAdaptive
from propagation.simulate_acoustics import SimAcoustic
from beamforming.array.microphone import Microphone
from utils.audio import save_wav
from beamforming.dereverberation.wpe import apply_wpe 
from beamforming.signal_model import near_field_steering_vector 

def generate_near_field_sv_matrix(freqs, Rs, mic_array, c=343.0):
    """
    Calcula la matriz de vectores direccionales de campo cercano con normalización robusta.
    """
    source_dist_origin = np.linalg.norm(Rs)
    distances = np.linalg.norm(Rs - mic_array, axis=1).reshape(-1, 1)
    distances = np.maximum(distances, 1e-12) # Prevenir división por cero

    delay_diff = (source_dist_origin - distances) / c
    
    # Matriz de fase (M, F)
    phase_term = np.exp(1j * 2 * np.pi * delay_diff @ freqs.reshape(1, -1))
    sv_matrix = phase_term / distances
    
    # Normalización espacial para evitar inestabilidad en la regularización
    sv_matrix_norm = np.linalg.norm(sv_matrix, axis=0)
    sv_matrix_norm = np.maximum(sv_matrix_norm, 1e-12)
    return sv_matrix / sv_matrix_norm

def ensure_folder(base_path, p, m1, m2, alpha, RT, loading, src_pos):
    pos_str = f"{src_pos[0]}_{src_pos[1]}_{src_pos[2]}"
    folder_name = f"P={p}_M={m1}x{m2}_RT={RT}_Src={pos_str}_Alpha={alpha}"
    full_path = os.path.join(base_path, folder_name)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    return full_path

import numpy as np
import scipy.signal as sig
# Asumo que tienes tus imports habituales aquí (SimAcoustic, ensure_folder, save_wav, etc.)

class OnlineSRIVE_N1:
    def __init__(self, num_mics, num_freqs, a_1, alpha=0.99, lambda_unit=10.0, lambda_scale=1e-4):
        self.M = num_mics
        self.F = num_freqs
        self.alpha = alpha
        self.lambda_unit = lambda_unit
        self.lambda_scale = lambda_scale
        self.a_1 = a_1 
        
        self.W = np.array([np.eye(self.M, dtype=complex) for _ in range(self.F)])
        self.Sigma_1 = np.zeros((self.F, self.M, self.M), dtype=complex)
        self.Sigma_noise = np.zeros((self.F, self.M, self.M), dtype=complex)
        
        for f in range(self.F):
            a_f = self.a_1[f].reshape(self.M, 1)
            self.Sigma_1[f] = self.lambda_scale * np.eye(self.M) + self.lambda_unit * (a_f @ a_f.conj().T)
            
        self.v_1 = 1.0 
        self.frame_count = 0 # Contador para estabilización inicial

    def process_frame(self, x_t):
        y_1_t = np.zeros(self.F, dtype=complex)
        
        # Paso 1: Salida actual
        for f in range(self.F):
            w_1 = self.W[f][:, 0] 
            y_1_t[f] = w_1.conj().T @ x_t[f]
            
        self.v_1 = np.mean(np.abs(y_1_t)**2)
        
        # Paso 2: Actualización (solo si ya pasaron M frames, según el artículo)
        if self.frame_count >= self.M:
            for f in range(self.F):
                x_f = x_t[f].reshape(self.M, 1)
                
                self.Sigma_1[f] = self.alpha * self.Sigma_1[f] + (1 - self.alpha) * (x_f @ x_f.conj().T) / (self.v_1 + 1e-10)
                self.Sigma_noise[f] = self.alpha * self.Sigma_noise[f] + (1 - self.alpha) * (x_f @ x_f.conj().T)
                
                a_f = self.a_1[f].reshape(self.M, 1)
                Pi_1 = self.Sigma_1[f] + self.lambda_scale * np.eye(self.M) + self.lambda_unit * (a_f @ a_f.conj().T)
                
                try:
                    Pi_1_inv = np.linalg.inv(Pi_1)
                    W_inv_H = np.linalg.inv(self.W[f].conj().T)
                except np.linalg.LinAlgError:
                    continue # Saltamos este frame en esta frecuencia si hay singularidad
                    
                e_1 = np.zeros((self.M, 1)); e_1[0] = 1.0
                
                w_1_unconstrained = Pi_1_inv @ W_inv_H @ e_1
                w_1_hat = self.lambda_unit * Pi_1_inv @ a_f
                
                h_1 = (w_1_unconstrained.conj().T @ Pi_1 @ w_1_unconstrained)[0,0].real
                h_hat_1 = (w_1_unconstrained.conj().T @ Pi_1 @ w_1_hat)[0,0]
                
                if np.abs(h_hat_1) == 0:
                    w_1_new = (1.0 / np.sqrt(h_1)) * w_1_unconstrained + w_1_hat
                else:
                    h_tilde_1 = (h_hat_1 / (2 * h_1)) * (-1 + np.sqrt(1 + (4 * h_1) / (np.abs(h_hat_1)**2)))
                    w_1_new = h_tilde_1 * w_1_unconstrained + w_1_hat
                    
                self.W[f][:, 0] = w_1_new.flatten()
                
                # Actualización de W_Z
                W_S = self.W[f][:, 0:1] 
                E_S = np.zeros((self.M, 1)); E_S[0] = 1.0
                E_Z = np.zeros((self.M, self.M-1)); np.fill_diagonal(E_Z[1:], 1.0)

                term1 = W_S.conj().T @ self.Sigma_noise[f] @ E_S
                term2 = W_S.conj().T @ self.Sigma_noise[f] @ E_Z

                # Aplicar la ecuación (25) y actualizar W
                W_Z_new = np.vstack([-np.linalg.inv(term1) @ term2, np.eye(self.M - 1)])
                self.W[f][:, 1:] = W_Z_new 

                # Añadir pequeña constante para estabilidad numérica
                Pi_1_stable = Pi_1 + 1e-6 * np.eye(self.M)
                try:
                    Pi_1_inv = np.linalg.inv(Pi_1_stable)
                    W_inv_H = np.linalg.inv(self.W[f].conj().T + 1e-6 * np.eye(self.M))
                except np.linalg.LinAlgError:
                    continue
                                    
        self.frame_count += 1

        return y_1_t

if __name__ == "__main__":
    FS = 48000
    M1, M2 = 8, 1          # Arreglo lineal de 4 micrófonos
    
    # Constante global para tu función space_delay
    speed_of_sound = 343.0 

    print("=== INTEGRATION TEST: IDEAL VS ONLINE-WPEXSRIVE ===")
    print("=== MODO: CAMPO LIBRE (ANECOICO) ===")
    
    # Configuramos carpetas y variables base
    RANK_P = 1 
    ALPHA_IVE = 0.99 
    MIN_LOADING = 0
    base_data_path = "tests/data"
    output_folder = ensure_folder(base_data_path, RANK_P, M1, M2, ALPHA_IVE, 0.0, MIN_LOADING, [0,0,0])
    
    mic_spacing = 0.021 
    M = M1 * M2 

    x = np.linspace(0, (M1-1)*mic_spacing, M1)
    mic_coords = np.column_stack([x, np.zeros(M), np.zeros(M)])
    array_center = np.array([1.25, 2.0, 1.25])
    mic_coords = mic_coords - np.mean(mic_coords, axis=0) + array_center
    
    r = 1.0 
    ang_target = np.deg2rad(130)
    ang_interf = np.deg2rad(50)
    
    source_pos = array_center + np.array([r * np.cos(ang_target), r * np.sin(ang_target), 0.0])
    interf_pos1 = array_center + np.array([r * np.cos(ang_interf), r * np.sin(ang_interf), 0.0])

    # Inicializamos la escena acústica (sin RT)
    acoustic_scene = SimAcoustic(mic_coords, array_mismatch=0.0, duration=10, fs=FS)

    source_path = "tools/data/signals/FA01_09.wav"
    int_path1 = "tools/data/signals/MC15_03.wav"

    acoustic_scene.set_source(source_path, gain=1, position=source_pos.reshape(1,3))
    acoustic_scene.set_interference(int_path1, gain=1, position=interf_pos1.reshape(1,3))

    print(" -> Calculando simulación en campo libre...")
    # Usamos tu método free_field en lugar de compute_room_ISB
    # mode="ideal" asegura que usemos las coordenadas perfectas sin mismatch
    room_input_ideal = acoustic_scene.free_field(iSIR_dB=0, normalize=True, mode="ideal")
    
    save_wav("1_input_room_IDEAL.wav", FS, room_input_ideal[0], output_folder)
    
    # =========================================================================
    # 1. PARÁMETROS STFT (Basados en el paper: 8 ms frame, 4 ms shift)
    # =========================================================================

    nperseg = int(FS * 0.032) # 32 ms (buena resolución frecuencial)
    noverlap = nperseg - int(FS * 0.016) # 16 ms shift
    nfft = nperseg
    
    print(" -> Aplicando STFT...")
    freqs, times, X_stft = sig.stft(room_input_ideal, fs=FS, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    
    F_bins = X_stft.shape[1]
    T_frames = X_stft.shape[2]
    
    # =========================================================================
    # 2. CONSTRUCCIÓN DEL STEERING VECTOR (CONVENCIÓN DE FASE CORREGIDA)
    # =========================================================================
    a_1 = np.zeros((F_bins, M), dtype=complex)
    c = speed_of_sound
    d = mic_spacing

    for f_idx, freq in enumerate(freqs):
        if freq == 0: 
            a_1[f_idx, :] = 1.0 / np.sqrt(M)
        else:
            # Ecuación (13) y (56) del documento
            tau = - np.arange(M) * d * np.cos(ang_target) / c
            a_1[f_idx, :] = np.exp(1j * 2 * np.pi * freq * tau) / np.sqrt(M)
    # 3. EJECUCIÓN DEL ALGORITMO ONLINE-SRIVE
    # =========================================================================
    print(" -> Ejecutando Online-SRIVE...")
    # Instanciamos con los parámetros base recomendados
    srive = OnlineSRIVE_N1(num_mics=M, num_freqs=F_bins, a_1=a_1, alpha=0.9997, lambda_unit=10.0, lambda_scale=1e-4)
    
    Y_out = np.zeros((F_bins, T_frames), dtype=complex)
    
    for t_idx in range(T_frames):
        x_t = X_stft[:, :, t_idx].T 
        Y_out[:, t_idx] = srive.process_frame(x_t)
        
        if t_idx % 500 == 0:
            print(f"    Procesado frame {t_idx}/{T_frames}")

    # =========================================================================
    # 4. RECONSTRUCCIÓN (iSTFT) Y GUARDADO
    # =========================================================================
    print(" -> Reconstruyendo señal en el tiempo...")
    _, y_time = sig.istft(Y_out, fs=FS, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    
    save_wav("2_output_SRIVE_FreeField.wav", FS, y_time, output_folder)
    print(" -> Proceso finalizado exitosamente.")