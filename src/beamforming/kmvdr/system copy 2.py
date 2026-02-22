import numpy as np
from scipy import signal
from scipy.io import wavfile
import os
import warnings
C_SOUND = 343.0
from beamforming.signal_model import steering_vector

import numpy as np
import scipy.signal as signal
import numpy as np
import scipy.signal as signal
from beamforming.signal_model import steering_vector

class LowRankAdaptive:
    def __init__(self, mic_array, fs, alpha=0.95):
        self.mic_array = mic_array
        self.fs = fs
        self.alpha = alpha
        self.ALS_iterations = 2
        
        # Estado del Beamformer
        self.R_cov = None 
        self.h1 = None
        self.h2 = None
        
        # --- ESTADO DEL VAD (MODIFICADO) ---
        self.blocking_matrix = None      # Se calculará al iniciar el bloque
        self.vad_threshold_db = 9.0      # Umbral de Ratio (Signal/Blocked) en dB
        self.vad_smoothing = 0.9         # Suavizado temporal de la decisión VAD
        self.vad_prob = 0.0              # Estado actual de probabilidad de voz
        
        # Parámetros para la Matriz de Bloqueo Robusta (Eigen-Blocking)
        self.svd_rank_reduction = 5      # Quedarse con los 3 componentes principales (Eigenbeams)
        self.protection_radius = 0.2     # Radio de incertidumbre (metros) alrededor del target

        # Grabación de pesos
        self.FPS_rec = 15
        self.h_record = None

        # banwith
        self.f_min = 100
        self.f_max = 8000

    def _band_pass_filter(self, x):
        band = [self.f_min, self.f_max]
        attenuation = 50
        gap = 25
        n_tabs = int( attenuation * self.fs / (22 *gap))
        edges = [0, self.f_min - gap, self.f_min, self.f_max, self.f_max + gap, 0.5 * self.fs]
        tabs = signal.remez(n_tabs, edges, [0, 1, 0], fs = self.fs )
        y = signal.lfilter(tabs, 1.0, x, axis = 1)
        return y

    def _compute_eigen_blocking_matrix(self, freqs, target_pos, num_points=50):
        """
        Calcula la Matriz de Bloqueo Robusta usando SVD (Eigen-Blocking).
        Genera una nube de puntos alrededor del target, extrae los componentes principales
        y construye un proyector ortogonal al subespacio de la señal.
        """
        F = len(freqs)
        M = self.mic_array.shape[0]
        
        # 1. Generar nube de puntos (Restricciones Múltiples)
        # Simulamos incertidumbre de posición alrededor del target
        perturbations = (np.random.rand(num_points, 3) - 0.5) * 2 * self.protection_radius
        cloud_positions = target_pos + perturbations
        # Aseguramos incluir el punto central exacto
        cloud_positions = np.vstack([target_pos, cloud_positions])
        
        # 2. Construir Matriz de Manifold A (Steering Vectors para toda la nube)
        # A tiene forma (F, M, N_points)
        A = np.zeros((F, M, len(cloud_positions)), dtype=complex)
        
        for i, pos in enumerate(cloud_positions):
            # Usamos la función existente steering_vector
            # Retorna (F, M), lo transponemos a (F, M, 1) conceptualmente
            sv = steering_vector(freqs, pos, self.fs, self.mic_array, c=343.0, squeeze=True, mode="near_field")
            A[:, :, i] = np.nan_to_num(sv)

        # 3. Aplicar SVD por frecuencia para encontrar el subespacio dominante
        # A = U * S * Vh
        # U contiene los "Eigenbeams" ortogonales
        U, S, Vh = np.linalg.svd(A, full_matrices=False)
        
        # 4. Selección de Rango (Low Rank Constraint)
        # Nos quedamos con los 'K' vectores singulares más fuertes (self.svd_rank_reduction)
        # U_signal: (F, M, K)
        K = min(self.svd_rank_reduction, M - 1) # Asegurar que K < M
        U_signal = U[:, :, :K]
        
        # 5. Construir el Proyector de Bloqueo (Blocking Matrix)
        # B = I - U_s * U_s^H
        # Esto proyecta cualquier señal al "Espacio Nulo" de la fuente (Solo Ruido)
        I = np.eye(M)[None, :, :]
        # Einsum: U_s x U_s^H -> (F, M, M)
        Projection_Signal = np.matmul(U_signal, U_signal.conj().transpose(0, 2, 1))
        Blocking_Matrix = I - Projection_Signal
        
        return Blocking_Matrix

    def block_process(self, 
                  input_signals, 
                  target_pos, M1: int, M2: int, P=2,
                  record_scene=True, mode="near_field", VAD=False,
                  min_loading=1e-6): 
        
        # Filter input 
        input_signals = self._band_pass_filter(input_signals)

        if P == 1 and mode == 'near_field':
            print("Advertencia: El rango P = 1 es incompatible con campo cercano")

        peak = np.max(np.abs(input_signals))
        if peak > 1e-9: input_signals /= peak

        n_window, n_overlap = 256, 192
        M = M1 * M2

        f, t, X = signal.stft(x=input_signals, fs=self.fs, nperseg=n_window, 
                              noverlap=n_overlap, window='hann', axis=1)
        X = X.transpose(1, 2, 0) # (F, T, M)
        F_bins, T_frames, _ = X.shape

        # Inicialización de grabación
        if record_scene:
            t_frame = n_overlap / self.fs
            estimated_records = int(T_frames / (self.fs/n_overlap / self.FPS_rec)) + 10
            self.h_record = np.zeros((F_bins, M, estimated_records), dtype=complex)
        record_idx = 0

        Y_stft = np.zeros((F_bins, T_frames), dtype=complex) 
        sv = steering_vector(f, target_pos, self.fs, self.mic_array, 1, squeeze=True, mode="near_field")
        sv = np.nan_to_num(sv) 

        # --- PRE-CÁLCULO: MATRIZ DE BLOQUEO ROBUSTA (EIGEN-BLOCKING) ---
        # Calculamos B una sola vez para el bloque (o actualizar si target es móvil)
        if self.blocking_matrix is None:
            self.blocking_matrix = self._compute_eigen_blocking_matrix(f, target_pos)

        # --- INICIALIZACIÓN DE ESTADO ---
        if self.R_cov is None:
             self.R_cov = np.zeros((F_bins, M, M), dtype=complex)
             self.R_cov[:] = np.eye(M)[None, :, :] * 1e-5
             self.h1 = np.random.randn(F_bins, M1, P) + 1j*np.random.randn(F_bins, M1, P)
             self.h2 = np.random.randn(F_bins, M2, P) + 1j*np.random.randn(F_bins, M2, P)
             self.current_delta = np.ones((F_bins, 1, 1)) * 0.1

        I_M = np.eye(M)[None, :, :] 

        # --- PARÁMETROS DEL LAZO DE CONTROL (WNG) ---
        target_wng_dB = -9.0                
        target_wng_lin = 10**(target_wng_dB/10)
        step_up = 1.05                      
        step_down = 0.98                    

        print(f"   -> Procesando {T_frames} tramas con Robustez Adaptativa y VAD Espacial...")

        with np.errstate(all='ignore'):

            for t_idx in range(T_frames):

                x_t = X[:, t_idx, :, None] # (F, M, 1)

                # --- 1. VAD ESPACIAL (SIGNAL-TO-BLOCKING RATIO) ---
                # A. Energía Camino Principal (Estimación Rápida)
                # Alineamos con el steering vector principal (Delay-and-Sum conceptual)
                y_fixed = np.sum(x_t.conj() * sv[:, :, None], axis=1) # (F, 1)
                p_signal = np.sum(np.abs(y_fixed)**2, axis=0) # Potencia total en todo el espectro (scalar)
                
                # B. Energía Camino Bloqueado (Blocking Matrix)
                # Proyectamos x_t sobre el espacio nulo B: x_blocked = B * x_t
                x_blocked = np.matmul(self.blocking_matrix, x_t) # (F, M, 1)
                p_blocked = np.sum(np.linalg.norm(x_blocked, axis=1)**2) # Potencia total residual (scalar)

                # C. Cálculo del Ratio y Decisión
                # Evitar división por cero
                p_blocked = np.maximum(p_blocked, 1e-12)
                ratio_db = 10 * np.log10((p_signal / p_blocked) + 1e-12)
                
                # Probabilidad instantánea (Sigmoide alrededor del umbral)
                inst_prob = 1.0 / (1.0 + np.exp(-1.0 * (ratio_db - self.vad_threshold_db)))
                
                # Suavizado temporal de la decisión
                self.vad_prob = self.vad_smoothing * self.vad_prob + (1 - self.vad_smoothing) * inst_prob

                # --- 2. ACTUALIZACIÓN R CONTROLADA ---
                # Si VAD es alto (Voz) -> alpha efectivo tiende a 1.0 (Congelar)
                # Si VAD es bajo (Ruido) -> alpha efectivo tiende a self.alpha (Aprender)
                effective_alpha = self.alpha + (1.0 - self.alpha) * self.vad_prob
                
                update_term = np.matmul(x_t, x_t.conj().transpose(0, 2, 1))
                self.R_cov = effective_alpha * self.R_cov + (1 - effective_alpha) * update_term
                
                # --- 3. APLICAR LOADING ADAPTATIVO ---
                tr_R = np.real(np.trace(self.R_cov, axis1=1, axis2=2))
                adaptive_loading = self.current_delta * (tr_R[:, None, None] / M)
                loading = np.maximum(adaptive_loading, min_loading)
                R_loaded = self.R_cov + I_M * loading

                # --- 4. OPTIMIZACIÓN ALS ---
                h1_curr, h2_curr = self.h1, self.h2
                
                for _ in range(self.ALS_iterations):
                    # Paso h1
                    H2 = np.einsum('ab, fcp -> facbp', np.eye(M1), h2_curr).reshape(F_bins, M, M1 * P)
                    Phi_y2 = H2.conj().transpose(0, 2, 1) @ R_loaded @ H2
                    d_2 = H2.conj().transpose(0, 2, 1) @ sv[:, :, None]
                    
                    h1_flat = np.linalg.pinv(Phi_y2, rcond=1e-5) @ d_2
                    den = d_2.conj().transpose(0, 2, 1) @ h1_flat
                    h1_flat = h1_flat / (den + 1e-12)
                    h1_curr = h1_flat.reshape(F_bins, M1, P)

                    # Paso h2
                    H1_raw = np.einsum('fap, cd -> facpd', h1_curr, np.eye(M2)).transpose(0, 1, 2, 4, 3) 
                    H1 = H1_raw.reshape(F_bins, M, M2 * P)
                    Phi_y1 = H1.conj().transpose(0, 2, 1) @ R_loaded @ H1
                    d_1 = H1.conj().transpose(0, 2, 1) @ sv[:, :, None]
                    
                    h2_flat = np.linalg.pinv(Phi_y1, rcond=1e-5) @ d_1
                    den = d_1.conj().transpose(0, 2, 1) @ h2_flat
                    h2_flat = h2_flat / (den + 1e-12)
                    h2_curr = h2_flat.reshape(F_bins, M2, P)

                self.h1, self.h2 = h1_curr, h2_curr
                h_total = np.einsum('fap, fbp -> fab', self.h1, self.h2).reshape(F_bins, M)

                # --- 5. FEEDBACK LOOP (WNG) ---
                w_norm2 = np.sum(np.abs(h_total)**2, axis=1)[:, None, None]
                current_wng = 1.0 / (w_norm2 + 1e-12)
                
                factor = np.where(current_wng < target_wng_lin, step_up, step_down)
                self.current_delta *= factor
                self.current_delta = np.clip(self.current_delta, 1e-6, 1.0)

                # --- 6. Grabación y Filtrado ---
                if record_scene:
                    rec_stride = int(self.fs / n_overlap / self.FPS_rec)
                    if (t_idx % rec_stride == 0) and (record_idx < self.h_record.shape[2]):
                        self.h_record[:, :, record_idx] = h_total
                        record_idx += 1

                Y_stft[:, t_idx] = (h_total.conj()[:, None, :] @ x_t)[:, 0, 0]

        _, y_out = signal.istft(Y_stft, fs=self.fs, window='hann', nperseg=n_window, noverlap=n_overlap)
        return y_out