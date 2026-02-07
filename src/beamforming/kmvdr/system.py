import numpy as np
from scipy import signal
from scipy.io import wavfile
import os
import warnings
C_SOUND = 343.0
from beamforming.signal_model import steering_vector

import numpy as np
import scipy.signal as signal

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
        
        # --- ESTADO DEL VAD (NUEVO) ---
        self.noise_level = 1e-3  # Estimación inicial del piso de ruido
        self.vad_hangover = 0    # Contador de resaca
        self.HANGOVER_MAX = 5    # Bloques a esperar después de que cese la voz (~50-100ms)
        self.VAD_THRESH_DB = 8   # Umbral: 8dB por encima del piso de ruido

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

        # Define lenght of the FIR filter
        n_tabs = int( attenuation * self.fs / (22 *gap))
        edges = [0, self.f_min - gap, self.f_min, self.f_max, self.f_max + gap, 0.5 * self.fs]
        tabs = signal.remez(n_tabs, edges, [0, 1, 0], fs = self.fs )

        # Apply the filter 
        y = signal.lfilter(tabs, 1.0, x, axis = 1)

        return y
    
    def block_process(self, 
                  input_signals, 
                  target_pos, M1: int, M2: int, P=2,
                  record_scene=True, mode="near_field", VAD=False,
                  min_loading=1e-6): # Bajamos el min_loading para permitir convergencia fina
        
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
        sv = steering_vector(f, target_pos, self.fs, self.mic_array, 1, squeeze=True, mode="far_field")
        sv = np.nan_to_num(sv) 

        # --- INICIALIZACIÓN DE ESTADO ---
        if self.R_cov is None:
             self.R_cov = np.zeros((F_bins, M, M), dtype=complex)
             self.R_cov[:] = np.eye(M)[None, :, :] * 1e-5
             self.h1 = np.random.randn(F_bins, M1, P) + 1j*np.random.randn(F_bins, M1, P)
             self.h2 = np.random.randn(F_bins, M2, P) + 1j*np.random.randn(F_bins, M2, P)
             
             # [NUEVO] Inicializamos el factor de carga adaptativo
             # Empezamos con un valor conservador (0.1) para evitar soplido al inicio
             self.current_delta = np.ones((F_bins, 1, 1)) * 0.1

        I_M = np.eye(M)[None, :, :] 

        # --- PARÁMETROS DEL LAZO DE CONTROL (WNG) ---
        target_wng_dB = -8.0                # Objetivo de robustez (industria estándar)
        target_wng_lin = 10**(target_wng_dB/10)
        step_up = 1.05                      # Subir 5% si estamos en peligro
        step_down = 0.98                    # Bajar 2% si estamos sobrados

        print(f"   -> Procesando {T_frames} tramas con Robustez Adaptativa...")

        with np.errstate(all='ignore'):

            for t_idx in range(T_frames):

                x_t = X[:, t_idx, :, None] # (F, M, 1)

                # --- 1. VAD / ACTUALIZACIÓN R ---
                # Nota: Mantengo tu lógica actual. Idealmente aquí iría el "if not is_speech"
                update_term = np.matmul(x_t, x_t.conj().transpose(0, 2, 1))
                self.R_cov = self.alpha * self.R_cov + (1 - self.alpha) * update_term
                
                # --- 2. APLICAR LOADING ADAPTATIVO (Pre-Cálculo) ---
                # Usamos self.current_delta que viene de la iteración anterior (o init)
                tr_R = np.real(np.trace(self.R_cov, axis1=1, axis2=2))
                
                # Normalizamos por la traza (potencia media) para que delta sea relativo
                # Shape broadcast: (F, 1, 1) * (F, 1, 1)
                adaptive_loading = self.current_delta * (tr_R[:, None, None] / M)
                
                # Aseguramos un piso mínimo numérico
                loading = np.maximum(adaptive_loading, min_loading)
                
                R_loaded = self.R_cov + I_M * loading

                # --- 3. OPTIMIZACIÓN ALS ---
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

                # Construcción del filtro final
                h_total = np.einsum('fap, fbp -> fab', self.h1, self.h2).reshape(F_bins, M)

                # --- 4. FEEDBACK LOOP (NUEVO) ---
                # Calculamos el WNG real obtenido con estos pesos
                # WNG = 1 / ||w||^2 (asumiendo ganancia unitaria en look direction)
                w_norm2 = np.sum(np.abs(h_total)**2, axis=1)[:, None, None] # (F, 1, 1)
                current_wng = 1.0 / (w_norm2 + 1e-12)
                
                # Lógica de Control:
                # Si WNG < target -> Peligro -> Subir Loading (STEP_UP)
                # Si WNG > target -> Seguro  -> Bajar Loading (STEP_DOWN)
                factor = np.where(current_wng < target_wng_lin, step_up, step_down)
                
                # Actualizamos y limitamos para seguridad
                self.current_delta *= factor
                self.current_delta = np.clip(self.current_delta, 1e-6, 1.0)

                # --- 5. Grabación y Filtrado ---
                if record_scene:
                    rec_stride = int(self.fs / n_overlap / self.FPS_rec)
                    if (t_idx % rec_stride == 0) and (record_idx < self.h_record.shape[2]):
                        self.h_record[:, :, record_idx] = h_total
                        record_idx += 1

                Y_stft[:, t_idx] = (h_total.conj()[:, None, :] @ x_t)[:, 0, 0]

        _, y_out = signal.istft(Y_stft, fs=self.fs, window='hann', nperseg=n_window, noverlap=n_overlap)
        return y_out
