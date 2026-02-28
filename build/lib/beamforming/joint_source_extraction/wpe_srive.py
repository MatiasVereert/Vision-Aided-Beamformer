import numpy as np

def generate_near_field_sv_matrix(freqs, Rs, mic_array, c=343.0):
    """
    Calcula la matriz de vectores direccionales de campo cercano.
    """
    source_dist_origin = np.linalg.norm(Rs)
    distances = np.linalg.norm(Rs - mic_array, axis=1).reshape(-1, 1)
    delay_diff = (source_dist_origin - distances) / c
    
    # Matriz de fase (M, F)
    phase_term = np.exp(1j * 2 * np.pi * delay_diff @ freqs.reshape(1, -1))
    sv_matrix = phase_term / distances
    
    # Normalización espacial
    sv_matrix_norm = np.linalg.norm(sv_matrix, axis=0)
    sv_matrix_norm[sv_matrix_norm == 0] = 1e-12
    return sv_matrix / sv_matrix_norm
import numpy as np

def generate_near_field_sv_matrix(freqs, Rs, mic_array, c=343.0):
    source_dist_origin = np.linalg.norm(Rs)
    distances = np.linalg.norm(Rs - mic_array, axis=1).reshape(-1, 1)
    delay_diff = (source_dist_origin - distances) / c
    
    phase_term = np.exp(1j * 2 * np.pi * delay_diff @ freqs.reshape(1, -1))
    sv_matrix = phase_term / distances
    
    sv_matrix_norm = np.linalg.norm(sv_matrix, axis=0)
    sv_matrix_norm[sv_matrix_norm == 0] = 1e-12
    return sv_matrix / sv_matrix_norm

class OnlineWPExSRIVE_N1_Vectorized:
    def __init__(self, M, L, D, Rs, mic_array, freqs, c=343.0, 
                 alpha=0.99, beta=0.9999, n_iter=2, lambda_unit=10.0, lambda_scale=1e-4):
        self.M = M
        self.L = L
        self.D = D
        self.alpha = alpha
        self.beta = beta
        self.n_iter = n_iter
        self.lambda_unit = lambda_unit
        self.lambda_scale = lambda_scale
        self.F = len(freqs)
        
        a_1_matrix = generate_near_field_sv_matrix(freqs, Rs, mic_array, c)
        self.a_1 = a_1_matrix.T[:, :, np.newaxis]
        
        self.W = np.tile(np.eye(M, dtype=complex), (self.F, 1, 1))
        self.W_invH = np.tile(np.eye(M, dtype=complex), (self.F, 1, 1))
        
        self.G = np.zeros((2, self.F, M * L, M), dtype=complex)
        self.R_inv = np.tile(np.eye(M * L, dtype=complex), (2, self.F, 1, 1))
        
        I_M = np.eye(M)[np.newaxis, :, :]
        a_1_H = self.a_1.conj().transpose(0, 2, 1)
        Pi_init = self.lambda_scale * I_M + self.lambda_unit * (self.a_1 @ a_1_H)
        self.Pi_inv = np.linalg.inv(Pi_init) 
        
        self.Sigma_Z = np.zeros((self.F, M, M), dtype=complex)
        self.v = np.ones(2)
        self.x_buffer = np.zeros((self.F, M, D + L), dtype=complex)
        
        self.e1 = np.zeros((self.F, M, 1), dtype=complex)
        self.e1[:, 0, 0] = 1.0
        self.E_Z = np.vstack((np.zeros((1, M - 1)), np.eye(M - 1)))[np.newaxis, :, :]

    def process_frame(self, x_t):
        self.x_buffer = np.roll(self.x_buffer, shift=1, axis=2)
        self.x_buffer[:, :, 0] = x_t.T 
        
        x_bar = self.x_buffer[:, :, self.D : self.D + self.L].reshape(self.F, self.M * self.L, 1)
        x_f = x_t.T[:, :, np.newaxis] 
        x_bar_H = x_bar.conj().transpose(0, 2, 1) 
        
        for iter_idx in range(self.n_iter):
            y_1 = x_f - self.G[0].conj().transpose(0, 2, 1) @ x_bar
            y_2 = x_f - self.G[1].conj().transpose(0, 2, 1) @ x_bar
            
            w_1 = self.W[:, :, 0:1] 
            W_Z = self.W[:, :, 1:]  
            
            s_hat = w_1.conj().transpose(0, 2, 1) @ y_1 
            z_hat = W_Z.conj().transpose(0, 2, 1) @ y_2 
            
            sz_vec = np.concatenate((s_hat, z_hat), axis=1) 
            y_comb = self.W_invH @ sz_vec 
            y_comb_H = y_comb.conj().transpose(0, 2, 1)
            
            if iter_idx == 0:
                self.v[0] = np.mean(np.abs(s_hat)**2)
                self.v[1] = 1.0
            
            # --- CORRECCIÓN WOODBURY ---
            num_pi = (1 - self.alpha) * (self.Pi_inv @ y_comb @ y_comb_H @ self.Pi_inv)
            den_pi = self.alpha**2 * self.v[0] + self.alpha * (1 - self.alpha) * (y_comb_H @ self.Pi_inv @ y_comb)
            self.Pi_inv = self.Pi_inv / self.alpha - num_pi / den_pi
            # ---------------------------
            
            self.Sigma_Z = self.alpha * self.Sigma_Z + (1 - self.alpha) * (y_comb @ y_comb_H) / self.v[1]
            
            w_1_old = w_1.copy()
            w_tilde = self.Pi_inv @ self.W_invH[:, :, 0:1]
            w_hat = self.lambda_unit * (self.Pi_inv @ self.a_1)
            
            w_tilde_H = w_tilde.conj().transpose(0, 2, 1)
            h_1 = (w_tilde_H @ self.W_invH[:, :, 0:1]).real
            h_hat_1 = w_tilde_H @ self.a_1 * self.lambda_unit
            
            mask = np.abs(h_hat_1) < 1e-12
            h_tilde_1 = np.where(mask, 1.0 / np.sqrt(np.maximum(h_1, 1e-12)), 
                                 (h_hat_1 / (2 * h_1)) * (-1 + np.sqrt(1 + 4 * h_1 / (np.abs(h_hat_1)**2 + 1e-12))))
            
            w_1_new = np.where(mask, w_tilde / np.sqrt(np.maximum(h_1, 1e-12)) + w_hat, 
                               h_tilde_1 * w_tilde + w_hat)
            self.W[:, :, 0:1] = w_1_new
            
            Delta_w = w_1_new - w_1_old
            Delta_w_H = Delta_w.conj().transpose(0, 2, 1)
            num_w = self.W_invH @ self.e1 @ Delta_w_H @ self.W_invH
            den_w = 1.0 + Delta_w_H @ self.W_invH[:, :, 0:1]
            self.W_invH = self.W_invH - num_w / den_w
            
            w_1_new_H = w_1_new.conj().transpose(0, 2, 1)
            term_s = w_1_new_H @ self.Sigma_Z @ self.e1
            term_z = w_1_new_H @ self.Sigma_Z @ self.E_Z
            
            top_block = -term_z / term_s 
            I_M_1 = np.tile(np.eye(self.M - 1, dtype=complex), (self.F, 1, 1))
            self.W[:, :, 1:] = np.concatenate((top_block, I_M_1), axis=1) 
            
            X = self.W[:, 0:1, 0:1]
            Y = self.W[:, 0:1, 1:]
            Z = self.W[:, 1:, 0:1]
            X_inv = 1.0 / (X - Y @ Z)
            
            top_row = np.concatenate((X_inv, -X_inv @ Y), axis=2)
            bot_row = np.concatenate((-Z @ X_inv, I_M_1 + Z @ X_inv @ Y), axis=2)
            self.W_invH = np.concatenate((top_row, bot_row), axis=1).conj().transpose(0, 2, 1)
            
            if iter_idx == 0:
                for n in range(2):
                    v_n = np.maximum(self.v[n], 1e-12)
                    R_inv_x = self.R_inv[n] @ x_bar 
                    den_k = self.beta * v_n + x_bar_H @ R_inv_x 
                    K_n = R_inv_x / den_k 
                    self.R_inv[n] = (self.R_inv[n] - K_n @ x_bar_H @ self.R_inv[n]) / self.beta
                    err_H = (x_f - self.G[n].conj().transpose(0, 2, 1) @ x_bar).conj().transpose(0, 2, 1)
                    self.G[n] = self.G[n] + K_n @ err_H
                    
        s_hat_final = self.W[:, :, 0:1].conj().transpose(0, 2, 1) @ (x_f - self.G[0].conj().transpose(0, 2, 1) @ x_bar)
        return s_hat_final.flatten()
class AcousticFrontend:
    """
    Contenedor Overlap-Add (OLA) para procesamiento en tiempo real.
    """
    def __init__(self, M, fs, n_fft, hop_length, Rs, mic_array, L=4, D=1):
        self.M = M
        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Condición estricta para reconstrucción perfecta con ventana de Hann
        assert hop_length == n_fft // 2, "hop_length debe ser n_fft / 2"
        
        # Ventana de análisis y síntesis (raíz de Hann para reconstrucción perfecta)
        hann_window = np.hanning(n_fft + 1)[:-1] 
        self.window = np.sqrt(hann_window)
        
        # Frecuencias de la RFFT
        self.freqs = np.fft.rfftfreq(n_fft, 1/fs)
        
        # Instancia del núcleo DSP
        self.dsp_core = OnlineWPExSRIVE_N1_Vectorized(
            M=M, L=L, D=D, Rs=Rs, mic_array=mic_array, freqs=self.freqs, c=343.0
        )
        
        # Buffers de estado temporal
        self.in_buffer = np.zeros((M, n_fft))
        self.out_buffer = np.zeros(n_fft)

    def process_chunk(self, audio_chunk):
        """
        Procesa un bloque de audio temporal de tamaño (M, hop_length).
        Devuelve el bloque filtrado de tamaño (hop_length,).
        """
        # 1. Actualizar buffer de entrada (FIFO)
        self.in_buffer = np.roll(self.in_buffer, -self.hop_length, axis=1)
        self.in_buffer[:, -self.hop_length:] = audio_chunk
        
        # 2. Enventanado y RFFT
        windowed_in = self.in_buffer * self.window
        X_f = np.fft.rfft(windowed_in, axis=1) # Salida: (M, F)
        
        # 3. Procesamiento en el núcleo DSP subespacial
        S_hat_f = self.dsp_core.process_frame(X_f) # Salida: (F,)
        
        # 4. IRFFT y Enventanado de síntesis
        s_hat_t = np.fft.irfft(S_hat_f, n=self.n_fft)
        windowed_out = s_hat_t * self.window
        
        # 5. Overlap-Add (OLA) en el buffer de salida
        self.out_buffer += windowed_out
        
        # 6. Extraer el bloque procesado y limpiar la zona extraída
        out_chunk = self.out_buffer[:self.hop_length].copy()
        self.out_buffer = np.roll(self.out_buffer, -self.hop_length)
        self.out_buffer[-self.hop_length:] = 0.0
        
        return out_chunk

if __name__ == '__main__':
    # Simulación de pipeline 
    fs = 16000
    duration = 4.0
    M = 4
    n_samples = int(fs * duration)
    
    # Geometría del arreglo y ubicación de la fuente
    mic_array = np.array([
        [0.00, 0, 0],
        [0.04, 0, 0],
        [0.08, 0, 0],
        [0.12, 0, 0]
    ])
    Rs = np.array([0.06, 1.0, 0.0]) # Fuente frente al centro del arreglo a 1 metro
    
    # Generación de señales de prueba crudas
    np.random.seed(42)
    x_time = np.random.randn(M, n_samples)
    
    # Configuración del bloque temporal
    n_fft = int(fs * 0.008) # 128 muestras (8 ms)
    hop_length = n_fft // 2 # 64 muestras (4 ms)
    
    frontend = AcousticFrontend(M, fs, n_fft, hop_length, Rs, mic_array)
    
    out_signal = np.zeros(n_samples)
    
    print(f"Iniciando procesamiento en bloques de {hop_length} muestras...")
    
    # Bucle principal (emulando la interrupción de un ADC DMA)
    for i in range(0, n_samples - hop_length, hop_length):
        chunk_in = x_time[:, i:i+hop_length]
        chunk_out = frontend.process_chunk(chunk_in)
        out_signal[i:i+hop_length] = chunk_out
        
    print(f"Procesamiento finalizado. Señal de salida generada con forma {out_signal.shape}.")  
    

import numpy as np
from scipy.signal import stft, istft

# (Asegurate de tener las clases generate_near_field_sv_matrix, 
# OnlineWPExSRIVE_N1_Vectorized y AcousticFrontend definidas arriba)

if __name__ == '__main__':
    # 1. Parámetros físicos y de simulación
    fs = 16000
    duration = 2.0
    M = 4
    c = 343.0
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    
    # 2. Geometría del arreglo (espaciado típico de 4 cm)
    mic_array = np.array([
        [0.00, 0, 0],
        [0.04, 0, 0],
        [0.08, 0, 0],
        [0.12, 0, 0]
    ])
    
    # Centro del arreglo para referencias
    array_center = np.mean(mic_array, axis=0)
    
    # 3. Definición de fuentes en coordenadas (x, y, z)
    # Target: Frente al arreglo a 0.5 metros
    Rs_target = array_center + np.array([0.0, 0.5, 0.0])
    freq_target = 1000.0
    s_target = np.sin(2 * np.pi * freq_target * t)
    
    # Interferer: Desplazado a la derecha y más cerca
    Rs_interf = array_center + np.array([0.4, 0.3, 0.0])
    freq_interf = 3000.0
    s_interf = 0.5 * np.sin(2 * np.pi * freq_interf * t) # Amplitud reducida
    
    # 4. Simulación de propagación de campo cercano a los micrófonos
    x_time = np.zeros((M, n_samples))
    
    for m in range(M):
        # Propagación del Target
        dist_t = np.linalg.norm(Rs_target - mic_array[m])
        delay_t = dist_t / c
        samples_delay_t = int(delay_t * fs)
        s_t_delayed = np.pad(s_target, (samples_delay_t, 0))[:n_samples] / dist_t
        
        # Propagación del Interferer
        dist_i = np.linalg.norm(Rs_interf - mic_array[m])
        delay_i = dist_i / c
        samples_delay_i = int(delay_i * fs)
        s_i_delayed = np.pad(s_interf, (samples_delay_i, 0))[:n_samples] / dist_i
        
        # Mezcla en el micrófono m
        x_time[m, :] = s_t_delayed + s_i_delayed
        
    # Añadir piso de ruido de los sensores MEMS (SNR alto)
    np.random.seed(42)
    x_time += 0.001 * np.random.randn(M, n_samples)
    
    # 5. Configuración del Frontend
    n_fft = 128
    hop_length = 64
    
    # Inicializamos el sistema apuntando exclusivamente a la coordenada del Target
    frontend = AcousticFrontend(M, fs, n_fft, hop_length, Rs_target, mic_array)
    out_signal = np.zeros(n_samples)
    
    # 6. Procesamiento OLA en tiempo real
    print("Iniciando filtrado espacial iterativo...")
    for i in range(0, n_samples - hop_length, hop_length):
        chunk_in = x_time[:, i:i+hop_length]
        out_signal[i:i+hop_length] = frontend.process_chunk(chunk_in)
        
    # 7. Evaluación de resultados (Métrica espectral)
    # Analizamos el último segundo para dar tiempo a que las matrices de covarianza converjan
    eval_samples = x_time[0, -fs:]
    eval_out = out_signal[-fs:]
    
    fft_in = np.abs(np.fft.rfft(eval_samples))
    fft_out = np.abs(np.fft.rfft(eval_out))
    freqs_eval = np.fft.rfftfreq(fs, 1/fs)
    
    # Encontrar índices de las frecuencias
    idx_target = np.argmin(np.abs(freqs_eval - freq_target))
    idx_interf = np.argmin(np.abs(freqs_eval - freq_interf))
    
    # Relación Señal a Interferencia (SIR) cruda en dB
    sir_in_db = 20 * np.log10(fft_in[idx_target] / fft_in[idx_interf])
    sir_out_db = 20 * np.log10(fft_out[idx_target] / fft_out[idx_interf])
    
    print("-" * 30)
    print("Resultados de la separación:")
    print(f"SIR de entrada (Mic 1): {sir_in_db:.2f} dB")
    print(f"SIR de salida:          {sir_out_db:.2f} dB")
    print(f"Mejora neta (SIR Gain): {sir_out_db - sir_in_db:.2f} dB")
    print("-" * 30)