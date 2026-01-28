import numpy as np
from scipy import signal
from scipy.io import wavfile
import os
import warnings
C_SOUND = 343.0
from beamforming.signal_model import steering_vector


# --- 2. CLASE KMVDR RECURSIVA (Lógica del Reference Code) ---
class LowRankAdaptive:
    def __init__(self, mic_array, fs, alpha=0.95):
        self.mic_array = mic_array
        self.fs = fs
        self.alpha = alpha  # Factor de olvido (0.95 es estándar para voz)
        self.ALS_iterations = 2
        
        # Estado persistente (Tensores de Frecuencia)
        # Se inicializan en la primera llamada para saber el tamaño F
        self.R_cov = None 
        self.h1 = None
        self.h2 = None
        self.FPS_rec = 15



    def block_process(self, input_signals, target_pos, M1: int, M2: int, P=2, record_scene = True, mode = "near_field"):
        
        # Advertencias de compatibilidad campo y rango
        if P==1 and mode == 'near_field':
            print("Advertencia: El rango P = 1 es incompatible con campo cercano")

        # Normalización
        peak = np.max(np.abs(input_signals))
        if peak > 1e-9: input_signals /= peak

        # Configuración STFT
        n_window = 1024
        n_overlap = 512
        M = M1 * M2

        
        
        # STFT devuelve: (Mics, Frecuencias, Tiempo) debido a axis=1
        f, t, X = signal.stft(x=input_signals, fs=self.fs, nperseg=n_window, 
                              noverlap=n_overlap, window='hann', axis=1)
        
        # CORRECCIÓN DE DIMENSIONES:
        # Queremos: (Frecuencias, Tiempo, Mics)
        # Origen:   (0:Mics, 1:Freqs, 2:Time)
        # Destino:  (1, 2, 0)
        X = X.transpose(1, 2, 0)
        
        F_bins, T_frames, _ = X.shape

        #Inicialize the weith recording vector
        t_frame = n_overlap * 1/ fs
        rec_per_frame = int( (1/self.FPS_rec)/ t_frame  ) 
        self.h_record = np.zeros((F_bins, M, int(T_frames/rec_per_frame)+2), dtype = complex)
        record_idx = 0
        
        
        # Validación de seguridad para evitar el error confuso
        if F_bins == M: 
             # Si esto pasa, las dimensiones siguen mal (caso de borde donde F_bins = Mics)
             # Pero con 1024 puntos, F_bins es 513, así que esto ya no debería ocurrir.
             pass

        Y_stft = np.zeros((F_bins, T_frames), dtype=complex)

        # --- A. Steering Vector ---

        sv = steering_vector(f, target_pos, self.fs, self.mic_array, 1, squeeze=True)
        sv = np.nan_to_num(sv)

        # --- B. Inicialización de Estado ---
        if self.R_cov is None:
            # R inicial: (F, M, M)
            self.R_cov = np.zeros((F_bins, M, M), dtype=complex)
            self.R_cov[:] = np.eye(M)[None, :, :] * 1e-5

            self.h1 = np.zeros((F_bins, M1, P), dtype=complex)
            self.h2 = np.zeros((F_bins, M2, P), dtype=complex)
            
            # Ahora sí el reshape funcionará: 4617 / 513 = 9 mics
            sv_matrix = sv.reshape(F_bins, M1, M2)
            
            try:
                u, s, vh = np.linalg.svd(sv_matrix)
                k_svd = min(P, s.shape[1])
                self.h1[:, :, :k_svd] = u[:, :, :k_svd]
                self.h2[:, :, :k_svd] = np.transpose(np.conj(vh[:, :k_svd, :]), (0, 2, 1))
            except:
                self.h1[:] = 1.0/M1
                self.h2[:] = 1.0/M2

        I_m1 = np.eye(M1)[None, :, :] 
        I_m2 = np.eye(M2)[None, :, :] 
        I_M  = np.eye(M)[None, :, :]  

        print(f"   -> Procesando {T_frames} tramas recursivamente (F={F_bins})...")

        diag_load_factor = 1e-3
        
        with np.errstate(all='ignore'):
            
            for t_idx in range(T_frames):
                # x_t shape: (F, M, 1)
                # X es (F, T, M), al indexar t_idx queda (F, M). Agregamos dimensión 1 al final.
                x_t = X[:, t_idx, :, None] 
                
                # Update RLS
                update_term = np.matmul(x_t, x_t.conj().transpose(0, 2, 1))
                self.R_cov = self.alpha * self.R_cov + (1 - self.alpha) * update_term
                
                # Loading
                tr_R = np.real(np.trace(self.R_cov, axis1=1, axis2=2))
                loading = (diag_load_factor * tr_R / M)[:, None, None]
                R_loaded = self.R_cov + I_M * loading

                h1_curr = self.h1
                h2_curr = self.h2
                
                for _ in range(self.ALS_iterations):
                    # --- Paso h1 ---
                    H2_raw = np.einsum('ab, fcp -> facbp', np.eye(M1), h2_curr)
                    H2 = H2_raw.reshape(F_bins, M, M1 * P)

                    Phi_y2 = H2.conj().transpose(0, 2, 1) @ R_loaded @ H2
                    d_2 = H2.conj().transpose(0, 2, 1) @ sv[:, :, None]

                    h1_flat = np.linalg.pinv(Phi_y2, rcond=1e-6) @ d_2
                    den = d_2.conj().transpose(0, 2, 1) @ h1_flat
                    h1_flat = h1_flat / (den + 1e-12)
                    
                    h1_curr = h1_flat.reshape(F_bins, M1, P)

                    # --- Paso h2 ---
                    H1_raw = np.einsum('fap, cd -> facpd', h1_curr, np.eye(M2))
                    H1_raw = H1_raw.transpose(0, 1, 2, 4, 3) 
                    H1 = H1_raw.reshape(F_bins, M, M2 * P)
                    
                    Phi_y1 = H1.conj().transpose(0, 2, 1) @ R_loaded @ H1
                    d_1 = H1.conj().transpose(0, 2, 1) @ sv[:, :, None]
                    
                    h2_flat = np.linalg.pinv(Phi_y1, rcond=1e-6) @ d_1
                    den = d_1.conj().transpose(0, 2, 1) @ h2_flat
                    h2_flat = h2_flat / (den + 1e-12)
                    
                    h2_curr = h2_flat.reshape(F_bins, M2, P)

                self.h1 = h1_curr
                self.h2 = h2_curr
                
                

                # Output
                h_kron = np.einsum('fap, fbp -> fab', self.h1, self.h2)
                h_total = h_kron.reshape(F_bins, M)

                #record weights
                resto = t_idx % rec_per_frame

                if record_scene == True and  resto == 0  :
                    record_idx = record_idx + 1
                    self.h_record[: , :, record_idx] = h_total

                
                
                y_val = h_total.conj()[:, None, :] @ x_t
                Y_stft[:, t_idx] = y_val[:, 0, 0]

        _, y_out = signal.istft(Y_stft, fs=self.fs, window='hann', nperseg=n_window, noverlap=n_overlap)
        return y_out


def load_audio_track(path, target_fs):
    """Carga, normaliza y remuestrea un archivo de audio."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
        
    fs_file, data = wavfile.read(path)
    
    # Conversión a float [-1, 1]
    if data.dtype == np.int16:
        data = data / 32768.0
    elif data.dtype == np.int32:
        data = data / 2147483648.0
    else:
        data = data.astype(float)
        data = data / np.max(np.abs(data)) if np.max(np.abs(data)) > 0 else data

    # Conversión a Mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)
        
    # Resample si es necesario
    if fs_file != target_fs: 
        num_samples = int(len(data) * target_fs / fs_file)
        data = signal.resample(data, num_samples)
        
    return data


def space_delay(src_signal, fs, pos_src, mic_array, c=343.0):
    n_mics = mic_array.shape[0]
    n_samples = len(src_signal)
    
    diff = mic_array - np.array(pos_src)
    dists = np.linalg.norm(diff, axis=1)
    dists = np.maximum(dists, 0.05) # Distancia mínima 5cm
    
    delays = dists / c
    output = np.zeros((n_mics, n_samples))
    
    N = len(src_signal)
    X = np.fft.rfft(src_signal)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    
    for m in range(n_mics):
        phase_shift = np.exp(-1j * 2 * np.pi * freqs * delays[m])
        output[m, :] = np.fft.irfft(X * phase_shift, n=N)
        
    return output, dists

# --- MAIN DE PRUEBA ---
if __name__ == "__main__":
    # Configuración igual a tu Reference Code
    fs = 48000
    mic_spacing = 0.04 
    M1, M2 = 3, 3  
    M = M1 * M2
    
    # Geometría
    x = np.linspace(0, (M2-1)*mic_spacing, M2)
    y = np.linspace(0, (M1-1)*mic_spacing, M1)
    xv, yv = np.meshgrid(x, y, indexing='xy') 
    mic_coords = np.column_stack([xv.flatten(), yv.flatten(), np.zeros(M)])
    
    pos_src = [1, 1, 0.5]
    pos_noise = [-1.5, 2.5, 0.5] 

    print("Cargando señales...")
    # (Tu código de carga aquí, abreviado)
    limit = int(3*fs)
    t = np.arange(limit)/fs
    

    
    # --- CARGA DE SEÑALES ---
    print("1. Cargando archivos de audio...")

    file_voice = "tools/data/signals/MF31_03.wav"
    file_interf = "tools/data/signals/FA01_09.wav"

    try:
        raw_voice = load_audio_track(file_voice, fs)
        raw_interf = load_audio_track(file_interf, fs)
        
        # Ajustar longitudes (cortar al más corto)
        min_len = min(len(raw_voice), len(raw_interf))
        # Limitamos a 5 segundos máximo para no eternizar la simulación
        limit_samples = min(min_len, 5 * fs) 
        
        source_signal = raw_voice[:limit_samples]
        # Amplificamos la interferencia para desafiar al beamformer (SIR bajo)
        noise_signal = raw_interf[:limit_samples] * 1.5 
        
        t = np.arange(limit_samples) / fs
        print(f"   -> Señales cargadas: {limit_samples/fs:.2f} segundos.")
        
    except FileNotFoundError as e:
        print(e)
        print("   -> Generando señales sintéticas de respaldo...")
        t = np.arange(0, 2.0, 1/fs)
        source_signal = np.sin(2 * np.pi * t * 440)
        noise_signal = np.random.normal(0, 1, len(t))

    print("2. Simulando propagación acústica (Near Field)...")
    src_raw, d_src = space_delay(source_signal, fs, pos_src, mic_coords, C_SOUND)
    noise_raw, d_noise = space_delay(noise_signal, fs, pos_noise, mic_coords, C_SOUND)

    
    array_input = (src_raw/d_src[:,None]) + (noise_raw/d_noise[:,None]) + np.random.normal(0, 0.0001, src_raw.shape)

    
    print("Procesando con Clase RLS...")
    # INSTANCIAMOS LA NUEVA CLASE
    bf = LowRankAdaptive(mic_coords, fs, alpha=0.95)
    
    # PROCESAMOS (P=2 es seguro aquí porque R se adapta bien)
    y_out = bf.block_process(array_input, pos_src, M1, M2, P=2)
    
    # Guardar
    m = np.max(np.abs(y_out))
    if m > 0: y_out /= m
    wavfile.write("entrada_class_rls.wav", fs, (array_input[0]*32767).astype(np.int16))
    wavfile.write("salida_class_rls.wav", fs, (y_out*32767).astype(np.int16))
    print("Guardado: salida_class_rls.wav")
