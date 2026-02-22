import numpy as np
import scipy.signal as signal
import scipy.fft as fft
import json
import matplotlib.pyplot as plt

class Microphone():

    def __init__(self, model="ideal", fs=44800):
        self.fs = fs 
        
        # --- 1. CARGA DE MODELOS ---
        # Ajusta la ruta si es necesario
        try:
            with open('src/beamforming/array/models.json', 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print("Error: No se encontró 'src/beamforming/array/models.json'.")
            raise

        models_dict = {m["name"]: m for m in data["models"]}

        if model not in models_dict:
            raise ValueError(f"Model not found. Available: {list(models_dict.keys())}")
        
        target_model_data = models_dict[model]
        params = target_model_data["parameters"]

        # Guardamos parámetros estadísticos (asumiendo que el JSON trae valores máx o 3-sigma)
        self.std_gain_dB = params["array_mismatch_dB"] / 3
        self.std_phase_deg = params["phase_mismatch_deg"] / 3
        self.snr_dB = params["snr_dB"]
        
        # --- 2. PERSISTENCIA DEL MISMATCH ---
        # Se inicializan vacíos y se calculan en la primera llamada a emulate
        self.fixed_gain_mismatch = None
        self.fixed_phase_mismatch = None
        self.last_M = 0

        # --- 3. DISEÑO DEL FILTRO FIR (Pasa-Banda 100Hz - 8kHz) ---
        # Se calcula una sola vez al instanciar para ahorrar cómputo
        f_min, f_max = 100, 8000
        attenuation = 50 # dB
        gap = 50 # Hz (Transition width)

        # Longitud del filtro FIR (Estimación para cumplir la atenuación)
        n_taps = int(attenuation * self.fs / (22 * gap))
        if n_taps % 2 == 0: n_taps += 1 # Impar es preferible para FIR

        edges = [0, f_min - gap, f_min, f_max, f_max + gap, 0.5 * self.fs]
        
        # Coeficientes del filtro (b)
        self.taps = signal.remez(n_taps, edges, [0, 1, 0], fs=self.fs)
        # En filtros FIR, el denominador 'a' es siempre 1.0
        self.a = 1.0

    def _apply_mismatch(self, array_input):
        """
        Aplica ganancia y fase aleatoria fija a los micrófonos.
        Simula el error de fabricación del hardware.
        """
        M = array_input.shape[0]

        # Si cambia la cantidad de mics o es la primera vez, generamos el perfil
        if self.fixed_gain_mismatch is None or self.last_M != M:
            # Ganancia (Distribución Log-Normal)
            random_gain_dB = np.random.normal(loc=0, scale=self.std_gain_dB, size=M)
            self.fixed_gain_mismatch = 10**(random_gain_dB / 20)
            
            # Fase (Convertimos a fasor complejo unitario)
            random_phase = np.random.normal(loc=0, scale=self.std_phase_deg, size=M)
            random_phase_rad = np.deg2rad(random_phase)
            self.fixed_phase_mismatch = np.exp(1j * random_phase_rad)
            
            self.last_M = M

        # 1. Aplicar Mismatch de Ganancia
        # Broadcasting: (M,) * (M, N)
        signal_gain = array_input * self.fixed_gain_mismatch[:, np.newaxis]

        # 2. Aplicar Mismatch de Fase (Vía FFT para precisión en banda ancha)
        signal_f = fft.rfft(signal_gain, axis=1)
        signal_f = signal_f * self.fixed_phase_mismatch[:, np.newaxis]
        signal_out = fft.irfft(signal_f, axis=1)
        
        return signal_out

    def _measure_in_band_rms(self, signal_in):
        """
        Filtra temporalmente una copia de la señal para medir su energía útil.
        No modifica la señal original.
        """
        # Filtramos copia de la señal solo para medir
        sig_filtered = signal.lfilter(self.taps, self.a, signal_in, axis=1)
        # Medimos RMS
        return np.sqrt(np.mean(sig_filtered**2))

    def emulate(self, array_input, show_plots=False):
        '''
        Simula la respuesta del array:
        1. Aplica Mismatch a la señal de entrada (Broadband).
        2. Genera Ruido Blanco y lo filtra (Band-limited).
        3. Ajusta la ganancia del ruido basándose en el SNR "En Banda".
        4. Retorna Señal (con mismatch) + Ruido (filtrado y escalado).
        
        Arguments:
            array_input (np.array) : Audio signals with shape (M, N_samples)
        Returns:
            np.array: Señal con imperfecciones y ruido añadido.
        '''
        M, N_samples = array_input.shape

        # --- 1. APLICAR MISMATCH ---
        # Afecta a la señal acústica entrante (Banda ancha)
        array_input_mismatched = self._apply_mismatch(array_input)
        
        # --- 2. MEDIR RMS DE LA SEÑAL ÚTIL ---
        # Filtramos la señal solo para saber cuánta energía hay en 100Hz-8kHz
        # Esto evita que el ruido de baja freq (rumble) afecte el cálculo del SNR
        signal_rms_in_band = self._measure_in_band_rms(array_input_mismatched)
        
        if signal_rms_in_band > 0:
            # --- 3. GENERAR RUIDO ---
            # Ruido blanco crudo
            noise_raw = np.random.normal(loc=0, scale=1, size=(M, N_samples))
            
            # --- 4. FILTRAR EL RUIDO ---
            # El ruido eléctrico se limita al ancho de banda del sistema
            noise_filtered = signal.lfilter(self.taps, self.a, noise_raw, axis=1)
            
            # --- 5. CALCULAR GANANCIA DEL RUIDO ---
            # Medimos RMS del ruido YA filtrado
            noise_rms_in_band = np.sqrt(np.mean(noise_filtered**2))
            
            # Objetivo: SNR = Signal_Band / Noise_Band
            target_noise_rms = signal_rms_in_band / (10**(self.snr_dB / 20))
            
            # Factor de escalado
            gain_factor = target_noise_rms / (noise_rms_in_band + 1e-12)
            
            # --- 6. SUMA FINAL ---
            # Señal (Mismatched) + Ruido (Filtrado y Escalado)
            array_output = array_input_mismatched + (noise_filtered * gain_factor)
        else:
            # Si la entrada es silencio digital
            array_output = array_input_mismatched

        # --- PLOTTING ---
        if show_plots:
            plt.figure(figsize=(10, 5))
            
            # Subplot 1: Señal temporal
            plt.subplot(2, 1, 1)
            plt.plot(array_output[0, :], label='Mic 0 Output')
            plt.title('Señal Temporal (Mic 0)')
            plt.grid(True)
            plt.legend()

            # Subplot 2: Espectrograma
            plt.subplot(2, 1, 2)
            plt.specgram(array_output[0,:], Fs=self.fs, cmap='viridis', NFFT=1024, noverlap=512)
            plt.title(f'Espectrograma (SNR Target: {self.snr_dB} dB en banda)')
            plt.ylabel('Frecuencia [Hz]')
            plt.xlabel('Tiempo [seg]')
            plt.colorbar(label='Intensidad [dB]')
            
            plt.tight_layout()
            plt.show()

        return array_output
if __name__ == "__main__":

    from propagation.simulate_acoustics import SimAcoustic
    import os 
    from utils.audio import save_wav

    # 1. SETUP & GEOMETRY
    fs = 48000
    mic_spacing = 0.04 
    folder_path = "tests/data"
    raw_audio_cache = os.path.join(folder_path, "raw_room_input.npy") # Archivo de caché
    
    if not os.path.exists(folder_path): 
        os.makedirs(folder_path)

    # --- Planar Array Config ---
    Mx, My, Mz = 3, 4, 1 
    M = Mx * My * Mz 
    
    x = np.linspace(0, (Mx-1)*mic_spacing, Mx)
    y = np.linspace(0, (My-1)*mic_spacing, My)
    z = np.array([0.0])
    xv, yv, zv = np.meshgrid(x, y, z, indexing='xy') 
    
    mic_coords = np.column_stack([xv.flatten(), yv.flatten(), zv.flatten()])
    array_center = np.array([1.25, 2.0, 1.25])
    mic_coords = mic_coords - np.mean(mic_coords, axis=0) + array_center
    
    print(f"[Setup] Planar Array Configured: {Mx}x{My}x{Mz} ({M} mics)")

    # 2. VERIFICACIÓN DE CACHÉ (Desacoplamiento)
    if os.path.exists(raw_audio_cache):
        print(f"[Cache] Cargando simulación acústica previa desde {raw_audio_cache}...")
        room_input = np.load(raw_audio_cache)
    else:
        print("[Sim] No se encontró caché. Iniciando simulación acústica (Etapa 1)...")
        
        # 3. SOURCE POSITIONS
        source_pos = (array_center + np.array([0.9, 0.0, 0.1])).reshape(1,3)
        interf_pos1 = (array_center + np.array([0.0, 1.2, 0.0])).reshape(1,3)
        interf_pos2 = (array_center + np.array([-0.6, 0.6, 0.0])).reshape(1,3)

        # 4. ACOUSTIC SCENE SETUP
        acoustic_scene = SimAcoustic(mic_coords, array_mismatch=1e-3, duration=4)
        room_dimensions = np.array([2.5, 4, 2.5])

        source_path = "tools/data/signals/FA01_09.wav"
        int_path1 = "tools/data/signals/MC15_03.wav"
        int_path2 = "tools/data/signals/MF31_03.wav"

        acoustic_scene.set_source(source_path, gain=1, position=source_pos)
        acoustic_scene.set_interference(int_path1, gain=1, position=interf_pos1)
        acoustic_scene.set_interference(int_path2, gain=1, position=interf_pos2)

        # Cálculo costoso
        room_input = acoustic_scene.compute_room_ISB(room_dimensions, desire_RT=1, iSIR_dB=5)
        
        # Guardar para la próxima vez
        np.save(raw_audio_cache, room_input)
        print(f"[Sim] Simulación completada y guardada en {raw_audio_cache}")

    # 5. ETAPA 2: EMULACIÓN DE MICRÓFONO (Iterativa)
    print("[Mic] Iniciando emulación de micrófono...")
    microphone = Microphone(model="MP34DT01-M", fs=fs)

    signals = microphone.emulate(room_input, show_plots = True)
    save_wav("test_micro.wav",
             rate = fs,
             data = signals[0,:],
             folder = "src/beamforming/array",
             )


    
