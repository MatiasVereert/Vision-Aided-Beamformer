import numpy as np
import scipy.signal as signal
import scipy.fft as fft
import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import scipy.fft as fft
import json
import matplotlib.pyplot as plt


import numpy as np
import scipy.signal as signal
import scipy.fft as fft
import json
import matplotlib.pyplot as plt


def a_weighting_magnitude(freqs):
    """
    Magnitud LINEAL de la ponderacion A (IEC 61672 / ANSI S1.4) evaluada
    ANALITICAMENTE en las frecuencias dadas (Hz), normalizada a 1.0 (0 dB) en
    1 kHz.

    A diferencia de un filtro digital (bilineal), esta expresion cerrada no sufre
    warping en frecuencia: es exacta hasta Nyquist a cualquier fs. Se usa para
    medir el nivel PONDERADO EN A por FFT (ver Microphone._measure_a_weighted_rms),
    tal como se especifica el SNR de los microfonos MEMS (dBA).

        R_A(f) = f4^2 * f^4 /
                 [ (f^2+f1^2) * sqrt((f^2+f2^2)(f^2+f3^2)) * (f^2+f4^2) ]
        W(f)   = R_A(f) / R_A(1000)      (0 dB en 1 kHz)
    """
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217

    def _R_A(f_sq):
        num = (f4 ** 2) * (f_sq ** 2)
        den = (f_sq + f1 ** 2) * np.sqrt((f_sq + f2 ** 2) * (f_sq + f3 ** 2)) * (f_sq + f4 ** 2)
        return num / den

    f_sq = np.asarray(freqs, dtype=float) ** 2
    return _R_A(f_sq) / _R_A(1000.0 ** 2)


class Microphone():

    def __init__(self, fs=48000, seed=0):
        """
        Initializes an ideal microphone by default.
        Errors and models can be set post-instantiation.

        `seed` fija TODA la aleatoriedad del sensor (patron de mismatch y ruido
        termico) de forma determinista. Ver set_seed() para el detalle del
        desacople patron/escala que hace reproducible y monotono el barrido de
        mismatch (Fig. 10).
        """
        self.fs = fs

        # Default ideal parameters (No mismatch, infinite SNR)
        # NOTA: snr_dB se interpreta PONDERADO EN A (dBA), como en los datasheets
        # de microfonos MEMS. Ver a_weighting_magnitude() y emulate().
        self.std_gain_dB = 0.0
        self.std_phase_deg = 0.0
        self.snr_dB = 120.0

        # --- DETERMINISTIC RANDOMNESS ---
        # Toda la aleatoriedad se deriva de esta semilla. El patron BASE (vectores
        # unitarios z ~ N(0, I_M) y el ruido termino de referencia) queda fijado
        # por (seed, M[, N]); la ESCALA (std / snr) se aplica sobre ese patron sin
        # re-sortear. Asi, al barrer std_gain/std_phase con la misma escena, todas
        # las celdas comparten el MISMO patron y solo cambian por el factor de
        # escala -> heatmap suave/monotono en vez de ruido de sorteo.
        self._seed = seed

        # Patron unitario de mismatch (depende solo de (seed, M)):
        self._z_gain = None
        self._z_phase = None
        self._pattern_M = None
        self._pattern_seed = None

        # Patron base de ruido termico (depende solo de (seed, M, N_samples)):
        self._noise_base = None
        self._noise_key = None

        # --- FIR FILTER DESIGN (Bandpass) ---
        f_min = 100
        attenuation = 50 # dB
        gap = 50 # Hz (Transition width)
        
        # Dynamically calculate f_max to ensure the transition band 
        # strictly fits below the Nyquist limit for any given sample rate.
        nyquist = 0.5 * self.fs
        
        # We target 8kHz, but if Nyquist is too low, we adapt f_max.
        # We subtract 'gap' and a small safety margin (10 Hz) from Nyquist.
        f_max = min(8000.0, nyquist - gap - 10.0)

        n_taps = int(attenuation * self.fs / (22 * gap))
        if n_taps % 2 == 0: n_taps += 1 

        edges = [0, f_min - gap, f_min, f_max, f_max + gap, nyquist]
        
        self.taps = signal.remez(n_taps, edges, [0, 1, 0], fs=self.fs)
        self.a = 1.0

        # --- A-WEIGHTING (IEC 61672) ---
        # El SNR del sensor se especifica en dBA. El nivel ponderado en A se mide
        # en el dominio de frecuencia con la magnitud analitica exacta
        # (a_weighting_magnitude), sin warping y exacta a cualquier fs. No se
        # necesita disenar ni almacenar un filtro.

    def set_custom_errors(self, std_gain_dB=0.0, std_phase_deg=0.0, snr_dB=60.0):
        """
        Manually sets the standard deviation for hardware errors and the target SNR.
        Overrides any previously loaded model.
        """
        self.std_gain_dB = std_gain_dB
        self.std_phase_deg = std_phase_deg
        self.snr_dB = snr_dB

        # IMPORTANTE: NO se invalida el patron base aqui. La std es SOLO una escala
        # que se aplica sobre el patron unitario fijo (ver _apply_mismatch). De este
        # modo, al barrer std_gain/std_phase, todas las celdas re-escalan el MISMO
        # sorteo z ~ N(0, I_M) en vez de sortear una realizacion nueva por celda
        # (ese re-sorteo era el bug que ensuciaba la Fig. 10).
        print(f"[Microphone] Custom errors set -> Gain std: {std_gain_dB}dB | Phase std: {std_phase_deg}deg | SNR: {snr_dB}dBA")

    def set_seed(self, seed):
        """
        Fija la semilla que determina el patron base de mismatch y de ruido termico.

        El patron base solo se re-sortea cuando cambia la semilla (o el numero de
        microfonos M / la longitud N de la senal), NUNCA cuando cambia la std o el
        SNR. El orquestador debe derivar esta semilla UNICAMENTE de la fisica de la
        escena (rt60, angulos target/interf, iSIR, ids de audio) para que:
          - la misma escena de siempre el mismo patron (reproducible), y
          - dentro del barrido de mismatch, las celdas difieran solo por la escala.
        """
        if seed != self._seed:
            self._seed = seed
            # Invalidar caches: se regeneran con la nueva semilla en la proxima
            # emulacion (los _ensure_* tambien lo detectan por su clave).
            self._z_gain = None
            self._noise_key = None


    def load_model(self, model_name, json_path='src/beamforming/array/models.json'):
        """
        Loads hardware specifications from a JSON file.
        Assumes the JSON provides 3-sigma maximum tolerance values.
        """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: JSON file not found at '{json_path}'.")

        models_dict = {m["name"]: m for m in data["models"]}

        if model_name not in models_dict:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(models_dict.keys())}")
        
        params = models_dict[model_name]["parameters"]

        # Apply the 3-sigma rule to get the standard deviation
        gain_std = params["array_mismatch_dB"] / 3.0
        phase_std = params["phase_mismatch_deg"] / 3.0
        snr = params["snr_dB"]

        self.set_custom_errors(gain_std, phase_std, snr)
        print(f"[Microphone] Loaded hardware profile: '{model_name}'")


    def _ensure_mismatch_pattern(self, M):
        """
        Garantiza que exista el patron unitario de mismatch z_gain, z_phase ~
        N(0, I_M), FIJADO por (seed, M). Se re-sortea SOLO si cambia la semilla o M,
        nunca por cambiar la std. Se usa un RNG dedicado y derivado de la semilla
        (rama 0) para desacoplarlo del ruido termico (rama 1).
        """
        if (self._z_gain is None or self._pattern_M != M
                or self._pattern_seed != self._seed):
            rng = np.random.default_rng([self._seed, 0])
            self._z_gain = rng.standard_normal(M)
            self._z_phase = rng.standard_normal(M)
            self._pattern_M = M
            self._pattern_seed = self._seed

    def _apply_mismatch(self, array_input):
        M = array_input.shape[0]
        self._ensure_mismatch_pattern(M)

        # ESCALA sobre el PATRON fijo: random_gain_dB = std_gain_dB * z_gain.
        # Con std_gain_dB = 0 -> mismatch nulo (gain = 1), como en Prueba A/C.
        random_gain_dB = self.std_gain_dB * self._z_gain
        gain_mismatch = 10 ** (random_gain_dB / 20)

        # 1. Apply Gain Mismatch
        signal_gain = array_input * gain_mismatch[:, np.newaxis]

        # 2. Apply Phase Mismatch
        if self.std_phase_deg > 0:
            random_phase_rad = np.deg2rad(self.std_phase_deg * self._z_phase)
            phase_mismatch = np.exp(1j * random_phase_rad)

            signal_f = fft.rfft(signal_gain, axis=1)

            # CRITICAL FIX: Ensure the DC bin remains purely real to avoid artifacts
            N_freqs = signal_f.shape[1]
            phasor_matrix = np.ones((M, N_freqs), dtype=np.complex128)
            phasor_matrix[:, 1:] = phase_mismatch[:, np.newaxis]

            signal_f = signal_f * phasor_matrix
            signal_out = fft.irfft(signal_f, n=array_input.shape[1], axis=1)
            return signal_out

        return signal_gain


    def _measure_a_weighted_rms(self, signal_in):
        """
        Mide el RMS PONDERADO EN A (IEC 61672) de la senal EN EL DOMINIO DE
        FRECUENCIA. El SNR del sensor se define en dBA, por lo que tanto la
        referencia de senal como el piso de ruido se ponderan en A.

        Se aplica la magnitud analitica exacta de la curva A por bin de la FFT y
        se integra la energia via Parseval (bins internos con peso x2). Es exacto
        hasta Nyquist a cualquier fs (sin el warping de un filtro bilineal).
        Devuelve un escalar (RMS global sobre todos los mics y muestras), igual
        que la version anterior.
        """
        sig_in = np.atleast_2d(signal_in)
        N = sig_in.shape[1]

        X = np.fft.rfft(sig_in, axis=1)
        f = np.fft.rfftfreq(N, d=1.0 / self.fs)
        W = a_weighting_magnitude(f)  # (K,) magnitud lineal, 0 dB @ 1 kHz

        # Pesos de Parseval: DC (y Nyquist si N es par) cuentan una vez; el resto x2
        parseval = np.full(f.shape, 2.0)
        parseval[0] = 1.0
        if N % 2 == 0:
            parseval[-1] = 1.0

        # mean(x^2) ponderado en A = sum( peso * |X*W|^2 ) / N^2   (por micro)
        mean_sq_per_mic = np.sum(parseval * (np.abs(X) * W) ** 2, axis=1) / (N ** 2)

        return np.sqrt(np.mean(mean_sq_per_mic))


    def _ensure_noise_base(self, M, N_samples):
        """
        Garantiza el patron BASE de ruido termico blanco ~ N(0, 1), FIJADO por
        (seed, M, N_samples). Se comparte entre celdas de la misma escena fisica
        (misma semilla): asi, en el barrido de mismatch, el ruido es el MISMO y las
        celdas difieren solo por el error barrido. El nivel (SNR) se aplica despues
        como escala, no aqui. RNG derivado de la semilla (rama 1) para desacoplarlo
        del patron de mismatch (rama 0).
        """
        key = (self._seed, M, N_samples)
        if self._noise_key != key:
            rng = np.random.default_rng([self._seed, 1, N_samples])
            self._noise_base = rng.standard_normal((M, N_samples))
            self._noise_key = key

    def emulate(self, array_input, show_plots=False):
        M, N_samples = array_input.shape

        array_input_mismatched = self._apply_mismatch(array_input)

        # Referencia de senal PONDERADA EN A (dBA)
        signal_rms_A = self._measure_a_weighted_rms(array_input_mismatched)

        if signal_rms_A > 0:
            # Ruido de sensor: blanco, independiente por microfono, limitado a la
            # banda de audio por el mismo pasabanda del sensor. El patron base es
            # determinista y compartido por escena (ver _ensure_noise_base).
            self._ensure_noise_base(M, N_samples)
            noise_raw = self._noise_base
            noise_filtered = signal.lfilter(self.taps, self.a, noise_raw, axis=1)

            # Nivel de ruido objetivo definido en el dominio PONDERADO EN A:
            #   SNR_dBA = 20*log10( A(senal)_rms / A(ruido)_rms )
            noise_rms_A = self._measure_a_weighted_rms(noise_filtered)
            target_noise_rms_A = signal_rms_A / (10**(self.snr_dB / 20))
            gain_factor = target_noise_rms_A / (noise_rms_A + 1e-12)

            array_output = array_input_mismatched + (noise_filtered * gain_factor)
        else:
            array_output = array_input_mismatched

        if show_plots:
            plt.figure(figsize=(10, 5))
            
            plt.subplot(2, 1, 1)
            plt.plot(array_output[0, :], label='Mic 0 Output')
            plt.title('Time-domain Signal (Mic 0)')
            plt.grid(True)
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.specgram(array_output[0,:], Fs=self.fs, cmap='viridis', NFFT=1024, noverlap=512)
            plt.title(f'Spectrogram (A-weighted Target SNR: {self.snr_dB} dBA)')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [s]')
            plt.colorbar(label='Intensity [dB]')
            
            plt.tight_layout()
            plt.show()

        return array_output
    
if __name__ == "__main__":

    from propagation.simulate_acoustics_v1 import SimAcoustic
    import os 
    from utils.audio import save_wav
    import numpy as np

    # 1. SETUP & GEOMETRY
    fs = 48000
    mic_spacing = 0.04 
    folder_path = "tests/data"
    raw_audio_cache = os.path.join(folder_path, "raw_room_input.npy") # Cache file path
    
    if not os.path.exists(folder_path): 
        os.makedirs(folder_path)

    # --- Planar Array Config ---
    Mx, My, Mz = 1, 4, 1 
    M = Mx * My * Mz 
    
    x = np.linspace(0, (Mx-1)*mic_spacing, Mx)
    y = np.linspace(0, (My-1)*mic_spacing, My)
    z = np.array([0.0])
    xv, yv, zv = np.meshgrid(x, y, z, indexing='xy') 
    
    mic_coords = np.column_stack([xv.flatten(), yv.flatten(), zv.flatten()])
    array_center = np.array([1.25, 2.0, 1.25])
    mic_coords = mic_coords - np.mean(mic_coords, axis=0) + array_center
    
    print(f"[Setup] Planar Array Configured: {Mx}x{My}x{Mz} ({M} mics)")

    # 2. CACHE VERIFICATION (Decoupled execution)
    if os.path.exists(raw_audio_cache):
        print(f"[Cache] Loading previous acoustic simulation from {raw_audio_cache}...")
        room_input = np.load(raw_audio_cache)
    else:
        print("[Sim] Cache not found. Starting acoustic simulation (Stage 1)...")
        
        # 3. SOURCE POSITIONS
        source_pos = (array_center + np.array([0.9, 0.0, 0.1])).reshape(1,3)
        interf_pos1 = (array_center + np.array([0.0, 1.2, 0.0])).reshape(1,3)
        interf_pos2 = (array_center + np.array([-0.6, 0.6, 0.0])).reshape(1,3)

        # 4. ACOUSTIC SCENE SETUP
        # Note: Added fs explicitly to match SimAcoustic initialization
        acoustic_scene = SimAcoustic(mic_coords, array_mismatch=1e-3, duration=4, fs=fs)
        room_dimensions = np.array([2.5, 4, 2.5])

        source_path = "tools/data/signals/FA01_09.wav"
        int_path1 = "tools/data/signals/MC15_03.wav"
        int_path2 = "tools/data/signals/MF31_03.wav"

        acoustic_scene.set_source(source_path, gain=1, position=source_pos)
        acoustic_scene.set_interference(int_path1, gain=1, position=interf_pos1)
        acoustic_scene.set_interference(int_path2, gain=1, position=interf_pos2)

        # Computationally expensive stage: Compute room physics and convolve
        acoustic_scene.compute_rirs(room_dimensions, desire_RT=1.0, ray_tracing=True)
        acoustic_scene.convolve_signals()
        
        # Fast stage: Matrix mixing for desired iSIR
        scene_data = acoustic_scene.mix_and_normalize(iSIR_dB=5)
        
        # Extract the mixed microphone signals to feed the sensor emulation
        room_input = scene_data["mic_signals"]
        
        # Save array to cache for future runs
        np.save(raw_audio_cache, room_input)
        print(f"[Sim] Simulation completed and saved to {raw_audio_cache}")

    # 5. STAGE 2: MICROPHONE EMULATION (Iterative)
    print("[Mic] Starting microphone emulation...")
    
    # Initialize the decoupled microphone
    microphone = Microphone(fs=fs)
    
    # Load the specific hardware profile (or you could use set_custom_errors here)
    microphone.load_model("MP34DT01-M")

    # Emulate hardware degradation and plot
    signals = microphone.emulate(room_input, show_plots=True)
    
    save_wav("test_micro.wav",
             rate=fs,
             data=signals[0,:],
             folder="src/beamforming/array"
             )
    print("[Done] Process finished successfully.")  