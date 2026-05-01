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

class Microphone():

    def __init__(self, fs=48000):
        """
        Initializes an ideal microphone by default. 
        Errors and models can be set post-instantiation.
        """
        self.fs = fs 
        
        # Default ideal parameters (No mismatch, infinite SNR)
        self.std_gain_dB = 0.0
        self.std_phase_deg = 0.0
        self.snr_dB = 120.0 
        
        # State variables for mismatch persistence
        self.fixed_gain_mismatch = None
        self.fixed_phase_mismatch = None
        self.last_M = 0

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

    def set_custom_errors(self, std_gain_dB=0.0, std_phase_deg=0.0, snr_dB=60.0):
        """
        Manually sets the standard deviation for hardware errors and the target SNR.
        Overrides any previously loaded model.
        """
        self.std_gain_dB = std_gain_dB
        self.std_phase_deg = std_phase_deg
        self.snr_dB = snr_dB
        
        # Reset mismatches so they are regenerated on the next emulation
        self.fixed_gain_mismatch = None
        print(f"[Microphone] Custom errors set -> Gain std: {std_gain_dB}dB | Phase std: {std_phase_deg}deg | SNR: {snr_dB}dB")


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


    def _apply_mismatch(self, array_input):
        M = array_input.shape[0]

        # Generate the hardware profile if it's the first run or if M changed
        if self.fixed_gain_mismatch is None or self.last_M != M:
            random_gain_dB = np.random.normal(loc=0, scale=self.std_gain_dB, size=M)
            self.fixed_gain_mismatch = 10**(random_gain_dB / 20)
            
            random_phase = np.random.normal(loc=0, scale=self.std_phase_deg, size=M)
            random_phase_rad = np.deg2rad(random_phase)
            self.fixed_phase_mismatch = np.exp(1j * random_phase_rad)
            
            self.last_M = M

        # 1. Apply Gain Mismatch
        signal_gain = array_input * self.fixed_gain_mismatch[:, np.newaxis]

        # 2. Apply Phase Mismatch 
        if self.std_phase_deg > 0:
            signal_f = fft.rfft(signal_gain, axis=1)
            
            # CRITICAL FIX: Ensure the DC bin remains purely real to avoid artifacts
            N_freqs = signal_f.shape[1]
            phasor_matrix = np.ones((M, N_freqs), dtype=np.complex128)
            phasor_matrix[:, 1:] = self.fixed_phase_mismatch[:, np.newaxis]
            
            signal_f = signal_f * phasor_matrix
            signal_out = fft.irfft(signal_f, n=array_input.shape[1], axis=1)
            return signal_out
            
        return signal_gain


    def _measure_in_band_rms(self, signal_in):
        sig_filtered = signal.lfilter(self.taps, self.a, signal_in, axis=1)
        return np.sqrt(np.mean(sig_filtered**2))


    def emulate(self, array_input, show_plots=False):
        M, N_samples = array_input.shape

        array_input_mismatched = self._apply_mismatch(array_input)
        signal_rms_in_band = self._measure_in_band_rms(array_input_mismatched)
        
        if signal_rms_in_band > 0:
            noise_raw = np.random.normal(loc=0, scale=1, size=(M, N_samples))
            noise_filtered = signal.lfilter(self.taps, self.a, noise_raw, axis=1)
            noise_rms_in_band = np.sqrt(np.mean(noise_filtered**2))
            
            target_noise_rms = signal_rms_in_band / (10**(self.snr_dB / 20))
            gain_factor = target_noise_rms / (noise_rms_in_band + 1e-12)
            
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
            plt.title(f'Spectrogram (In-Band Target SNR: {self.snr_dB} dB)')
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