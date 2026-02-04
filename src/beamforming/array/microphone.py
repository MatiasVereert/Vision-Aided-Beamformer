import numpy as np 
import scipy as sc 
from matplotlib import pyplot as plt 
import json



class Microphone():

    def __init__(self, model="ideal", fs=44800):
        self.fs = fs 
        # LOAD MODELS 
        with open('src/beamforming/array/models.json', 'r') as f:
            data = json.load(f)

        # 1. Crear un diccionario de búsqueda (Lookup Table)
        # Convertimos la lista de modelos en un diccionario donde la clave es el 'name'
        models_dict = {m["name"]: m for m in data["models"]}

        names = list(models_dict.keys())

        # AGREGAR IMPORTACION
        if model not in names:
            print("Model not found, the avalable list is:")
            print(names)
        else: 
            self.model = model
            # 2. Seleccionar el objeto del modelo
            target_model_data = models_dict[model]
            
            # 3. Acceder a la sub-clave 'parameters' (según tu JSON)
            params = target_model_data["parameters"]

            #save an convert to standar deviation MDE/3
            self.gain_mismatch = params["array_mismatch_dB"] /3
            self.phase_mismatch_deg = params["phase_mismatch_deg"] /3
            self.snr_dB = params["snr_dB"] 
    
    def emulate(self, array_input, show_plots = False):
        '''
            This mode emulates the respose of the microphone adding the most 
            important caracteristics that afect the beamforming performance, including
            noise, gain and phase mismatch.

            Arguments:
                array_input (np.array) : Audio signals with shape (M, N_len )
                fs (int): sample rate
            Returns:
                array_input
        '''
        # 1  -----   Apply Gain Mismatch
        M = np.shape(array_input[:,0])
        random_gain_dB = np.random.normal(loc =0, scale = self.gain_mismatch, size = M)
        random_gain = 10**(random_gain_dB / 20 )
        random_phase = np.random.normal(loc =0, scale = self.phase_mismatch_deg , size = M)
        random_phase_rad = np.deg2rad( random_phase)

        # unify in a phasor
        unitary_phasor = np.exp(1j * random_phase_rad)
        mismatch_phasor = random_gain * unitary_phasor

        #Transform to frecuency domain to apply phase deviation with acurracy
        array_input_f = sc.fft.rfft(array_input, axis = 1)

        #Apply mismatchs
        array_input_f = array_input_f * mismatch_phasor[ :, np.newaxis]
        
        #Transform to time
        array_input = np.fft.irfft(array_input_f, axis = 1)
        
        # 2 ---- Add Gaussian noise acording to the SNR  
        noise_signals = np.random.normal(loc =0, scale = 1, size = np.shape(array_input))#shape( M, N_samples)

        #Measure Signal power
        input_rms = np.sqrt(np.mean(array_input**2))
        snr = 10**( self.snr_dB /20)
        n_gain = input_rms / snr
        array_input = array_input + noise_signals * n_gain
        
        # --- filtrate signals bandwid
        # Filter parameters
        f_min = 100
        f_max = 8000
        band = [f_min, f_max]

        attenuation = 50
        gap = 50

        # Define lenght of the FIR filter
        n_tabs = int( attenuation * self.fs / (22 *gap))
        edges = [0, f_min - gap, f_min, f_max, f_max + gap, 0.5 * self.fs]
        tabs = sc.signal.remez(n_tabs, edges, [0, 1, 0], fs = self.fs )

        # Apply the filter 
        array_input = sc.signal.lfilter(tabs, 1.0, array_input, axis = 1)

        # Plot spectrogram
        if show_plots:
            plt.specgram(array_input[0,:], Fs=fs, cmap='viridis')
            plt.title('Espectrograma')
            plt.ylabel('Frecuencia [Hz]')
            plt.xlabel('Tiempo [seg]')
            plt.colorbar(label='Intensidad [dB]')
            plt.show()

        return array_input


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


    
