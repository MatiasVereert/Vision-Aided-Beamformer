import numpy as np 
import scipy as sc
from matplotlib import pyplot as plt 
from utils.audio import load_audio_source, save_wav
from propagation.free_field import space_delay
import pyroomacoustics as pra
from scipy.spatial.distance import pdist
import numpy as np


def split_rir_early_late(rir_vector, fs, t_early=0.050):
    """
    Splits a Room Impulse Response (RIR) into early and late components 
    relative to the direct path arrival.
    """
    # Find the index of the direct path (maximum absolute value)
    peak_idx = np.argmax(np.abs(rir_vector))
    
    # Calculate how many samples correspond to the early time window
    split_offset = int(t_early * fs)
    split_idx = peak_idx + split_offset
    
    # Prevent index out of bounds if the RIR is shorter than the window
    split_idx = min(split_idx, len(rir_vector))
    
    # Initialize zero arrays to maintain the original temporal alignment
    h_early = np.zeros_like(rir_vector)
    h_late = np.zeros_like(rir_vector)
    
    # Populate the early and late sections
    h_early[:split_idx] = rir_vector[:split_idx]
    h_late[split_idx:] = rir_vector[split_idx:]
    
    return h_early, h_late

def get_hybrid_sim_params(room_dims, mic_spacing, t_mix=0.08):
    """
    Calcula parámetros óptimos para simulación híbrida (ISM + Ray Tracing)
    siguiendo el protocolo de validación para Beamforming Adaptativo.
    
    Args:
        room_dims (array): Dimensiones [x, y, z]
        mic_spacing (float): Distancia mínima entre micrófonos (m). CRÍTICO.
        t_mix (float): Tiempo de mezcla objetivo. 0.08s (80ms) es robusto para fase.
    """
    c = 343.0
    volumen = np.prod(room_dims)
    min_dim = np.min(room_dims)
    
    # 1. ISM: Extendido para cubrir el Tiempo de Mezcla
    # La investigación indica que para arrays compactos necesitamos fase exacta
    # al menos hasta 50-80ms.
    dist_mix = c * t_mix
    max_order_ism = int(np.ceil(dist_mix / min_dim))
    
    # "Recomendación: Configurar max_order en el rango de 10 a 15" [cite: 123]
    # Quitamos el límite de 5. Ponemos un tope de 15 por seguridad de RAM.
    max_order_ism = np.clip(max_order_ism, 10, 15)

    # 2. RAY TRACING: Alta Resolución Espacial
    
    # "Radio del Receptor: Debe ser <= d_min_mic / 2" [cite: 129]
    # Si d=0.04, radio debe ser <= 0.02. Usamos un margen de seguridad (factor 0.45)
    rec_radius = mic_spacing * 0.45
    
    # Al reducir el radio, la probabilidad de que un rayo pegue baja cuadráticamente.
    # Necesitamos aumentar MASIVAMENTE los rayos para compensar.
    # "Valores del orden de 10^5 a 10^6 son comunes" [cite: 133]
    
    # Densidad base alta para compensar el radio pequeño
    # Heurística ajustada: Volumen * Densidad * Factor_Escala_Radio
    # (Si el radio es muy chico, subimos los rayos)
    base_rays = 200000 # Base sólida
    
    # Factor de corrección si el volumen es grande
    n_rays = int(base_rays * (volumen / 30.0)) 
    n_rays = np.clip(n_rays, 100000, 500000) # Tope en 500k para no colgar la CPU

    print(f"[SimConfig] ISM Order: {max_order_ism} (para cubrir {t_mix*1000:.0f}ms)")
    print(f"[SimConfig] RT Radius: {rec_radius*100:.2f}cm (evita solapamiento)")
    print(f"[SimConfig] N Rays: {n_rays} (Alta densidad)")

    return {
        "max_order": max_order_ism,
        "ray_tracing": {
            "n_rays": n_rays,
            "receiver_radius": rec_radius,
            "energy_thres": 1e-7
        }
    }

class SimAcoustic():

    def __init__(self, 
                 array_geometry, 
                 array_mismatch = 1e-3, 
                 duration= 4, 
                 fs = 48000 ):
        
        #define the scene atributes 
        self.ideal_array = array_geometry
        self.duration = duration #s 
        self.array_mismatch = array_mismatch
        self.M = array_geometry.shape[0]

        #Ramdomize position error
        geo_shape = np.shape(array_geometry)
        random_diferences = (- 2* array_mismatch) * np.random.random_sample(geo_shape) - array_mismatch
        self.real_array = self.ideal_array + random_diferences

        #Inicialice audio
        self.N = 0 #number of sources
        self.audio_interferences = []
        self.audio_sources = []

        self.fs = fs

        #Contador auxiliar para que no se defina mas de una fuente
        self.S_count = 0

  

    def _load_audio_struct(self, audio_path, gain, position,  type):
        #Count the number of sound sources 
        self.N += 1 
        audio_signal = load_audio_source(audio_path, self.fs, self.duration) * gain

        data = {
            "audio" : audio_signal,
            "position": position}

        if type == 's':
            self.audio_sources.append(data)
        else:
            self.audio_interferences.append(data)


    def set_source( self, audio_path, gain, position):
        
        if self.S_count >= 1:
            return print("A source has already defined.")
        self._load_audio_struct( audio_path, gain, position, type= "s")
        self.S_count += 1


        
    def set_interference( self, audio_path, gain, position):

        self._load_audio_struct( audio_path, gain, position, type = "i")



    def compute_rirs(self, room_dimensions, desire_RT=0.0, ray_tracing=False):
            """
            Generates the acoustic environment and computes the Room Impulse Responses (RIRs).
            If desire_RT is 0.0 or less, it simulates an anechoic chamber (free field) 
            by forcing the maximum image source order to 0.
            """
            # Determine absorption and maximum order based on the desired reverberation time
            if desire_RT <= 0.0:
                # Anechoic setup (Free Field)
                alpha = 1.0
                max_order = 0
                material = pra.Material(alpha)
                
                # Disable ray tracing for pure anechoic conditions to save processing time
                room = pra.ShoeBox(
                    room_dimensions, 
                    self.fs, 
                    materials=material, 
                    max_order=max_order,
                    ray_tracing=False
                )
                print("[SimAcoustic] Mode: Anechoic Chamber (Free Field)")
                
            else:
                # Reverberant setup using Sabine's formula
                alpha, max_order = pra.inverse_sabine(desire_RT, room_dimensions)
                
                if ray_tracing:
                    # Obtain minimum distance between sensors to optimize ray tracing radius
                    distances = pdist(self.real_array)
                    min_spacing = np.min(distances[distances > 0])

                    # Get optimal settings for hybrid simulation
                    params_dic = get_hybrid_sim_params(room_dimensions, min_spacing, t_mix=0.06)
                    max_order = params_dic["max_order"]

                    material = pra.Material(alpha, scattering=0.2)        
                    room = pra.ShoeBox(
                        room_dimensions, 
                        self.fs, 
                        materials=material,
                        max_order=max_order,
                        ray_tracing=True,
                    )
                    room.set_ray_tracing(**params_dic["ray_tracing"])
                    print(f"[SimAcoustic] Mode: Hybrid ISM+RT (RT60 = {desire_RT}s)")
                else:
                    # Pure ISM simulation
                    material = pra.Material(alpha)        
                    room = pra.ShoeBox(
                        room_dimensions, 
                        self.fs, 
                        materials=material, 
                        max_order=max_order
                    )
                    print(f"[SimAcoustic] Mode: Pure ISM (RT60 = {desire_RT}s)")

            # Add target source to the room
            source_pos = self.audio_sources[0]["position"] 
            room.add_source(source_pos.T)
            
            # Add all interference sources to the room
            for interference in self.audio_interferences:
                room.add_source(interference["position"].T)

            # Add microphone array 
            # Note: We use the real array geometry to capture spatial mismatches accurately
            room.add_microphone_array(self.real_array.T)

            # Compute and extract the Room Impulse Responses
            room.compute_rir()
            
            # Store the RIRs as a class attribute for later convolution and mixing stages
            # room.rir structure: list of lists -> room.rir[mic_index][source_index]
            self.rirs = room.rir
            
            print("[SimAcoustic] RIRs successfully computed and stored in 'self.rirs'.")

    def _split_rir(self, rir_vector, t_early=0.050):
        # Find the index of the direct path arrival
        peak_idx = np.argmax(np.abs(rir_vector))
        
        # Calculate the sample index for the 50ms threshold
        split_offset = int(t_early * self.fs)
        split_idx = min(peak_idx + split_offset, len(rir_vector))
        
        # Initialize arrays with zeros to maintain temporal alignment
        h_early = np.zeros_like(rir_vector)
        h_late = np.zeros_like(rir_vector)
        
        # Separate the early reflections and late reverberation
        h_early[:split_idx] = rir_vector[:split_idx]
        h_late[split_idx:] = rir_vector[split_idx:]
        
        return h_early, h_late

    def convolve_signals(self, t_early=0.050):
        """
        Convolves the defined audio sources and interferences with the previously 
        computed Room Impulse Responses (RIRs). Splits the results into early and 
        late components for evaluation purposes.
        Signals are stored unscaled to allow rapid iSIR mixing later.
        """
        if not hasattr(self, 'rirs'):
            raise ValueError("RIRs not found. Call compute_rirs() before convolving.")

        # Define output length in samples
        N_samples = int(self.duration * self.fs)

        # Initialize storage arrays for unscaled signals
        self.target_early_unscaled = np.zeros((self.M, N_samples))
        self.target_late_unscaled = np.zeros((self.M, N_samples))
        self.interf_early_sum_unscaled = np.zeros((self.M, N_samples))
        self.interf_late_sum_unscaled = np.zeros((self.M, N_samples))

        # Get source audio
        source_pos = self.audio_sources[0]["position"]
        source_sig = self.audio_sources[0]["audio"].flatten()

        # Convolve Target (Source 0)
        for i in range(self.M):
            h_early, h_late = self._split_rir(self.rirs[i][0], t_early)
            
            sig_early = sc.signal.fftconvolve(h_early, source_sig)
            sig_late = sc.signal.fftconvolve(h_late, source_sig)
            
            self.target_early_unscaled[i, :] = sig_early[:N_samples]
            self.target_late_unscaled[i, :] = sig_late[:N_samples]

        # Convolve Interferences
        for i, interference in enumerate(self.audio_interferences):
            audio = interference["audio"].flatten()
            
            # Temporary arrays for current interference across all mics
            curr_interf_early = np.zeros((self.M, N_samples))
            curr_interf_late = np.zeros((self.M, N_samples))
            
            for j in range(self.M):
                h_early, h_late = self._split_rir(self.rirs[j][i+1], t_early)
                
                sig_early = sc.signal.fftconvolve(h_early, audio)
                sig_late = sc.signal.fftconvolve(h_late, audio)
                
                curr_interf_early[j, :] = sig_early[:N_samples]
                curr_interf_late[j, :] = sig_late[:N_samples]
            
            # Accumulate interferences
            self.interf_early_sum_unscaled += curr_interf_early
            self.interf_late_sum_unscaled += curr_interf_late

        # Compute pure anechoic target using free field propagation
        # We use the real array geometry to perfectly match the RIR conditions
        target_anechoic = space_delay(
            signal_in=source_sig,
            fs=self.fs,
            source_pos=source_pos,
            mic_array=self.real_array 
        )
        
        # Ensure length matches N_samples (pad with zeros if shorter, crop if longer)
        if target_anechoic.shape[1] < N_samples:
            target_anechoic = np.pad(target_anechoic, ((0, 0), (0, N_samples - target_anechoic.shape[1])))
        else:
            target_anechoic = target_anechoic[:, :N_samples]
            
        self.target_anechoic_unscaled = target_anechoic
        
        print("[SimAcoustic] Signals successfully convolved and split (Early/Late).")


    def mix_and_normalize(self, iSIR_dB=0, inter_normalization=True, vad_threshold_db=-20):
        """
        Mixes the unscaled convolved signals according to the desired iSIR.
        Applies global normalization to prevent clipping and computes the VAD oracle.
        Returns the final evaluation dictionary. This method is extremely fast.
        """
        if not hasattr(self, 'target_early_unscaled'):
            raise ValueError("Unscaled signals not found. Call convolve_signals() first.")

        # Create copies to avoid mutating the unscaled base signals
        # This allows calling this method multiple times with different iSIRs
        target_early = self.target_early_unscaled.copy()
        target_late = self.target_late_unscaled.copy()
        interf_early = self.interf_early_sum_unscaled.copy()
        interf_late = self.interf_late_sum_unscaled.copy()
        target_anechoic = self.target_anechoic_unscaled.copy()

        if inter_normalization:
            # Calculate RMS of the total target signal at mic 0 for reference
            target_total_mic0 = target_early[0, :] + target_late[0, :]
            target_rms = np.sqrt(np.mean(target_total_mic0**2)) + 1e-10
            
            # Calculate RMS of the total interference sum at mic 0
            interf_total_mic0 = interf_early[0, :] + interf_late[0, :]
            interf_rms = np.sqrt(np.mean(interf_total_mic0**2)) + 1e-10
            
            # Apply interference scaling to achieve the desired iSIR
            iSIR_linear = 10**(iSIR_dB / 20)
            sir_scaling_factor = (target_rms / interf_rms) * (1 / iSIR_linear)
            
            interf_early *= sir_scaling_factor
            interf_late *= sir_scaling_factor

        # Construct the final unscaled mixture
        array_input = target_early + target_late + interf_early + interf_late

        # Determine the global peak across all microphones in the mixture
        global_max = np.max(np.abs(array_input))
        
        # Calculate global scaling factor to prevent clipping (leaving 1% headroom)
        global_scale = 0.99 / (global_max + 1e-10)
        
        # Apply the exact same global scaling to ALL components to preserve iSIR ratios
        array_input *= global_scale
        target_early *= global_scale
        target_late *= global_scale
        interf_early *= global_scale
        interf_late *= global_scale
        target_anechoic *= global_scale

        # Compute binary VAD ORACLE
        # Compute Hilbert transform of the early isolated target signal
        hilbert_target_early = sc.signal.hilbert(target_early[0, :])
        hilbert_target_early_energy = np.abs(hilbert_target_early) + 1e-12 
        voice_peak = np.max(hilbert_target_early_energy)

        # Define the VAD threshold based on the signal peak
        VAD_threshold = 10**(vad_threshold_db / 20) * voice_peak

        # Compare and generate boolean mask
        vad_oracle = (hilbert_target_early_energy > VAD_threshold).astype(bool)

        print(f"[SimAcoustic] Mixture completed with iSIR: {iSIR_dB} dB.")

        # Return a structured dictionary with all evaluation components
        return {
            "mic_signals": array_input,               
            "target_anechoic": target_anechoic,       
            "target_early": target_early,             
            "target_late": target_late,               
            "interference_early": interf_early,   
            "interference_late": interf_late,      
            "VAD": vad_oracle
        }



if __name__ == "__main__":

    fs = 48000
    mic_spacing = 0.04 
    M1, M2 = 3, 3  
    M = M1 * M2
    
    # Array geometry definition
    x = np.linspace(0, (M2-1)*mic_spacing, M2)
    y = np.linspace(0, (M1-1)*mic_spacing, M1)
    xv, yv = np.meshgrid(x, y, indexing='xy') 
    mic_coords = np.column_stack([xv.flatten(), yv.flatten(), np.zeros(M)])
    
    array_position = np.array([1, 1, 1]).reshape(1,3)
    mic_coords = array_position + mic_coords

    # Simulation settings
    mic_array_mismatch = 1e-3
    sim_duration = 4

    # Room characteristics
    room_dimensions = np.array([2.5, 4, 2.5])

    # Initialize the acoustic scene
    acoustic_scene = SimAcoustic(mic_coords, mic_array_mismatch, duration=sim_duration, fs=fs)

    # Signals paths
    source_path = "tools/data/signals/FA01_09.wav"
    interference_path = "tools/data/signals/MC15_03.wav"
    interference_path1 = "tools/data/signals/MF31_03.wav"

    # Signals positions
    source_pos = np.array([1, 1, .1]).reshape(1,3)
    interf_pos = np.array([2, 1.3, .4]).reshape(1,3)
    interf_pos1 = np.array([1.4, 2, 1]).reshape(1,3)

    # Load audio into the scene
    acoustic_scene.set_source(source_path, gain=1, position=source_pos)
    acoustic_scene.set_interference(interference_path, gain=1, position=interf_pos)
    acoustic_scene.set_interference(interference_path1, gain=1, position=interf_pos1)

    folder_path = "tests/data"

    # ==========================================
    # TEST 1: REVERBERANT ENVIRONMENT (DATASET SCALABILITY)
    # ==========================================
    print("\n--- Running Reverberant Simulation ---")
    
    # 1. Compute physical room properties just ONCE
    acoustic_scene.compute_rirs(room_dimensions, desire_RT=0.5, ray_tracing=True)
    
    # 2. Apply convolution just ONCE
    acoustic_scene.convolve_signals(t_early=0.050)
    
    # 3. Generate multiple iSIR conditions instantly
    isirs_to_test = [0, 5, 10]
    
    for isir in isirs_to_test:
        # Fast matrix mixing
        scene_data = acoustic_scene.mix_and_normalize(iSIR_dB=isir)
        
        # Save the mixture using the reference microphone (mic 0)
        mix_filename = f"eval_mic_signals_{isir}dB.wav"
        save_wav(mix_filename, fs, scene_data["mic_signals"][0], folder_path)
        
        # Save the isolated components (only needed once, or per iSIR if scaled differently)
        if isir == 0:
            save_wav("eval_target_early.wav", fs, scene_data["target_early"][0], folder_path)
            save_wav("eval_target_late.wav", fs, scene_data["target_late"][0], folder_path)
            save_wav("eval_interf_early.wav", fs, scene_data["interference_early"][0], folder_path)
            save_wav("eval_interf_late.wav", fs, scene_data["interference_late"][0], folder_path)
            save_wav("eval_target_anechoic.wav", fs, scene_data["target_anechoic"][0], folder_path)

    # ==========================================
    # TEST 2: ANECHOIC CHAMBER (FREE FIELD)
    # ==========================================
    print("\n--- Running Free Field Simulation ---")
    
    # Re-compute using desire_RT = 0 to trigger the Anechoic logic
    acoustic_scene.compute_rirs(room_dimensions, desire_RT=0.0, ray_tracing=False)
    
    # Convolve with the new direct-path-only RIRs
    acoustic_scene.convolve_signals()
    
    # Mix
    scene_data_ff = acoustic_scene.mix_and_normalize(iSIR_dB=0)
    
    save_wav("free_field_mic_signals.wav", fs, scene_data_ff["mic_signals"][0], folder_path)
    
    print("\nAll evaluation signals have been successfully processed and saved.")