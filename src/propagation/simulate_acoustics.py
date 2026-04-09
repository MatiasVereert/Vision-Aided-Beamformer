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


    def free_field(self, iSIR_dB , normalize = True, mode = "real", VAD = False):
        
        #Selects geometry (real/ideal)
        if mode == "real":
            array_geometry = self.real_array
        else:
            array_geometry = self.ideal_array

        samples_lenght = self.duration * self.fs
        

        #Inicialice a list to save the signals
        interference_signals = []
        interference_sum = np.zeros((self.M, samples_lenght ))

        for element in self.audio_interferences:
            audio = element['audio']
            position = element['position']


            #Propagates delay and cut to speicided time
            audio_propagated = space_delay(audio, self.fs, position, array_geometry)
            #Normalize
            if normalize:
                rms = np.sqrt(np.mean(audio_propagated[0,:]**2))
                audio_propagated = audio_propagated / rms


            interference_sum = interference_sum + audio_propagated

        #Proces source dealy
        source_delayed = space_delay( signal_in = self.audio_sources[0]["audio"], 
                                fs = self.fs,
                                source_pos = self.audio_sources[0]["position"], 
                                mic_array = array_geometry)
        
        #Cut to obtain specify lenght
        source_delayed = source_delayed[:samples_lenght]


        #Normalize signal and interfernce sumation to set SIR
        interference_rms = np.sqrt(np.mean(interference_sum[0,:]**2))
        interference_input_norm = interference_sum / interference_rms #Sets the input signal to 0 RMSfs

        source_rms = np.sqrt(np.mean(source_delayed[0,:]**2))
        source_input_norm = source_delayed / source_rms #Sets the input signal to 0 RMSfs



        #Mix signal and interfernce with the desire iSIR
        iSIR = 10**(iSIR_dB / 20)
        array_input = source_input_norm  +  interference_input_norm *(1/iSIR)

        if VAD:
            # Compute binary VAD ORACLE
            # Compute Hilbert transform of the early isolated target signal
            hilbert_target_early = sc.signal.hilbert(source_delayed[0,:] )
            hilbert_target_early_energy = np.abs(hilbert_target_early) + 1e-12 
            voice_peak = np.max(hilbert_target_early_energy)

            # Define the VAD treshole
            VAD_treshole_dB  = -30 #dB 
            VAD_treshole = 10**(VAD_treshole_dB / 10) * voice_peak

            # Compare
            vad_oracle = (hilbert_target_early_energy > VAD_treshole).astype(bool)
            
            return array_input, vad_oracle
        else:
            return array_input
    

    def compute_room_ISB(  self, room_dimensions, desire_RT, iSIR_dB =0 , mode ="real" , inter_normalization = True , ray_tracing = True ):
        """
        This program computes the RIRs using image source methond and convolve to obtain 
        each sensor room responese
        For simplicity this method asumes there is only one source signal. But is intended to expand
        for future testing.
        """
        #Define lenght
        N_samples = self.duration * self.fs

        #estimate absortion, maximun order of sources,set the material
        alpha, max_order = pra.inverse_sabine(desire_RT, room_dimensions)
        

        if ray_tracing:
            # obtain minimun distance between sensors
            distances = pdist(self.real_array)
            min_spacing = np.min(distances[distances > 0])

            # get best setings for the room
            params_dic = get_hybrid_sim_params(room_dimensions, min_spacing, t_mix=0.06)
            max_order = params_dic["max_order"]

            ray_dic = params_dic["ray_tracing"]
            n_rays = ray_dic["n_rays"]
            receiver_radius = ray_dic["receiver_radius"]
            energy_thres = ray_dic["energy_thres"]

            material = pra.Material(alpha, scattering = 0.2)        
            

            room = pra.ShoeBox(room_dimensions, 
                               self.fs, 
                               materials = material,
                               max_order=max_order,

                               ray_tracing = True,
                               )
            room.set_ray_tracing(**params_dic["ray_tracing"])

        else:
            

            material = pra.Material(alpha)        
            room = pra.ShoeBox(room_dimensions, self.fs, materials = material, max_order=max_order)

        #Set Source 
        source_pos = self.audio_sources[0]["position"].T
        source_sig = self.audio_sources[0]["audio"].T
        
        room.add_source( source_pos )
        
        #Set interferene positions
        for i ,interference  in enumerate(self.audio_interferences):
            room.add_source(interference["position"].T)

        #Set mic array acording to the mode
        if mode == "real":
            mic_array = self.real_array.T
        else:
            mic_array = self.ideal_array.T 
        room.add_microphone_array( mic_array)

        #Compute RIRs
        room.compute_rir( )
        conv_source_signals = []

        #inicialice vectors to save results
        source_signals = np.zeros((self.M,N_samples ))

        #Convolve Source signal
        for i in range(self.M):
            sig = sc.signal.fftconvolve( room.rir[i][0], source_sig )
            source_signals[i, :] = sig[:N_samples] 
            conv_source_signals.append(sig)

        interference_signals = np.zeros((self.M, len(self.audio_interferences), N_samples))    
        
        #Convolve interference signals
        for i, interference  in enumerate(self.audio_interferences):
            audio = interference["audio"]
            

            for j in range(self.M):
                sig = sc.signal.fftconvolve( room.rir[j][i+1], audio)

                #Cut and save into specified duration
                interference_signals[j,i,:] = sig[:self.fs*self.duration] 
        
        #--    Normalize RMS   --
        if inter_normalization:
            #The source only has one signal 
            signal_rms = np.sqrt(np.mean(source_signals[0,:]**2))
            source_signals_norm = source_signals / signal_rms

            #Brodcast the interfences
            #Separate M = 0 signals to refer the rs
            ref_signals = interference_signals[0,:,:]
            interference_rms_inverse = 1 / np.sqrt( np.mean(ref_signals**2, axis =1 ))
            interference_signals = np.einsum( 'mst, s->mt',interference_signals, interference_rms_inverse )

        # Re normalize interference mix (ineficient but robust)
        rms_interference_sum = np.sqrt(np.mean( interference_signals[0,:]**2))
        interference_signals_norm = interference_signals / rms_interference_sum
        
        
        # Mix signals and interference with the desired iSIR
        iSIR = 10 ** (iSIR_dB / 20)
        array_input = source_signals_norm  +  interference_signals_norm *(1/iSIR)
        return array_input
    
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

    def get_eval_scene(self, room_dimensions, desire_RT, iSIR_dB=0, mode="real", inter_normalization=True, ray_tracing=True):
            # Define output length in samples
            N_samples = self.duration * self.fs

            # Estimate absorption and maximum order of sources
            alpha, max_order = pra.inverse_sabine(desire_RT, room_dimensions)
            
            if ray_tracing:
                distances = pdist(self.real_array)
                min_spacing = np.min(distances[distances > 0])
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
            else:
                material = pra.Material(alpha)        
                room = pra.ShoeBox(room_dimensions, self.fs, materials=material, max_order=max_order)

            # Set source and interferences
            source_pos = self.audio_sources[0]["position"] # Shape: (1, 3)
            source_sig = self.audio_sources[0]["audio"]
            room.add_source(source_pos.T)
            
            for interference in self.audio_interferences:
                room.add_source(interference["position"].T)

            # Set mic array
            mic_array = self.real_array if mode == "real" else self.ideal_array
            room.add_microphone_array(mic_array.T)

            # Compute RIRs
            room.compute_rir()

            # Initialize storage arrays
            target_early = np.zeros((self.M, N_samples))
            target_late = np.zeros((self.M, N_samples))
            interf_early_sum = np.zeros((self.M, N_samples))
            interf_late_sum = np.zeros((self.M, N_samples))

            # Convolve Source signal (Target)
            for i in range(self.M):
                h_early, h_late = self._split_rir(room.rir[i][0])
                
                sig_early = sc.signal.fftconvolve(h_early, source_sig.flatten())
                sig_late = sc.signal.fftconvolve(h_late, source_sig.flatten())
                
                target_early[i, :] = sig_early[:N_samples]
                target_late[i, :] = sig_late[:N_samples]

            # Convolve interference signals
            for i, interference in enumerate(self.audio_interferences):
                audio = interference["audio"].flatten()
                
                # Temporary arrays for current interference across all mics
                curr_interf_early = np.zeros((self.M, N_samples))
                curr_interf_late = np.zeros((self.M, N_samples))
                
                for j in range(self.M):
                    h_early, h_late = self._split_rir(room.rir[j][i+1])
                    
                    sig_early = sc.signal.fftconvolve(h_early, audio)
                    sig_late = sc.signal.fftconvolve(h_late, audio)
                    
                    curr_interf_early[j, :] = sig_early[:N_samples]
                    curr_interf_late[j, :] = sig_late[:N_samples]
                
                # Accumulate interferences
                interf_early_sum += curr_interf_early
                interf_late_sum += curr_interf_late

            # Compute pure anechoic target using free field propagation
            target_anechoic = space_delay(
                signal_in=source_sig.flatten(),
                fs=self.fs,
                source_pos=source_pos,
                mic_array=mic_array
            )
            
            # Ensure length matches N_samples (pad with zeros if shorter, crop if longer)
            if target_anechoic.shape[1] < N_samples:
                target_anechoic = np.pad(target_anechoic, ((0, 0), (0, N_samples - target_anechoic.shape[1])))
            else:
                target_anechoic = target_anechoic[:, :N_samples]

            # Normalization and Mixing
            if inter_normalization:
                # Calculate RMS of the total target signal at mic 0 for reference
                target_total_mic0 = target_early[0, :] + target_late[0, :]
                target_rms = np.sqrt(np.mean(target_total_mic0**2)) + 1e-10
                
                # Calculate RMS of the total interference sum at mic 0
                interf_total_mic0 = interf_early_sum[0, :] + interf_late_sum[0, :]
                interf_rms = np.sqrt(np.mean(interf_total_mic0**2)) + 1e-10
                
                # Apply interference scaling to achieve the desired iSIR
                iSIR_linear = 10**(iSIR_dB / 20)
                sir_scaling_factor = (target_rms / interf_rms) * (1 / iSIR_linear)
                
                interf_early_sum *= sir_scaling_factor
                interf_late_sum *= sir_scaling_factor

                # Construct the final unscaled mixture
                array_input = target_early + target_late + interf_early_sum + interf_late_sum

                # Determine the global peak across all microphones in the mixture
                global_max = np.max(np.abs(array_input))
                
                # Calculate global scaling factor to prevent clipping (leaving 1% headroom)
                global_scale = 0.99 / (global_max + 1e-10)
                
                # Apply the exact same global scaling to ALL components to preserve iSIR
                array_input *= global_scale
                target_early *= global_scale
                target_late *= global_scale
                interf_early_sum *= global_scale
                interf_late_sum *= global_scale
                
                # Apply exact same scale to the anechoic reference
                target_anechoic *= global_scale
            else:
                array_input = target_early + target_late + interf_early_sum + interf_late_sum

            # Compute binary VAD ORACLE
            # Compute hilbert transform of the early isolated signal
            hilbert_target_early = sc.signal.hilbert(target_early[0,:] )
            hilbert_target_early_energy = np.abs(hilbert_target_early) + 1e-12 
            voice_peak = np.max(hilbert_target_early_energy)
        

            # Define the VAD treshole
            VAD_treshole_dB  = -30 #dB 
            VAD_treshole = 10**(VAD_treshole_dB / 20) * voice_peak

            # Compare
            vad_oracle = (hilbert_target_early_energy > VAD_treshole).astype(bool)

            






            # Return a structured dictionary with all evaluation components
            return {
                "mic_signals": array_input,               # The actual input for your MPDR-WPE
                "target_anechoic": target_anechoic,       # Pure direct path (Anechoic) for PESQ
                "target_early": target_early,             # Target early reflections
                "target_late": target_late,               # Target reverberation (WPE target)
                "interference_early": interf_early_sum,   # Directional interference (MPDR target)
                "interference_late": interf_late_sum,      # Diffuse background noise
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
    source_postion_mismatch = .1
    sim_duration = 4

    # Room characteristics
    room_dimensions = np.array([2.5, 4, 2.5])
    desire_RT = 1

    acoustic_scene = SimAcoustic(mic_coords, mic_array_mismatch , duration = sim_duration)

    # Signals paths
    source_path =       "tools/data/signals/FA01_09.wav"
    interference_path = "tools/data/signals/MC15_03.wav"
    interference_path1 = "tools/data/signals/MF31_03.wav"

    # Signals positions
    source_pos = np.array([1,1,.1]).reshape(1,3)
    interf_pos = np.array([2,1.3,.4]).reshape(1,3)
    interf_pos1 = np.array([1.4,2,1]).reshape(1,3)

    acoustic_scene.set_source(source_path, gain= 1, position= source_pos)
    acoustic_scene.set_interference(interference_path, gain = 1,position =  interf_pos )
    acoustic_scene.set_interference(interference_path1, gain = 1,position =  interf_pos1 )

    folder_path = "tests/data"

    # FREE FIELD
    array_input = acoustic_scene.free_field( iSIR_dB = 0)
    save_wav("Array_input.wav", fs, array_input[0], folder_path)

    # ROOM ACOUSTIC SIMULATION FOR EVALUATION
    # Execute the evaluation scene method to get the split signals dictionary
    scene_data = acoustic_scene.get_eval_scene(room_dimensions, desire_RT = .5, iSIR_dB=0)

    # ROOM ACOUSTIC SIMULATION FOR EVALUATION
    # Execute the evaluation scene method to get the split signals dictionary
    scene_data = acoustic_scene.get_eval_scene(room_dimensions, desire_RT=0.5, iSIR_dB=0)

    # Save all the evaluation components using the reference microphone (mic 0)
    # 1. The dirty mixture that feeds the beamformer
    save_wav("eval_mic_signals.wav", fs, scene_data["mic_signals"][0], folder_path)
    
    # 2. Target with early reflections (Recommended PESQ/STOI reference)
    save_wav("eval_target_early.wav", fs, scene_data["target_early"][0], folder_path)
    
    # 3. Target late reverberation (What the WPE tries to cancel)
    save_wav("eval_target_late.wav", fs, scene_data["target_late"][0], folder_path)
    
    # 4. Directional early interference (What the MPDR tries to cancel)
    save_wav("eval_interference_early.wav", fs, scene_data["interference_early"][0], folder_path)
    
    # 5. Diffuse late interference (Background noise)
    save_wav("eval_interference_late.wav", fs, scene_data["interference_late"][0], folder_path)

    print("All evaluation signals have been successfully saved.")
