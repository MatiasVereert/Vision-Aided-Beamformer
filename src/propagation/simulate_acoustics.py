import numpy as np 
import scipy as sc
from matplotlib import pyplot as plt 
from utils.audio import load_audio_source, save_wav
from propagation.free_field import space_delay
import pyroomacoustics as pra




class SimAcoustic():

    def __init__(self, 
                 array_geometry, 
                 array_mismatch = 1e-3, 
                 duration= 4 ):
        
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

        self.fs = 48000

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


    def free_field(self, iSIR_db , normalize = True, mode = "real"):
        
        #Selects geometry (real/ideal)
        if mode == "real":
            array_geometry = self.real_array
        else:
            array_geomtery = self.ideal_array

        samples_lenght = self.duration * self.fs
        

        #Inicialice a list to save the signals
        interference_signals = []
        interference_sum = np.zeros((self.M, samples_lenght ))

        for element in self.audio_interferences:
            audio = element['audio']
            position = element['position']


            #Propagates delay and cut to speicided time
            audio_propagated = space_delay(audio, self.fs, position, array_geometry)
            audio_propagated = audio_propagated[:samples_lenght]
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
        iSIR = 10**(iSIR_db / 20)
        array_input = source_input_norm  +  interference_input_norm *(1/iSIR)

        return array_input
    

    def compute_room_ISB(  self, room_dimensions, desire_RT, iSIR_dB =0 , mode ="real" , inter_normalization = True  ):
        """
        This program computes the RIRs using image source methond and convolve to obtain 
        each sensor room responese
        For simplicity this method asumes there is only one source signal. But is intended to expand
        for future testing.
        
        """
        #estimate absortion, maximun order of sources,set the material
        alpha, max_order = pra.inverse_sabine(desire_RT, room_dimensions)
        material = pra.Material(alpha)

        #Define lenght
        N_samples = self.duration * self.fs

        #instace roome 
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

        #Plot RIRs
        room.plot_rir()
        plt.show()

        #inicialice vectors to save results
        source_signals = np.zeros((self.M,N_samples ))

        #Convolve Source signal
        for i in range(self.M):
            sig = sc.signal.fftconvolve( room.rir[i][0], source_sig )
            source_signals[i, :] = sig[:N_samples] 
            conv_source_signals.append(sig)

        interference_signals = np.zeros((self.M, len(self.audio_interferences), N_samples))    
        
        #convolve interference signals
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
        iSIR = 10**(iSIR_dB / 20)
        array_input = source_signals_norm  +  interference_signals_norm *(1/iSIR)
        return array_input
    

if __name__ == "__main__":

    fs = 48000
    mic_spacing = 0.04 
    M1, M2 = 3, 3  
    M = M1 * M2
    
    # Geometría
    x = np.linspace(0, (M2-1)*mic_spacing, M2)
    y = np.linspace(0, (M1-1)*mic_spacing, M1)
    xv, yv = np.meshgrid(x, y, indexing='xy') 
    mic_coords = np.column_stack([xv.flatten(), yv.flatten(), np.zeros(M)])
    print(mic_coords.shape)

    array_position = np.array([1, 1, 1]).reshape(1,3)
    mic_coords = array_position + mic_coords

    #Sim settings
    mic_array_mismatch = 1e-3
    source_postion_mismatch = .1
    sim_duration = 4

    #Room caracteristics
    room_dimensions = np.array([2.5, 4, 2.5])
    desire_RT = 1


    acoustic_scene = SimAcoustic(mic_coords, mic_array_mismatch , duration = sim_duration)

    #Signals
    source_path =       "tools/data/signals/FA01_09.wav"
    interference_path = "tools/data/signals/MC15_03.wav"
    interference_path1 = "tools/data/signals/MF31_03.wav"

    #Signals Position
    source_pos = np.array([1,1,.1]).reshape(1,3)
    interf_pos = np.array([2,1.3,.4]).reshape(1,3)
    interf_pos1 = np.array([1.4,2,1]).reshape(1,3)

    acoustic_scene.set_source(source_path, gain= 1, position= source_pos)
    acoustic_scene.set_interference(interference_path, gain = 1,position =  interf_pos )
    acoustic_scene.set_interference(interference_path1, gain = 1,position =  interf_pos1 )

    #Compute free field simulation
    array_input = acoustic_scene.free_field( iSIR_db = 0)
    folder_path = "tests/data"
    save_wav("Array_input.wav", fs, array_input[0], folder_path)


    #Compute room simulation
    isb_input = acoustic_scene.compute_room_ISB(room_dimensions, desire_RT = .5)

    save_wav("isb.wav", fs, isb_input[0], folder_path)

    



    

        
        
        
    
    
