from beamforming.signal_model import near_field_steering_vector
from scipy import signal, fft
import numpy as np
from propagation.free_field import space_delay

class LowRankAdaptive:

    def __init__(self, mic_array, fmin, fmax, fs):
        #save atributes
        self.mic_array = mic_array
        self.fmin = fmin
        self.fmax = fmax
        self.fs = fs

    def narrow_steer(self, target_pos, f):
        self.target_pos
        #Narrow
        self.steering_v = near_field_steering_vector(f,
                                                          self.traget_pos,
                                                          self.fs,
                                                          self.mic_array,
                                                          )

    def block_process(self,
                      intput_signals,
                      M1 : int,
                      M2: int,
                      P = 1):
        n_window = 1024
        n_overlap = 512

        
        #verify dimensions 
        M = M1 * M2
        if M != intput_signals.shape[0]:
            ValueError("M1xM2 must be equal to M, please retry")
            return

        print(np.shape(intput_signals))

        #Expected signal shape (M, N_length)
        sample_lenght = intput_signals.shape[1]
        #First we obtain the f domain shape()
        f, t, X = signal.stft(x = intput_signals,
                                        fs =fs,
                                        nperseg=n_window,
                                        noverlap=n_overlap,
                                        window='hann',
                                        axis=1
                                        )
        #X shape is (M,F,n_windows)
        f_len = len(f)
        t_len = len(t)


        print(f"the shape is {X.shape}")

        #Inicialice the filters
        h1 = np.zeros((M1, f_len))
        h2 = np.random.random((M2, f_len))
        H1 = np.zeros((M, P*M2,f_len ))
        H2 = np.zeros((M, P*M1, f_len))

        #Snapshot window

        #Output in f domain

        #Iniciate process loop
        for i in range(t_len):
            snapshot = X[:,:,i]

            #ALS loop






        return 0
    
#-------------- TEST ---------------------------------------


if __name__ == "__main__":
    fs = 48000
    f_test = 3000.0 
    C_SOUND = 343
    fmin = 200
    fmax= 10000

    # Geometría Paper: 9 micros, espaciado lambda/2
    LAMBDA_REF = C_SOUND / f_test
    M = 9
    D = LAMBDA_REF / 2
    mic_x = np.linspace(0, (M - 1) * D, M) - (M - 1) * D / 2
    mic_array = np.stack([mic_x, np.zeros(M), np.zeros(M)], axis=1)
    source_pos = np.array([1,1,0])

    #Low rank setup
    M1 = 3
    M2 = 3

    #Define a test signal
    tot_time = 10
    t = np.arange(0, tot_time, 1/fs)
    source_signal = np.sin(t)

    #Space propagation
    array_input = space_delay(signal=source_signal,
                              fs = fs,
                              source_pos= source_pos,
                              mic_array= mic_array)
    
    
    print(np.shape(array_input))

    #Instance beamformer
    low_rank_bf = LowRankAdaptive(mic_array = mic_array,
                                  fmin = fmin,
                                  fmax = fmax,
                                  fs = fs
    )

    test = low_rank_bf.block_process(intput_signals=array_input,
                                     M1 = M1,
                                     M2 = M2)
    
    print(np.shape(test))












