from matplotlib import pyplot as plt 
import numpy as np
from scipy.constants import speed_of_sound
from .utils.geo



def calculate_polar(f, mic_array, K_taps, fs, c= speed_of_sound, beamformer = 'narrowband_LCMV', propagation = "free_field"  ):
    """
    
    """
    #Define the test signal
    duration_s = 1  # 1 s of duration
    t = np.arange(0, duration_s, 1/fs)
    signal_source = np.sin(2 * np.pi * f * t)

    #Calculate signal arribal from diferent angles 
    source_points = 



    


    
    return 


def simple_plot( gain, angle ):
    """ 
    


    """
    return 