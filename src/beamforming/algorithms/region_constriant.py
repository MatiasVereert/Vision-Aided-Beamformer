import numpy as np 
from beamforming.signal_model import near_field_steering_vector_multi
from scipy.constants import speed_of_sound



def covariance_matrix(f_vector, R_vector, fs, mic_array, K=1, c=speed_of_sound):
    P = len(f_vector)
    J = len(R_vector)
    R= 1/ (P * J) 

    #Calculate the steering vector for P points and J frecuencies 
    a = near_field_steering_vector_multi(f_vector, R_vector, fs, mic_array, K=1, c=speed_of_sound)
    
    #Shape [F, P, N] @ [F, P, N] -> [N,N]
    Rss = np.einsum('fpn,fpn->nm', a ,a.conj()) / R

    return Rss

    