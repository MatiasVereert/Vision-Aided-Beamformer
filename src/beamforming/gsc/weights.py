from beamforming.signal_model import near_field_steering_vector
import numpy as np 
from ..array.mic_array import MicArray


def compute_fixed_weights_optimized(Constrains, Target_gain):
    '''
    Fixed weights computation given by the equation
             w_q = C(C^[H]C)^-1 h.
    Uses np.linalg.solve for improved numerical stability and efficiency 
    compared to explicit matrix inversion.
    '''
    C_H = Constrains.conj().T
    
    # M = C^H C (La matriz que va a ser invertida)
    M = C_H @ Constrains
    
    # V = h (El vector de la derecha)
    V = Target_gain
    
    # X = (C^H C)^-1 h  --> Resuelve M X = V para X
    X = np.linalg.solve(M, V)
    
    # w_q = C X
    fixed_weights = Constrains @ X

    return fixed_weights

