from beamforming.signal_model import near_field_steering_vector
import numpy as np 
from .optimizer import WeightOptimizer
from .constrains import ConstrainGenerator
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

import numpy as np

class LmcvOptimizer(WeightOptimizer):
    def __init__(self, constrains_generator : ConstrainGenerator):
        '''
        Indicates constrain strategy.
        '''
        self.constrains_generator = constrains_generator
    
    def calculate(self, array_obj: MicArray, K: int, fs: float, **kwargs):
        """
        Calcula los pesos del beamformer.
        
        Args:
            array_obj (MicrophoneArray): El objeto de la geometría del arreglo.
            K (int): Número de taps.
            fs (float): Frecuencia de muestreo.
            **kwargs: Este optimizador no los usa directamente, pero los pasa
                      al generador de restricciones si este los necesita.
        """
        C, H = self.constrains_generator.generate(array_obj, K, fs, **kwargs)

        weights = compute_fixed_weights_optimized(C, H)
        return weights
