from beamforming.signal_model import near_field_steering_vector, near_field_steering_vector_multi
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple
from ..array.mic_array import MicArray



def point_constraint( target_point , K_taps, mic_array , f, fs ):
    #steering vector of the specific point (shape (M.K , 1))
    steering_vector = near_field_steering_vector(f, target_point, fs, mic_array, K=K_taps)
    
    constrains_C = np.hstack([np.real(steering_vector), np.imag(steering_vector)]) 

    # 3. Definir el vector de respuesta deseada h. (Forma: 2 x 1)
    # [Ganancia Real Deseada (1)]
    # [Ganancia Imaginaria Deseada (0)]
    target_gain_h = np.vstack([1.0, 0.0]) # [1, 0]^T

    return constrains_C, target_gain_h

class ConstrainGenerator(ABC):
    '''
    Abstract Class to generate to generate constrains
    '''
    @abstractmethod
    def generate(self, array_obj: MicArray, K: int, fs: float, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        pass

class NarrowbandPointConstrain(ConstrainGenerator):
    '''
    Defines constrainst structure for unitary gain at focal focal point with f_target. 
    '''
    def __init__(self,  f_target : int, **kwargs):

        self.f_target = f_target

    def generate(self, array_obj: MicArray, K: int, fs: float, **kwargs):

        #search the kwargs 
        if 'focal_point' not in kwargs:
            raise ValueError("NarrowbandPointConstrain requires 'focal_point' in kwargs")
        
        focal_point = kwargs['focal_point']

        mic_array = array_obj.coordinates

        C, H = point_constraint( target_point = focal_point,
                                 K_taps= K,
                                 mic_array = mic_array,
                                 f = self.f_target,
                                 fs = fs  )
        return C, H
    

    


