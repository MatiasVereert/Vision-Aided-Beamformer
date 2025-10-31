from beamforming.signal_model import near_field_steering_vector, near_field_steering_vector_multi

from abc import ABC, abstractmethod
import numpy as np
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
    Abstract Class to generate diferent constraints
    '''
    @abstractmethod
    def generate(self, array_obj: MicArray, K: int, fs: float) -> Tuple[np.ndarray, np.ndarray]:
        pass


class NarrowbandPointConstrain(ConstrainGenerator):
    '''
    '''
    def __init__(self, focal_point : np.ndarray, f_target : int ):
        self.focal_point = focal_point
        self.f_target = f_target

    def generate(self, array_obj: MicArray, K: int, fs: float):
        mic_array = array_obj.coordinates

        C, H = point_constraint(self.focal_point, K, mic_array, self.f_target, fs  )
        return C, H
    
    

#class BroadbandPointConstrain(ConstrainGenerator):
    '''
    Work in progress
    '''
    


