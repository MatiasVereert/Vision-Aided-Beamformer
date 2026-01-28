from beamforming.signal_model import near_field_steering_vector, steering_vector
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple
from ..array.mic_array import MicArray
from beamforming.gsc.region_constriant import build_region_constraints

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
    


    from .region_constriant import build_region_constraints # Importa tu función existente

class RegionalBroadbandConstrain(ConstrainGenerator):
    """
    Genera restricciones robustas (Regionales) usando SVD.
    Envuelve la lógica de 'build_region_constraints'.
    """
    def __init__(self, 
                 f_min: float, 
                 f_max: float, 
                 num_freqs: int = 50,
                 num_points: int = 50,
                 delta_r: float = 0.1,
                 delta_azimut: float = np.deg2rad(4),
                 delta_elevation: float = np.deg2rad(2)):
        
        self.f_min = f_min
        self.f_max = f_max
        self.num_freqs = num_freqs
        self.num_points = num_points
        self.delta_r = delta_r
        self.delta_azimut = delta_azimut
        self.delta_elevation = delta_elevation
        self.Ca = None # Guardaremos la matriz de bloqueo aquí para uso futuro

    def generate(self, array_obj: MicArray, K: int, fs: float, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        
        if 'focal_point' not in kwargs:
            raise ValueError("RegionalBroadbandConstrain requiere 'focal_point' en kwargs")
        
        focal_point = kwargs['focal_point']
        mic_array = array_obj.coordinates

        # Llamamos a tu función existente (que devuelve C, h, Ca)
        C, h_vec, Ca = build_region_constraints(
            Rs=focal_point,
            delta_r=self.delta_r,
            delta_azimut=self.delta_azimut,
            delta_elevation=self.delta_elevation,
            mic_array=mic_array,
            fs=fs,
            K=K,
            f_min=self.f_min,
            f_max=self.f_max,
            num_points=self.num_points,
            num_freqs=self.num_freqs
        )
        
        # Guardamos Ca en la instancia para que el Beamformer pueda pedirla después
        self.Ca = Ca
        
        # Aplanamos h para que coincida con lo que espera compute_fixed_weights_optimized
        return C, h_vec.flatten()
    
    def get_blocking_matrix(self):
        if self.Ca is None:
            raise RuntimeError("Debe llamar a generate() antes de pedir la matriz de bloqueo.")
        return self.Ca

    


