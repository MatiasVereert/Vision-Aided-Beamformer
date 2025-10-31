from .array.mic_array import MicArray 
from .algorithms.optimizer import WeightOptimizer
import numpy as np
from  beamforming.evaluation.gain import analytical_gain

class Beamformer():
    """

    """

    def __init__(self, 
                 array_obj: MicArray,
                 optimization_obj: WeightOptimizer,
                 K: int,
                 fs : int ):
        
        self.array_obj = array_obj
        self.optimization = optimization_obj
        self.K = K
        self.fs = fs
    
    def compute_weights( self, **kwargs ):

        self.focal_point = kwargs['focal_point']
        print("Computing weights")

        weights = self.optimization.calculate(
            self.array_obj,
            self.K,
            self.fs,
            **kwargs
        )
        print("The weights has been updated")
        self.weights = weights

    def compute_gain(self, frecs: np.ndarray, points: np.ndarray,  **kwargs ):
        mic_array = self.array_obj.coordinates 
        gains = analytical_gain(frecs = frecs,          
                                source_points = points,         
                                fs = self.fs,   
                                mic_array = mic_array,
                                weights= self.weights
                                )
        return gains 
        



