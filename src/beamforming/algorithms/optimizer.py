from abc import ABC, abstractmethod
from ..array.mic_array import MicArray
from typing import Tuple
import numpy as np

class WeightOptimizer(ABC):
    """
    Clase base abstracta (Interfaz) para todas las estrategias de cálculo de pesos.
    Define el contrato que todos los algoritmos de optimización deben seguir.
    """
    @abstractmethod
    def calculate(self, array_obj: MicArray, K: int, fs: float, **kwargs) -> np.ndarray:
        """
        Este método debe ser implementado por cada algoritmo específico.

        Args:
            array_obj (MicrophoneArray): El objeto que describe la geometría del arreglo.
            K (int): El número de taps por filtro.
            fs (float): La frecuencia de muestreo.
            **kwargs: Argumentos adicionales específicos del algoritmo.
                      Por ejemplo, LcmvOptimizer esperará recibir 'focal_point'.

        Returns:
            np.ndarray: El vector de pesos calculado, de forma (M*K,).
        """
        pass