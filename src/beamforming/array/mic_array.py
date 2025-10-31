import numpy as np
import matplotlib.pyplot as plt

class MicArray:
    def __init__(self, M: int, coordinates: np.ndarray):
        if coordinates.shape != (M,3):
            raise ValueError("Array Coordinates must be (M,3)")
        
        self.M = M
        self.coordinates = coordinates 

    def Plot_Geometry(self):
            print("Ploting Array")
            x = self.coordinates[:,0]
            y = self.coordinates[:,1]
            plt.figure()
            plt.scatter(x, y)
            plt.show
            plt.title("Geometría del Arreglo de Micrófonos")
            plt.xlabel("Posición X (m)")
            plt.ylabel("Posición Y (m)")
            plt.grid(True)
            plt.axis('equal') # Para que las escalas en X e Y sean iguales
            plt.show()    

class ULA(MicArray): 
    def __init__(self, M: int, d: float):
        x_position = np.linspace(-(M-1)*d/2, (M-1)*d/2, M)
        coordinates = np.stack([x_position, np.zeros(M) , np.zeros(M)], axis = 1)
        super().__init__(M=M, coordinates=coordinates)
        print("   -> Geometría específica: ULA")

class custom(MicArray):
    def __init__(self, coordinates: np.ndarray):
        self.coordinates = coordinates
        M = coordinates.shape[0]
        super().__init__(M=M, coordinates=coordinates)
        print("  -> Custom Array Build")


mic_array = np.stack([np.arange(0,3,1), np.zeros(3), np.zeros(3)], axis = 1)


            
            
