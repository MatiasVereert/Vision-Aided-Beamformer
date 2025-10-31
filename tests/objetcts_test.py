from beamforming.array.mic_array import MicArray, custom, ULA 
from beamforming.algorithms.optimizer import WeightOptimizer
from beamforming.algorithms.lmcv import LmcvOptimizer
from beamforming.algorithms.constrains import ConstrainGenerator, NarrowbandPointConstrain
from beamforming.beamformer import Beamformer
import numpy as np

from utils.geometry import source_rotation
from utils.polar_plot import plot_polar_pattern

arreglo = ULA(10, 0.1)
arreglo.Plot_Geometry()
K = 20
fs = 44000

focal_point = np.array([0, -1,0])
distance = np.linalg.norm(focal_point)

f_target = 1000.0 #Hz

narrow_point_constrain = NarrowbandPointConstrain( f_target)
optimizador = LmcvOptimizer( narrow_point_constrain  )

my_beamformer = Beamformer(arreglo, optimizador, K, fs )

points, angles = source_rotation(distance, 360, 'h')


my_beamformer.compute_weights(focal_point=focal_point)

gains = my_beamformer.compute_gain(frecs = f_target, points =points.T )

plot_polar_pattern(
    gains_list=[gains[0]], 
    angles_deg= angles, 
    labels_list=["Ganancia Analítica"]
)

