from matplotlib import pyplot as plt 
import numpy as np
from scipy.constants import speed_of_sound

from utils.geometry import source_rotation
from beamforming.processors import beamforming
from propagation.free_field import space_delay
from beamforming.processors import snapshots
# EN: utils/polar_plot.py
import matplotlib.pyplot as plt
import numpy as np
from typing import List

def plot_polar_pattern(gains_list: List[np.ndarray], angles_deg: np.ndarray, labels_list: List[str], title: str = ""):
    """
    Grafica uno o más patrones polares superpuestos.

    Args:
        gains_list (List[np.ndarray]): Una lista de arrays de ganancia en dB.
        angles_deg (np.ndarray): Un array de ángulos en grados.
        labels_list (List[str]): Una lista de etiquetas para la leyenda, una para cada array de ganancia.
        title (str): El título del gráfico.
    """
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    # --- INICIO DE LA LÓGICA MODIFICADA ---
    # Bucle para dibujar cada patrón de ganancia con su etiqueta
    for gains_db, label in zip(gains_list, labels_list):
        ax.plot(np.deg2rad(angles_deg), gains_db, label=label)
    
    # Añadir la leyenda al gráfico para que aparezcan las etiquetas
    ax.legend()
    # --- FIN DE LA LÓGICA MODIFICADA ---

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    
    # Ajustar límites para que se vean bien todos los plots
    # (asumiendo que todos están normalizados a 0 dB)
    ax.set_rlim(-50, 5) 
    ax.set_rticks(np.arange(-40, 1, 10))
    ax.set_title(title, va='bottom', pad=20)
    plt.tight_layout()
    plt.show()
