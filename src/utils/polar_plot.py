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

    # --- Bucle para dibujar cada patrón ---
    for gains_db, label in zip(gains_list, labels_list):
        # Asume que angles_deg es 0=Eje X, 90=Eje Y
        ax.plot(np.deg2rad(angles_deg), gains_db, label=label)
    
    # Mueve la leyenda fuera del gráfico para que no tape los datos
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))
    
    # --- INICIO DE MODIFICACIONES ---

    # 1. Ajustar el eje angular (Dirección X/Y)
    # Pone 0° en el Este (eje X), que es la convención matemática estándar
    ax.set_theta_zero_location('E') 
    # Define los grids angulares y sus etiquetas para X e Y
    ax.set_thetagrids(
        [0, 90, 180, 270], 
        labels=['X (0°)', 'Y (90°)', '-X (180°)', '-Y (270°)']
    )

    # 2. Ajustar el eje radial (Etiqueta de Ganancia)
    ax.set_rlim(-50, 5) # Límite radial
    ax.set_rticks(np.arange(-40, 1, 10)) # Marcas radiales
    
    # Posiciona los números de los ticks (ej. a 22.5 grados)
    ax.set_rlabel_position(22.5) 
    
    # Añade la etiqueta "Ganancia (dB)" al eje radial
    # Se usa ax.text para un control preciso de la posición
    ax.text(np.deg2rad(22.5), ax.get_rmin() - 15, 'Ganancia (dB)', 
            ha='center', va='center', fontsize=9, color='gray')
    
    # --- FIN DE MODIFICACIONES ---
    
    ax.set_title(title, va='bottom', pad=20)
    plt.tight_layout()
    plt.show()