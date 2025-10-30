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

#def plot_polar_pattern(gain_db, angles_deg, title="Patrón de Directividad del Beamformer"):
    """
    Generates a 2D polar plot of a beamformer's directivity pattern.

    This function takes gain data in decibels and corresponding angles in
    degrees and creates a standard polar plot. It is designed to visualize
    the output of functions like `calculate_gain_at_points` after normalization.

    Args:
        gain_db (np.ndarray): 
            A 1D array containing the beampattern gain in decibels (dB).
            It is assumed that this data is already normalized, so the peak is at 0 dB.
        angles_deg (np.ndarray): 
            A 1D array of the corresponding angles in degrees. Must have the
            same length as `gain_db`.
        title (str, optional): 
            The title for the polar plot. Defaults to 
            "Patrón de Directividad del Beamformer".
    """
    """# 1. Convert angles from degrees to radians, as required by Matplotlib's polar plot.
    angles_rad = np.deg2rad(angles_deg)

    # 2. Create a figure and a subplot with a polar projection.
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    # 3. Plot the gain data against the angles.
    ax.plot(angles_rad, gain_db)

    # 4. Configure the plot for better readability.
    # Set the radial limits (r-axis) to show the dynamic range clearly.
    # For example, from -40 dB up to a little above 0 dB.
    ax.set_rmin(np.min(gain_db) - 5 if np.min(gain_db) > -100 else -40)
    ax.set_rmax(np.max(gain_db) + 5 if np.max(gain_db) < 5 else 5)
    
    # Set the title for the plot. 'va' positions it nicely.
    ax.set_title(title, va='bottom', fontsize=14)

    # Set the label for the radial axis (the gain).
    ax.set_ylabel("Ganancia (dB)", labelpad=-40)
    
    # Set the 0-degree angle to be at the top ("North").
    ax.set_theta_zero_location("N")
    
    # Set the angular direction to be clockwise.
    ax.set_theta_direction(-1)

    # Enable the grid for easier reading.
    ax.grid(True)

    # 5. Display the plot.
    plt.show()

    return"""