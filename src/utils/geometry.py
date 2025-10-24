import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def source_rotation(radius, samples, axis = 'h'):
    """
    Generates an array of source points distributed uniformly 
    in a circle around the origin (assumed location of the mic array).

    The function outputs a matrix of Cartesian coordinates (3, N) 
    and the corresponding array of azimuthal angles.

    Parameters
    ----------
    radius : float
        The radius of the source circle (distance from the source to the array center) in meters.
    samples : int
        The number of source points (samples) to generate along the circle.
    axis : {'h', 'v', 'l'}, optional
        Defines the plane of rotation:
        - 'h' (horizontal, default): XY plane (Z=0). Rotation around the Z-axis.
        - 'v' (vertical): XZ plane (Y=0). Rotation around the Y-axis.
        - 'l' (lateral): YZ plane (X=0). Rotation around the X-axis.

    Returns
    -------
    points : numpy.ndarray
        Matrix of Cartesian coordinates for the source points. 
        Shape: (3, samples), where Row 0=X, Row 1=Y, Row 2=Z.
    degrees : numpy.ndarray
        1D array of the azimuthal angles used to generate the points, 
        expressed in degrees. Shape: (samples,).
    """

    angles = np.arange(0, 2*np.pi, 2*np.pi/samples)

    cos = np.cos(angles)
    sin = np.sin(angles)
    zeros = np.zeros(len(angles))

    degrees = np.degrees(angles)

    if axis == "h":
        points = np.stack([cos, sin, zeros], axis = 0)
    elif axis == "v": 
        points = np.stack([cos, zeros, sin], axis = 0)
    elif axis == "l":
        points = np.stack([zeros, sin,  cos], axis = 0)
    else:
        # Error
        raise ValueError("Parameter must be: 'h', 'v' o 'l'.")

    points = radius * points

    return points, degrees 
    
def sferical_to_coord(radius, azimut, elevation):
    # Función de conversión (usamos la que ya definiste)
    coords = np.array([ np.cos(azimut) * np.sin(elevation),
                        np.sin(elevation) * np.sin(azimut),
                        np.cos(elevation) ])

    return radius * coords

def source_sphere_grid(radius, samples_azimut, samples_elevation):
    """
    Genera puntos de fuente en una Malla de Cuadrícula Esférica.

    Parameters
    ----------
    radius : float
        Radio de la esfera.
    samples_azimut : int
        Número de divisiones angulares para el ángulo Azimutal (phi).
    samples_elevation : int
        Número de divisiones angulares para el ángulo de Elevación (theta).

    Returns
    -------
    coords_grid (ndarray): Matriz de coordenadas (3, N*M).
    azimut_flat (ndarray): Vector 1D de todos los ángulos azimutales usados (N*M).
    elevation_flat (ndarray): Vector 1D de todos los ángulos de elevación usados (N*M).
    """

    # 1. Crear vectores 1D para cada eje angular (Delta angular constante)
    # Azimut (phi): 0 a 2*pi
    azimut_1d = np.linspace(0, 2 * np.pi, samples_azimut, endpoint=False) # No incluye 2*pi
    
    # Elevación (theta): 0 a pi (desde el eje Z)
    elevation_1d = np.linspace(0, np.pi, samples_elevation)

    # 2. Generar la Malla de Cuadrícula (Grid)
    # Genera dos matrices 2D (azimut_mesh, elevation_mesh)
    azimut_mesh, elevation_mesh = np.meshgrid(azimut_1d, elevation_1d)

    # 3. Aplanar las mallas a vectores 1D
    # Esto crea los dos vectores de longitud (N*M) con todas las combinaciones.
    azimut_flat = azimut_mesh.flatten()
    elevation_flat = elevation_mesh.flatten()

    # 4. Uso de Broadcasting
    # La función vectorizada (sferical_to_coord) usa estos vectores N*M para generar la cuadrícula.
    coords_grid = sferical_to_coord(radius, azimut_flat, elevation_flat)
    
    # coords_grid tendrá forma (3, N*M)
    return coords_grid, azimut_flat, elevation_flat

def plot_3d_points(points, title, ax):
    """Auxiliary function to plot 3D points from a (3, N) matrix."""
    
    # Los puntos están en formato (3, N): Fila 0=X, Fila 1=Y, Fila 2=Z
    X = points[0, :]
    Y = points[1, :]
    Z = points[2, :]

    ax.scatter(X, Y, Z, s=15, alpha=0.8) # s es el tamaño del punto
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    # Ajustar límites para mantener la escala esférica
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def main_test_geometry():
    """Ejecuta las pruebas de geometría y grafica los resultados."""
    
    R = 1.5  # Radio (distancia de la fuente) en metros
    N_circle = 30  # 30 puntos para el círculo
    N_grid = 15  # 15 divisiones para cada ángulo de la malla (15*15 = 225 puntos)

    print(f"--- Prueba de Generación de Geometrías ---")
    print(f"Radio (R): {R} m")
    print(f"Malla esférica: {N_grid}x{N_grid} = {N_grid*N_grid} puntos.")
    print("-" * 35)

    # 1. PRUEBA DE SOURCE_ROTATION (CÍRCULOS)
    points_h, _ = source_rotation(R, N_circle, 'h')
    points_v, _ = source_rotation(R, N_circle, 'v')
    points_l, _ = source_rotation(R, N_circle, 'l')

    # 2. PRUEBA DE SOURCE_SPHERE_GRID (MALLA ESFÉRICA)
    points_grid, _, _ = source_sphere_grid(R, N_grid, N_grid)
    
    # ------------------------------------------------------------
    # VISUALIZACIÓN
    fig = plt.figure(figsize=(12, 9))
    
    # Subplot 1: Círculo Horizontal (XY)
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    plot_3d_points(points_h, f'Círculo Horizontal (XY) - {N_circle} pts', ax1)

    # Subplot 2: Círculo Vertical (XZ)
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    plot_3d_points(points_v, f'Círculo Vertical (XZ) - {N_circle} pts', ax2)
    
    # Subplot 3: Círculo Lateral (YZ)
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    plot_3d_points(points_l, f'Círculo Lateral (YZ) - {N_circle} pts', ax3)
    
    # Subplot 4: Malla Esférica (Grid)
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    plot_3d_points(points_grid, f'Malla Esférica ({N_grid}x{N_grid}) - {N_grid*N_grid} pts', ax4)
    

    plt.tight_layout()
    plt.show()

# Ejecutar el main
if __name__ == "__main__":
    main_test_geometry()