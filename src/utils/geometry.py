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
    
import numpy as np

def spherical_to_cartesian(
    radius: np.ndarray, 
    azimuth: np.ndarray, 
    elevation: np.ndarray
) -> np.ndarray:
    """
    Converts arrays of Spherical coordinates to Cartesian (x, y, z).

    This function assumes the ISO 80000-2 standard (physics convention):
    - radius (r): Distance from the origin.
    - azimuth (θ): Angle in the XY-plane from the X-axis (in radians).
    - elevation (φ): Angle from the positive Z-axis (in radians).

    Args:
        radius (np.ndarray): Array of radial distances. Shape (N,).
        azimuth (np.ndarray): Array of azimuth angles (theta) in radians. Shape (N,).
        elevation (np.ndarray): Array of elevation angles (phi) in radians. Shape (N,).

    Returns:
        np.ndarray: Array of (x, y, z) Cartesian coordinates. Shape (N, 3).
    """
    
    # Calculate Cartesian coordinates using element-wise operations
    x = radius * np.sin(elevation) * np.cos(azimuth)
    y = radius * np.sin(elevation) * np.sin(azimuth)
    z = radius * np.cos(elevation)
    
    # Stack the (N,) coordinate arrays as columns to create an (N, 3) matrix
    cartesian_coords = np.stack([x, y, z], axis=1)
    
    return cartesian_coords



def cartesian_to_spherical(cartesian_coords: np.ndarray) -> np.ndarray:
    """
    Converts an array of Cartesian coordinates (x, y, z) to Spherical.

    Args:
        cartesian_coords (np.ndarray): Array of (x, y, z) points. Shape: (N, 3).

    Returns:
        np.ndarray: Array of (radius, azimuth, elevation) points. Shape: (N, 3).
                    Azimuth and elevation are in radians.
    """
    x = cartesian_coords[:, 0]
    y = cartesian_coords[:, 1]
    z = cartesian_coords[:, 2]

    radius = np.linalg.norm(cartesian_coords, axis=1)

    # Avoid division by zero at the origin (r=0)
    radius_safe = np.where(radius == 0, 1e-12, radius)
    
    # Calculate elevation (polar angle, φ, from the Z-axis) [0, pi]
    elevation = np.arccos(np.clip(z / radius_safe, -1.0, 1.0))
    
    # Calculate azimuth (angle, θ, in the XY-plane from the X-axis) [-pi, pi]
    azimuth = np.arctan2(y, x)

    # Handle the origin case (0, 0, 0)
    elevation[radius == 0] = 0.0
    azimuth[radius == 0] = 0.0
    
    spherical_coords = np.stack([radius, azimuth, elevation], axis=1)
    
    return spherical_coords

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


