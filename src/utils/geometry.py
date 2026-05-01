import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def generate_source_and_interferences(N_interferences: int, radius: float, delta_ang_deg: float, array_center: np.ndarray) -> tuple:
    """
    Generates the 3D coordinates for the target source and N interferences.
    The target source is fixed at 0 degrees.
    Interferences are placed alternately at +/- multiples of delta_ang.
    
    Parameters:
    N_interferences: Number of interference sources to generate
    radius: Distance from the array center to the sources (meters)
    delta_ang_deg: Angular spacing between interferences (degrees)
    array_center: 1D array [x, y, z] with the array's geometric center
    
    Returns:
    source_pos: (3,) numpy array with the target source coordinates
    interferences_pos: (N_interferences, 3) numpy array with the interference coordinates
    """
    # Convert delta angle to radians for trigonometric functions
    delta_ang_rad = np.deg2rad(delta_ang_deg)
    
    # The target source is always at 0 degrees relative to the center.
    # Assuming 0 degrees aligns with the positive X-axis in the 2D plane:
    # x = x_c + r*cos(0), y = y_c + r*sin(0)
    source_pos = np.copy(array_center)
    source_pos[0] += radius 
    
    # Initialize the array to hold all interference coordinates
    interferences_pos = np.zeros((N_interferences, 3))
    
    for i in range(N_interferences):
        # Calculate the sequence multiplier: 1, 1, 2, 2, 3, 3...
        # Using integer division: (0//2)+1=1, (1//2)+1=1, (2//2)+1=2...
        multiplier = (i // 2) + 1
        
        # Alternate the sign: +, -, +, -, ...
        # Even indices (0, 2, 4...) get +, odd indices (1, 3, 5...) get -
        sign = 1 if i % 2 == 0 else -1
        
        # Calculate the current angle in radians
        angle_rad = sign * multiplier * delta_ang_rad
        
        # Calculate the cartesian coordinates for the current interference
        x = array_center[0] + radius * np.cos(angle_rad)
        y = array_center[1] + radius * np.sin(angle_rad)
        z = array_center[2] # Maintain the same height (Z) as the array center
        
        interferences_pos[i] = [x, y, z]
        
    return source_pos, interferences_pos

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
    inclination: np.ndarray
) -> np.ndarray:
    """
    Converts arrays of Spherical coordinates to Cartesian (x, y, z).

    This function assumes the ISO 80000-2 standard (physics convention):
    - radius (r): Distance from the origin.
    - azimuth (θ): Angle in the XY-plane from the X-axis (in radians).
    - inclination (φ): Angle from the positive Z-axis (in radians).

    Args:
        radius (np.ndarray): Array of radial distances. Shape (N,).
        azimuth (np.ndarray): Array of azimuth angles (theta) in radians. Shape (N,).
        inclination (np.ndarray): Array of inclination angles (phi) in radians. Shape (N,).

    Returns:
        np.ndarray: Array of (x, y, z) Cartesian coordinates. Shape (N, 3).
    """
    
    # Calculate Cartesian coordinates using element-wise operations
    x = radius * np.sin(inclination) * np.cos(azimuth)
    y = radius * np.sin(inclination) * np.sin(azimuth)
    z = radius * np.cos(inclination)
    
    # Stack the (N,) coordinate arrays as columns to create an (N, 3) matrix
    cartesian_coords = np.stack([x, y, z], axis=1)
    
    return cartesian_coords


import numpy as np

def cartesian_to_spherical(cartesian_coords: np.ndarray) -> np.ndarray:
    """
    Converts Cartesian coordinates (x, y, z) to Spherical (radius, azimuth, inclination).

    This function robustly handles both a single point (1D array of shape (3,))
    and a batch of points (2D array of shape (N, 3)).

    Args:
        cartesian_coords (np.ndarray): Array of (x, y, z) points.
                                     Shape can be (3,) or (N, 3).

    Returns:
        np.ndarray: Array of (radius, azimuth, inclination) points.
                    Shape will be (3,) or (N, 3), matching the input shape.
                    Azimuth and inclination are in radians.
    """
    # 1. Convertir a un array de NumPy (por si el usuario pasó una lista)
    coords = np.asarray(cartesian_coords)
    
    # 2. Detectar si la entrada es 1D (un solo punto)
    if coords.ndim == 1:
        if coords.shape[0] != 3:
            raise ValueError(f"Single point must have shape (3,), but got {coords.shape}")
        # Es 1D. Guardamos este hecho y la promovemos a 2D para el cálculo.
        was_1d = True
        coords_2d = coords.reshape(1, 3)
    elif coords.ndim == 2:
        if coords.shape[1] != 3:
            raise ValueError(f"Array of points must have shape (N, 3), but got {coords.shape}")
        # Es 2D, la usamos tal cual.
        was_1d = False
        coords_2d = coords
    else:
        raise ValueError(f"Input must be a 1D or 2D array, but got {coords.ndim} dimensions.")

    # --- 3. Cálculos (Este bloque es idéntico al anterior) ---
    # (Ahora 'coords_2d' está garantizado que es 2D)
    x = coords_2d[:, 0]
    y = coords_2d[:, 1]
    z = coords_2d[:, 2]

    radius = np.linalg.norm(coords_2d, axis=1)
    radius_safe = np.where(radius == 0, 1e-12, radius)
    
    inclination = np.arccos(np.clip(z / radius_safe, -1.0, 1.0))
    azimuth = np.arctan2(y, x)

    inclination[radius == 0] = 0.0
    azimuth[radius == 0] = 0.0
    
    spherical_coords = np.stack([radius, azimuth, inclination], axis=1)
    
    # --- 4. Devolver el formato original ---
    if was_1d:
        # Si la entrada era 1D, devolvemos un array 1D (shape (3,))
        return spherical_coords.squeeze()
    else:
        # Si la entrada era 2D, devolvemos el array 2D (shape (N, 3))
        return spherical_coords

def source_sphere_grid(radius, samples_azimut, samples_inclination):
    """
    Genera puntos de fuente en una Malla de Cuadrícula Esférica.

    Parameters
    ----------
    radius : float
        Radio de la esfera.
    samples_azimut : int
        Número de divisiones angulares para el ángulo Azimutal (phi).
    samples_inclination : int
        Número de divisiones angulares para el ángulo de Elevación (theta).

    Returns
    -------
    coords_grid (ndarray): Matriz de coordenadas (3, N*M).
    azimut_flat (ndarray): Vector 1D de todos los ángulos azimutales usados (N*M).
    inclination_flat (ndarray): Vector 1D de todos los ángulos de elevación usados (N*M).
    """

    # 1. Crear vectores 1D para cada eje angular (Delta angular constante)
    # Azimut (phi): 0 a 2*pi
    azimut_1d = np.linspace(0, 2 * np.pi, samples_azimut, endpoint=False) # No incluye 2*pi
    
    # Elevación (theta): 0 a pi (desde el eje Z)
    inclination_1d = np.linspace(0, np.pi, samples_inclination)

    # 2. Generar la Malla de Cuadrícula (Grid)
    # Genera dos matrices 2D (azimut_mesh, inclination_mesh)
    azimut_mesh, inclination_mesh = np.meshgrid(azimut_1d, inclination_1d)

    # 3. Aplanar las mallas a vectores 1D
    # Esto crea los dos vectores de longitud (N*M) con todas las combinaciones.
    azimut_flat = azimut_mesh.flatten()
    inclination_flat = inclination_mesh.flatten()

    # 4. Uso de Broadcasting
    # La función vectorizada (sferical_to_coord) usa estos vectores N*M para generar la cuadrícula.
    coords_grid = spherical_to_cartesian(radius, azimut_flat, inclination_flat)
    
    # coords_grid tendrá forma (3, N*M)
    return coords_grid, azimut_flat, inclination_flat




def spatial_grid(delta_radius, delta_azimut, delta_inclination ,center, points, mode = 'cart'):

    #defines a set of equi agled points for each dimention

    radius = np.linspace( -delta_radius/2 + center[0],
                                delta_radius/2 + center[0],
                                points
                                )
    
    azimut = np.linspace( -delta_azimut/ 2 + center[2],
                                delta_azimut/ 2 + center[2],
                                points
                                )
    
    inclination = np.linspace( -delta_inclination/ 2 + center[1],
                                delta_inclination/2 + center[1],
                                points
                                )

    R, Inc, Az = np.meshgrid(radius, azimut, inclination, indexing='ij')

    spatial_grid = np.stack([R.flatten(), Inc.flatten(), Az.flatten()], axis=1)

    if mode == "sphr":
        return spatial_grid
    
    elif mode=="cart":
        return spherical_to_cartesian( spatial_grid[:, 0],
                                       spatial_grid[:, 1],
                                       spatial_grid[:, 2])
        

