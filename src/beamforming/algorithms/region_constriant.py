import numpy as np 
from scipy.constants import speed_of_sound
from beamforming.signal_model import near_field_steering_vector_multi 
from utils.geometry import cartesian_to_spherical, spherical_to_cartesian
from typing import Tuple


def set_domains(Rs, delta_r, delta_azimut, delta_elevation, fmin, fmax, P, J):

    #Frecuencya array 
    f_array = np.linspace(fmin, fmax, J)
    
    
    Rs_sferical = cartesian_to_spherical(Rs)

    #Generate P random points inside the domain
    radius = np.random.uniform(Rs_sferical[0] - delta_r/2,  Rs_sferical[0] + delta_r/2, P )
    azimut = np.random.uniform(Rs_sferical[1] - delta_azimut/2, Rs_sferical[1]+ delta_azimut/2, P)
    elevation = np.random.uniform( Rs_sferical[2] - delta_elevation/2, Rs_sferical[2]+ delta_elevation/2, P)

    #generate a single vector with all the points in cartesian coordinates 
    points = spherical_to_cartesian(radius, azimut, elevation)

    return f_array, points

def build_A_and_g(
    freqs: np.ndarray, 
    source_points: np.ndarray, 
    fs: int, 
    mic_array: np.ndarray, 
    K: int, 
    c: float = speed_of_sound
) -> Tuple[np.ndarray, np.ndarray]: 
    """
    Calculates the real-valued constraint matrix A (Eq. 13) and
    the real-valued desired response vector g (Eq. 15).

    Both are sampled at J frequencies and I spatial points.

    Args:
        freqs (np.ndarray): Array of J frequency samples (Hz). Shape: (J,).
        source_points (np.ndarray): Array of I spatial sample points (m). Shape: (I, 3).
        fs (int): Sampling frequency (Hz), used to determine tap delays.
        mic_array (np.ndarray): Array of M microphone coordinates (m). Shape: (M, 3).
        K (int): Number of taps per microphone (defines N = M * K).
        c (float, optional): Speed of sound (m/s).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            A (np.ndarray): The real-valued constraint matrix A.
                            Shape: (N, 2*P) where N=M*K, P=J*I.
            g (np.ndarray): The real-valued desired response vector g.
                            Shape: (2*P, 1).
    """
    
    num_freqs = len(freqs)      # J
    num_points = len(source_points) # I
    
    if num_freqs == 0 or num_points == 0:
        raise ValueError("Frequency and source point arrays cannot be empty.")
        
    P_total = num_freqs * num_points # P = J * I

    # --- 1. Construcción de la Matriz A (Tu código original) ---
    
    a_tensor_complex = near_field_steering_vector_multi(
        f=freqs, 
        Rs=source_points, 
        fs=fs,
        mic_array=mic_array,
        K=K,
        c=c
    )
    
    N = a_tensor_complex.shape[2] 
    a_transpose_complex = np.transpose(a_tensor_complex, axes=(2, 0, 1))
    a_complex_flat = a_transpose_complex.reshape(N, -1)
    A = np.hstack([np.real(a_complex_flat), np.imag(a_complex_flat)])

    # --- 2. Construcción del Vector g (NUEVO) ---
    #    Basado en la Ecuación 15: g = [g_real, g_imag]^T
    
    # [cite_start]a. Establecer la ganancia de amplitud g_ij = 1 (para ganancia unitaria) [cite: 219]
    gain_amplitude = 1.0

    # b. [cite_start]Establecer el retardo de grupo 'tau' al centro temporal del filtro [cite: 219]
    #    tau = (K-1)/2 * T_s = (K-1)/(2*fs)
    tau_center = ((K - 1) / 2) / fs
    
    # c. Crear un array de frecuencias (shape P,) que coincida con el orden
    #    de las columnas de A. 'reshape' en NumPy (orden C) hace que el
    #    índice más a la derecha ('I') cambie más rápido.
    #    Orden: [f_0p_0, f_0p_1, ..., f_0p_I, f_1p_0, ...]
    #    Necesitamos repetir cada frecuencia 'num_points' (I) veces.
    freqs_P = np.repeat(freqs, num_points) # Shape (P_total,)

    # d. Calcular los componentes de fase (Eq. 15)
    phi = 2 * np.pi * freqs_P * tau_center
    
    g_real = gain_amplitude * np.cos(phi) # Shape (P_total,)
    g_imag = gain_amplitude * np.sin(phi) # Shape (P_total,)

    # e. Concatenar para formar g (shape (2*P_total,)) y darle forma de vector columna
    g = np.hstack([g_real, g_imag]).reshape(-1, 1) # Shape (2*P, 1)

    # --- 3. Devolver ambos ---
    return A, g


def compute_svd_and_rank(
    A: np.ndarray, 
    energy_threshold: float = 0.999
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Performs SVD on the real matrix A and calculates the rank L and error.

    This function implements the core of the robust beamformer design:
    1. Performs SVD on the (N, 2P) real matrix A.
    2. Calculates the eigenvalues (lambda_n) of the underlying Rss.
    3. Determines the rank L (number of components to keep) based on a
       cumulative energy threshold.
    4. Calculates the approximation error epsilon(L) (sum of discarded energy).

    Args:
        A (np.ndarray): The (N, 2P) real constraint matrix from build_A_matrix.
        energy_threshold (float, optional): The fraction of total energy to
                                            preserve (e.g., 0.999 for 99.9%).

    Returns:
        tuple:
            L (int): The calculated rank (number of components).
            U (np.ndarray): The left singular vectors (V in the paper). Shape (N, N).
            s (np.ndarray): The singular values (sigma_n). Shape (N,).
            Vh (np.ndarray): The right singular vectors (U^T in the paper). Shape (N, 2P).
            epsilon_L (float): The approximation error (sum of discarded eigenvalues).
    """
    
    # 1. Infer P (I*J) from the shape of A (N, 2P)
    N, two_P = A.shape
    if two_P % 2 != 0:
        raise ValueError("Matrix A has an odd number of columns. Should be (N, 2*P).")
    P = two_P // 2

    # 2. Perform SVD on the real matrix A
    # We set full_matrices=False for efficiency.
    # A is (N, 2P). Assuming N < 2P:
    # U will be (N, N), s will be (N,), Vh will be (N, 2P)
    # This U corresponds to V in the paper (Eq. 16).
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    
    # 3. Calculate the eigenvalues (lambda_n) of the related Rss
    # The paper shows the error epsilon(L) is based on the sum of 
    # eigenvalues of Rss. These are related to the singular values (s)
    # of A by: lambda_n = (sigma_n^2) / P.
    
    # We square the singular values 's' (sigma_n)
    lambdas = (s**2) / P
    
    # 4. Calculate total energy and find L
    total_energy = np.sum(lambdas)
    
    if total_energy < 1e-12:
        # Avoid division by zero if the signal is all zeros
        return 0, U, s, Vh, 0.0

    # Calculate cumulative energy (already sorted by svd)
    cumulative_energy = np.cumsum(lambdas)
    
    # Find the first index 'L' where cumulative energy exceeds the threshold
    # np.searchsorted finds where to insert to maintain order
    # We add 1 because index 0 represents L=1 component.
    L = np.searchsorted(cumulative_energy / total_energy, energy_threshold) + 1
    
    # Ensure L does not exceed the total number of components
    L = min(L, N)
    
    # 5. Calculate the approximation error epsilon(L)
    # This is the sum of all energy (eigenvalues) we are discarding
    epsilon_L = np.sum(lambdas[L:])
    
    return L, U, s, Vh, epsilon_L

def build_region_constraints(
    Rs: np.ndarray, 
    delta_r: float, 
    delta_azimut: float, 
    delta_elevation: float, 
    mic_array: np.ndarray, 
    fs: int, 
    K: int,
    f_min: int = 200,
    f_max: int = 10000,
    num_points: int = 100, # I
    num_freqs: int = 100,  # J
    c: float = speed_of_sound
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Orchestrates the calculation of the robust constraints C (Eq. 18)
    and h (Eq. 18) based on the SVD method.
    """
    
    # 1. Define sample points for the integration
    freqs, points = set_domains(Rs, delta_r, delta_azimut, delta_elevation, f_min, f_max, num_points, num_freqs)

    # 2. Build the A matrix and g vector based on these points
    A, g = build_A_and_g(freqs, points, fs, mic_array, K, c)

    # 3. Perform SVD and find the rank L
    # CORREGIDO: La llamada a la función no necesita P_total
    L, U, s, Vh, epsilon = compute_svd_and_rank(A, energy_threshold=0.999)

    # --- 4. Build C and h (Equation 18) ---
    
    # a. Build C = V_L
    # V_L (paper) = U (numpy) truncated to L columns [cite: 233]
    C = U[:, :L] # Shape (N, L)

    # b. Build h = (Sigma_L)^-1 * (U_L)^T * g [cite: 234]
    
    # (Sigma_L)^-1
    # CORREGIDO: Se usa la variable 's' (minúscula) y se trunca a 'L'
    s_L_inv = 1.0 / (s[:L] + 1e-12)
    Sigma_L_inv = np.diag(s_L_inv) # Shape (L, L)
    
    # (U_L)^T
    # CORREGIDO: U^T (paper) = Vh (numpy). U_L^T is the first L rows of Vh. [cite: 230]
    U_L_T = Vh[:L, :] # Shape (L, 2P)

    # Calculate h
    # (L, L) @ (L, 2P) @ (2P, 1) -> (L, 1)
    h = Sigma_L_inv @ U_L_T @ g

    return C, h   
