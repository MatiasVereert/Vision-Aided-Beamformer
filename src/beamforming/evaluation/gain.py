import numpy as np
from scipy.constants import speed_of_sound

from utils.geometry import source_rotation
from beamforming.processors import beamforming, snapshots
from beamforming.models import near_field_steering_vector, near_field_steering_vector_multi
from propagation.free_field import space_delay




def polar_gain(f, fs, mic_array, weights, focal_point, points=360):
    """Calculates the 2D polar pattern of a time-domain beamformer.

    This function evaluates the directivity of a given beamformer by simulating
    a sinusoidal signal arriving from multiple angles in a single plane.
    It processes the simulated signals through the beamformer pipeline 
    (snapshots and filtering) and computes the output power for each angle.

    The number of microphones (M) and FIR taps (K) are automatically inferred
    from the shapes of the input arrays (`mic_array` and `weights`). The final
    output is normalized relative to the maximum output power, resulting in a
    beampattern with a peak of 0 dB.

    Args:
        f (float): The test frequency of the source signal in Hz.
        fs (float): The sampling frequency in Hz.
        mic_array (np.ndarray): A matrix of microphone coordinates, with shape (M, 3).
        weights (np.ndarray): The beamformer's weight vector, with shape (M * K, 1).
        focal_point (np.ndarray): The Cartesian coordinates (x, y, z) of the beamformer's focal
                                  point. The norm of this vector is used as the radius.
        points (int, optional): The number of angular points to compute. Defaults to 360.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - gain_db (np.ndarray): A 1D array of the beampattern gain in dB.
            - angles (np.ndarray): A 1D array of the corresponding angles in degrees.
    """
    # 1. Validar entradas y derivar M y K (tu código original está bien aquí)
    if mic_array.ndim != 2 or mic_array.shape[1] != 3:
        raise ValueError("mic_array must be a 2D array of shape (M, 3)")
    M = mic_array.shape[0]

    if weights.ndim != 2 or weights.shape[1] != 1:
        raise ValueError("weights must be a 2D vector with shape (M*K, 1)")
    M_K = weights.shape[0]
    
    if M == 0: raise ValueError("mic_array cannot be empty.")
    if M_K % M != 0:
        raise ValueError(f"The length of the weights vector ({M_K}) is not divisible by the number of microphones ({M}).")
    K = int(M_K / M)

    # 2. Definir la señal de prueba
    duration_s = 0.1
    t = np.arange(int(duration_s * fs)) / fs
    signal_source = np.sin(2 * np.pi * f * t)

    # 3. Definir los puntos de evaluación del patrón polar
    distance = np.linalg.norm(focal_point)
    # Asumo que tienes una función source_rotation, si no, hay que definirla
    # source_points debería tener forma (points, 3)
    angles_rad = np.linspace(0, 2 * np.pi, points, endpoint=False)
    source_points = np.zeros((points, 3))
    source_points[:, 0] = distance * np.cos(angles_rad)
    source_points[:, 1] = distance * np.sin(angles_rad)
    # Asumimos un plano z=0 para el patrón 2D
    angles_deg = np.rad2deg(angles_rad)

    # 4. Simular la propagación para TODOS los puntos a la vez.
    # space_delay debe devolver un array 3D de forma (points, M, N_samples)
    signals_array, _, _ = space_delay(signal_source, fs, source_points, mic_array)

    # Verificar que la salida es 3D. Si P=1, space_delay devuelve 2D, hay que corregirlo.
    if signals_array.ndim == 2 and points == 1:
        # Añadir la dimensión que fue eliminada por .squeeze()
        signals_array = signals_array[np.newaxis, :, :]

    if signals_array.shape[0] != points:
        raise ValueError("La salida de space_delay no tiene la dimensión de puntos esperada.")

    # 5. Bucle para procesar cada punto y calcular su potencia de salida
    output_power = np.zeros(points)
    for i in range(points):
        # Extraemos las señales de los M micrófonos para el i-ésimo ángulo
        mic_signals_at_angle_i = signals_array[i, :, :]  # Esto ahora es seguro
        snapshots_matrix = snapshots(mic_signals_at_angle_i, K)
        beamformer_output = beamforming(snapshots_matrix, weights)
        output_power[i] = np.mean(np.real(beamformer_output)**2)

    # 6. Normalización para el patrón polar (el pico más alto es 0 dB)
    max_power = np.max(output_power)
    if max_power > 1e-12: # Usar umbral para evitar division por cero
        gain_db = 10 * np.log10(output_power / max_power + 1e-12)
    else:
        gain_db = np.full_like(output_power, -120.0) # Potencia muy baja si todo es cero

    return gain_db, angles_deg

def synthetic_gain(f, fs, mic_array, weights, source_points):
    """
    Calculates the beamformer's time-domain gain at a set of specified spatial points.

    This function simulates a source signal from each point in `source_points`,
    processes the resulting array signals through the beamformer pipeline, and
    computes the output power. The gain is returned relative to the input
    signal's power.

    Args:
        f (float): The test frequency in Hz.
        fs (float): The sampling frequency in Hz.
        mic_array (np.ndarray): Microphone array coordinates of shape (M, 3).
        weights (np.ndarray): Beamformer weight vector of shape (M*K, 1).
        source_points (np.ndarray): A matrix of Cartesian coordinates (x, y, z)
                                     for each point to be tested. Shape: (P, 3).

    Returns:
        np.ndarray: A 1D array of gains in decibels (dB) for each of the P points.
    """
    # 1. Derivar M y K a partir de las formas de los arrays (como hiciste antes)
    M = mic_array.shape[0]
    M_K = weights.shape[0]
    # ... (añadir las comprobaciones de error)
    K = int(M_K / M)
    
    # 2. Definir la señal de prueba y su potencia de referencia
    duration_s = 0.1
    t = np.arange(int(duration_s * fs)) / fs
    signal_source = np.sin(2 * np.pi * f * t)
    source_power_ref = np.mean(signal_source**2)

    # 3. Simular la propagación para TODOS los puntos a la vez
    # Tu función space_delay ya maneja esto de forma eficiente
    signals_array, _, _ = space_delay(signal_source, fs, source_points, mic_array)
    
    num_points = source_points.shape[0]
    output_powers = np.zeros(num_points)

    # 4. Bucle para procesar cada punto y calcular su potencia de salida
    for i in range(num_points):
        mic_signals = signals_array[i, :, :]
        U_snapshots = snapshots(mic_signals, K)
        y_output = beamforming(U_snapshots, weights)
        output_powers[i] = np.mean(np.real(y_output)**2)
        
    # 5. Calcular la ganancia en dB relativa a la potencia de ENTRADA
    # Esto devuelve la "ganancia de array", que es más general.
    # La normalización para el patrón polar se hará en la función de ploteo.
    gain_db = 10 * np.log10(output_powers / source_power_ref + 1e-12)

    return gain_db

def analytical_gain(frecs, fs, mic_array, weights, source_points):
    # 1. Derivar M y K a partir de las formas de los arrays (como hiciste antes)
    M = mic_array.shape[0]
    M_K = weights.shape[0]
    if weights.size % mic_array.shape[0] != 0:
        raise ValueError("The length of weights must be a multiple of the number of mics.")
    K = int(M_K / M)

    #Caclulate the steering vector for each pointa and frecuency (F, P, KxM)
    steering_vectors = near_field_steering_vector_multi(frecs, source_points, fs, mic_array, K )

    #hermitian traspoese fo the weights (as dim: 1D no need to transpose)
    w_H = np.conj(weights).flatten()

    #Calculate response using ecuation:
    # b(f,x) = w^T a(f,x) 
    b = np.einsum('k,fpk->fp', w_H, steering_vectors)

    epsilon = 1e-20
    gain_dB = 20 * np.log10(np.abs(b)+ epsilon)

    return gain_dB 