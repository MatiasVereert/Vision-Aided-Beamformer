import numpy as np
from scipy.constants import speed_of_sound
from beamforming.signal_model import near_field_steering_vector
from numpy.lib.stride_tricks import sliding_window_view

def snapshots(array_signals, K):
    """
    Creates the snapshot matrix U for a multi-channel time-domain beamformer.

    This function efficiently transforms a 2D array of microphone signals 
    (M mics, N samples) into the 2D snapshot matrix required by a 
    tapped-delay-line (FIR) beamformer. Each column of the output matrix 
    represents a concatenated snapshot vector u(k).

    Args:
        array_signals (np.ndarray): 
            Input signal matrix of shape (M, N_samples). M is the number 
            of microphones, N_samples is the number of time samples.
        K (int): 
            The number of FIR taps per microphone (the window size).

    Returns:
        np.ndarray: 
            The snapshot matrix U of shape (M * K, N_snapshots), where 
            N_snapshots = N_samples - K + 1.
    """
    M, _ = array_signals.shape

    # Create a sliding window view of size K along the time axis (axis=1).
    # This results in a 3D tensor of shape (M, N_snapshots, K).
    signals_window = sliding_window_view(array_signals, window_shape=K, axis=1)

    # Reverse the tap axis to ensure a causal FIR filter structure [x(n), x(n-1), ...].
    # The steering vector is defined for this order.
    reversed_taps = signals_window[:, :, ::-1]

    # Reorder axes to group taps by microphone, then reshape to the final
    # (M*K, N_snapshots) matrix by concatenating all tap blocks.
    snapshot_matrix = reversed_taps.transpose(0, 2, 1).reshape(M * K, -1)

    return snapshot_matrix

import numpy as np

def beamforming(signals, weights):
    """
    Applies a time-domain beamformer to a snapshot matrix of signals.

    This function performs the core beamforming operation by taking the dot product
    of the Hermitian transpose of the weight vector with the snapshot matrix. This
    is equivalent to applying a multi-channel FIR filter and summing the result.

    Args:
        signals (np.ndarray): 
            The snapshot matrix U, where each column is a concatenated snapshot
            vector u(k). Expected shape: (N, T), where N = M * K is the total 
            number of weights, and T is the number of time snapshots.
        weights (np.ndarray): 
            The beamformer weight vector w. Expected shape: (N, 1).

    Returns:
        np.ndarray: 
            The single-channel output signal y. Shape: (1, T).
    """
    # Calculate the beamformer output via matrix multiplication (y = w^H * U).
    # The Hermitian transpose (.conj().T) is used for potentially complex weights,
    # but works correctly for real-valued weights as well (where it's just .T).
    y = weights.conj().T @ signals
    
    return y


def gsc_adaptive_beamformer(
    input_signal: np.ndarray, 
    w_q: np.ndarray, 
    Ca: np.ndarray, 
    K: int, 
    mu: float = 0.1,
    epsilon: float = 1e-6
):
    """
    Implementación funcional del Generalized Sidelobe Canceller (GSC) con NLMS.
    
    Args:
        input_signal: Señal multicanal de entrada (M micrófonos x T muestras).
        w_q: Pesos fijos del beamformer (vector de N = M*K).
        Ca: Matriz de bloqueo (N x N_adapt). Ortogonal a C.
        K: Número de Taps (retardos temporales).
        mu: Paso de adaptación (Step size) para NLMS.
        epsilon: Factor de regularización para evitar división por cero.
        
    Returns:
        y_out: Señal de salida del beamformer (1D array de longitud T).
        w_log: Historial de los pesos adaptativos (para análisis).
    """
    
    M, T = input_signal.shape
    N = w_q.shape[0] # N = M * K
    
    # Verificación de dimensiones
    if N != M * K:
        raise ValueError(f"Error de dimensiones: w_q tiene {N}, pero debería ser M*K={M*K}")

    # Inicialización
    y_out = np.zeros(T, dtype=complex)
    
    # El filtro adaptativo wa tiene el tamaño del espacio nulo (columnas de Ca)
    n_adapt = Ca.shape[1]
    wa = np.zeros(n_adapt, dtype=complex) 
    
    # Buffer para la línea de retardo (TDL) de cada micrófono
    # Almacena las últimas K muestras de los M micrófonos
    # Lo aplanaremos para formar el vector u(k) de tamaño N
    buffer = np.zeros((M, K), dtype=complex)
    
    # Historial de pesos (opcional, para ver convergencia)
    # Guardamos un snapshot cada 100 muestras para no llenar la RAM
    w_log = [] 

    print(f"Procesando {T} muestras con GSC-NLMS (mu={mu})...")

    # --- Bucle Temporal (Muestra a Muestra) ---
    for k in range(T):
        
        # 1. Actualizar Buffer (TDL)
        # Desplazamos el buffer y metemos la nueva muestra al final
        buffer[:, 1:] = buffer[:, :-1]     # Shift derecha
        buffer[:, 0]  = input_signal[:, k] # Nueva muestra en posición 0
        
        # 2. Construir vector de entrada u(k)
        # Aplanamos el buffer para que coincida con la estructura de los pesos
        # (Mic1_t0, Mic1_t1..., Mic2_t0...)
        u_k = buffer.reshape(-1) # Vector de (N,)
        
        # -----------------------------------------------------------
        # ESTRUCTURA GSC
        # -----------------------------------------------------------
        
        # 3. Rama Superior (Fixed Beamformer)
        # y_q = w_q^H * u(k)
        y_q = np.dot(np.conj(w_q), u_k)
        
        # 4. Rama Inferior (Blocking Matrix)
        # x_a = Ca^H * u(k) -> Señal de referencia de ruido/interferencia
        x_a = np.dot(np.conj(Ca.T), u_k)
        
        # 5. Filtro Adaptativo
        # y_a = wa^H * x_a -> Estimación del ruido en la salida fija
        y_a = np.dot(np.conj(wa), x_a)
        
        # 6. Salida del Sistema (y cálculo del error)
        # e(k) = y_q - y_a
        e = y_q - y_a
        y_out[k] = e # La salida final ES el error de la cancelación
        
        # 7. Actualización NLMS de los pesos adaptativos
        # wa_new = wa + mu * (x_a * e*) / (|x_a|^2 + eps)
        energy = np.real(np.dot(np.conj(x_a), x_a))
        step = (mu / (energy + epsilon)) * x_a * np.conj(e)
        wa = wa + step
        
        # Logueo (opcional)
        if k % 1000 == 0:
            w_log.append(np.abs(wa).mean()) # Guardamos la magnitud media

    return y_out, w_log


class AdaptiveGSC:

    """
    Procesador Generalized Sideslobe Canceler
    """

    def __init__(self, w_q, ):
        pass
    
