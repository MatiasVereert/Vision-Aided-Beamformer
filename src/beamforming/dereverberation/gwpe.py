import numpy as np 
from scipy import signal
from numpy.lib.stride_tricks import sliding_window_view
from matplotlib import pyplot as plt 


def create_tapped_delay_line(X, K, Delta_frames, axis=2):
    """
        Generates a tapped delay line using a sliding window view with causal padding.

        Applies zero-padding to the left of the temporal axis equivalent to
        (Delta_frames + K - 1) to preserve system causality. Constructs the observation
        matrix by temporally aligning the delays and truncates the excess of
        generated windows to maintain the original block length (T).

        Parameters
        ----------
        X : numpy.ndarray
            Multidimensional input tensor representing the observation signal.
        K : int
            Prediction filter order (number of temporal taps).
        Delta_frames : int
            Initial prediction delay (\Delta_frames).
        axis : int, optional
            Index of the axis corresponding to the temporal dimension in tensor X.
            Defaults to 2.

        Returns
        -------
        numpy.ndarray
            Tensor containing the memory view of the shifted history. If X has
            dimensions (F, M, T) and axis=2, the returned tensor will have
            dimensions (F, M, K, T). The K axis is ordered such that index 0
            contains the sample with delay Delta_frames, and index K-1 contains the
            oldest sample (Delta_frames + K - 1).
    """
    T = X.shape[axis]
    dim = X.ndim
    pad_with = []

    for i in range(dim):
        if i == axis:
            pad_with.append((Delta_frames + K - 1, 0))
        else: 
            pad_with.append((0, 0))

    # Zero pads to obtain consistent lenght
    Y = np.pad(X, pad_with, 'constant', constant_values=0)

    # Delay K samples the input signal, add zeros to the begining 
    Y_delays = sliding_window_view(Y, K, axis=axis) 

    # Transpose
    Y_delays_T = np.swapaxes(Y_delays, axis, -1)

    # Mirror axes to order data: Y1 = Y[t-tao], Y2 = Y[t-tao-1] ...
    Y_window_view = np.flip(Y_delays_T, axis=axis)

    # Trim to original lenght
    Y_window_view = Y_window_view[..., :T]
    
    return Y_window_view

def batch_dereverb(Y, fs, K):
    
    # Constants
    n_window = 1024
    n_overlap = 768
    d_loading = 1e-2
    iterations = 4
    Delta_frames = 5  # OBLIGATORIO: Retardo de 1280 muestras (> ventana de 1024)
    K = 12

    # Tranform to frecuency domain
    f, t, Y = signal.stft(Y, 
                          fs=fs, 
                          nperseg=n_window, 
                          noverlap=n_overlap, 
                          window='hann', 
                          axis=1)
    
    Y = Y.transpose(1, 0, 2)
    F = Y.shape[0]
    M = Y.shape[1]

    # 1) -- Inicialice filter G = 0
    g_hat = np.zeros((F, M, K, M), dtype=complex)

    # Obtaing K tapped and delayed signals
    Y_windowed = create_tapped_delay_line(Y, K=K, Delta_frames=Delta_frames, axis=2)

    # Start Iterations
    for i in range(iterations):
        print(f"Comenzando iteracion {i}")

        # 2) -- Compute De-Reverberation Y(t) = yl(t) - sum( G* yl ) --
        Y_tilde = np.einsum('fmkn, fmkt -> fnt', g_hat, Y_windowed)

        # Obtain dereverbated output
        X_hat = Y - Y_tilde

        # 3) -- Obtain Spatial Correlation Matrix Lamda_hat
        power = np.mean(np.abs(X_hat)**2, axis=1)
        
        # Calculamos un piso de ruido dinámico: -40 dB (1e-4) por debajo del pico de energía de esa frecuencia
        power_floor = np.amax(power, axis=1, keepdims=True) * 1e-4
        
        # Enmascaramos las tramas de silencio forzando el piso
        lamda = np.maximum(power, power_floor)
        lamda_inv = 1 / lamda

        # 4) -- Obtain R and r tensors
        # R_tensor: (F, M, K, M, K) -> Contraemos el tiempo con lamda_inv
        R_tensor = np.einsum('fmkt, ft, fplt -> fmkpl', Y_windowed.conj(), lamda_inv, Y_windowed, optimize=True)

        # r_tensor: (F, M, K, M_out) -> Correlación cruzada con la señal objetivo
        r_tensor = np.einsum('fmkt, ft, fnt -> fmkn', Y_windowed.conj(), lamda_inv, Y, optimize=True)

        # R_matrix: (F, M*K, M*K)
        R_matrix = R_tensor.reshape(F, M*K, M*K)
        
        # 1. Calcular la traza de R por cada frecuencia (F,)
        trace_R = np.trace(R_matrix, axis1=1, axis2=2)
        
        # 2. Generar una matriz de carga proporcional a la energía promedio de la diagonal
        # El escalar d_loading (ej. 1e-6) actúa ahora como un ratio relativo, no absoluto.
        relative_loading = (d_loading * trace_R / (M * K))[:, np.newaxis, np.newaxis]
        
        # 3. Sumar la regularización dinámica
        R_matrix += relative_loading * np.eye(M*K)
        r_matrix = r_tensor.reshape(F, M*K, M)

        # 5) -- Obtain Optimized weights
        # Solve Linear System 
        g_matrix = np.linalg.solve(R_matrix, r_matrix)

        # Restauración topológica coherente con el aplanamiento previo
        g_hat = g_matrix.reshape(F, M, K, M) 
        
    # Compute last output
    Y_tilde = np.einsum('fmkn, fmkt -> fnt', g_hat, Y_windowed, optimize=True)
    X_hat = Y - Y_tilde
    X_hat = X_hat.transpose(1, 0, 2)

    # Inverse Transform
    _, x_out = signal.istft(X_hat, fs=fs, window='hann', nperseg=n_window, noverlap=n_overlap)

    return x_out #shape (N, T)


if __name__ == "__main__":
    print("--- Iniciando Test del Pipeline GWPE ---")
    
    # 1. Configuración de parámetros de prueba
    fs = 16000
    duration_sec = 1.0
    n_samples = int(fs * duration_sec)
    n_mics = 4
    K_test = 10 # Orden del filtro
    
    # 2. Generar señal de prueba (Ruido Blanco multicanal)
    # Shape esperado por batch_dereverb: (Mics, Samples)
    rng = np.random.default_rng(seed=42)
    source_signal = rng.standard_normal(n_samples)
    
    # Simulamos que llega a los 4 micrófonos con un pequeño delay y atenuación
    # (Esto simula una reverberación extremadamente simple para testear dimensiones)
    room_input_mic = np.zeros((n_mics, n_samples))
    for m in range(n_mics):
        delay = m * 5  # cada mic recibe la señal un poco después
        room_input_mic[m, delay:] = source_signal[:n_samples-delay] * (1.0 - m*0.1)

    print(f"Forma de la señal de entrada: {room_input_mic.shape}") # (4, 16000)

    # 3. Ejecutar el pipeline
    try:
        print("Ejecutando batch_dereverb...")
        # Llamamos a tu función con los parámetros
        dereverb_signal = batch_dereverb(room_input_mic, fs, K=K_test)
        
        print("\n¡Éxito! El pipeline terminó sin errores.")
        print(f"Forma de la señal de salida: {dereverb_signal.shape}")
        
        # 4. Verificación rápida de valores
        if np.isnan(dereverb_signal).any():
            print("AVISO: La salida contiene valores NaN. Revisa la estabilidad de la matriz R.")
        else:
            print("La salida no contiene NaNs. El sistema lineal se resolvió correctamente.")

    except Exception as e:
        print(f"\nERROR en el pipeline: {e}")
        import traceback
        traceback.print_exc()

    # 5. Visualización básica
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(room_input_mic[0, :500], label="Entrada (Mic 0)")
    plt.title("Señal de Entrada Original")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(dereverb_signal[0, :500], label="Salida (Procesada)", color='orange')
    plt.title("Señal de Salida (WPE)")
    plt.legend()
    
    plt.tight_layout()
    print("Mostrando gráfico de comparación (primeras 500 muestras)...")
    plt.show()
