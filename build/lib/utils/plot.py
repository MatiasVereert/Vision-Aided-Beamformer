import numpy as np
import matplotlib.pyplot as plt

def plot_beamforming_results(input_signal, output_signal, ref_signal=None, fs=48000, title="Resultados GSC", zoom_ms=50):
    """
    Genera un dashboard de 3 filas con:
    1. Dominio del tiempo (Zoom centrado).
    2. Convergencia de Energía (Log).
    3. Comparativa de Espectrogramas (Input vs Output).
    
    Args:
        input_signal (np.array): Señal sucia (micrófono).
        output_signal (np.array): Señal procesada (salida beamformer).
        ref_signal (np.array, optional): Señal limpia de referencia (target).
        fs (int): Frecuencia de muestreo.
        title (str): Título del gráfico.
        zoom_ms (float): Ventana de tiempo en milisegundos para el zoom del primer plot.
    """
    
    plt.figure(figsize=(14, 10))
    plt.suptitle(title, fontsize=16)
    
    # Asegurar arrays 1D
    input_signal = input_signal.flatten()
    output_signal = output_signal.flatten()
    
    # --- 1. DOMINIO DEL TIEMPO (ZOOM) ---
    plt.subplot(3, 1, 1)
    plt.title(f"Dominio del Tiempo (Zoom {zoom_ms}ms)")
    
    # Calcular índices para el zoom en el centro del audio
    center_idx = len(output_signal) // 2
    half_window = int((zoom_ms / 1000) * fs / 2)
    start = max(0, center_idx - half_window)
    end = min(len(output_signal), center_idx + half_window)
    
    # Eje de tiempo en ms relativo al zoom
    t_zoom = np.linspace(0, (end-start)/fs*1000, end-start)
    
    plt.plot(t_zoom, input_signal[start:end], label="Mic (Input)", color='silver', alpha=0.7)
    plt.plot(t_zoom, output_signal[start:end], label="Beamformer (Output)", color='green', linewidth=1.5)
    
    if ref_signal is not None:
        ref_signal = ref_signal.flatten()
        # Ajustar longitud si difiere
        limit_ref = min(len(ref_signal), end)
        if limit_ref > start:
            plt.plot(t_zoom[:limit_ref-start], ref_signal[start:limit_ref]*0.8, 
                    label="Clean Reference", color='black', linestyle='--', alpha=0.6)
            
    plt.legend(loc='upper right')
    plt.ylabel("Amplitud")
    plt.xlabel("Tiempo (ms)")
    plt.grid(True, alpha=0.3)

    # --- 2. ENERGÍA (CONVERGENCIA) ---
    plt.subplot(3, 1, 2)
    plt.title("Evolución de Energía (Media Móvil)")
    
    # Ventana de suavizado (aprox 20ms)
    window_size = int(0.02 * fs) 
    kernel = np.ones(window_size) / window_size
    
    # Convolución segura
    energy_in = np.convolve(input_signal**2, kernel, mode='same')
    energy_out = np.convolve(output_signal**2, kernel, mode='same')
    
    # Eje de tiempo completo en segundos
    t_full = np.arange(len(energy_in)) / fs
    
    plt.semilogy(t_full, energy_in, label="Input Energy", color='gray', alpha=0.5)
    plt.semilogy(t_full, energy_out, label="Output Energy", color='green')
    plt.ylabel("Energía (dB Scale)")
    plt.xlabel("Tiempo (s)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)

    # --- 3. ESPECTROGRAMAS ---
    # Input
    plt.subplot(3, 2, 5)
    plt.title("Espectrograma: Input (Mic)")
    plt.specgram(input_signal, Fs=fs, NFFT=1024, noverlap=512, cmap='inferno')
    plt.ylabel("Freq (Hz)")
    plt.xlabel("Tiempo (s)")

    # Output
    plt.subplot(3, 2, 6)
    plt.title("Espectrograma: Output (Beamformer)")
    plt.specgram(output_signal, Fs=fs, NFFT=1024, noverlap=512, cmap='inferno')
    plt.ylabel("Freq (Hz)")
    plt.xlabel("Tiempo (s)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()