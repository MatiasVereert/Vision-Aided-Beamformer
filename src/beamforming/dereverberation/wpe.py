import numpy as np
from scipy import signal
# Asegurate de tener instalado: pip install nara_wpe
from nara_wpe.wpe import wpe

def apply_wpe(
    audio_multichannel: np.ndarray, 
    fs: int, 
    taps: int = 10, 
    delay: int = 3, 
    iterations: int = 3,
    n_fft: int = 512,
    hop_length: int = 128
) -> np.ndarray:
    """
    Aplica dereverberación WPE a una señal multicanal en el dominio del tiempo.
    """
    
    # 1. Validar dimensiones
    if audio_multichannel.ndim != 2:
        raise ValueError("El audio debe tener forma (Canales, Muestras)")
        
    M, N = audio_multichannel.shape
    
    # 2. STFT
    # Eliminamos 'axis=1'. Por defecto scipy toma el último eje como tiempo, 
    # lo cual es correcto para tu entrada (Canales, Muestras).
    f, t, Y = signal.stft(audio_multichannel, fs=fs, nperseg=n_fft, noverlap=n_fft-hop_length)
    
    # Y tiene forma (Canales, Frecuencias, Tiempo) -> (D, F, T)
    # nara_wpe espera: (Frecuencias, Canales, Tiempo) -> (F, D, T)
    Y_transposed = Y.transpose(1, 0, 2)
    
    # 3. Aplicar WPE
    Z_transposed = wpe(
        Y_transposed,
        taps=taps,
        delay=delay,
        iterations=iterations,
        statistics_mode='full'
    )
    
    # 4. Transponer de vuelta a (Canales, Frecuencias, Tiempo)
    Z = Z_transposed.transpose(1, 0, 2)
    
    # 5. iSTFT
    # Eliminamos 'axis=1'. Al entrar Z con forma (M, F, T), istft por defecto
    # entiende correctamente que los últimos dos ejes son Frecuencia y Tiempo.
    _, audio_dereverberated = signal.istft(Z, fs=fs, nperseg=n_fft, noverlap=n_fft-hop_length)
    
    # Recortar o rellenar para coincidir con la longitud original N
    if audio_dereverberated.shape[1] > N:
        audio_dereverberated = audio_dereverberated[:, :N]
    elif audio_dereverberated.shape[1] < N:
        pad_width = N - audio_dereverberated.shape[1]
        audio_dereverberated = np.pad(audio_dereverberated, ((0,0), (0, pad_width)))
        
    return audio_dereverberated


# ... (código anterior de la función apply_wpe) ...

if __name__ == "__main__":
    print("=== TEST UNITARIO: DEREVERBERATION WPE ===")
    
    # 1. Configuración de prueba
    fs_test = 16000
    duration = 1.0     # segundos
    M_test = 4         # micrófonos
    N_test = int(fs_test * duration)
    
    # 2. Generar señal sintética (Ruido + Eco simple)
    # Creamos ruido blanco
    clean_noise = np.random.randn(M_test, N_test) * 0.1
    # Simulamos un "eco" retardando la señal 500 muestras y atenuándola
    echo = np.roll(clean_noise, shift=500, axis=1) * 0.5
    # Señal de entrada "sucia"
    input_signal = clean_noise + echo
    
    print(f"[INPUT]  Forma de señal: {input_signal.shape} (M={M_test}, N={N_test})")
    
    # 3. Ejecutar la función
    try:
        print("[PROCESS] Ejecutando apply_wpe...")
        output_signal = apply_wpe(
            input_signal, 
            fs=fs_test, 
            taps=10, 
            delay=3, 
            iterations=2 # Pocas iteraciones para el test rápido
        )
        
        # 4. Verificaciones
        print(f"[OUTPUT] Forma de señal: {output_signal.shape}")
        
        # A. Chequeo de Dimensionalidad
        if input_signal.shape == output_signal.shape:
            print("  ✅ [PASS] La dimensionalidad se conserva correctamente.")
        else:
            print(f"  ❌ [FAIL] Discrepancia de dimensiones: Entró {input_signal.shape}, Salió {output_signal.shape}")
            
        # B. Chequeo de Estabilidad Numérica (NaNs o Infinitos)
        if np.isnan(output_signal).any() or np.isinf(output_signal).any():
            print("  ❌ [FAIL] La salida contiene NaNs o Infs (inestabilidad numérica).")
        else:
            print("  ✅ [PASS] La salida contiene valores numéricos válidos.")

        # C. Chequeo simple de energía (WPE debería reducir la energía al quitar reverb)
        energy_in = np.sum(input_signal**2)
        energy_out = np.sum(output_signal**2)
        print(f"  ℹ️ [INFO] Energía Entrada: {energy_in:.2f} -> Salida: {energy_out:.2f}")
        
    except Exception as e:
        print(f"  ❌ [CRITICAL ERROR] Excepción durante la ejecución:\n{e}")
        import traceback
        traceback.print_exc()