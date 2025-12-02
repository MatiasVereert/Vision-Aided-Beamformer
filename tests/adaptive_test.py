import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

# --- Imports del Sistema ---
# Ajusta los imports según tu estructura de carpetas real
from beamforming.array.mic_array import ULA
from beamforming.system import AdaptiveBeamformer
from propagation.free_field import space_delay
from beamforming.gsc.region_constriant import build_region_constraints
from beamforming.gsc.weights import compute_fixed_weights_optimized

def generate_synthetic_voice(fs, duration, f0=120.0, f_max=4000.0):
    """
    Genera una señal armónica que imita la estructura de la voz (Buzz).
    f0: Frecuencia fundamental (aprox 120Hz para voz masculina grave).
    f_max: Ancho de banda máximo.
    """
    t = np.arange(int(duration * fs)) / fs
    signal = np.zeros_like(t)
    
    # Suma de armónicos para cubrir el espectro de voz (Sawtooth-like)
    current_f = f0
    while current_f < f_max:
        # Spectral tilt: La amplitud cae con la frecuencia (1/f)
        amp = 1.0 / (1 + (current_f/500)**1.5) 
        signal += amp * np.sin(2 * np.pi * current_f * t)
        current_f += f0
        
    # Normalizar
    signal = signal / np.max(np.abs(signal))
    
    # Aplicar una envolvente (silabas) para simular el habla
    # Modulación lenta (1.5 Hz) + Silencios
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 1.5 * t)) 
    
    return signal * envelope

def save_wav(filename, rate, data, folder="resultados_test"):
    if not os.path.exists(folder): os.makedirs(folder)
    # Normalización segura a int16 para evitar clipping
    data = np.real(data)
    m = np.max(np.abs(data))
    if m > 0: 
        # Dejamos un margen de seguridad (-1dB aprox)
        data = data / m * 0.9
    wavfile.write(os.path.join(folder, filename), rate, (data * 32767).astype(np.int16))
    print(f"-> Guardado: {filename}")

def run_test():
    print("=== INICIANDO PRUEBA DE SISTEMA ADAPTATIVO (SINTÉTICO) ===")

    # 1. Parámetros de Simulación
    FS = 48000
    K = 25          # Taps del filtro
    DURATION = 4.0  # Segundos
    MU_TEST = 0.1   # <--- SUBIMOS EL MU A 0.1 (Convergencia más rápida para el test)
    
    # Geometría (ULA 9 mics, 4cm spacing)
    M_MICS = 9
    D_SPACING = 0.04 
    mic_array_obj = ULA(M=M_MICS, d=D_SPACING)
    
    # Banda de diseño (Voz humana estándar)
    F_MIN = 100.0
    F_MAX = 4000.0
    
    # 2. Instanciar el Sistema
    print(f"[Sistema] Inicializando Beamformer (M={M_MICS}, K={K})...")
    bf = AdaptiveBeamformer(
        MicArrayObj=mic_array_obj,
        K=K,
        fs=FS,
        fmin=F_MIN,
        fmax=F_MAX
    )
    
    # --- INYECCIÓN DE PARÁMETRO MU ---
    # Sobrescribimos el valor por defecto de la clase para este test
    bf.MU = MU_TEST
    print(f"[Config] MU establecido en: {bf.MU}")

    # 3. Generar Señales (Fuente y Ruido)
    print("\n[Generación] Creando escenario acústico virtual...")
    
    # POSICIONES (Coordenadas Cartesianas)
    # Target: Frente (90 grados, 2 metros) -> (0, 2, 0)
    target_pos = np.array([0.0, 2.0, 0.0]) 
    
    # Interferencia: Derecha (45 grados, 2 metros)
    # x = r*cos(theta), y = r*sin(theta) -> theta=45 deg
    angle_int = np.deg2rad(45)
    interf_pos = np.array([2.0 * np.cos(angle_int), 2.0 * np.sin(angle_int), 0.0])
    
    # AUDIOS MONO
    target_clean = generate_synthetic_voice(FS, DURATION)
    
    # Interferencia: Ruido Blanco + un tono molesto de 1kHz
    noise_len = len(target_clean)
    t = np.arange(noise_len) / FS
    interf_clean = 0.3 * np.random.randn(noise_len) + 0.1 * np.sin(2*np.pi*1000*t)
    
    # 4. Simulación de Propagación (Broadband)
    print("[Propagación] Simulando retardos fraccionarios (space_delay)...")
    
    # Propagar Target hacia los micros
    # space_delay retorna (Fuentes, Mics, Muestras), tomamos fuente 0
    mics_target, _, _ = space_delay(target_clean, FS, target_pos, mic_array_obj.coordinates)
    mics_target = mics_target[0] 
    
    # Propagar Interferencia
    mics_interf, _, _ = space_delay(interf_clean, FS, interf_pos, mic_array_obj.coordinates)
    mics_interf = mics_interf[0]

    # Suma en el micrófono (Mezcla)
    # Alinear longitudes con padding si es necesario
    len_max = max(mics_target.shape[1], mics_interf.shape[1])
    input_signal = np.zeros((M_MICS, len_max), dtype=np.float32)
    
    input_signal[:, :mics_target.shape[1]] += mics_target
    input_signal[:, :mics_interf.shape[1]] += mics_interf
    
    # Añadir piso de ruido térmico (blanco, no correlacionado) - Importante para robustez
    input_signal += 0.002 * np.random.randn(M_MICS, len_max)
    
    print(f"[Input] Señal mezclada lista. Shape: {input_signal.shape}")

    # 5. Calcular Pesos (Bypass de generate_bank)
    # Calculamos los filtros óptimos para la posición del target 'on-the-fly'
    print("\n[Configuración] Calculando filtros LCMV Regionales (Bypass)...")
    
    C, h, Ca = build_region_constraints(
        Rs=target_pos,
        delta_r=0.2,            # +/- 20cm
        delta_azimut=np.deg2rad(5), # +/- 5 grados
        delta_elevation=np.deg2rad(2),
        mic_array=mic_array_obj.coordinates,
        fs=FS,
        K=K,
        f_min=F_MIN,
        f_max=F_MAX,
        num_points=40, 
        num_freqs=30
    )
    
    # Resolver pesos fijos w_q
    w_q = compute_fixed_weights_optimized(C, h).flatten()
    
    # Inyectar estado al sistema (Simulamos que el Tracker encontró al target)
    bf.current_wq = w_q.astype(np.float32) # Asegurar float32
    bf.current_Ca = Ca.astype(np.float32)
    bf.active_coords = target_pos
    # Reset del filtro adaptativo
    bf.current_wa = None 
    
    print(" -> Pesos inyectados correctamente.")

    # 6. Procesamiento
    print("\n[Procesamiento] Ejecutando GSC (process_block)...")
    
    # Procesar todo el bloque de audio de una vez
    output_signal = bf.process_block(input_signal)
    
    # 7. Resultados y Gráficas
    print("\n[Resultados] Generando reporte...")
    
    # Guardar audio
    save_wav("TEST_1_Input_Mic0.wav", FS, input_signal[0,:])
    save_wav("TEST_2_Output_GSC.wav", FS, output_signal)
    save_wav("TEST_3_Ref_Target.wav", FS, target_clean)
    
    # Graficar
    plt.figure(figsize=(14, 10))
    
    # A. Dominio del Tiempo (Zoom en una transición)
    plt.subplot(3, 1, 1)
    plt.title(f"Dominio del Tiempo (Zoom) - mu={MU_TEST}")
    start_s = int(FS * 1.5) # Mirar al segundo 1.5
    end_s = start_s + 4000  # 4000 muestras
    t_zoom = np.arange(4000) / FS * 1000 # ms
    
    plt.plot(t_zoom, input_signal[0, start_s:end_s], label="Mic 0 (Sucio)", color='lightgray')
    plt.plot(t_zoom, output_signal[start_s:end_s], label="Salida GSC", color='green', linewidth=1.5)
    plt.plot(t_zoom, target_clean[start_s:end_s]*0.8, label="Target (Ref)", color='black', linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    plt.ylabel("Amplitud")
    plt.xlabel("Tiempo (ms)")
    plt.grid(True, alpha=0.3)
    
    # B. Convergencia (Aproximada por energía del error)
    plt.subplot(3, 1, 2)
    plt.title("Energía de Salida (Convergencia)")
    # Media móvil de la energía
    window_size = 1000
    energy_in = np.convolve(input_signal[0]**2, np.ones(window_size)/window_size, mode='same')
    energy_out = np.convolve(output_signal**2, np.ones(window_size)/window_size, mode='same')
    t_full = np.arange(len(energy_in)) / FS
    
    plt.semilogy(t_full, energy_in, label="Energía Entrada", color='gray', alpha=0.6)
    plt.semilogy(t_full, energy_out, label="Energía Salida", color='green')
    plt.ylabel("Energía (Log)")
    plt.xlabel("Tiempo (s)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # C. Espectrogramas (Comparativa)
    plt.subplot(3, 2, 5)
    plt.title("Espectro Entrada (Mic 0)")
    plt.specgram(input_signal[0], Fs=FS, NFFT=1024, noverlap=512, cmap='inferno')
    plt.ylabel("Frecuencia (Hz)")
    
    plt.subplot(3, 2, 6)
    plt.title("Espectro Salida (GSC)")
    plt.specgram(output_signal, Fs=FS, NFFT=1024, noverlap=512, cmap='inferno')
    plt.ylabel("Frecuencia (Hz)")
    
    plt.tight_layout()
    plt.show()
    print("Prueba finalizada.")

if __name__ == "__main__":
    run_test()