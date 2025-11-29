import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import os

# --- Imports del Sistema ---
from beamforming.array.mic_array import ULA
from beamforming.system import AdaptiveBeamformer
from propagation.free_field import space_delay
from beamforming.algorithms.region_constriant import build_region_constraints
from beamforming.algorithms.weights import compute_fixed_weights_optimized
from utils.audio import load_audio_source, save_wav



def run_test():
    print("=== INICIANDO PRUEBA DE SISTEMA ADAPTATIVO (VOZ REAL) ===")

    # 1. Parámetros de Simulación
    FS = 48000
    K = 25          
    DURATION = 4.0  # Segundos (El audio se recortará o repetirá a esto)
    MU_TEST = 0.1   
    
    # --- CONFIGURACIÓN DEL ARCHIVO DE AUDIO ---
    # Cambia esto por la ruta de tu archivo wav
    AUDIO_FILENAME = "tests/FK62_01.wav" 
    
    # Crear un archivo dummy si no existe para probar el script
    if not os.path.exists(AUDIO_FILENAME):
        print(f"[Alerta] No existe {AUDIO_FILENAME}. Generando uno de prueba temporal...")
        t_dummy = np.linspace(0, 1, 48000)
        dummy_data = np.sin(2*np.pi*440*t_dummy) * 0.5
        wavfile.write(AUDIO_FILENAME, 48000, (dummy_data*32767).astype(np.int16))

    # Geometría (ULA 9 mics, 4cm spacing)
    M_MICS = 9
    D_SPACING = 0.04 
    mic_array_obj = ULA(M=M_MICS, d=D_SPACING)
    mic_array = mic_array_obj.coordinates
    
    F_MIN = 100.0
    F_MAX = 4000.0
    
    # 2. Instanciar el Sistema
    print(f"[Sistema] Inicializando Beamformer (M={M_MICS}, K={K})...")
    bf = AdaptiveBeamformer(
        mic_array =mic_array,
        K=K,
        fs=FS,
        fmin=F_MIN,
        fmax=F_MAX
    )
    bf.MU = MU_TEST

    # 3. Generar Señales (Fuente y Ruido)
    print("\n[Generación] Creando escenario acústico virtual...")
    
    # POSICIONES
    target_pos = np.array([0.0, 2.0, 0.0]) # Frente
    
    angle_int = np.deg2rad(45)
    interf_pos = np.array([2.0 * np.cos(angle_int), 2.0 * np.sin(angle_int), 0.0])
    
    # --- CARGA DE VOZ REAL ---
    print(f"[Carga] Leyendo archivo: {AUDIO_FILENAME}")
    try:
        target_clean = load_audio_source(AUDIO_FILENAME, FS, DURATION)
    except Exception as e:
        print(f"[Error Crítico] Fallo al cargar audio: {e}")
        return
    
    # Interferencia: Ruido Blanco + tono 1kHz
    noise_len = len(target_clean)
    t = np.arange(noise_len) / FS
    interf_clean = 0.3 * np.random.randn(noise_len) + 0.1 * np.sin(2*np.pi*1000*t)
    
    # 4. Simulación de Propagación (Broadband)
    print("[Propagación] Simulando retardos fraccionarios...")
    
    # Propagar Target (Voz Real)
    mics_target, _, _ = space_delay(target_clean, FS, target_pos, mic_array_obj.coordinates)
    mics_target = mics_target[0] 
    
    # Propagar Interferencia
    mics_interf, _, _ = space_delay(interf_clean, FS, interf_pos, mic_array_obj.coordinates)
    mics_interf = mics_interf[0]

    # Mezcla
    len_max = max(mics_target.shape[1], mics_interf.shape[1])
    input_signal = np.zeros((M_MICS, len_max), dtype=np.float32)
    
    input_signal[:, :mics_target.shape[1]] += mics_target
    input_signal[:, :mics_interf.shape[1]] += mics_interf
    
    # Ruido térmico
    input_signal += 0.002 * np.random.randn(M_MICS, len_max)
    
    print(f"[Input] Señal mezclada lista. Shape: {input_signal.shape}")

    # 5. Calcular Pesos (Bypass)
    print("\n[Configuración] Calculando filtros LCMV Regionales (Bypass)...")
    C, h, Ca = build_region_constraints(
        Rs=target_pos,
        delta_r=0.2,            
        delta_azimut=np.deg2rad(5), 
        delta_elevation=np.deg2rad(2),
        mic_array=mic_array_obj.coordinates,
        fs=FS,
        K=K,
        f_min=F_MIN,
        f_max=F_MAX,
        num_points=40, 
        num_freqs=30
    )
    
    w_q = compute_fixed_weights_optimized(C, h).flatten()
    
    # Inyectar estado
    bf.current_wq = w_q.astype(np.float32)
    bf.current_Ca = Ca.astype(np.float32)
    bf.active_coords = target_pos
    bf.current_wa = None 
    
    # 6. Procesamiento
    print("\n[Procesamiento] Ejecutando GSC (process_block)...")
    
    # NOTA: Verifica si tu método en AdaptiveBeamformer se llama 
    # 'process_block' o 'procces_block' (error tipográfico en src/beamforming/system.py)
    if hasattr(bf, 'process_block'):
        output_signal = bf.process_block(input_signal)
    elif hasattr(bf, 'procces_block'):
        output_signal = bf.procces_block(input_signal)
    else:
        raise AttributeError("No se encontró el método process_block en AdaptiveBeamformer")
    
    # 7. Resultados y Gráficas
    print("\n[Resultados] Generando reporte...")
    
    save_wav("TEST_VOZ_1_Input_Mic0.wav", FS, input_signal[0,:])
    save_wav("TEST_VOZ_2_Output_GSC.wav", FS, output_signal)
    save_wav("TEST_VOZ_3_Ref_Target.wav", FS, target_clean)
    
    # Graficar
    plt.figure(figsize=(14, 10))
    
    # A. Dominio del Tiempo
    plt.subplot(3, 1, 1)
    plt.title(f"Dominio del Tiempo - Voz Real - mu={MU_TEST}")
    # Zoom arbitrario en el medio del audio
    center_idx = len(output_signal) // 2
    window_view = 4000 
    start_s = center_idx 
    end_s = start_s + window_view
    if end_s > len(output_signal): end_s = len(output_signal)
    
    t_zoom = np.arange(end_s - start_s) / FS * 1000 # ms
    
    plt.plot(t_zoom, input_signal[0, start_s:end_s], label="Mic 0 (Sucio)", color='lightgray')
    plt.plot(t_zoom, output_signal[start_s:end_s], label="Salida GSC", color='green', linewidth=1.5)
    plt.plot(t_zoom, target_clean[start_s:end_s]*0.8, label="Voz Limpia (Ref)", color='black', linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    plt.ylabel("Amplitud")
    plt.xlabel("Tiempo (ms)")
    plt.grid(True, alpha=0.3)
    
    # B. Convergencia
    plt.subplot(3, 1, 2)
    plt.title("Energía de Salida (Convergencia)")
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
    
    # C. Espectrogramas
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


