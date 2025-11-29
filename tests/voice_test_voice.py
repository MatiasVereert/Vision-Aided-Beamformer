import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.io import wavfile
from scipy import signal
import os

# --- Imports del Sistema ---
# Asegúrate de que las rutas de importación coincidan con tu estructura de carpetas
from beamforming.array.mic_array import ULA
from beamforming.system import AdaptiveBeamformer
from propagation.free_field import space_delay
from beamforming.algorithms.region_constriant import build_region_constraints
from beamforming.algorithms.weights import compute_fixed_weights_optimized
from utils.audio import load_audio_source, save_wav

def run_test_presentation():
    print("=== INICIANDO PRUEBA DE SISTEMA ADAPTATIVO (FORMATO PRESENTACIÓN) ===")

    # 1. Parámetros de Simulación
    FS = 48000
    K = 60          
    DURATION = 5
    MU_TEST = 0.4
    
    # Archivos de Audio
    AUDIO_FILENAME = "tests/FK62_01.wav" 
    INTERF_FILENAME = "tools/data/signals/MC15_03.wav" # Nueva interferencia solicitada
    
    # --- Generación de Dummies (Seguridad para que corra el script) ---
    # 1. Target Dummy
    if not os.path.exists(AUDIO_FILENAME):
        print(f"[Alerta] Generando archivo dummy en {AUDIO_FILENAME}...")
        t_dummy = np.linspace(0, 1, 48000)
        dummy_data = np.sin(2*np.pi*440*t_dummy) * 0.5
        os.makedirs(os.path.dirname(AUDIO_FILENAME), exist_ok=True)
        wavfile.write(AUDIO_FILENAME, 48000, (dummy_data*32767).astype(np.int16))

    # 2. Interferencia Dummy (Si no existe el archivo MC15_03)
    if not os.path.exists(INTERF_FILENAME):
        print(f"[Alerta] Generando archivo dummy en {INTERF_FILENAME}...")
        t_dummy = np.linspace(0, 1, 48000)
        # Usamos una señal distinta (Sawtooth) para diferenciarla auditivamente del seno
        dummy_data = 0.3 * signal.sawtooth(2 * np.pi * 150 * t_dummy)
        os.makedirs(os.path.dirname(INTERF_FILENAME), exist_ok=True)
        wavfile.write(INTERF_FILENAME, 48000, (dummy_data*32767).astype(np.int16))


    # Geometría (ULA 9 mics, 4cm spacing)
    M_MICS = 15
    D_SPACING = 0.02
    mic_array_obj = ULA(M=M_MICS, d=D_SPACING)
    mic_array_coords = mic_array_obj.coordinates
    
    F_MIN = 50.0
    F_MAX = 10000.0
    
    # 2. Instanciar el Sistema
    bf = AdaptiveBeamformer(
        mic_array=mic_array_coords,
        K=K,
        fs=FS,
        fmin=F_MIN,
        fmax=F_MAX
    )
    bf.MU = MU_TEST

    # 3. Generar Señales
    # Target: 0.8 metros (Campo Cercano)
    target_pos = np.array([0.0, 0.8, 0.0]) 
    
    angle_int = np.deg2rad(45)
    interf_pos = np.array([2.0 * np.cos(angle_int), 2.0 * np.sin(angle_int), 0.0])
    
    # Carga de Audios
    try:
        print(f"[Carga] Fuente deseada: {AUDIO_FILENAME}")
        target_clean = load_audio_source(AUDIO_FILENAME, FS, DURATION)
        
        print(f"[Carga] Interferencia: {INTERF_FILENAME}")
        interf_clean = load_audio_source(INTERF_FILENAME, FS, DURATION)
        
    except Exception as e:
        print(f"[Error] Fallo al cargar audio: {e}")
        return
    
    # 4. Propagación
    print("[Propagación] Simulando retardos...")
    mics_target, _, _ = space_delay(target_clean, FS, target_pos, mic_array_coords)
    mics_target = mics_target[0] 
    
    mics_interf, _, _ = space_delay(interf_clean, FS, interf_pos, mic_array_coords)
    mics_interf = mics_interf[0]

    # Mezcla
    len_max = max(mics_target.shape[1], mics_interf.shape[1])
    input_signal = np.zeros((M_MICS, len_max), dtype=np.float32)
    
    input_signal[:, :mics_target.shape[1]] += mics_target
    input_signal[:, :mics_interf.shape[1]] += mics_interf
    
    # 5. Calcular Pesos (Bypass - Beamformer Fijo Inicial)
    C, h, Ca = build_region_constraints(
        Rs=target_pos,
        delta_r=0.01,            
        delta_azimut=.1, 
        delta_elevation=.1,
        mic_array=mic_array_coords,
        fs=FS,
        K=K,
        f_min=F_MIN,
        f_max=F_MAX,
        num_points=40, 
        num_freqs=30
    )
    
    w_q = compute_fixed_weights_optimized(C, h).flatten()
    
    bf.current_wq = w_q.astype(np.float32)
    bf.current_Ca = Ca.astype(np.float32)
    bf.active_coords = target_pos
    bf.current_wa = None 


    
    # 6. Procesamiento
    print("[Procesamiento] Ejecutando GSC...")
    if hasattr(bf, 'process_block'):
        output_signal = bf.process_block(input_signal)
    elif hasattr(bf, 'procces_block'): # Typos handling
        output_signal = bf.procces_block(input_signal)
    else:
        raise AttributeError("Método process_block no encontrado")

    # --- EXPORTAR AUDIOS ---
    print(f"[Audios] Exportando resultados a 'audios_presentacion'...")
    FOLDER_OUT = "audios_presentacion"
    save_wav("Voice_Presentacion_Input_Mic0.wav", FS, input_signal[0,:], folder=FOLDER_OUT)
    save_wav("Voice_Presentacion_Output_GSC.wav", FS, output_signal, folder=FOLDER_OUT)
    save_wav("Voice_Presentacion_Ref_Target.wav", FS, target_clean, folder=FOLDER_OUT)
    
    # 7. GRÁFICOS PARA PRESENTACIÓN
    print("[Gráficos] Generando figura optimizada...")

    # --- Configuración de Estilo Global ---
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'lines.linewidth': 1.5
    })

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    
    # GridSpec: 2 filas, 2 columnas. 
    # La columna 0 (Izquierda) ocupará ambas filas.
    # La columna 1 (Derecha) tendrá dos gráficos apilados.
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1, 1.2]) 

    # --- A. MAPA DE GEOMETRÍA (Izquierda Completa) ---
    ax_geo = fig.add_subplot(gs[:, 0]) # Spans both rows
    ax_geo.set_title("Escenario Acústico (Vista Superior)", fontweight='bold')
    
    # Plot Mics
    ax_geo.scatter(mic_array_coords[:, 0], mic_array_coords[:, 1], 
                   c='black', marker='s', s=100, label='Micrófonos (ULA)', zorder=3)
    
    # Plot Target
    ax_geo.scatter(target_pos[0], target_pos[1], 
                   c='#1f77b4', marker='*', s=500, label='Fuente Deseada (Voz)', zorder=3)
    # Línea visual hacia el target
    ax_geo.plot([0, target_pos[0]], [0, target_pos[1]], '--', color='#1f77b4', alpha=0.3)

    # Plot Interferencia
    ax_geo.scatter(interf_pos[0], interf_pos[1], 
                   c='#d62728', marker='X', s=250, label='Interferencia', zorder=3)
    # Línea visual hacia la interferencia
    ax_geo.plot([0, interf_pos[0]], [0, interf_pos[1]], '--', color='#d62728', alpha=0.3)

    # Ajustes estéticos Geo
    ax_geo.set_aspect('equal')
    ax_geo.set_xlabel("Eje X (metros)")
    ax_geo.set_ylabel("Eje Y (metros)")
    ax_geo.grid(True, linestyle=':', alpha=0.6)
    ax_geo.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=12)
    
    # Limites dinámicos
    max_range = max(np.linalg.norm(target_pos), np.linalg.norm(interf_pos)) * 1.2
    ax_geo.set_xlim(-max_range/2, max_range)
    ax_geo.set_ylim(-0.5, max_range)


    # --- B. ESPECTROGRAMAS (Derecha Apilados) ---
    
    # 1. Espectrograma Entrada (Arriba Derecha)
    ax_spec_in = fig.add_subplot(gs[0, 1])
    ax_spec_in.set_title("Espectrograma: Entrada (Mic 0 - Sucia)", fontweight='bold')
    Pxx, freqs, bins, im1 = ax_spec_in.specgram(input_signal[0], NFFT=1024, Fs=FS, noverlap=512, cmap='inferno')
    ax_spec_in.set_ylabel("Frecuencia (Hz)")
    ax_spec_in.set_xlabel("") # Quitamos label x del de arriba para limpiar
    ax_spec_in.set_xticklabels([]) # Quitamos ticks x

    # 2. Espectrograma Salida (Abajo Derecha)
    ax_spec_out = fig.add_subplot(gs[1, 1])
    ax_spec_out.set_title("Espectrograma: Salida Beamformer (Limpia)", fontweight='bold')
    Pxx, freqs, bins, im2 = ax_spec_out.specgram(output_signal, NFFT=1024, Fs=FS, noverlap=512, cmap='inferno')
    ax_spec_out.set_ylabel("Frecuencia (Hz)")
    ax_spec_out.set_xlabel("Tiempo (s)")

    # Título Principal ELIMINADO
    # plt.suptitle(...)
    
    # Guardar
    plt.savefig("Resultados_Presentacion.png", dpi=300, bbox_inches='tight')
    print("[Gráficos] Imagen guardada como 'Resultados_Presentacion.png'")
    
    plt.show()

if __name__ == "__main__":
    run_test_presentation()