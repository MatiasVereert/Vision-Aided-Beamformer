import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from scipy.io import wavfile
from scipy import signal
import os

# --- Imports del Sistema ---
from beamforming.array.mic_array import ULA
from beamforming.gsc.system import AdaptiveBeamformer
from propagation.free_field import space_delay
from beamforming.gsc.region_constriant import build_region_constraints
from beamforming.gsc.weights import compute_fixed_weights_optimized
from utils.audio import load_audio_source, save_wav
# Imports necesarios para la animación
from beamforming.evaluation.gain import analytical_gain
from utils.geometry import source_rotation

# ==============================================================================
# CLASE ANIMATOR (Multi-Frecuencia)
# ==============================================================================
class BeampatternAnimator:
    def __init__(self, beamformer, mic_array, target_pos, interf_pos, fs, freqs_to_plot=[500, 1500, 3000]):
        self.bf = beamformer
        self.data_log = beamformer.data_log
        self.mic_array = mic_array
        self.target_pos = target_pos
        self.interf_pos = interf_pos
        self.fs = fs
        self.freqs = np.array(freqs_to_plot) 
        
        # Pre-calcular puntos del círculo para el patrón polar (0 a 360 grados)
        self.circle_points, self.angles_deg = source_rotation(radius=2.0, samples=360, axis='h')
        self.circle_points = self.circle_points.T 
        
        self.lines = []

    def _compute_total_weights(self, snapshot):
        w_q = snapshot['wq']
        w_a = snapshot['wa']
        C_a = snapshot['ca']
        adaptive_component = C_a @ w_a
        w_total = w_q - adaptive_component
        return w_total

    def update_plot(self, frame_idx, ax_polar, text_iter):
        if frame_idx >= len(self.data_log): return self.lines + [text_iter]

        snapshot = self.data_log[frame_idx]
        time_sec = snapshot['time']
        
        # 1. Pesos Totales
        w_total = self._compute_total_weights(snapshot)
        
        # 2. Ganancia Analítica Vectorizada
        gains_db_matrix = analytical_gain(
            frecs=self.freqs,
            fs=self.fs,
            mic_array=self.mic_array,
            weights=w_total[:, np.newaxis], 
            source_points=self.circle_points
        )
        
        # 3. Actualizar líneas
        for i, line in enumerate(self.lines):
            gains_db = gains_db_matrix[i]
            # Normalizar respecto al máximo para ver la forma del lóbulo
            gains_norm = gains_db - np.max(gains_db)
            line.set_ydata(gains_norm)
        
        text_iter.set_text(f"Tiempo: {time_sec:.2f} s | Frame: {frame_idx}")
        return self.lines + [text_iter]

    def run_animation(self, filename="animation_voice_noise.mp4"):
        print(f"Generando animación multi-banda... ({len(self.data_log)} frames)")
        
        if len(self.data_log) == 0:
            print("[Error] Log vacío. Verifica record_weights=True.")
            return

        fig = plt.figure(figsize=(15, 8))
        gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1])
        
        # --- GEO ---
        ax_geo = fig.add_subplot(gs[0], aspect='equal')
        ax_geo.set_title("Escenario Acústico")
        ax_geo.scatter(self.mic_array[:,0], self.mic_array[:,1], c='k', marker='s', s=80, label='Mics')
        ax_geo.scatter(self.target_pos[0], self.target_pos[1], c='b', marker='*', s=200, label='Voz')
        ax_geo.scatter(self.interf_pos[0], self.interf_pos[1], c='r', marker='X', s=100, label='Ruido')
        
        max_dist = max(np.linalg.norm(self.target_pos), np.linalg.norm(self.interf_pos)) * 1.2
        ax_geo.set_xlim(-max_dist, max_dist)
        ax_geo.set_ylim(-0.5, max_dist)
        ax_geo.grid(True, linestyle=':')
        ax_geo.legend(loc='lower left')

        # --- POLAR ---
        ax_polar = fig.add_subplot(gs[1], projection='polar')
        ax_polar.set_title("Evolución del Patrón Polar")
        ax_polar.set_theta_zero_location('E')
        ax_polar.set_ylim(-40, 5)
        
        theta_rad = np.deg2rad(self.angles_deg)
        colors = ['#1f77b4', '#2ca02c', '#d62728']
        
        self.lines = []
        for i, f in enumerate(self.freqs):
            l, = ax_polar.plot(theta_rad, np.zeros_like(theta_rad), 
                               label=f'{f} Hz', color=colors[i%3], linewidth=2, alpha=0.8)
            self.lines.append(l)

        # Referencias angulares
        ang_t = np.arctan2(self.target_pos[1], self.target_pos[0])
        ang_i = np.arctan2(self.interf_pos[1], self.interf_pos[0])
        ax_polar.axvline(ang_t, color='b', linestyle='--', alpha=0.3)
        ax_polar.axvline(ang_i, color='r', linestyle=':', alpha=0.5)
        
        ax_polar.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1))
        text_iter = ax_polar.text(0.05, 0.95, '', transform=ax_polar.transAxes)

        ani = animation.FuncAnimation(fig, self.update_plot, frames=len(self.data_log), 
                                      fargs=(ax_polar, text_iter), interval=50, blit=True)
        try:
            ani.save(filename, writer='ffmpeg', fps=20, dpi=100)
            print(f"--> Video guardado: {filename}")
        except Exception as e:
            print(f"No se pudo guardar video (falta ffmpeg?): {e}")
            plt.show()

# ==============================================================================
# SCRIPT PRINCIPAL
# ==============================================================================
def run_test_presentation():
    print("=== INICIANDO PRUEBA VOZ + RUIDO (CON ANIMACIÓN) ===")

    # 1. Parámetros
    FS = 48000
    K = 25          
    DURATION = 4.0
    
    # IMPORTANTE: Cambiado de 0 a 0.1 para que haya movimiento en la animación
    MU_TEST = 0.1    
    
    AUDIO_FILENAME = "tests/FK62_01.wav" 
    
    # Generar dummy si no existe
    if not os.path.exists(AUDIO_FILENAME):
        print(f"[Alerta] Generando archivo dummy en {AUDIO_FILENAME}...")
        t_dummy = np.linspace(0, 1, 48000)
        dummy_data = np.sin(2*np.pi*440*t_dummy) * 0.5
        os.makedirs(os.path.dirname(AUDIO_FILENAME), exist_ok=True)
        wavfile.write(AUDIO_FILENAME, 48000, (dummy_data*32767).astype(np.int16))

    # Geometría
    M_MICS = 9
    D_SPACING = 0.04 
    mic_array_obj = ULA(M=M_MICS, d=D_SPACING)
    mic_array_coords = mic_array_obj.coordinates
    
    F_MIN = 100.0
    F_MAX = 4000.0
    
    # 2. Instanciar Sistema
    bf = AdaptiveBeamformer(mic_array_coords, K, FS, F_MIN, F_MAX)
    bf.MU = MU_TEST
    bf.FPS = 15 # Configurar FPS para la grabación

    # 3. Señales
    target_pos = np.array([0.0, 0.8, 0.0]) # Frente cercano
    angle_int = np.deg2rad(45)
    interf_pos = np.array([2.0 * np.cos(angle_int), 2.0 * np.sin(angle_int), 0.0])
    
    try:
        target_clean = load_audio_source(AUDIO_FILENAME, FS, DURATION)
    except Exception as e:
        print(f"[Error] {e}")
        return
    
    # Generar Interferencia (Ruido sintético)
    noise_len = len(target_clean)
    t = np.arange(noise_len) / FS
    interf_clean = 0.3 * np.random.randn(noise_len) + 0.1 * np.sin(2*np.pi*1000*t)
    
    # 4. Propagación
    mics_target, _, _ = space_delay(target_clean, FS, target_pos, mic_array_coords)
    mics_target = mics_target[0] 
    
    mics_interf, _, _ = space_delay(interf_clean, FS, interf_pos, mic_array_coords)
    mics_interf = mics_interf[0]

    # Mezcla Robusta
    len_max = max(mics_target.shape[1], mics_interf.shape[1])
    input_signal = np.zeros((M_MICS, len_max), dtype=np.float32)
    input_signal[:, :mics_target.shape[1]] += mics_target
    input_signal[:, :mics_interf.shape[1]] += mics_interf
    
    # 5. Pesos Fijos
    C, h, Ca = build_region_constraints(
        Rs=target_pos, delta_r=0.2, delta_azimut=np.deg2rad(5), delta_elevation=np.deg2rad(2),
        mic_array=mic_array_coords, fs=FS, K=K, f_min=F_MIN, f_max=F_MAX, num_points=40, num_freqs=30
    )
    
    w_q = compute_fixed_weights_optimized(C, h).flatten()
    
    bf.current_wq = w_q.astype(np.float32)
    bf.current_Ca = Ca.astype(np.float32)
    bf.active_coords = target_pos
    bf.current_wa = None 
    
    # 6. Procesamiento con Grabación
    print("[Procesamiento] Ejecutando GSC con record_weights=True...")
    # Usamos process_block estándar pero activando grabación
    output_signal = bf.process_block_vad(input_signal, record_weights=True)

    # 7. Generar Animación
    print("[Animación] Iniciando renderizado...")
    animator = BeampatternAnimator(
        beamformer=bf,
        mic_array=mic_array_coords,
        target_pos=target_pos,
        interf_pos=interf_pos,
        fs=FS,
        freqs_to_plot=[500, 1000, 2500] # Frecuencias relevantes para voz
    )
    animator.run_animation("Voice_Noise_Animation.mp4")

    # 8. Gráficos Estáticos (Originales)
    print("[Gráficos] Generando figura estática de resumen...")
    FOLDER_OUT = "audios_presentacion"
    save_wav("Voice_Sim_Output.wav", FS, output_signal, folder=FOLDER_OUT)
    
    plt.rcParams.update({'font.size': 10})
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1.2]) 

    # Mapa Geo
    ax_geo = fig.add_subplot(gs[:, 0])
    ax_geo.set_title("Escenario")
    ax_geo.scatter(mic_array_coords[:, 0], mic_array_coords[:, 1], c='k', marker='s', label='Mics')
    ax_geo.scatter(target_pos[0], target_pos[1], c='b', marker='*', s=200, label='Voz')
    ax_geo.scatter(interf_pos[0], interf_pos[1], c='r', marker='X', s=100, label='Ruido')
    ax_geo.legend()
    ax_geo.set_aspect('equal')
    ax_geo.grid(True, linestyle=':')

    # Espectrogramas
    ax_spec_in = fig.add_subplot(gs[0, 1])
    ax_spec_in.set_title("Entrada (Mic 0)")
    ax_spec_in.specgram(input_signal[0], NFFT=1024, Fs=FS, noverlap=512, cmap='inferno')
    ax_spec_in.set_ylabel("Hz")

    ax_spec_out = fig.add_subplot(gs[1, 1])
    ax_spec_out.set_title("Salida Beamformer")
    ax_spec_out.specgram(output_signal, NFFT=1024, Fs=FS, noverlap=512, cmap='inferno')
    ax_spec_out.set_ylabel("Hz")
    ax_spec_out.set_xlabel("Segundos")

    plt.tight_layout()
    plt.savefig("Resultados_Voice_Sim.png", dpi=150)
    print("Imagen guardada.")
    plt.show()

if __name__ == "__main__":
    run_test_presentation()