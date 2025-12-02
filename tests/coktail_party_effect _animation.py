import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import os

# --- Imports del Sistema ---
from beamforming.array.mic_array import ULA
from beamforming.system import AdaptiveBeamformer
from propagation.free_field import space_delay
from beamforming.gsc.region_constriant import build_region_constraints
from beamforming.gsc.weights import compute_fixed_weights_optimized
from utils.audio import load_audio_source, save_wav
from beamforming.evaluation.gain import analytical_gain
from utils.geometry import source_rotation

class BeampatternAnimator:
    def __init__(self, beamformer, mic_array, target_pos, interf_pos, fs, freqs_to_plot=[500, 1500, 3000]):
        """
        Clase auxiliar para animar la evolución del GSC en MULTIPLES FRECUENCIAS.
        """
        self.bf = beamformer
        self.data_log = beamformer.data_log
        self.mic_array = mic_array
        self.target_pos = target_pos
        self.interf_pos = interf_pos
        self.fs = fs
        # Convertimos a array numpy para vectorización
        self.freqs = np.array(freqs_to_plot) 
        
        # Pre-calcular puntos del círculo para el patrón polar (0 a 360 grados)
        self.circle_points, self.angles_deg = source_rotation(radius=2.0, samples=360, axis='h')
        self.circle_points = self.circle_points.T 
        
        # Almacenará las referencias a las líneas de matplotlib
        self.lines = []

    def _compute_total_weights(self, snapshot):
        w_q = snapshot['wq']
        w_a = snapshot['wa']
        C_a = snapshot['ca']
        adaptive_component = C_a @ w_a
        w_total = w_q - adaptive_component
        return w_total

    def update_plot(self, frame_idx, ax_polar, text_iter):
        if frame_idx >= len(self.data_log):
            return self.lines + [text_iter]

        snapshot = self.data_log[frame_idx]
        time_sec = snapshot['time']
        
        # 1. Calcular Pesos Totales
        w_total = self._compute_total_weights(snapshot)
        
        # 2. Calcular Ganancia Analítica Vectorizada
        # Retorna matriz de forma (N_freqs, 360_puntos)
        gains_db_matrix = analytical_gain(
            frecs=self.freqs,
            fs=self.fs,
            mic_array=self.mic_array,
            weights=w_total[:, np.newaxis], 
            source_points=self.circle_points
        )
        
        # 3. Actualizar cada línea (una por frecuencia)
        for i, line in enumerate(self.lines):
            gains_db = gains_db_matrix[i]
            # Normalizar individualmente o globalmente. 
            # Aquí normalizamos respecto al máximo de ESA frecuencia para ver la forma del lóbulo claramente
            gains_norm = gains_db - np.max(gains_db)
            line.set_ydata(gains_norm)
        
        text_iter.set_text(f"Tiempo: {time_sec:.2f} s | Frame: {frame_idx}")
        
        return self.lines + [text_iter]

    def run_animation(self, filename="animation_beamforming_multi.mp4"):
        print(f"Generando animación multi-frecuencia ({self.freqs} Hz) con {len(self.data_log)} cuadros...")
        
        if len(self.data_log) == 0:
            print("[Error] No hay datos en el log.")
            return

        fig = plt.figure(figsize=(15, 8)) # Un poco más ancho
        gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1])
        
        # --- SUBPLOT 1: GEOMETRÍA ---
        ax_geo = fig.add_subplot(gs[0], aspect='equal')
        ax_geo.set_title("Escenario Acústico")
        
        ax_geo.scatter(self.mic_array[:,0], self.mic_array[:,1], c='k', marker='s', s=80, label='Mics')
        ax_geo.scatter(self.target_pos[0], self.target_pos[1], c='b', marker='*', s=200, label='Target')
        ax_geo.scatter(self.interf_pos[0], self.interf_pos[1], c='r', marker='X', s=100, label='Interferencia')
        
        ax_geo.arrow(0, 0, self.target_pos[0], self.target_pos[1], color='b', alpha=0.2, width=0.01)
        ax_geo.arrow(0, 0, self.interf_pos[0], self.interf_pos[1], color='r', alpha=0.2, width=0.01)
        
        max_coord = max(np.max(np.abs(self.target_pos)), np.max(np.abs(self.interf_pos))) * 1.2
        ax_geo.set_xlim(-max_coord, max_coord)
        ax_geo.set_ylim(-0.5, max_coord)
        ax_geo.grid(True, linestyle=':')
        ax_geo.legend(loc='lower left')

        # --- SUBPLOT 2: BEAMPATTERN MULTI-FRECUENCIA ---
        ax_polar = fig.add_subplot(gs[1], projection='polar')
        ax_polar.set_title(f"Evolución del Patrón Polar")
        ax_polar.set_theta_zero_location('E')
        ax_polar.set_ylim(-40, 5)
        
        theta_rad = np.deg2rad(self.angles_deg)
        
        # Inicializar líneas vacías para cada frecuencia
        colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd'] # Azul, Verde, Rojo, Violeta
        self.lines = []
        for i, f in enumerate(self.freqs):
            color = colors[i % len(colors)]
            # Usamos alpha=0.8 para ver solapamientos
            l, = ax_polar.plot(theta_rad, np.zeros_like(theta_rad), 
                               label=f'{f} Hz', color=color, linewidth=2, alpha=0.8)
            self.lines.append(l)

        # Ángulos de referencia
        ang_target = np.arctan2(self.target_pos[1], self.target_pos[0])
        ang_interf = np.arctan2(self.interf_pos[1], self.interf_pos[0])
        ax_polar.axvline(ang_target, color='k', linestyle='--', alpha=0.3, label='Target Dir')
        ax_polar.axvline(ang_interf, color='k', linestyle=':', alpha=0.5, label='Interf Dir')
        
        ax_polar.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1), fontsize='small')
        text_iter = ax_polar.text(0.05, 0.95, '', transform=ax_polar.transAxes)

        ani = animation.FuncAnimation(
            fig, 
            self.update_plot, 
            frames=len(self.data_log), 
            fargs=(ax_polar, text_iter),
            interval=50, 
            blit=True
        )
        
        try:
            ani.save(filename, writer='ffmpeg', fps=20, dpi=100)
            print(f"Animación guardada en: {filename}")
        except Exception as e:
            print(f"No se pudo guardar video: {e}")
            plt.show()

# ==============================================================================
# SCRIPT PRINCIPAL
# ==============================================================================
def run_test_presentation():
    print("=== INICIANDO PRUEBA COCKTAIL PARTY (ANIMACIÓN MULTI-BAND) ===")

    FS = 48000
    K = 40 # Aumenté un poco los taps para mejorar la resolución en baja frecuencia
    DURATION = 3.0 
    MU_TEST = 0.05 
    
    AUDIO_FILENAME = "tests/FK62_01.wav" 
    INTERF_FILENAME = "tools/data/signals/MC15_03.wav" 

    M_MICS = 10
    mic_array_coords = np.stack( 
        [[-.12,-.5, 0, 0.15, 0.03, 0.1, 0.12, 0.15, 0.3, 0.6], 
         np.zeros(M_MICS), 
         np.zeros(M_MICS)], axis=1 
    )
    
    F_MIN = 200.0
    F_MAX = 4000.0 
    
    bf = AdaptiveBeamformer(mic_array_coords, K, FS, F_MIN, F_MAX)
    bf.MU = MU_TEST
    bf.FPS = 15 

    target_pos = np.array([0.5, 1.5, 0.0]) 
    angle_int = np.deg2rad(40)
    interf_pos = np.array([2.0 * np.cos(angle_int), 2.0 * np.sin(angle_int), 0.0])
    
    try:
        target_clean = load_audio_source(AUDIO_FILENAME, FS, DURATION)
        interf_clean = load_audio_source(INTERF_FILENAME, FS, DURATION)
    except Exception as e:
        print(f"[Error] {e}")
        return
    
    print("[Propagación] Generando mezcla...")
    mics_target, _, _ = space_delay(target_clean, FS, target_pos, mic_array_coords)
    mics_interf, _, _ = space_delay(interf_clean, FS, interf_pos, mic_array_coords)
    
    n_samples_target = mics_target.shape[2]
    n_samples_interf = mics_interf.shape[2]
    len_max = max(n_samples_target, n_samples_interf)


    
    input_signal = np.zeros((M_MICS, len_max), dtype=np.float32)
    input_signal[:, :n_samples_target] += mics_target[0]
    input_signal[:, :n_samples_interf] += mics_interf[0]
    input_signal += 0.001 * np.random.randn(*input_signal.shape)
    
    C, h, Ca = build_region_constraints(
        Rs=target_pos,
        delta_r=0.1, delta_azimut=np.deg2rad(4), delta_elevation=np.deg2rad(2),
        mic_array=mic_array_coords, fs=FS, K=K,
        f_min=F_MIN, f_max=F_MAX, num_points=50, num_freqs=30
    )
    w_q = compute_fixed_weights_optimized(C, h).flatten()
    
    bf.current_wq = w_q.astype(np.float32)
    bf.current_Ca = Ca.astype(np.float32)
    bf.active_coords = target_pos
    bf.current_wa = None 

    print("[Procesamiento] Ejecutando GSC con grabación de pesos...")
    if hasattr(bf, 'process_block_vad'):
        output_signal = bf.process_block_vad(input_signal, record_weights=True)
    else:
        output_signal = bf.process_block(input_signal, record_weights=True)

    print("[Animación] Iniciando renderizado MULTI-BAND...")
    
    # AQUÍ DEFINES LAS FRECUENCIAS A VISUALIZAR
    freqs_visualizacion = [500, 1500, 3000] 
    
    animator = BeampatternAnimator(
        beamformer=bf,
        mic_array=mic_array_coords,
        target_pos=target_pos,
        interf_pos=interf_pos,
        fs=FS,
        freqs_to_plot=freqs_visualizacion 
    )
    
    animator.run_animation("Cocktail_Party_MultiBand.mp4")
    
    if not os.path.exists("audios_presentacion"):
        os.makedirs("audios_presentacion")
    save_wav("CP_Output.wav", FS, output_signal, "audios_presentacion")

if __name__ == "__main__":
    run_test_presentation()