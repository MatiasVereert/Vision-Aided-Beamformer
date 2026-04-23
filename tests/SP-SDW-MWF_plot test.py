# This model is based on the paper titled: 
# "Frequency-domain criterion for the speech distortion weighted multichannel Wiener filter for robust noise reduction"
# doi: 10.1016/j.specom.2007.02.001
# by Simon Doclo *, Ann Spriet, Jan Wouters, Marc Moonen

import numpy as np
from numba import njit
# Local 
from beamforming.signal_model import steering_vector, compute_rtf_steering_vector
from scipy.spatial.distance import pdist

import numpy as np
import matplotlib.pyplot as plt

# Normalize signals to range [-0.99, 0.99] to prevent clipping when saving as WAV
def normalize_signal(sig):
    max_abs = np.max(np.abs(sig))
    if max_abs > 0:
        return sig * (0.99 / max_abs)
    return sig

from beamforming.MWF.SP_SDW_MWF import sdw_mwf
import os
from propagation.simulate_acoustics import SimAcoustic
from utils.audio import save_wav
# Import the fixed branch function we built

if __name__ == "__main__":
    FS = 16000
    M1, M2 = 12, 1          
    M = M1 * M2
    speed_of_sound = 343.0 
    
    print("=== INTEGRATION TEST: FIXED BRANCH (SDW-MWF) ===")
    
    output_folder = "tests/data/sdw_mwf_output"
    os.makedirs(output_folder, exist_ok=True)
    
    # === REPLACEMENT: Non-Uniform Linear Array (Logarithmic Spacing) ===
    max_length = 0.30  # Maximum array length of 30 cm

    # Create logarithmic spacing to cluster microphones for high frequencies
    # while spreading them out for low frequencies. Ideal for broadband speech.
    if M > 1:
        base = 2.0  # Dispersion factor (can be tuned between 1.5 and 3.0)
        indices = np.arange(M)
        
        # Normalize from 0 to 1 and scale to maximum length
        x_norm = (base**indices - 1) / (base**(M - 1) - 1)
        x = x_norm * max_length
    else:
        x = np.array([0.0])

    # Assemble 3D coordinate matrix (assuming array is along the X-axis)
    mic_coords = np.column_stack([x, np.zeros(M), np.zeros(M)])

    # Translate the array to the desired center position
    array_center = np.array([1.25, 2.0, 1.25])
    mic_coords = mic_coords - np.mean(mic_coords, axis=0) + array_center
    # ===================================================================
    
    r = 1.0 
    ang_target = np.deg2rad(130)
    ang_interf = np.deg2rad(50)
    
    source_pos = array_center + np.array([r * np.cos(ang_target), r * np.sin(ang_target), 0.0])
    interf_pos1 = array_center + np.array([r * np.cos(ang_interf), r * np.sin(ang_interf), 0.0])

    print(" -> Initializing acoustic scene...")
    acoustic_scene = SimAcoustic(mic_coords, array_mismatch=0.0, duration=40, fs=FS)

    source_path = "tools/data/signals/FA01_09.wav"
    int_path1 = "tools/data/signals/MC15_03.wav"

    acoustic_scene.set_source(source_path, gain=1, position=source_pos.reshape(1,3))
    acoustic_scene.set_interference(int_path1, gain=1, position=interf_pos1.reshape(1,3))

    print(" -> Computing free field simulation...")
    # room_input_ideal shape is expected to be (M, N_samples)
    room_input_ideal, vad_oracle = acoustic_scene.free_field(iSIR_dB=0, normalize=True, mode="ideal", VAD = True)
    save_wav("1_input_mix_mic0.wav", FS, room_input_ideal[0], output_folder)
    
    print(" -> Applying Fixed Branch (SDW-MWF Delay-and-Sum)...")
    # Ensure source_pos is shape (1, 3) for the steering vector broadcasting
    source_pos_2d = source_pos.reshape(1, 3)
    
    # Execute the delay-and-sum fixed branch in frequency domain
    # Mapping: u = room_input_ideal, target_pos = mic_coords, source_pos = source_pos_2d
    z_fixed, post_block , z_noise, output, weights = sdw_mwf(room_input_ideal,
                               vad_oracle, 
                               mic_coords, 
                               source_pos_2d, 
                               FS,
                               ouput_weights=True)
    
    print(" -> Normalizing and saving reconstructed time-domain signals...")

    print(z_noise)


    z_fixed_norm = normalize_signal(z_fixed)
    post_block_norm = normalize_signal(post_block)
    z_noise_norm = normalize_signal(z_noise)
    output_norm = normalize_signal(output)
    

    save_wav("2_output_SDW_MWF_fixed.wav", FS, z_fixed_norm, output_folder)
    save_wav("2_output_SDW_MWF_post_block.wav", FS, post_block_norm, output_folder)
    save_wav("2_output_SDW_MWF_noise.wav", FS, z_noise_norm, output_folder)
    save_wav("2_output_SDW_MWF_output.wav", FS, output_norm, output_folder)

    import matplotlib
    # Force Matplotlib to use a non-interactive backend to prevent rendering conflicts
    matplotlib.use('Agg') 
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    print("=== GENERATING POLAR PATTERN EVOLUTION VIDEO (30 FPS) ===")
    
    # Define 3 target frequencies for visualization
    target_freqs = [500.0, 1000.0, 2000.0]
    colors = ['blue', 'orange', 'purple']
    
    # Reconstruct frequency vector based on the weights matrix dimensions
    F_bins = weights.shape[1]
    frecs_eval = np.linspace(0, FS / 2, F_bins)
    
    # Find the closest frequency indices
    freq_indices = [np.argmin(np.abs(frecs_eval - tf)) for tf in target_freqs]
    actual_freqs = [frecs_eval[idx] for idx in freq_indices]

    # Angles for the polar plot (0 to 2*pi)
    angles = np.linspace(0, 2 * np.pi, 360)
    
    # Preallocate steering vector matrices for each target frequency
    sv_matrices = [np.zeros((len(angles), M), dtype=np.complex128) for _ in target_freqs]

    print(f" -> Precomputing steering vectors for frequencies: {[round(f, 1) for f in actual_freqs]} Hz...")
    for i, ang in enumerate(angles):
        r_far = 100.0 
        pos = array_center + np.array([r_far * np.cos(ang), r_far * np.sin(ang), 0.0])
        pos_2d = pos.reshape(1, 3)
        
        sv = compute_rtf_steering_vector(frecs_eval, pos_2d, mic_coords, ref_mic_idx=0, mode="far_field", squeeze=True)
        
        for j, f_idx in enumerate(freq_indices):
            sv_matrices[j][i, :] = sv[f_idx, :]

    # Setup figure
    fig = plt.figure(figsize=(14, 6))
    
    # Subplot 1: 2D Spatial map setup (Static)
    ax1 = fig.add_subplot(121)
    ax1.scatter(mic_coords[:, 0], mic_coords[:, 1], c='blue', label='Microphones', alpha=0.6)
    ax1.scatter(source_pos[0], source_pos[1], c='green', marker='*', s=200, label='Target Source')
    ax1.scatter(interf_pos1[0], interf_pos1[1], c='red', marker='X', s=150, label='Interference')
    
    ax1.plot([array_center[0], source_pos[0]], [array_center[1], source_pos[1]], 'g--', alpha=0.3)
    ax1.plot([array_center[0], interf_pos1[0]], [array_center[1], interf_pos1[1]], 'r--', alpha=0.3)
    
    ax1.set_aspect('equal')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_title('Spatial Configuration')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    # Subplot 2: Polar pattern setup
    ax2 = fig.add_subplot(122, projection='polar')

    # Frame decimation to speed up plotting drastically
    tot_frames = weights.shape[0]
    step = 10 
    frames_to_render = list(range(0, tot_frames, step))

    def animate(m):
        # Bulletproof method: Clear the axis and rebuild it every frame
        ax2.clear()
        ax2.set_ylim([-30, 5]) 
        ax2.set_yticks([-30, -20, -10, 0])
        
        # Redraw static reference lines
        ax2.plot([ang_target, ang_target], [-30, 5], color='green', linestyle='-', linewidth=2, label='Target Angle')
        ax2.plot([ang_interf, ang_interf], [-30, 5], color='red', linestyle='-', linewidth=2, label='Interf Angle')

        for j, f_idx in enumerate(freq_indices):
            w_frame = weights[m, f_idx, :]
            
            # Safety Check: Prevent NaNs from wiping out the plot
            if np.isnan(w_frame).any() or np.isinf(w_frame).any():
                w_frame = np.zeros_like(w_frame)
                print("invalid weights detected, replacing with zeros")
                
            resp = np.abs(np.einsum('m, am -> a', w_frame, sv_matrices[j]))
            current_resp_db = 20 * np.log10(resp + 1e-12)
            current_resp_db = np.clip(current_resp_db, -30, 5)
            
            # Draw the dynamic polar line for this frequency
            ax2.plot(angles, current_resp_db, color=colors[j], linewidth=2, label=f'{actual_freqs[j]:.1f} Hz')
            
        L = 128
        time_sec = (m * L) / FS 
        ax2.set_title(f"Polar Pattern Evolution\nFrame: {m}/{tot_frames} (Sim Time: {time_sec:.3f}s)")
        ax2.legend(loc='lower right', bbox_to_anchor=(1.3, -0.1))

    print(f" -> Rendering {len(frames_to_render)} frames... (Decimated by {step})")
    
    # Init function is no longer needed with the clear() strategy
    ani = animation.FuncAnimation(fig, animate, frames=frames_to_render, blit=False)

    video_path = os.path.join(output_folder, "polar_evolution_30fps.mp4")
    
    try:
        ani.save(video_path, writer='ffmpeg', fps=30)
        print(f" -> Animation successfully saved at: {video_path}")
    except Exception as e:
        print(" -> ERROR: Could not save video. Make sure 'ffmpeg' is installed and in your system PATH.")
        print(e)
        
    plt.close(fig)