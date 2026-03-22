import numpy as np
from numba import njit, prange

# ============================================================================
# 1. Numba Core Computation Kernels
# ============================================================================

@njit(parallel=True, fastmath=True, cache=True)
def _compute_rtf_core(f_arr: np.ndarray, Rs_arr: np.ndarray, mic_arr: np.ndarray, 
                      ref_mic_idx: int, c: float, is_near_field: bool) -> np.ndarray:
    """
    Core math engine for the steering vector. 
    Explicit loops prevent massive RAM allocations during broadcasting.
    """
    F = f_arr.shape[0]
    P = Rs_arr.shape[0]
    M = mic_arr.shape[0]
    
    out = np.zeros((F, P, M), dtype=np.complex128)

    for p in prange(P):
        # Calculate Euclidean distances for point P to all M mics
        dists = np.zeros(M, dtype=np.float64)
        for m in range(M):
            dx = Rs_arr[p, 0] - mic_arr[m, 0]
            dy = Rs_arr[p, 1] - mic_arr[m, 1]
            dz = Rs_arr[p, 2] - mic_arr[m, 2]
            dists[m] = np.sqrt(dx*dx + dy*dy + dz*dz)

        d_ref = dists[ref_mic_idx]

        # Evaluate transfer function over all frequencies
        for f in range(F):
            freq = f_arr[f]
            for m in range(M):
                delta_d = dists[m] - d_ref
                phase = -2.0 * np.pi * freq * delta_d / c
                
                # Euler's formula representation: exp(j * phase)
                val = np.cos(phase) + 1j * np.sin(phase)

                if is_near_field:
                    val *= (d_ref / dists[m])

                out[f, p, m] = val
                
    return out


@njit(parallel=True, fastmath=True, cache=True)
def _compute_spatial_response_core(sv: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Replaces the np.einsum('fpm,fkm->fpk') operation. 
    Performs manual complex conjugate multiplication to save memory and scale across CPU cores.
    """
    F, P, M = sv.shape
    _, K, _ = weights.shape
    
    db_out = np.zeros((F, P, K), dtype=np.float32)
    
    for f in prange(F):
        for p in range(P):
            for k in range(K):
                summ_r = 0.0
                summ_i = 0.0
                
                for m in range(M):
                    v = sv[f, p, m]
                    w = weights[f, k, m]
                    
                    # Manual complex conjugate dot product: v * conj(w)
                    # (a + bi)(c - di) = (ac + bd) + i(bc - ad)
                    summ_r += v.real * w.real + v.imag * w.imag
                    summ_i += v.imag * w.real - v.real * w.imag
                    
                mag_sq = summ_r * summ_r + summ_i * summ_i
                
                # Convert power directly to dB within the loop
                db_out[f, p, k] = 10.0 * np.log10(mag_sq + 1e-12)
                
    return db_out


@njit(cache=True)
def _get_closest_freq_indices(freqs: np.ndarray, valid_bands: np.ndarray) -> np.ndarray:
    """
    Quickly finds the closest STFT frequency bin index for each standard 1/3 octave band.
    """
    f_indices = np.zeros(valid_bands.shape[0], dtype=np.int64)
    for i in range(valid_bands.shape[0]):
        f_indices[i] = np.argmin(np.abs(freqs - valid_bands[i]))
    return f_indices

# ============================================================================
# 2. Python API Wrappers
# ============================================================================

@njit(cache=True)
def get_sphere_of_points(radius: float, N_azimuth: int) -> np.ndarray:
    """
    Generates a spherical grid. Pre-compiled for instant generation.
    """
    N_elevation = N_azimuth // 2 + 1
    azimuths = np.linspace(0.0, 2 * np.pi, N_azimuth)
    elevations = np.linspace(-np.pi / 2, np.pi / 2, N_elevation)
    
    N_points = N_azimuth * N_elevation
    points = np.zeros((N_points, 3), dtype=np.float64)
    
    idx = 0
    for i in range(N_elevation):
        el = elevations[i]
        cos_el = np.cos(el)
        sin_el = np.sin(el)
        for j in range(N_azimuth):
            az = azimuths[j]
            cos_az = np.cos(az)
            sin_az = np.sin(az)
            
            points[idx, 0] = radius * cos_el * cos_az
            points[idx, 1] = radius * cos_el * sin_az
            points[idx, 2] = radius * sin_el
            idx += 1
            
    return points


def compute_rtf_steering_vector(f, Rs, mic_array, ref_mic_idx=0, c=343.0, mode="near_field", squeeze=True):
    """
    Python wrapper to interface with the optimized RTF Numba core.
    """
    f_arr = np.atleast_1d(f)
    Rs_arr = np.atleast_2d(Rs)
    is_near_field = (mode == "near_field")
    
    # Offload heavy math to Numba
    rtf_tensor = _compute_rtf_core(f_arr, Rs_arr, mic_array, ref_mic_idx, c, is_near_field)
    
    if squeeze:
        return np.squeeze(rtf_tensor)
    return rtf_tensor


def subsample_weights(weights: np.ndarray, freqs: np.ndarray, fs: int, hop_length: int, target_fps: int = 24):
    """
    Sub-samples the complex weights tensor using NumPy's C-backend for speed.
    """
    F_bins, T_frames, M_mics = weights.shape
    
    # 1. Temporal Downsampling
    audio_duration = T_frames * hop_length / fs
    target_frames = max(1, int(audio_duration * target_fps))
    t_indices = np.round(np.linspace(0, T_frames - 1, target_frames)).astype(int)
    
    # 2. Frequency Downsampling
    standard_1_3_octave = np.array([
        31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 
        800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 
        10000, 12500, 16000, 20000
    ])
    valid_bands = standard_1_3_octave[standard_1_3_octave <= (fs / 2)]
    
    # Get indices using the Numba helper
    f_indices = _get_closest_freq_indices(freqs, valid_bands)
    f_indices = np.unique(f_indices)
    
    # 3. Apply Sub-sampling via NumPy Advanced Indexing (instant operation)
    w_subsampled = weights[np.ix_(f_indices, t_indices, np.arange(M_mics))]
    freqs_subsampled = freqs[f_indices]
    
    return w_subsampled, freqs_subsampled


def precompute_quantized_spatial_response(weights, freqs, mic_pos, source_radius, N_azimuth, min_dB=-50.0):
    """
    Orchestrates the Numba core computation to return the final quantized 3D response matrix.
    Normalization is performed PER FRAME to preserve dynamic shape in visualization.
    """
    points = get_sphere_of_points(source_radius, N_azimuth)
    
    # --- FIX: Translate the evaluation grid to the physical location of the array ---
    # The 'points' are generated around the origin [0, 0, 0]. 
    # We must shift them to the array's acoustic center before calculating the RTF,
    # otherwise the steering vectors will evaluate completely skewed incoming angles.
    array_center = np.mean(mic_pos, axis=0)
    eval_points = points + array_center
    
    # Guarantee a 3D tensor output for the JIT compiler, regardless of single frequencies
    # Pass 'eval_points' to calculate accurate phase delays
    steering_vectors = compute_rtf_steering_vector(freqs, eval_points, mic_pos, mode="near_field", squeeze=False)
    
    # Execute heavily parallelized complex dot product and dB conversion
    db_gain = _compute_spatial_response_core(steering_vectors, weights)
    
    # --- Per-Frame Normalization ---
    # Find the max dB value for each time slice (k) across all points (p)
    # The result has shape (F, 1, K) which allows for broadcasting
    max_db_per_frame = np.max(db_gain, axis=1, keepdims=True)

    # Clip only the bottom, as each frame's top is its own max
    clipped_db = np.clip(db_gain, min_dB, None) 
    
    # Normalize each frame's gain relative to its own peak and the global min
    # Add a small epsilon to avoid division by zero if a frame is silent
    normalized = (clipped_db - min_dB) / (max_db_per_frame - min_dB + 1e-12)
    quantized_gain = (np.clip(normalized, 0, 1) * 255.0).astype(np.uint8)
    
    # Return the origin-centered 'points' so the Plotly mesh generation in dashboard.py stays intact
    return quantized_gain, points, np.squeeze(max_db_per_frame, axis=1)

if __name__ == "__main__":
    # ------------------------------------------------------------------------
    # A. Setup Mock Environment
    # ------------------------------------------------------------------------
    M = 12       # Number of microphones
    F = 2        # Number of frequencies
    K = 40       # Number of time frames (40 frames for a smooth rotation)
    
    # Create a 2D Uniform Circular Array (UCA) on the XY plane
    array_radius = 0.1 # 10 cm radius
    angles = np.linspace(0, 2 * np.pi, M, endpoint=False)
    mic_array = np.zeros((M, 3))
    mic_array[:, 0] = array_radius * np.cos(angles)
    mic_array[:, 1] = array_radius * np.sin(angles)
    
    freqs = np.array([1000.0, 4000.0])
    
    # Define rotating target source positions over K frames
    source_radius = 1.5
    angles_k = np.linspace(0, 2 * np.pi, K, endpoint=False)
    rotating_targets = np.zeros((K, 3))
    rotating_targets[:, 0] = source_radius * np.cos(angles_k)
    rotating_targets[:, 1] = source_radius * np.sin(angles_k)
    rotating_targets[:, 2] = 0.5 # Slightly elevated Z
    
    # ------------------------------------------------------------------------
    # B. Compute Dynamic Delay-and-Sum Weights
    # ------------------------------------------------------------------------
    # Pass the entire Kx3 array of positions. 
    # The optimized Numba function returns a tensor of shape (F, K, M)
    sv_rotating = compute_rtf_steering_vector(freqs, rotating_targets, mic_array, mode="near_field")
    
    # Delay-and-Sum weights are simply the conjugated steering vector divided by M
    # Note: Since the core einsum/Numba computes response with conj(weights), 
    # and we want response = |V|^2, we can just use the plain SV divided by M.
    weights_tensor = sv_rotating / M 

    # ------------------------------------------------------------------------
    # C. Precompute Quantized Grid (HIGH RESOLUTION)
    # ------------------------------------------------------------------------
    N_azimuth = 90
    min_dB = -30.0 
    
    print("Precomputing volumetric spatial response...")
    quantized_gain, points, max_dB = precompute_quantized_spatial_response(
        weights=weights_tensor,
        freqs=freqs,
        mic_pos=mic_array,
        source_radius=source_radius,
        N_azimuth=N_azimuth,
        min_dB=min_dB
    )
    print("Computation complete. Building Plotly animation...")

    # ============================================================================
    # D. Plot Opaque Surface with Temporal Slider (Plotly WebGL)
    # ============================================================================
    import plotly.graph_objects as go
    
    f_idx = 0 # Visualize the 1000 Hz bin

    # Helper function to generate surface data for a specific time frame
    def get_frame_surface_data(k_idx):
        q_slice = quantized_gain[f_idx, :, k_idx]
        db_gain_recovered = (q_slice / 255.0) * (max_dB - min_dB) + min_dB
        linear_magnitude = np.clip((db_gain_recovered - min_dB) / (max_dB - min_dB), 0.0, 1.0)
        
        lobe_radii = linear_magnitude * source_radius
        unit_vectors = points / source_radius
        lobe_points = unit_vectors * lobe_radii[:, np.newaxis]

        N_elevation = N_azimuth // 2 + 1
        X = lobe_points[:, 0].reshape((N_elevation, N_azimuth))
        Y = lobe_points[:, 1].reshape((N_elevation, N_azimuth))
        Z = lobe_points[:, 2].reshape((N_elevation, N_azimuth))
        C = linear_magnitude.reshape((N_elevation, N_azimuth))
        
        return X, Y, Z, C

    # 1. Base Figure (Frame 0)
    X0, Y0, Z0, C0 = get_frame_surface_data(0)
    fig = go.Figure()

    # Trace 0: The Volumetric Lobe
    fig.add_trace(go.Surface(
        x=X0, y=Y0, z=Z0, surfacecolor=C0,      
        colorscale='Viridis', cmin=0, cmax=1,      
        showscale=True, opacity=1.0,         
        contours=dict(
            x=dict(show=True, color='white', width=1, size=0.1, highlight=False),
            y=dict(show=True, color='white', width=1, size=0.1, highlight=False),
            z=dict(show=True, color='white', width=1, size=0.1, highlight=False)
        ),
        colorbar=dict(title='Normalized Linear Gain'),
        lighting=dict(ambient=0.7, diffuse=0.5, roughness=0.9, specular=0.1, fresnel=0.1)
    ))

    # Trace 1: The Microphones
    fig.add_trace(go.Scatter3d(
        x=mic_array[:, 0], y=mic_array[:, 1], z=mic_array[:, 2],
        mode='markers', marker=dict(size=4, color='black'), name='Microphones'
    ))

    # Trace 2: The Target Source
    fig.add_trace(go.Scatter3d(
        x=[rotating_targets[0, 0]], y=[rotating_targets[0, 1]], z=[rotating_targets[0, 2]],
        mode='markers', marker=dict(size=8, color='red', symbol='diamond'), name='Target Source'
    ))

    # 2. Generate Frames for Animation
    frames = []
    for k in range(K):
        Xk, Yk, Zk, Ck = get_frame_surface_data(k)
        
        # We must update Trace 0 (Surface) and Trace 2 (Target Scatter)
        frames.append(go.Frame(
            data=[
                go.Surface(x=Xk, y=Yk, z=Zk, surfacecolor=Ck),
                go.Scatter3d(x=mic_array[:, 0], y=mic_array[:, 1], z=mic_array[:, 2]),
                go.Scatter3d(x=[rotating_targets[k, 0]], y=[rotating_targets[k, 1]], z=[rotating_targets[k, 2]])
            ],
            name=str(k)
        ))
    fig.frames = frames

    # 3. Add Slider and Play Button controls
    sliders = [dict(
        steps=[dict(
            method='animate', 
            args=[[str(k)], dict(mode='immediate', frame=dict(duration=100, redraw=True), transition=dict(duration=0))], 
            label=str(k)
        ) for k in range(K)],
        active=0, transition=dict(duration=0),
        x=0, y=0, currentvalue=dict(font=dict(size=14), prefix='Time Frame: ', visible=True, xanchor='left')
    )]
    
    fig.update_layout(
        title=f"Dynamic Beamforming Simulation @ {freqs[f_idx]} Hz",
        sliders=sliders,
        updatemenus=[dict(
            type='buttons', showactive=False, y=0, x=-0.05, xanchor='right', yanchor='top',
            buttons=[
                dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=100, redraw=True), transition=dict(duration=0), fromcurrent=True, mode='immediate')]),
                dict(label='Pause', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate', transition=dict(duration=0))])
            ]
        )],
        scene=dict(
            xaxis=dict(title='X (m)', range=[-source_radius * 1.2, source_radius * 1.2]),
            yaxis=dict(title='Y (m)', range=[-source_radius * 1.2, source_radius * 1.2]),
            zaxis=dict(title='Z (m)', range=[-source_radius * 1.2, source_radius * 1.2]),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.write_html("dynamic_opaque_surface.html", auto_open=True)