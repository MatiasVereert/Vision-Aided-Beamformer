import os
import h5py
import numpy as np
import streamlit as st
import plotly.graph_objects as go
# Configure page to maximize screen space
st.set_page_config(page_title="Acoustic Benchmark", layout="wide")

@st.cache_data
def load_benchmark_data(h5_filepath: str) -> dict:
    """
    Reads the HDF5 file and extracts all relevant data into a nested dictionary.
    Cached by Streamlit to prevent disk reads on every UI interaction. 
    """
    data = {"audio": {}, "metadata": {}, "geometry": {}, "metrics": {}, "spatial": {}, "weights": {}}
    
    with h5py.File(h5_filepath, 'r') as f:
        # Load standard structured sections
        for section in ["audio", "metadata", "geometry", "metrics", "weights"]:
            if section in f:
                for key in f[section].keys():
                    data[section][key] = f[section][key][()]
                for attr_key, attr_val in f[section].attrs.items():
                    data[section][attr_key] = attr_val
        
        # Discover and load all precomputed spatial groups
        for key in f.keys():
            if key.startswith("spatial_"):
                proc_name = key.replace("spatial_", "")
                data["spatial"][proc_name] = {}
                for dset_key in f[key].keys():
                    data["spatial"][proc_name][dset_key] = f[key][dset_key][()]
                for attr_key, attr_val in f[key].attrs.items():
                    data["spatial"][proc_name][attr_key] = attr_val
                    
    return data

def display_metrics_row(metrics_dict: dict, processor_name: str):
    """
    Displays metrics using Streamlit native metric columns for a cleaner UI.
    Injects CSS to reduce font sizes for a more compact view.
    """
    st.markdown("""
    <style>
    /* Reduce font size and padding for Streamlit's metric widget */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem;
        padding-bottom: 0.1rem;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.8rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    base = {k.replace("baseline_", ""): v for k, v in metrics_dict.items() if "baseline" in k}
    proc = {k.replace(f"{processor_name}_", ""): v for k, v in metrics_dict.items() if processor_name in k and "Delta" not in k}
    delta = {k.replace(f"{processor_name}_Delta_", ""): v for k, v in metrics_dict.items() if "Delta" in k}
    
    keys = list(base.keys())
    if not keys: return
    
    cols = st.columns(len(keys))
    for idx, key in enumerate(keys):
        with cols[idx]:
            val = proc.get(key, np.nan)
            d_val = delta.get(key, np.nan)
            st.metric(label=key, value=f"{val:.3f}", delta=f"{d_val:.3f}")

def get_room_wireframe_traces(room_dims: list, **kwargs) -> list:
    """
    Generates a list of Plotly Scatter3d traces to draw a correct wireframe box.
    Defines 8 vertices and connects them with 12 edge lines.
    """
    lx, ly, lz = room_dims
    # Define the 8 vertices of the cube
    vertices = np.array([
        [0, 0, 0], [lx, 0, 0], [lx, ly, 0], [0, ly, 0],
        [0, 0, lz], [lx, 0, lz], [lx, ly, lz], [0, ly, lz]
    ])
    
    # Define the 12 edges connecting the vertices
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4), # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)  # Vertical edges
    ]

    x_coords, y_coords, z_coords = [], [], []
    for p1_idx, p2_idx in edges:
        p1 = vertices[p1_idx]
        p2 = vertices[p2_idx]
        x_coords.extend([p1[0], p2[0], None]) # 'None' creates a break in the line
        y_coords.extend([p1[1], p2[1], None])
        z_coords.extend([p1[2], p2[2], None])

    lines = go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords,
        mode='lines',
        line=dict(color='grey', width=2),
        showlegend=False,
        name='Room',
        **kwargs
    )
    return [lines]

def build_native_plotly_animation(data: dict, proc_name: str, f_idx: int, user_scale: float, decimator: int) -> go.Figure:
    """
    Constructs a Plotly figure with native JavaScript animation frames.
    This entirely offloads the playback loop to the client's WebGL browser engine,
    bypassing Streamlit's st.rerun() bottleneck and preserving camera state.
    """
    spatial_data = data["spatial"][proc_name]
    q_gain = spatial_data["quantized_gain"]
    points = spatial_data["points"]
    # Use per-frame max dB if available, otherwise fall back to the old global max_dB attribute
    max_db_per_frame = spatial_data.get("max_db_per_frame")
    if max_db_per_frame is None:
        # Legacy fallback
        # Legacy fallback: Create a compatible 2D array from the global max_dB attribute
        num_freqs = q_gain.shape[0]
        num_frames = q_gain.shape[2]
        global_max_db = spatial_data.get("max_dB", 0)
        max_db_per_frame = np.full((num_freqs, num_frames), global_max_db)
        
    min_dB = spatial_data.get("min_dB", -30.0)
    n_azimuth = spatial_data.get("N_azimuth", 90)
    target_fps = spatial_data.get("target_fps", 24) # Fallback to 24 if not in H5

    # Geometry extraction
    geom = data["geometry"]
    mic_coords = np.atleast_2d(geom.get("mic_coords", [[0, 0, 0]]))
    source_pos = np.atleast_2d(geom.get("source_pos", [[1, 1, 1]]))
    interf_pos = geom.get("interferences_pos", None)
    room_dims = geom.get("room_dims", [5.0, 5.0, 3.0])

    array_center = np.mean(mic_coords, axis=0)
    dist_to_source = np.linalg.norm(source_pos[0] - array_center)
    max_allowed_radius = np.min(room_dims) / 2.0
    lobe_scale = min(dist_to_source, max_allowed_radius) * user_scale

    unit_vectors = points / np.linalg.norm(points[0]) 
    n_elevation = n_azimuth // 2 + 1
    T = q_gain.shape[2]
    time_indices = range(0, T, decimator)

    # Helper to generate mesh data for a specific time step
    def get_frame_mesh(t_idx, max_db_for_frame):
        q_slice = q_gain[f_idx, :, t_idx]
        
        # This formula reverses the PER-FRAME quantization process
        # 1. De-quantize from uint8 to a normalized float [0, 1]
        normalized_gain = q_slice / 255.0
        # 2. Rescale back to the original clipped dB range FOR THIS FRAME
        db_gain_recovered = normalized_gain * (max_db_for_frame - min_dB) + min_dB
        # 3. Re-normalize the dB gain to a [0, 1] linear scale for coloring and sizing the lobe.
        #    Because we use the frame's own max_db, the peak will always be 1.0.
        linear_magnitude = np.clip((db_gain_recovered - min_dB) / (max_db_for_frame - min_dB + 1e-9), 0.0, 1.0)

        lobe_radii = linear_magnitude * lobe_scale
        vertices = (unit_vectors * lobe_radii[:, np.newaxis]) + array_center

        X = vertices[:, 0].reshape((n_elevation, n_azimuth))
        Y = vertices[:, 1].reshape((n_elevation, n_azimuth))
        Z = vertices[:, 2].reshape((n_elevation, n_azimuth))
        C = linear_magnitude.reshape((n_elevation, n_azimuth))
        return X, Y, Z, C

    # 1. Base Figure (Time Step 0)
    fig = go.Figure()
    X0, Y0, Z0, C0 = get_frame_mesh(0, max_db_per_frame[f_idx, 0])

    # Trace 0: Room Wireframe (static)
    for trace in get_room_wireframe_traces(room_dims):
        fig.add_trace(trace)

    # Trace 1: The dynamic polar pattern surface (Optimized for animation)
    fig.add_trace(go.Surface(
        uid='dynamic-surface', # Add a unique ID for object constancy
        x=X0, y=Y0, z=Z0, surfacecolor=C0,
        colorscale='Viridis', cmin=0, cmax=1, showscale=False, opacity=1.0,
        contours=dict(
            # Disable dynamic contours to prevent CPU bottleneck and visual lag
            x=dict(show=False),
            y=dict(show=False),
            z=dict(show=False)
        ),
        # Enhance lighting parameters to define the 3D volume without needing wireframes
        lighting=dict(ambient=0.7, diffuse=0.5, roughness=0.9, specular=0.2, fresnel=0.2),
        name="Polar Pattern"
    ))

    # Static Traces: Mics and targets
    fig.add_trace(go.Scatter3d(x=mic_coords[:, 0], y=mic_coords[:, 1], z=mic_coords[:, 2], mode='markers', marker=dict(size=4, color='black'), name='Mics'))
    fig.add_trace(go.Scatter3d(x=source_pos[:, 0], y=source_pos[:, 1], z=source_pos[:, 2], mode='markers', marker=dict(size=8, color='green', symbol='diamond'), name='Target'))
    if interf_pos is not None:
        fig.add_trace(go.Scatter3d(x=interf_pos[:, 0], y=interf_pos[:, 1], z=interf_pos[:, 2], mode='markers', marker=dict(size=6, color='red', symbol='x'), name='Interf'))

    # 2. Build precomputed frames for JS engine
    frames = []
    # Create an explicit list of frame names to handle decimation correctly
    frame_names = [str(t) for t in time_indices]
    for t_idx, frame_name in zip(time_indices, frame_names):
        Xt, Yt, Zt, Ct = get_frame_mesh(t_idx, max_db_per_frame[f_idx, t_idx])
        # By passing a dict instead of a full go.Surface object, we only update the
        # data arrays, which is much faster and compatible with redraw=False. We must
        # explicitly provide the 'type' for Plotly to correctly interpret the dict.
        frames.append(go.Frame(
            data=[{'uid': 'dynamic-surface', 'type': 'surface', 'x': Xt, 'y': Yt, 'z': Zt, 'surfacecolor': Ct}],
            traces=[1], # This MUST match the index of the dynamic surface trace
            name=frame_name
        ))
    fig.frames = frames

    # 3. Configure Animation Controls (Play/Pause/Slider)
    # Create slider steps based on time in seconds, not frame indices
    slider_steps = []
    time_step_s = 0.5 # Show a mark every half second
    
    for t_idx in time_indices:
        time_sec = t_idx / target_fps
        # Add a label only for major time steps
        label = f"{time_sec:.2f}s" if (time_sec % time_step_s < (1/target_fps)) else ""
        
        slider_steps.append(dict(
            method='animate',
            # Force WebGL redraw to update the 3D surface.
            args=[[str(t_idx)], dict(mode='immediate', frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
            label=label
        ))

    sliders = [dict(
        steps=slider_steps,
        active=0, 
        transition=dict(duration=0),
        x=0.05, y=0, len=0.9, 
        currentvalue=dict(font=dict(size=14), prefix='Time: ', visible=True, xanchor='right')
    )]

    # 4. Configure Play and Pause buttons
    # The 'fromcurrent=True' argument ensures playback starts from the active slider position.
    # The 'redraw=True' parameter in the Play button is mandatory for WebGL surfaces.
    updatemenus = [dict(
        type='buttons',
        showactive=False,
        y=0,
        x=-0.05,
        xanchor='right',
        yanchor='top',
        buttons=[
            dict(
                label='Play',
                method='animate',
                args=[None, dict(frame=dict(duration=100, redraw=True), transition=dict(duration=0), fromcurrent=True, mode='immediate')]
            ),
            dict(
                label='Pause',
                method='animate',
                args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate', transition=dict(duration=0))]
            )
        ]
    )]

    fig.update_layout(
        scene=dict( 
            xaxis=dict(title='X (m)', range=[0, room_dims[0]], showgrid=False, zeroline=False, backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(title='Y (m)', range=[0, room_dims[1]], showgrid=False, zeroline=False, backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(title='Z (m)', range=[0, room_dims[2]], showgrid=False, zeroline=False, backgroundcolor="rgba(0,0,0,0)"),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=700,
        sliders=sliders,
        updatemenus=updatemenus, # Inject the play/pause buttons into the layout
        uirevision='constant_view' # This preserves UI state (like camera) during data updates, fixing the animation.
    )

    return fig

def main():
    # --- SIDEBAR ---
    st.sidebar.title("Acoustic Benchmark")
    st.sidebar.markdown("---")
    
    results_dir = r"tests/data/benchmark_results" 
    if not os.path.exists(results_dir):
        st.sidebar.error(f"Directory not found: {results_dir}")
        return
        
    h5_files = [f for f in os.listdir(results_dir) if f.endswith('.h5')]
    if not h5_files:
        st.sidebar.warning("No HDF5 files found.")
        return
        
    selected_file = st.sidebar.selectbox("Select File", h5_files)
    data = load_benchmark_data(os.path.join(results_dir, selected_file))
    
    available_processors = list(data["spatial"].keys())
    if not available_processors:
        st.sidebar.warning("No spatial data found.")
        return
        
    selected_proc = st.sidebar.selectbox("Processor", available_processors)
    spatial_data = data["spatial"][selected_proc]
    fs = data["metadata"].get("fs", 16000)
    total_frames = spatial_data["quantized_gain"].shape[2]

    st.sidebar.markdown("---")
    st.sidebar.subheader("Animation Controls")
    scale_factor = st.sidebar.slider("Visual Scale", 0.1, 2.0, 1.0)
    frame_decimator = st.sidebar.slider("Frame Decimator", min_value=1, max_value=max(1, total_frames // 10), value=1, help="Higher values skip more frames to improve performance. '1' means no skipping.")

    if "audio" in data:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Playback")
        st.sidebar.caption("Reference Mic (Mix):")
        mic_mix = data["audio"].get("mic_signals", np.zeros((1, 100)))[0]
        st.sidebar.audio(np.int16(mic_mix / np.max(np.abs(mic_mix)) * 32767), sample_rate=fs)
        
        proc_key = f"processed_{selected_proc}"
        if proc_key in data["audio"]:
            st.sidebar.caption(f"Processed ({selected_proc}):")
            y_proc = data["audio"][proc_key]
            st.sidebar.audio(np.int16(y_proc / np.max(np.abs(y_proc)) * 32767), sample_rate=fs)

    # --- MAIN PANEL ---
    if data["metrics"]:
        display_metrics_row(data["metrics"], selected_proc)
        
    st.markdown("---")

    freqs = spatial_data.get("freqs", [1000.0])
    freq_labels = [f"{f:.0f} Hz" for f in freqs]

    # --- FREQUENCY CONTROL ---
    col_freq, _ = st.columns([1, 3])
    with col_freq:
        # Default to a mid-range frequency
        default_f_idx = len(freqs) // 2
        sel_label = st.selectbox("Frequency Band", freq_labels, index=default_f_idx)
        f_idx = freq_labels.index(sel_label)

    # --- 3D RENDER ---
    fig = build_native_plotly_animation(data, selected_proc, f_idx, scale_factor, decimator=frame_decimator)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()  