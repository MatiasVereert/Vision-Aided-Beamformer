import os
import h5py
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# Configure page to maximize screen space
st.set_page_config(page_title="Acoustic Benchmark", layout="wide")

@st.cache_data
def load_benchmark_tree(h5_filepath: str) -> dict:
    """
    Scans the HDF5 file to build the hierarchical tree for the sidebar dropdowns
    without loading the heavy tensors into memory.
    """
    tree = {}
    try:
        with h5py.File(h5_filepath, 'r') as f:
            for proc in f.keys():
                tree[proc] = {}
                for met in f[proc].keys():
                    tree[proc][met] = list(f[proc][met].keys())
    except Exception as e:
        st.error(f"Error reading file structure: {e}")
    return tree

@st.cache_data
def load_benchmark_case(h5_filepath: str, path: str) -> dict:
    """
    Loads all data (audio, spatial, metrics) strictly for the selected case path.
    """
    data = {"audio": {}, "metadata": {}, "geometry": {}, "metrics": {}, "spatial": {}, "weights": {}}
    
    with h5py.File(h5_filepath, 'r') as f:
        if path not in f:
            return data
            
        target = f[path]
        
        # Load standard structured sections
        for section in ["audio", "metadata", "geometry", "metrics", "weights"]:
            if section in target:
                for key in target[section].keys():
                    data[section][key] = target[section][key][()]
                for attr_key, attr_val in target[section].attrs.items():
                    data[section][attr_key] = attr_val
        
        # Discover and load the spatial group specific to this processor
        proc_name = path.split('/')[0]
        spat_key = f"spatial_{proc_name}"
        
        if spat_key in target:
            data["spatial"][proc_name] = {}
            for dset_key in target[spat_key].keys():
                data["spatial"][proc_name][dset_key] = target[spat_key][dset_key][()]
            for attr_key, attr_val in target[spat_key].attrs.items():
                data["spatial"][proc_name][attr_key] = attr_val
                    
    return data

def display_metrics_row(metrics_dict: dict, processor_name: str):
    """
    Displays metrics using Streamlit native metric columns for a cleaner UI.
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
    
    base = {k.replace("base_", ""): v for k, v in metrics_dict.items() if "base_" in k}
    proc = {k.replace(f"proc_", ""): v for k, v in metrics_dict.items() if "proc_" in k}
    delta = {k.replace(f"Delta_", ""): v for k, v in metrics_dict.items() if "Delta_" in k}
    
    keys = list(base.keys())
    if not keys: return
    
    cols = st.columns(len(keys))
    for idx, key in enumerate(keys):
        with cols[idx]:
            val = proc.get(key, np.nan)
            d_val = delta.get(key, np.nan)
            
            # Formateo condicional para strings como "NaN"
            if isinstance(val, str) or isinstance(d_val, str) or np.isnan(val):
                st.metric(label=key, value="NaN", delta="NaN")
            else:
                # Invert delta color for Cepstral Distance (lower is better)
                if key == "CD":
                    st.metric(label=key, value=f"{val:.3f}", delta=f"{d_val:.3f}", delta_color="inverse")
                else:
                    st.metric(label=key, value=f"{val:.3f}", delta=f"{d_val:.3f}")

def get_room_wireframe_traces(room_dims: list, **kwargs) -> list:
    """Generates wireframe box traces."""
    lx, ly, lz = room_dims
    vertices = np.array([
        [0, 0, 0], [lx, 0, 0], [lx, ly, 0], [0, ly, 0],
        [0, 0, lz], [lx, 0, lz], [lx, ly, lz], [0, ly, lz]
    ])
    
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    x_coords, y_coords, z_coords = [], [], []
    for p1_idx, p2_idx in edges:
        p1 = vertices[p1_idx]
        p2 = vertices[p2_idx]
        x_coords.extend([p1[0], p2[0], None]) 
        y_coords.extend([p1[1], p2[1], None])
        z_coords.extend([p1[2], p2[2], None])

    lines = go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords, mode='lines',
        line=dict(color='grey', width=2), showlegend=False, name='Room', **kwargs
    )
    return [lines]

def build_native_plotly_animation(data: dict, proc_name: str, f_idx: int, user_scale: float, decimator: int) -> go.Figure:
    """Constructs a Plotly figure with native JavaScript animation frames."""
    spatial_data = data["spatial"][proc_name]
    q_gain = spatial_data["quantized_gain"]
    points = spatial_data["points"]
    
    max_db_per_frame = spatial_data.get("max_db_per_frame")
    if max_db_per_frame is None:
        num_freqs = q_gain.shape[0]
        num_frames = q_gain.shape[2]
        global_max_db = spatial_data.get("max_dB", 0)
        max_db_per_frame = np.full((num_freqs, num_frames), global_max_db)
        
    min_dB = spatial_data.get("min_dB", -30.0)
    n_azimuth = spatial_data.get("N_azimuth", 90)
    target_fps = spatial_data.get("target_fps", 24)

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

    def get_frame_mesh(t_idx, max_db_for_frame):
        q_slice = q_gain[f_idx, :, t_idx]
        normalized_gain = q_slice / 255.0
        db_gain_recovered = normalized_gain * (max_db_for_frame - min_dB) + min_dB
        linear_magnitude = np.clip((db_gain_recovered - min_dB) / (max_db_for_frame - min_dB + 1e-9), 0.0, 1.0)

        lobe_radii = linear_magnitude * lobe_scale
        vertices = (unit_vectors * lobe_radii[:, np.newaxis]) + array_center

        X = vertices[:, 0].reshape((n_elevation, n_azimuth))
        Y = vertices[:, 1].reshape((n_elevation, n_azimuth))
        Z = vertices[:, 2].reshape((n_elevation, n_azimuth))
        C = linear_magnitude.reshape((n_elevation, n_azimuth))
        return X, Y, Z, C

    fig = go.Figure()
    X0, Y0, Z0, C0 = get_frame_mesh(0, max_db_per_frame[f_idx, 0])

    for trace in get_room_wireframe_traces(room_dims):
        fig.add_trace(trace)

    fig.add_trace(go.Surface(
        uid='dynamic-surface', x=X0, y=Y0, z=Z0, surfacecolor=C0,
        colorscale='Viridis', cmin=0, cmax=1, showscale=False, opacity=1.0,
        contours=dict(x=dict(show=False), y=dict(show=False), z=dict(show=False)),
        lighting=dict(ambient=0.7, diffuse=0.5, roughness=0.9, specular=0.2, fresnel=0.2),
        name="Polar Pattern"
    ))

    fig.add_trace(go.Scatter3d(x=mic_coords[:, 0], y=mic_coords[:, 1], z=mic_coords[:, 2], mode='markers', marker=dict(size=4, color='black'), name='Mics'))
    fig.add_trace(go.Scatter3d(x=source_pos[:, 0], y=source_pos[:, 1], z=source_pos[:, 2], mode='markers', marker=dict(size=8, color='green', symbol='diamond'), name='Target'))
    if interf_pos is not None:
        fig.add_trace(go.Scatter3d(x=interf_pos[:, 0], y=interf_pos[:, 1], z=interf_pos[:, 2], mode='markers', marker=dict(size=6, color='red', symbol='x'), name='Interf'))

    frames = []
    frame_names = [str(t) for t in time_indices]
    for t_idx, frame_name in zip(time_indices, frame_names):
        Xt, Yt, Zt, Ct = get_frame_mesh(t_idx, max_db_per_frame[f_idx, t_idx])
        frames.append(go.Frame(
            data=[{'uid': 'dynamic-surface', 'type': 'surface', 'x': Xt, 'y': Yt, 'z': Zt, 'surfacecolor': Ct}],
            traces=[1], name=frame_name
        ))
    fig.frames = frames

    slider_steps = []
    time_step_s = 0.5 
    
    for t_idx in time_indices:
        time_sec = t_idx / target_fps
        label = f"{time_sec:.2f}s" if (time_sec % time_step_s < (1/target_fps)) else ""
        slider_steps.append(dict(
            method='animate',
            args=[[str(t_idx)], dict(mode='immediate', frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
            label=label
        ))

    sliders = [dict(
        steps=slider_steps, active=0, transition=dict(duration=0),
        x=0.05, y=0, len=0.9, currentvalue=dict(font=dict(size=14), prefix='Time: ', visible=True, xanchor='right')
    )]

    updatemenus = [dict(
        type='buttons', showactive=False, y=0, x=-0.05, xanchor='right', yanchor='top',
        buttons=[
            dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=100, redraw=True), transition=dict(duration=0), fromcurrent=True, mode='immediate')]),
            dict(label='Pause', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate', transition=dict(duration=0))])
        ]
    )]

    fig.update_layout(
        scene=dict( 
            xaxis=dict(title='X (m)', range=[0, room_dims[0]], showgrid=False, zeroline=False, backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(title='Y (m)', range=[0, room_dims[1]], showgrid=False, zeroline=False, backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(title='Z (m)', range=[0, room_dims[2]], showgrid=False, zeroline=False, backgroundcolor="rgba(0,0,0,0)"),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0), height=700, sliders=sliders, updatemenus=updatemenus, uirevision='constant_view'
    )

    return fig

def main():
    st.sidebar.title("Acoustic Benchmark")
    st.sidebar.markdown("---")
    
    # 1. SETUP DE DIRECTORIO
    results_dir = r"tests\dataset_out\Exp_B" 
    if not os.path.exists(results_dir):
        st.sidebar.error(f"Directory not found: {results_dir}")
        return
        
    h5_files = [f for f in os.listdir(results_dir) if f.endswith('.h5')]
    if not h5_files:
        st.sidebar.warning("No HDF5 files found.")
        return
        
    selected_file = st.sidebar.selectbox("Select Master File", h5_files)
    h5_path = os.path.join(results_dir, selected_file)
    
    # 2. CARGAR EL ÁRBOL JERÁRQUICO
    tree = load_benchmark_tree(h5_path)
    if not tree:
        st.sidebar.warning("Empty or invalid file structure.")
        return

    # 3. SELECTORES EN CASCADA
    st.sidebar.markdown("### Seleccione el Caso")
    sel_proc = st.sidebar.selectbox("Procesador", list(tree.keys()))
    
    metrics_available = list(tree[sel_proc].keys())
    sel_met = st.sidebar.selectbox("Métrica Analizada", metrics_available)
    
    cases_available = tree[sel_proc][sel_met]
    sel_case = st.sidebar.selectbox("Récord a Visualizar", cases_available)
    
    # Construir el path exacto y cargar los datos
    case_path = f"{sel_proc}/{sel_met}/{sel_case}"
    data = load_benchmark_case(h5_path, case_path)
    
    if "spatial" not in data or sel_proc not in data["spatial"]:
        st.sidebar.warning("Datos espaciales no encontrados para este caso.")
        return

    spatial_data = data["spatial"][sel_proc]
    fs = data["metadata"].get("fs", 16000)
    total_frames = spatial_data["quantized_gain"].shape[2]

    st.sidebar.markdown("---")
    st.sidebar.subheader("Animation Controls")
    scale_factor = st.sidebar.slider("Visual Scale", 0.1, 2.0, 1.0)
    frame_decimator = st.sidebar.slider("Frame Decimator", min_value=1, max_value=max(1, total_frames // 10), value=1)

    if "audio" in data:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Playback")
        st.sidebar.caption("Reference Mic (Mix):")
        mic_mix = data["audio"].get("mic_signals", np.zeros((1, 100)))[0]
        st.sidebar.audio(np.int16(mic_mix / np.max(np.abs(mic_mix)) * 32767), sample_rate=fs)
        
        proc_key = f"processed_{sel_proc}"
        if proc_key in data["audio"]:
            st.sidebar.caption(f"Processed ({sel_proc}):")
            y_proc = data["audio"][proc_key]
            st.sidebar.audio(np.int16(y_proc / np.max(np.abs(y_proc)) * 32767), sample_rate=fs)

    # --- MAIN PANEL ---
    st.markdown(f"### Visualizando Récord: `{sel_case}` basado en `{sel_met}`")
    st.markdown(f"**Escenario Acústico:** RT60 = {data['metadata'].get('rt60', 'N/A')}s | iSIR = {data['metadata'].get('isir_db', 'N/A')} dB | Mics = {data['metadata'].get('M', 'N/A')} | WPE = {data['metadata'].get('use_wpe', 'N/A')}")
    
    if data["metrics"]:
        display_metrics_row(data["metrics"], sel_proc)
        
    st.markdown("---")

    freqs = spatial_data.get("freqs", [1000.0])
    freq_labels = [f"{f:.0f} Hz" for f in freqs]

    # --- FREQUENCY CONTROL ---
    col_freq, _ = st.columns([1, 3])
    with col_freq:
        default_f_idx = len(freqs) // 2
        sel_label = st.selectbox("Frequency Band", freq_labels, index=default_f_idx)
        f_idx = freq_labels.index(sel_label)

    # --- 3D RENDER ---
    fig = build_native_plotly_animation(data, sel_proc, f_idx, scale_factor, decimator=frame_decimator)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()