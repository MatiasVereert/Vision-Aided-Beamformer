import os
import h5py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Initialize Streamlit configuration
st.set_page_config(page_title="Acoustic Benchmark", layout="wide")

@st.cache_data
def load_benchmark_data(h5_filepath: str) -> dict:
    """
    Load HDF5 benchmark data including spatial precomputed datasets.
    """
    data = {"audio": {}, "metadata": {}, "geometry": {}, "metrics": {}, "spatial": {}, "weights": {}}
    with h5py.File(h5_filepath, 'r') as f:
        # Load main simulation sections
        for section in ["audio", "metadata", "geometry", "metrics", "weights"]:
            if section in f:
                for key in f[section].keys():
                    data[section][key] = f[section][key][()]
                for attr_key, attr_val in f[section].attrs.items():
                    data[section][attr_key] = attr_val
                    
        # Load precomputed spatial matrices for fast rendering
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
    Renders the metric evaluation comparisons in Streamlit columns.
    """
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

def build_native_plotly_animation(data: dict, proc_name: str, f_idx: int, user_scale: float, fps: int = 24) -> go.Figure:
    """
    Constructs a Plotly figure with native JavaScript animation frames.
    This entirely offloads the playback loop to the client's WebGL browser engine,
    bypassing Streamlit's st.rerun() bottleneck.
    """
    spatial_data = data["spatial"][proc_name]
    q_gain = spatial_data["quantized_gain"] 
    points = spatial_data["points"]         
    max_dB = spatial_data["max_dB"]
    min_dB = spatial_data.get("min_dB", -30.0)
    N_azimuth = spatial_data.get("N_azimuth", 90)
    
    geom = data["geometry"]
    mic_coords = np.atleast_2d(geom.get("mic_coords", [[0,0,0]]))
    source_pos = np.atleast_2d(geom.get("source_pos", [[1,1,1]]))
    interf_pos = geom.get("interferences_pos", None)
    
    array_center = np.mean(mic_coords, axis=0)
    dist_to_source = np.linalg.norm(source_pos[0] - array_center)
    room_dims = geom.get("room_dims", [5.0, 5.0, 3.0])
    max_allowed_radius = np.min(room_dims) / 2.0
    
    lobe_scale = min(dist_to_source, max_allowed_radius) * user_scale
    unit_vectors = points / np.linalg.norm(points[0]) 
    N_elevation = N_azimuth // 2 + 1
    
    T = q_gain.shape[2]
    
    # Helper to generate mesh data for a specific time step
    def get_frame_mesh(t_idx):
        q_slice = q_gain[f_idx, :, t_idx] 
        db_gain_recovered = (q_slice / 255.0) * (max_dB - min_dB) + min_dB
        linear_magnitude = np.clip((db_gain_recovered - min_dB) / (max_dB - min_dB), 0.0, 1.0)
        
        lobe_radii = linear_magnitude * lobe_scale
        vertices = (unit_vectors * lobe_radii[:, np.newaxis]) + array_center
        
        X = vertices[:, 0].reshape((N_elevation, N_azimuth))
        Y = vertices[:, 1].reshape((N_elevation, N_azimuth))
        Z = vertices[:, 2].reshape((N_elevation, N_azimuth))
        C = linear_magnitude.reshape((N_elevation, N_azimuth))
        return X, Y, Z, C

    # 1. Base Figure (Time Step 0)
    fig = go.Figure()
    X0, Y0, Z0, C0 = get_frame_mesh(0)
    
    # Trace 0: The dynamic polar pattern surface
    fig.add_trace(go.Surface(
        x=X0, y=Y0, z=Z0, surfacecolor=C0,
        colorscale='Viridis', cmin=0, cmax=1, showscale=False, opacity=1.0, 
        contours=dict(
            x=dict(show=True, color='white', width=1, size=lobe_scale/10),
            y=dict(show=True, color='white', width=1, size=lobe_scale/10),
            z=dict(show=True, color='white', width=1, size=lobe_scale/10)
        ),
        lighting=dict(ambient=0.7, diffuse=0.5, roughness=0.9),
        name="Polar Pattern"
    ))

    # Static Traces: Mics and targets
    fig.add_trace(go.Scatter3d(x=mic_coords[:, 0], y=mic_coords[:, 1], z=mic_coords[:, 2], mode='markers', marker=dict(size=4, color='black'), name='Mics'))
    fig.add_trace(go.Scatter3d(x=source_pos[:, 0], y=source_pos[:, 1], z=source_pos[:, 2], mode='markers', marker=dict(size=8, color='green', symbol='diamond'), name='Target'))
    if interf_pos is not None:
        fig.add_trace(go.Scatter3d(x=interf_pos[:, 0], y=interf_pos[:, 1], z=interf_pos[:, 2], mode='markers', marker=dict(size=6, color='red', symbol='x'), name='Interf'))

    # 2. Build precomputed frames for JS engine
    frames = []
    for t in range(T):
        Xt, Yt, Zt, Ct = get_frame_mesh(t)
        # We explicitly target traces=[0] so Plotly only updates the surface mesh
        frames.append(go.Frame(
            data=[go.Surface(x=Xt, y=Yt, z=Zt, surfacecolor=Ct)],
            traces=[0],
            name=str(t)
        ))
    fig.frames = frames

    # 3. Configure Animation Controls (Play/Pause/Slider)
    frame_duration_ms = int(1000 / fps)
    
    sliders = [dict(
        steps=[dict(
            method='animate', 
            args=[[str(t)], dict(mode='immediate', frame=dict(duration=frame_duration_ms, redraw=True), transition=dict(duration=0))], 
            label=f"{t}"
        ) for t in range(T)],
        active=0, transition=dict(duration=0),
        x=0.05, y=0, currentvalue=dict(font=dict(size=14), prefix='Time Frame: ', visible=True, xanchor='left')
    )]
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X (m)', range=[0, room_dims[0]], showgrid=False, zeroline=False),
            yaxis=dict(title='Y (m)', range=[0, room_dims[1]], showgrid=False, zeroline=False),
            zaxis=dict(title='Z (m)', range=[0, room_dims[2]], showgrid=False, zeroline=False),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=700,
        updatemenus=[dict(
            type='buttons', showactive=False, x=0.0, y=0, xanchor='left', yanchor='top',
            buttons=[
                dict(label='▶ Play', method='animate', args=[None, dict(frame=dict(duration=frame_duration_ms, redraw=True), transition=dict(duration=0), fromcurrent=True, mode='immediate')]),
                dict(label='⏸ Pause', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate', transition=dict(duration=0))])
            ]
        )],
        sliders=sliders
    )
    
    return fig

def main():
    # --- SIDEBAR CONFIGURATION ---
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
    
    proc_keys = [k.replace("processed_", "") for k in data["weights"].keys()]
    if not proc_keys:
        st.sidebar.warning("No processors found.")
        return
        
    selected_proc = st.sidebar.selectbox("Processor", proc_keys)
    proc_key = f"processed_{selected_proc}"
    weights = data["weights"][proc_key]
    
    fs = data["metadata"].get("fs", 16000)
    F_bins = weights.shape[0]
    nfft = (F_bins - 1) * 2
    freqs = np.fft.rfftfreq(nfft, d=1.0/fs)
    
    standard_1_3 = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000])
    valid_bands = standard_1_3[standard_1_3 <= (fs / 2)]
    freq_indices = [int(np.argmin(np.abs(freqs - f))) for f in valid_bands]
    freq_labels = [f"{freqs[i]:.0f} Hz" for i in freq_indices]
    
    scale_factor = st.sidebar.slider("Visual Scale", 0.1, 2.0, 1.0)
    playback_speed = st.sidebar.select_slider("Playback FPS", options=[12, 24, 48, 60], value=24)
    
    if "audio" in data:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Audio Output")
        st.sidebar.caption("Reference Mic (Mix):")
        mic_mix = data["audio"].get("mic_signals", np.zeros((1, 100)))[0]
        st.sidebar.audio(np.int16(mic_mix / np.max(np.abs(mic_mix)) * 32767), sample_rate=fs)
        
        if proc_key in data["audio"]:
            st.sidebar.caption(f"Processed ({selected_proc}):")
            y_proc = data["audio"][proc_key]
            st.sidebar.audio(np.int16(y_proc / np.max(np.abs(y_proc)) * 32767), sample_rate=fs)

    # --- MAIN CONTENT AREA ---
    if data["metrics"]:
        display_metrics_row(data["metrics"], selected_proc)
        
    st.markdown("---")
    
    col_freq, _ = st.columns([1, 3])
    with col_freq:
        sel_label = st.selectbox("Frequency Band", freq_labels, key="f_selector")
        f_idx = freq_indices[freq_labels.index(sel_label)]

    # Render native Plotly animation (Zero Streamlit reruns!)
    fig = build_native_plotly_animation(data, selected_proc, f_idx, scale_factor, fps=playback_speed)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()