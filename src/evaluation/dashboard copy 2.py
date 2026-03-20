import os
import time
import h5py
import numpy as np
import pandas as pd
import streamlit as st
import pyvista as pv
from stpyvista import stpyvista

# Streamlit config
st.set_page_config(page_title="Acoustic Benchmark", layout="wide")

# State variables for the temporal player
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False
if "t_idx" not in st.session_state:
    st.session_state.t_idx = 0

@st.cache_data
def load_benchmark_data(h5_filepath: str) -> dict:
    """
    Load HDF5 benchmark data including spatial precomputed datasets.
    """
    data = {"audio": {}, "metadata": {}, "geometry": {}, "metrics": {}, "spatial": {}, "weights": {}}
    with h5py.File(h5_filepath, 'r') as f:
        for section in ["audio", "metadata", "geometry", "metrics", "weights"]:
            if section in f:
                for key in f[section].keys():
                    data[section][key] = f[section][key][()]
                for attr_key, attr_val in f[section].attrs.items():
                    data[section][attr_key] = attr_val
                    
        # Load precomputed spatial data for PyVista/Trame rendering
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
    Displays the benchmark metrics using Streamlit native columns.
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

def create_trame_plotter_frame(data: dict, proc_name: str, f_idx: int, t_idx: int, user_scale: float) -> pv.Plotter:
    """
    Builds the PyVista Plotter for a specific time frame (t_idx).
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
    array_center = np.mean(mic_coords, axis=0)
    dist_to_source = np.linalg.norm(source_pos[0] - array_center)
    room_dims = geom.get("room_dims", [5.0, 5.0, 3.0])
    max_allowed_radius = np.min(room_dims) / 2.0
    
    lobe_scale = min(dist_to_source, max_allowed_radius) * user_scale
    original_radius = np.linalg.norm(points[0]) 
    unit_vectors = points / original_radius 
    
    # Calculate vertices for the specific time frame
    q_slice = q_gain[f_idx, :, t_idx] 
    db_gain_recovered = (q_slice / 255.0) * (max_dB - min_dB) + min_dB
    linear_magnitude = np.clip((db_gain_recovered - min_dB) / (max_dB - min_dB), 0.0, 1.0)
    
    lobe_radii = linear_magnitude * lobe_scale
    vertices = (unit_vectors * lobe_radii[:, np.newaxis]) + array_center

    # Initialize PyVista Plotter
    plotter = pv.Plotter(window_size=[800, 600], off_screen=True)
    plotter.set_background("black")
    
    # Static Geometry: Room
    Lx, Ly, Lz = room_dims
    room_box = pv.Box(bounds=(0, Lx, 0, Ly, 0, Lz))
    plotter.add_mesh(room_box, style='wireframe', color='gray', line_width=2)
    
    # Static Geometry: Mics and Sources
    plotter.add_mesh(pv.PolyData(mic_coords), color='white', render_points_as_spheres=True, point_size=8)
    plotter.add_mesh(pv.PolyData(source_pos), color='green', render_points_as_spheres=True, point_size=15)
    
    interf = geom.get("interferences_pos", None)
    if interf is not None:
        plotter.add_mesh(pv.PolyData(np.atleast_2d(interf)), color='red', render_points_as_spheres=True, point_size=10)

    # Dynamic Volumetric Mesh
    N_elevation = N_azimuth // 2 + 1
    polar_mesh = pv.StructuredGrid()
    polar_mesh.points = vertices
    polar_mesh.dimensions = (N_azimuth, N_elevation, 1)
    polar_mesh['magnitude'] = linear_magnitude
    
    plotter.add_mesh(
        polar_mesh, 
        scalars='magnitude', 
        cmap='viridis',
        show_edges=True,
        edge_color='white',
        line_width=0.5,
        smooth_shading=True,
        show_scalar_bar=False,
        clim=[0.0, 1.0] 
    )

    plotter.reset_camera()
    return plotter

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
    
    proc_keys = [k.replace("processed_", "") for k in data["weights"].keys()]
    if not proc_keys:
        st.sidebar.warning("No processors found.")
        return
        
    selected_proc = st.sidebar.selectbox("Processor", proc_keys)
    proc_key = f"processed_{selected_proc}"
    weights = data["weights"][proc_key]
    
    F_bins, n_frames, M_mics = weights.shape
    fs = data["metadata"].get("fs", 16000)
    
    nfft = (F_bins - 1) * 2
    freqs = np.fft.rfftfreq(nfft, d=1.0/fs)
    
    standard_1_3 = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000])
    valid_bands = standard_1_3[standard_1_3 <= (fs / 2)]
    freq_indices = [int(np.argmin(np.abs(freqs - f))) for f in valid_bands]
    freq_labels = [f"{freqs[i]:.0f} Hz" for i in freq_indices]
    
    scale_factor = st.sidebar.slider("Visual Scale", 0.1, 2.0, 1.0)
    
    if "audio" in data:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Playback")
        st.sidebar.caption("Reference Mic (Mix):")
        mic_mix = data["audio"].get("mic_signals", np.zeros((1, 100)))[0]
        st.sidebar.audio(np.int16(mic_mix / np.max(np.abs(mic_mix)) * 32767), sample_rate=fs)
        
        if proc_key in data["audio"]:
            st.sidebar.caption(f"Processed ({selected_proc}):")
            y_proc = data["audio"][proc_key]
            st.sidebar.audio(np.int16(y_proc / np.max(np.abs(y_proc)) * 32767), sample_rate=fs)

    # --- MAIN PANEL ---
    if data["metrics"]:
        display_metrics_row(data["metrics"], selected_proc)
        
    st.markdown("---")

    # Lower Controls
    col_freq, col_time, col_play, col_speed = st.columns([2, 5, 1, 1])
    
    with col_freq:
        sel_label = st.selectbox("Frequency", freq_labels, key="f_selector")
        f_idx = freq_indices[freq_labels.index(sel_label)]
        
    with col_time:
        # CRITICAL FIX: Removed key="t_idx". 
        # Using a local variable and assigning the state value prevents Streamlit API exceptions.
        user_t = st.slider("Time Frame", 0, n_frames - 1, value=st.session_state.t_idx)
        
        # Sync user manual sliding with internal state
        if not st.session_state.is_playing and user_t != st.session_state.t_idx:
            st.session_state.t_idx = user_t

    with col_play:
        st.write("") 
        st.write("")
        if st.button("⏯️ Play/Pause"):
            st.session_state.is_playing = not st.session_state.is_playing
            st.rerun()
            
    with col_speed:
        speed_map = {"x1": 1.0, "x2": 2.0, "x4": 4.0}
        speed_str = st.selectbox("Speed", list(speed_map.keys()))
        speed_factor = speed_map[speed_str]

    # Render PyVista/Trame
    # In order to force stpyvista to refresh the iframe when t_idx changes, we make the key dynamic
    plotter = create_trame_plotter_frame(data, selected_proc, f_idx, st.session_state.t_idx, scale_factor)
    stpyvista(plotter, key=f"pv_plotter_{st.session_state.t_idx}")

    # Playback Logic Loop
    if st.session_state.is_playing:
        time.sleep(0.1 / speed_factor)
        if st.session_state.t_idx < n_frames - 1:
            st.session_state.t_idx += 1
        else:
            st.session_state.t_idx = 0
            st.session_state.is_playing = False
        st.rerun()

if __name__ == "__main__":
    main()