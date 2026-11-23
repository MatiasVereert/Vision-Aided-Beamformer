import os
import re
import h5py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Configure page to maximize screen space
st.set_page_config(page_title="Acoustic Benchmark", layout="wide")

# CSS to compact Streamlit's metric widgets (injected once in main)
METRIC_CSS = """
<style>
[data-testid="stMetricValue"] { font-size: 1.5rem; }
[data-testid="stMetricLabel"] { font-size: 0.8rem; padding-bottom: 0.1rem; }
[data-testid="stMetricDelta"] { font-size: 0.8rem; }
</style>
"""

def prep_audio(sig):
    """
    Normalize a signal to int16 for Streamlit playback.
    Handles both 1D and 2D arrays (takes the first channel if 2D).
    """
    if isinstance(sig, np.ndarray) and len(sig.shape) > 1:
        sig = sig[0]  # Take the first channel if it is a 2D array
    max_val = np.max(np.abs(sig))
    if max_val == 0:
        return np.int16(sig)
    return np.int16(sig / max_val * 32767)

@st.cache_data
def get_h5_structure(h5_filepath: str, mtime: float) -> dict:
    """
    Scans the HDF5 file to build a nested dictionary of available paths:
    { Processor : { Metric : [Cases] } }
    This drives the cascaded dropdown menus in the sidebar.
    """
    structure = {}
    with h5py.File(h5_filepath, 'r') as f:
        for proc in f.keys():
            structure[proc] = {}
            for metric in f[proc].keys():
                structure[proc][metric] = list(f[proc][metric].keys())
    return structure

def _decode_attr(v):
    """Inverse of the benchmark's _h5_attr: 'None' -> None, 'NaN' -> nan,
    numpy scalars -> python."""
    if isinstance(v, bytes):
        v = v.decode()
    if isinstance(v, str):
        if v == "None":
            return None
        if v == "NaN":
            return np.nan
        return v
    if isinstance(v, np.generic):
        return v.item()
    return v

@st.cache_data
def load_benchmark_case(h5_filepath: str, proc_name: str, metric_name: str, case_type: str, mtime: float) -> dict:
    """
    Reads a specific case path from the HDF5 file and extracts the data.
    Cached by Streamlit to prevent disk reads on every UI interaction.
    """
    data = {"audio": {}, "metadata": {}, "geometry": {}, "metrics": {}, "spatial": {}}
    case_path = f"{proc_name}/{metric_name}/{case_type}"

    with h5py.File(h5_filepath, 'r') as f:
        if case_path not in f:
            return data

        grp = f[case_path]

        # Load standard structured groups
        for section in ["audio", "metadata", "geometry"]:
            if section in grp:
                for key in grp[section].keys():
                    data[section][key] = grp[section][key][()]
                for attr_key, attr_val in grp[section].attrs.items():
                    data[section][attr_key] = attr_val

        # Load metrics from attributes
        if "metrics" in grp:
            for k, v in grp["metrics"].attrs.items():
                # Handle NaN strings saved by the benchmark script
                if isinstance(v, str) and v == "NaN":
                    data["metrics"][k] = np.nan
                else:
                    data["metrics"][k] = v

        # OPTIONAL groups written only by measured-dataset runs (Kria). MIRD
        # catalogs do not have them, and then data["scene"] / data["rirs"]
        # stay empty and nothing below changes.
        if "scene" in grp:
            data["scene"] = {k: _decode_attr(v) for k, v in grp["scene"].attrs.items()}
        if "rirs" in grp:
            data["rirs"] = {k: grp["rirs"][k][()] for k in grp["rirs"].keys()}
            data["rirs"].update({k: v for k, v in grp["rirs"].attrs.items()})

        # Load spatial group for the specific processor
        spat_key = f"spatial_{proc_name}"
        if spat_key in grp:
            data["spatial"][proc_name] = {}
            for dset_key in grp[spat_key].keys():
                data["spatial"][proc_name][dset_key] = grp[spat_key][dset_key][()]
            for attr_key, attr_val in grp[spat_key].attrs.items():
                data["spatial"][proc_name][attr_key] = attr_val

    return data

def display_metrics_row(metrics_dict: dict):
    """
    Displays metrics dynamically, adapting to the new prefix-based naming
    convention and multiple ground truth references (e.g., anechoic, early).
    """
    # Extract unique metric names by stripping the 'base_' prefix (CD excluded)
    base_keys = [k for k in metrics_dict.keys() if k.startswith("base_") and "CD" not in k]
    all_metric_names = [k.replace("base_", "") for k in base_keys]

    if not all_metric_names: return

    # Extract unique reference ground truths (e.g., 'early', 'anechoic')
    references = list(set([m.split('_')[-1] for m in all_metric_names if '_' in m]))
    selected_ref = ""
    metric_names = all_metric_names

    if references:
        selected_ref = st.selectbox("Ground Truth Reference for Metrics", sorted(references))
        # Filter metric names for the selected reference
        metric_names = [m for m in all_metric_names if m.endswith(f"_{selected_ref}")]

    if not metric_names: return

    # Row 1: Beamformer Performance vs Baseline
    st.markdown("### Spatial Processing Performance")
    cols_bf = st.columns(len(metric_names))
    for idx, m in enumerate(metric_names):
        with cols_bf[idx]:
            display_name = m.replace(f"_{selected_ref}", "") if selected_ref else m
            val = metrics_dict.get(f"proc_{m}", np.nan)
            delta = metrics_dict.get(f"Delta_tot_{m}", np.nan)
            st.metric(label=f"{display_name} (BF)", value=f"{val:.3f}", delta=f"{delta:.3f}")

    # Row 2: Standalone Neural Performance (DTLN Alone) vs Baseline
    has_dtln_alone = any(k.startswith("dtln_alone_") for k in metrics_dict.keys())
    if has_dtln_alone:
        st.markdown("### Standalone DTLN Performance (Single-Mic)")
        cols_alone = st.columns(len(metric_names))
        for idx, m in enumerate(metric_names):
            with cols_alone[idx]:
                display_name = m.replace(f"_{selected_ref}", "") if selected_ref else m
                val = metrics_dict.get(f"dtln_alone_{m}", np.nan)
                base_val = metrics_dict.get(f"base_{m}", np.nan)

                # Compute total absolute delta improvement against raw baseline
                delta_tot_alone = val - base_val if not np.isnan(val) and not np.isnan(base_val) else np.nan

                st.metric(label=f"{display_name} (DTLN Alone)", value=f"{val:.3f}", delta=f"{delta_tot_alone:.3f}" if not np.isnan(delta_tot_alone) else None)

    # Row 3: Full Pipeline (including DTLN) vs Baseline, if available
    has_dtln = any(k.startswith("dtln_post_") for k in metrics_dict.keys())
    if has_dtln:
        st.markdown("### Full Pipeline Performance (BF + DTLN)")
        cols_dnn = st.columns(len(metric_names))
        for idx, m in enumerate(metric_names):
            with cols_dnn[idx]:
                display_name = m.replace(f"_{selected_ref}", "") if selected_ref else m
                val = metrics_dict.get(f"dtln_post_{m}", np.nan)
                delta = metrics_dict.get(f"Delta_tot_pipeline_{m}", np.nan)

                st.metric(label=f"{display_name} (Pipeline)", value=f"{val:.3f}", delta=f"{delta:.3f}")

# Human-readable labels for known metadata keys (unknown keys fall back to Title Case).
_META_LABELS = {
    "rt60": ("RT60", "s"),
    "isir_db": ("iSIR", "dB"),
    "target_angle": ("Ángulo objetivo", "°"),
    "target_dist": ("Distancia objetivo", "m"),
    "interf_configs": ("Interferencias (áng°, dist m)", ""),
    "N_interferences": ("# Interferencias", ""),
    "error_angle_deg": ("Error de ángulo", "°"),
    "error_distance_m": ("Error de distancia", "m"),
    "mismatch_gain": ("Mismatch ganancia", ""),
    "mismatch_phase": ("Mismatch fase", ""),
    "mismatch_pos": ("Mismatch posición", ""),
    "use_wpe": ("WPE (dereverb.)", ""),
    "M": ("# Micrófonos", ""),
    "fs": ("Frecuencia de muestreo", "Hz"),
}

def _fmt_meta_value(key: str, val) -> str:
    """Format a metadata attribute value for display."""
    if isinstance(val, (bool, np.bool_)):
        return "Sí" if val else "No"
    if isinstance(val, (float, np.floating)):
        return f"{val:g}"
    return str(val)

def display_experiment_conditions(data: dict, h5_structure: dict, selected_proc: str,
                                  selected_metric: str, selected_case: str):
    """
    Renders the searchgrid parameters and processor context for the case in view,
    adapting to whatever metadata keys are present (works for both the simulated
    and MIRD benchmark variants).
    """
    meta = data.get("metadata", {})
    geom = data.get("geometry", {})

    with st.expander("🔬 Condiciones del experimento", expanded=True):
        # Processor / metric / scenario context
        all_procs = list(h5_structure.keys())
        all_metrics = list(h5_structure.get(selected_proc, {}).keys())
        st.markdown(
            f"**Procesador:** `{selected_proc}`  &nbsp;·&nbsp;  "
            f"**Métrica optimizada:** `{selected_metric}`  &nbsp;·&nbsp;  "
            f"**Escenario:** `{selected_case}`"
        )
        st.caption(
            f"Procesadores disponibles: {', '.join(all_procs)}  |  "
            f"Métricas del searchgrid: {', '.join(all_metrics)}"
        )

        # Build the parameter list: geometry-derived facts first, then metadata attrs
        params = []
        mic_coords = np.atleast_2d(geom.get("mic_coords", [])) if "mic_coords" in geom else None
        if mic_coords is not None and mic_coords.size:
            params.append(("# Micrófonos", str(mic_coords.shape[0])))
        room_dims = geom.get("room_dims")
        if room_dims is not None:
            params.append(("Sala (X×Y×Z)", " × ".join(f"{d:g}" for d in np.ravel(room_dims)) + " m"))
        if "interferences_pos" in geom:
            params.append(("# Interferencias", str(np.atleast_2d(geom["interferences_pos"]).shape[0])))

        skip = {"fs"}  # fs is shown in the audio section; keep this panel focused on the grid
        for key, val in meta.items():
            if key in skip:
                continue
            label, unit = _META_LABELS.get(key, (key.replace("_", " ").title(), ""))
            # Measured datasets reuse the 'rt60' axis for the ROOM NAME (a str);
            # MIRD stores a number there and keeps the RT60 label.
            if key == "rt60" and isinstance(val, (str, bytes)):
                label, unit = "Sala", ""
            text = _fmt_meta_value(key, val)
            params.append((label, f"{text} {unit}".strip()))

        # Render as a responsive grid, 4 items per row
        per_row = 4
        for i in range(0, len(params), per_row):
            cols = st.columns(per_row)
            for col, (label, value) in zip(cols, params[i:i + per_row]):
                col.markdown(f"**{label}**  \n{value}")

_BAND_RE = re.compile(r"^(T|EDT|DRR|CDR)_(\d+)$")

def _scene_prefixes(scene: dict) -> list:
    """[('Objetivo', 'tgt_'), ('Interf 1', 'int1_'), ...] present in the scene."""
    out = [("Objetivo", "tgt_")]
    j = 1
    while any(k.startswith(f"int{j}_") for k in scene):
        out.append((f"Interf {j}", f"int{j}_"))
        j += 1
    return out

def _fmt_cell(v, nd=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)

def display_measured_scene(data: dict):
    """
    Extra panel for catalogs built from MEASURED RIRs (full_benchmark_test_dtln_kria.py):
    reads the optional 'scene' / 'rirs' groups. MIRD catalogs lack them and
    this function returns without drawing anything.
    """
    scene = data.get("scene") or {}
    if not scene:
        return
    rirs = data.get("rirs") or {}

    with st.expander("📐 Escena medida (RIRs reales)", expanded=False):
        st.caption(f"`{scene.get('scene_id', '')}` · aro `{scene.get('array', '')}` · "
                   f"centro {scene.get('arr_center_origin', '?')}")
        tab_src, tab_bands, tab_rir, tab_arr, tab_q = st.tabs(
            ["Fuentes", "Acústica por banda", "RIRs", "Aro y sala", "Calidad"])

        prefixes = _scene_prefixes(scene)

        # --- Fuentes: one row per source -----------------------------------
        with tab_src:
            cols = [("src", "Fuente"), ("dist_m", "Dist (m)"), ("dist_origin", "Origen dist"),
                    ("az_deg", "Az (°)"), ("el_deg", "El (°)"), ("dir_origin", "Origen dir"),
                    ("sep_deg", "Sep. objetivo (°)"), ("T_wide", "T (s)"), ("T_mid", "T mid (s)"),
                    ("DRR_wide", "DRR (dB)"), ("MSC_rms", "MSC rms"), ("audio", "Audio")]
            rows = []
            for role, p in prefixes:
                r = {"Rol": role}
                for k, lab in cols:
                    v = scene.get(f"{p}{k}")
                    if k == "src" and p == "tgt_":
                        v = data.get("metadata", {}).get("target_src", v)
                    r[lab] = _fmt_cell(v)
                rows.append(r)
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            tof_bad = [role for role, p in prefixes if scene.get(f"{p}tof_rejected")]
            if tof_bad:
                st.warning("ToF descartado (fuera de rango) en: " + ", ".join(tof_bad) +
                           f" → distancia nominal. Crudo: " +
                           ", ".join(_fmt_cell(scene.get(f"{p}tof_dist_raw_m"))
                                     for role, p in prefixes if scene.get(f"{p}tof_rejected")) + " m")

        # --- Acústica por banda --------------------------------------------
        with tab_bands:
            desc = st.radio("Descriptor", ["T", "EDT", "DRR", "CDR"], horizontal=True,
                            key="scene_band_desc")
            fig = go.Figure()
            table = {}
            for role, p in prefixes:
                pts = []
                for k, v in scene.items():
                    if not k.startswith(p):
                        continue
                    m = _BAND_RE.match(k[len(p):])
                    if m and m.group(1) == desc and v is not None:
                        pts.append((int(m.group(2)), float(v)))
                pts.sort()
                if not pts:
                    continue
                fig.add_trace(go.Scatter(x=[b for b, _ in pts], y=[v for _, v in pts],
                                         mode="lines+markers", name=role))
                table[role] = {f"{b} Hz": v for b, v in pts}
            if table:
                unit = "s" if desc in ("T", "EDT") else "dB"
                fig.update_layout(xaxis=dict(title="Banda (Hz)", type="log"),
                                  yaxis=dict(title=f"{desc} ({unit})"),
                                  height=320, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, width="stretch")
                st.dataframe(pd.DataFrame(table).T.round(3), width="stretch")
            else:
                st.info(f"No hay {desc} por banda en este caso.")

        # --- RIRs ------------------------------------------------------------
        with tab_rir:
            names = [k for k in rirs if isinstance(rirs[k], np.ndarray)]
            if not names:
                st.info("Este catálogo no guardó las RIRs.")
            else:
                fs_rir = int(rirs.get("fs", data.get("metadata", {}).get("fs", 16000)))
                active = [int(c) for c in str(scene.get("arr_active") or "").split(",") if c != ""]
                ref_ch = scene.get("arr_ref_ch")
                ref_idx = active.index(int(ref_ch)) if (ref_ch is not None and int(ref_ch) in active) else 0
                n_ch = rirs[names[0]].shape[1]
                ch = st.selectbox("Canal (índice dentro de los activos)", list(range(n_ch)),
                                  index=min(ref_idx, n_ch - 1), key="scene_rir_ch",
                                  format_func=lambda i: f"{i} (ch{active[i]})" if i < len(active) else str(i))
                fig = go.Figure()
                for nm in names:
                    h = np.asarray(rirs[nm])[:, ch]
                    db = 20 * np.log10(np.abs(h) / (np.max(np.abs(h)) + 1e-30) + 1e-12)
                    fig.add_trace(go.Scatter(x=np.arange(len(h)) / fs_rir * 1e3, y=db,
                                             mode="lines", name=nm, line=dict(width=1)))
                fig.update_layout(xaxis=dict(title="t (ms)"), yaxis=dict(title="|h| (dB re pico)",
                                  range=[-80, 3]), height=320, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, width="stretch")

        # --- Aro y sala ------------------------------------------------------
        with tab_arr:
            items = [(k, v) for k, v in scene.items() if k.startswith(("arr_", "room_"))]
            per_row = 4
            for i in range(0, len(items), per_row):
                cs = st.columns(per_row)
                for c, (k, v) in zip(cs, items[i:i + per_row]):
                    c.markdown(f"**{k}**  \n{_fmt_cell(v, 3)}")

        # --- Calidad de las mediciones ---------------------------------------
        with tab_q:
            qcols = [("file", "Archivo"), ("when", "Cuándo"), ("reps_used", "Reps"),
                     ("rep_corr_min", "Corr. mín"), ("eps_ppm", "ε (ppm)"),
                     ("noise_rms_dbfs", "Ruido (dBFS)"), ("peak_ir_db", "Pico IR (dB)"),
                     ("n_warnings", "# avisos")]
            rows = []
            for role, p in prefixes:
                r = {"Rol": role}
                r.update({lab: _fmt_cell(scene.get(f"{p}{k}"), 3) for k, lab in qcols})
                rows.append(r)
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            for role, p in prefixes:
                w = scene.get(f"{p}warnings")
                if w:
                    st.warning(f"**{role}:** " + str(w).replace(" | ", "\n\n"))

def get_room_wireframe_traces(room_dims: list, **kwargs) -> list:
    """
    Generates a list of Plotly Scatter3d traces to draw a correct wireframe box.
    """
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
    """
    spatial_data = data["spatial"][proc_name]
    q_gain = spatial_data["quantized_gain"]
    points = spatial_data["points"]
    max_db_per_frame = spatial_data.get("max_db_per_frame")
    min_dB = spatial_data.get("min_dB", -30.0)
    n_azimuth = spatial_data.get("N_azimuth", 90)
    target_fps = spatial_data.get("target_fps", 24)

    # Geometry extraction
    geom = data["geometry"]
    mic_coords = np.atleast_2d(geom.get("mic_coords", [[0, 0, 0]]))
    source_pos = np.atleast_2d(geom.get("source_pos", [[1, 1, 1]]))
    interf_pos = geom.get("interferences_pos", None)
    if interf_pos is not None:
        interf_pos = np.atleast_2d(interf_pos)
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
        uid='dynamic-surface',
        x=X0, y=Y0, z=Z0, surfacecolor=C0,
        colorscale='Viridis', cmin=0, cmax=1, showscale=False, opacity=1.0,
        contours=dict(x=dict(show=False), y=dict(show=False), z=dict(show=False)),
        lighting=dict(ambient=0.7, diffuse=0.5, roughness=0.9, specular=0.2, fresnel=0.2),
        name="Polar Pattern"
    ))

    fig.add_trace(go.Scatter3d(x=mic_coords[:, 0], y=mic_coords[:, 1], z=mic_coords[:, 2], mode='markers', marker=dict(size=4, color='black'), name='Mics'))
    fig.add_trace(go.Scatter3d(x=source_pos[:, 0], y=source_pos[:, 1], z=source_pos[:, 2], mode='markers', marker=dict(size=8, color='green', symbol='diamond'), name='Target'))
    if interf_pos is not None and len(interf_pos) > 0:
        fig.add_trace(go.Scatter3d(x=interf_pos[:, 0], y=interf_pos[:, 1], z=interf_pos[:, 2], mode='markers', marker=dict(size=6, color='red', symbol='x'), name='Interf'))

    frames = []
    frame_names = [str(t) for t in time_indices]
    for t_idx, frame_name in zip(time_indices, frame_names):
        Xt, Yt, Zt, Ct = get_frame_mesh(t_idx, max_db_per_frame[f_idx, t_idx])
        frames.append(go.Frame(
            data=[{'uid': 'dynamic-surface', 'type': 'surface', 'x': Xt, 'y': Yt, 'z': Zt, 'surfacecolor': Ct}],
            traces=[1],
            name=frame_name
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
    st.markdown(METRIC_CSS, unsafe_allow_html=True)
    st.sidebar.title("Acoustic Benchmark")
    st.sidebar.markdown("---")

    results_dir = os.environ.get("BENCHMARK_RESULTS_DIR", "/home/matias/Downloads")
    if not os.path.exists(results_dir):
        st.sidebar.error(f"Directory not found: {results_dir}")
        return

    h5_files = [f for f in os.listdir(results_dir) if f.endswith('.h5')]
    if not h5_files:
        st.sidebar.warning("No HDF5 files found.")
        return

    selected_file = st.sidebar.selectbox("Select Benchmark File", h5_files)
    h5_path = os.path.join(results_dir, selected_file)
    h5_mtime = os.path.getmtime(h5_path)  # cache key: invalidates when the file changes

    # 1. Parse H5 Structure dynamically
    h5_structure = get_h5_structure(h5_path, h5_mtime)
    if not h5_structure:
        st.sidebar.warning("File is empty or not in the expected nested format.")
        return

    # 2. Cascaded Dropdowns for Navigation
    st.sidebar.subheader("Select Evaluation Case")
    selected_proc = st.sidebar.selectbox("Processor", list(h5_structure.keys()))

    available_metrics = list(h5_structure[selected_proc].keys())
    selected_metric = st.sidebar.selectbox("Optimized Metric", available_metrics)

    available_cases = h5_structure[selected_proc][selected_metric]
    selected_case = st.sidebar.selectbox("Scenario", available_cases)

    # 3. Load specific data chunk
    data = load_benchmark_case(h5_path, selected_proc, selected_metric, selected_case, h5_mtime)

    if not data["spatial"]:
        st.error(f"Could not load spatial data for path: {selected_proc}/{selected_metric}/{selected_case}")
        return

    spatial_data = data["spatial"][selected_proc]
    fs = data["metadata"].get("fs", 16000)
    total_frames = spatial_data["quantized_gain"].shape[2]

    st.sidebar.markdown("---")
    st.sidebar.subheader("Animation Controls")
    scale_factor = st.sidebar.slider("Visual Scale", 0.1, 2.0, 1.0)

    # --- BUG FIX: Streamlit slider range restriction ---
    # Enforce that max_decimator is strictly strictly greater than min_value
    max_decimator = max(2, total_frames // 10)
    if total_frames > 1:
        frame_decimator = st.sidebar.slider("Frame Decimator", min_value=1, max_value=max_decimator, value=1)
    else:
        frame_decimator = 1

    # 4. Comprehensive Audio Playback matching new H5 structure
    if "audio" in data and data["audio"]:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Audio Comparison")

        audio_dict = data["audio"]

        # Target Reference
        if "target_reference" in audio_dict:
            st.sidebar.caption("🎯 Target Reference:")
            st.sidebar.audio(prep_audio(audio_dict["target_reference"]), sample_rate=fs)

        # Baseline Mic
        if "mic_signals" in audio_dict:
            st.sidebar.caption("🎙️ Reference Mic (Degraded Mixture):")
            st.sidebar.audio(prep_audio(audio_dict["mic_signals"]), sample_rate=fs)

        # Beamformer Output
        proc_key = f"processed_{selected_proc}"
        if proc_key in audio_dict:
            st.sidebar.caption(f"⚙️ Processed ({selected_proc}):")
            st.sidebar.audio(prep_audio(audio_dict[proc_key]), sample_rate=fs)

        # DTLN Outputs (if neural enhancement was active)
        if "processed_dtln_alone" in audio_dict:
            st.sidebar.caption("🧠 Single-Mic DTLN (No Beamforming):")
            st.sidebar.audio(prep_audio(audio_dict["processed_dtln_alone"]), sample_rate=fs)

        dtln_post_key = f"processed_{selected_proc}_dtln"
        if dtln_post_key in audio_dict:
            st.sidebar.caption(f"🚀 Full Pipeline ({selected_proc} + DTLN):")
            st.sidebar.audio(prep_audio(audio_dict[dtln_post_key]), sample_rate=fs)

    # --- MAIN PANEL ---
    st.title(f"Visualizing: {selected_proc} ({selected_case.replace('_', ' ').title()})")
    st.caption(f"Optimized for extreme variance in: **{selected_metric}**")

    display_experiment_conditions(data, h5_structure, selected_proc, selected_metric, selected_case)
    display_measured_scene(data)   # no-op for MIRD catalogs (no 'scene' group)

    if data["metrics"]:
        display_metrics_row(data["metrics"])

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
    fig = build_native_plotly_animation(data, selected_proc, f_idx, scale_factor, decimator=frame_decimator)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()