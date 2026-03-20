import os
import sys
import h5py
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyvista as pv
from pyvistaqt import QtInteractor
import sounddevice as sd

# We only need the spherical coordinate generator from your polar_plots.py
from evaluation.polar_plots import get_sphere_of_points



import os
import sys
import h5py
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyvista as pv
from pyvistaqt import QtInteractor
import sounddevice as sd

# Import the exact spherical generator from your logic
from polar_plots import get_sphere_of_points

def load_benchmark_data(h5_filepath: str) -> dict:
    """
    Load HDF5 benchmark data.
    """
    data = {"audio": {}, "metadata": {}, "geometry": {}, "metrics": {}, "spatial": {}, "weights": {}}
    with h5py.File(h5_filepath, 'r') as f:
        for section in ["audio", "metadata", "geometry", "metrics", "weights"]:
            if section in f:
                for key in f[section].keys():
                    data[section][key] = f[section][key][()]
                for attr_key, attr_val in f[section].attrs.items():
                    data[section][attr_key] = attr_val
        for key in f.keys():
            if key.startswith("spatial_"):
                proc_name = key.replace("spatial_", "")
                data["spatial"][proc_name] = {}
                for dset_key in f[key].keys():
                    data["spatial"][proc_name][dset_key] = f[key][dset_key][()]
                for attr_key, attr_val in f[key].attrs.items():
                    data["spatial"][proc_name][attr_key] = attr_val
    return data


class AcousticDashboardApp(QtWidgets.QMainWindow):
    def __init__(self, results_dir):
        super().__init__()
        self.setWindowTitle("Acoustic Benchmark - PyVista Engine")
        self.resize(1200, 800)
        self.results_dir = results_dir
        
        self.data = None
        self.spatial_data = None
        self.precomputed_vertices = None
        self.precomputed_magnitudes = None
        self.current_frame = 0
        self.n_frames = 1
        self.is_playing = False
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.on_timer_tick)

        self.init_ui()
        self.load_directory()

    def load_directory(self):
        if not os.path.exists(self.results_dir):
            print(f"Error: Directory not found: {self.results_dir}")
            return
        files = [f for f in os.listdir(self.results_dir) if f.endswith('.h5')]
        self.cb_file.addItems(files)

    def on_file_selected(self):
        filename = self.cb_file.currentText()
        if not filename: return
        
        filepath = os.path.join(self.results_dir, filename)
        self.data = load_benchmark_data(filepath)
        
        self.cb_proc.blockSignals(True)
        self.cb_proc.clear()
        self.cb_proc.addItems(list(self.data["spatial"].keys()))
        self.cb_proc.blockSignals(False)
        
        self.draw_static_geometry()
        
        if self.cb_proc.count() > 0:
            self.on_processor_selected()

    def draw_static_geometry(self):
        geom = self.data["geometry"]
        Lx, Ly, Lz = geom.get("room_dims", [5.0, 5.0, 3.0])
        
        bounds = (0, Lx, 0, Ly, 0, Lz)
        room_box = pv.Box(bounds=bounds)
        if self.actor_room: self.plotter.remove_actor(self.actor_room)
        self.actor_room = self.plotter.add_mesh(room_box, style='wireframe', color='gray', line_width=2)
        
        mics = np.atleast_2d(geom.get("mic_coords", [[0,0,0]]))
        mic_cloud = pv.PolyData(mics)
        if self.actor_mics: self.plotter.remove_actor(self.actor_mics)
        self.actor_mics = self.plotter.add_mesh(mic_cloud, color='black', render_points_as_spheres=True, point_size=10)
        
        src = np.atleast_2d(geom.get("source_pos", [[1,1,1]]))
        src_cloud = pv.PolyData(src)
        if self.actor_src: self.plotter.remove_actor(self.actor_src)
        self.actor_src = self.plotter.add_mesh(src_cloud, color='green', render_points_as_spheres=True, point_size=15)
        
        interf = geom.get("interferences_pos", None)
        if interf is not None:
            interf_cloud = pv.PolyData(np.atleast_2d(interf))
            if self.actor_interf: self.plotter.remove_actor(self.actor_interf)
            self.actor_interf = self.plotter.add_mesh(interf_cloud, color='red', render_points_as_spheres=True, point_size=10)
            
        self.plotter.reset_camera()

    def on_processor_selected(self):
        proc_name = self.cb_proc.currentText()
        if not proc_name: return
        self.spatial_data = self.data["spatial"][proc_name]
        
        freqs = self.spatial_data.get("freqs", [1000.0])
        self.cb_freq.blockSignals(True)
        self.cb_freq.clear()
        self.cb_freq.addItems([f"{f:.0f} Hz" for f in freqs])
        self.cb_freq.blockSignals(False)
        
        # q_gain shape is (F, P, T)
        q_gain = self.spatial_data["quantized_gain"]
        self.n_frames = q_gain.shape[2]
        self.slider_frame.setRange(0, self.n_frames - 1)
        
        self.on_freq_selected()

    def on_freq_selected(self):
        self.recompute_all_frames()
        self.current_frame = 0
        self.slider_frame.setValue(0)
        self.update_mesh_to_current_frame()
        
    def init_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)
        
        # Left panel layout setup
        control_panel = QtWidgets.QVBoxLayout()
        control_panel.setAlignment(QtCore.Qt.AlignTop)
        control_panel.setContentsMargins(10, 10, 10, 10)
        
        control_panel.addWidget(QtWidgets.QLabel("Select File:"))
        self.cb_file = QtWidgets.QComboBox()
        self.cb_file.currentIndexChanged.connect(self.on_file_selected)
        control_panel.addWidget(self.cb_file)
        
        control_panel.addWidget(QtWidgets.QLabel("Processor:"))
        self.cb_proc = QtWidgets.QComboBox()
        self.cb_proc.currentIndexChanged.connect(self.on_processor_selected)
        control_panel.addWidget(self.cb_proc)
        
        control_panel.addWidget(QtWidgets.QLabel("Frequency:"))
        self.cb_freq = QtWidgets.QComboBox()
        self.cb_freq.currentIndexChanged.connect(self.on_freq_selected)
        control_panel.addWidget(self.cb_freq)
        
        control_panel.addWidget(QtWidgets.QLabel("Visual Scale:"))
        self.slider_scale = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_scale.setRange(1, 20)
        self.slider_scale.setValue(10)
        self.slider_scale.valueChanged.connect(self.recompute_all_frames)
        control_panel.addWidget(self.slider_scale)
        
        play_layout = QtWidgets.QHBoxLayout()
        self.btn_play = QtWidgets.QPushButton("Play / Pause")
        self.btn_play.clicked.connect(self.toggle_playback)
        play_layout.addWidget(self.btn_play)
        
        self.cb_speed = QtWidgets.QComboBox()
        self.cb_speed.addItems(["x1", "x2", "x4"])
        self.cb_speed.currentIndexChanged.connect(self.update_timer_interval)
        play_layout.addWidget(self.cb_speed)
        control_panel.addLayout(play_layout)
        
        self.slider_frame = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_frame.setRange(0, 0)
        self.slider_frame.valueChanged.connect(self.on_frame_slider_changed)
        control_panel.addWidget(self.slider_frame)
        self.lbl_frame = QtWidgets.QLabel("Frame: 0")
        control_panel.addWidget(self.lbl_frame)
        
        control_panel.addSpacing(20)
        
        control_panel.addWidget(QtWidgets.QLabel("Audio (sounddevice):"))
        self.btn_play_ref = QtWidgets.QPushButton("Play Reference Mic")
        self.btn_play_ref.clicked.connect(lambda: self.play_audio("ref"))
        control_panel.addWidget(self.btn_play_ref)
        
        self.btn_play_proc = QtWidgets.QPushButton("Play Processed")
        self.btn_play_proc.clicked.connect(lambda: self.play_audio("proc"))
        control_panel.addWidget(self.btn_play_proc)
        
        control_widget = QtWidgets.QWidget()
        control_widget.setLayout(control_panel)
        control_widget.setFixedWidth(250)
        main_layout.addWidget(control_widget)
        
        # Right panel: PyVista 3D Plotter
        self.plotter = QtInteractor(self)
        self.plotter.set_background('white')
        
        # IMPORTANT FIX: Initialize as None. 
        # We will create the mesh when the first batch of real data is processed.
        self.polar_mesh = None
        self.actor_polar = None
        
        self.actor_mics = None
        self.actor_src = None
        self.actor_interf = None
        self.actor_room = None
        
        main_layout.addWidget(self.plotter.interactor)

    def recompute_all_frames(self):
        """
        Calculates all vertices for the animation loop and initializes the mesh correctly.
        """
        if self.spatial_data is None: return
        f_idx = self.cb_freq.currentIndex()
        if f_idx < 0: return
        
        q_gain = self.spatial_data["quantized_gain"] 
        points = self.spatial_data["points"]         
        max_dB = self.spatial_data["max_dB"]
        min_dB = self.spatial_data.get("min_dB", -30.0)
        N_azimuth = self.spatial_data.get("N_azimuth", 90)
        
        user_scale = self.slider_scale.value() / 10.0
        geom = self.data["geometry"]
        mic_coords = np.atleast_2d(geom.get("mic_coords", [[0,0,0]]))
        source_pos = np.atleast_2d(geom.get("source_pos", [[1,1,1]]))
        array_center = np.mean(mic_coords, axis=0)
        dist_to_source = np.linalg.norm(source_pos[0] - array_center)
        room_dims = geom.get("room_dims", [5.0, 5.0, 3.0])
        max_allowed_radius = np.min(room_dims) / 2.0
        
        lobe_scale = min(dist_to_source, max_allowed_radius) * user_scale
        
        original_radius = np.linalg.norm(points[0]) 
        unit_vectors = points / original_radius 
        
        P = points.shape[0]
        T = self.n_frames
        
        # Allocate memory for all precomputed frames
        self.precomputed_vertices = np.zeros((T, P, 3), dtype=np.float32)
        self.precomputed_magnitudes = np.zeros((T, P), dtype=np.float32)
        
        q_slice = q_gain[f_idx, :, :] 
        
        db_gain_recovered = (q_slice / 255.0) * (max_dB - min_dB) + min_dB
        linear_magnitude = np.clip((db_gain_recovered - min_dB) / (max_dB - min_dB), 0.0, 1.0)
        
        # Calculate scaling and offsets for each time step
        for t in range(T):
            mag_t = linear_magnitude[:, t]
            self.precomputed_magnitudes[t, :] = mag_t
            
            lobe_radii = mag_t * lobe_scale
            lobe_points = (unit_vectors * lobe_radii[:, np.newaxis]) + array_center
            self.precomputed_vertices[t, :, :] = lobe_points

        # VTK Mesh Topology Setup
        N_elevation = N_azimuth // 2 + 1
        
        # Create the mesh only if it doesn't exist yet
        if self.polar_mesh is None:
            self.polar_mesh = pv.StructuredGrid()
            
        # VTK REQUIREMENT: Assign points FIRST, then dimensions, then scalars!
        self.polar_mesh.points = self.precomputed_vertices[0]
        self.polar_mesh.dimensions = (N_azimuth, N_elevation, 1)
        self.polar_mesh['magnitude'] = self.precomputed_magnitudes[0]
        
        # Add the actor to the plotter only once
        if self.actor_polar is None:
            self.actor_polar = self.plotter.add_mesh(
                self.polar_mesh, 
                scalars='magnitude', 
                cmap='viridis',
                show_edges=False,
                smooth_shading=True,
                show_scalar_bar=False,
                clim=[0.0, 1.0] 
            )

    def toggle_playback(self):
        if self.is_playing:
            self.timer.stop()
        else:
            self.update_timer_interval()
            self.timer.start()
        self.is_playing = not self.is_playing

    def update_timer_interval(self):
        fps_base = 24
        speed_factor = [1, 2, 4][self.cb_speed.currentIndex()]
        interval_ms = int(1000 / (fps_base * speed_factor))
        self.timer.setInterval(interval_ms)

    def on_timer_tick(self):
        self.current_frame = (self.current_frame + 1) % self.n_frames
        self.slider_frame.blockSignals(True)
        self.slider_frame.setValue(self.current_frame)
        self.slider_frame.blockSignals(False)
        self.lbl_frame.setText(f"Frame: {self.current_frame}")
        self.update_mesh_to_current_frame()

    def on_frame_slider_changed(self):
        if not self.is_playing:
            self.current_frame = self.slider_frame.value()
            self.lbl_frame.setText(f"Frame: {self.current_frame}")
            self.update_mesh_to_current_frame()

    def update_mesh_to_current_frame(self):
        if self.precomputed_vertices is None: return
        t = self.current_frame
        
        # Inject precomputed values directly into GPU mapped memory
        self.polar_mesh.points = self.precomputed_vertices[t]
        self.polar_mesh['magnitude'] = self.precomputed_magnitudes[t]
        
        self.plotter.render()

    def play_audio(self, source_type):
        if not self.data or "audio" not in self.data: return
        fs = self.data["metadata"].get("fs", 16000)
        
        if source_type == "ref":
            audio_data = self.data["audio"].get("mic_signals", np.zeros((1, 100)))[0]
        else:
            proc_name = self.cb_proc.currentText()
            proc_key = f"processed_{proc_name}"
            audio_data = self.data["audio"].get(proc_key, np.zeros(100))
            
        audio_norm = np.float32(audio_data / np.max(np.abs(audio_data)))
        sd.play(audio_norm, samplerate=fs)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    
    # Update to your actual local path
    target_dir = r"tests/data/benchmark_results" 
    
    window = AcousticDashboardApp(target_dir)
    window.show()
    sys.exit(app.exec_())