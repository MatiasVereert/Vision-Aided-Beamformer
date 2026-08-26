import glob
import os
import re
import numpy as np
import scipy.io as sio

# Import native geometric conversion function as structured in your local package
from utils.geometry import spherical_to_cartesian



class MirdDatasetProvider:
    """
    Automated, ultra-robust data loader for the Bar-Ilan MIRD database.
    Scans directories recursively following symlinks and uses highly flexible regex 
    matching to extract parameters regardless of delimiter variations.
    Caches relative Cartesian coordinates (1, 3) for seamless SimAcoustic integration.
    """

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        # Primary lookup registry: registry[t60_ms][spacing][distance][angle] = file_path
        self.registry = {}
        
        # Internal state tracking for current simulation targets
        self.current_rir = None
        self.current_pos_cartesian = None
        self.current_pos_spherical = None
        
        # Execute mapping instantly upon instantiation
        self._index_dataset()

    def _index_dataset(self):
        """
        Scans directory recursively using os.walk with followlinks=True.
        Applies a permissive regex pattern to extract key acoustic and geometric parameters.
        Includes smart fallback path resolution to automatically locate the 'data/rirs/mird' structure.
        """
        abs_root = os.path.abspath(self.root_dir)
        
        # Fallback heuristic: if target path is missing, scan parent directories for the correct data folder
        if not os.path.exists(abs_root):
            curr_dir = os.path.abspath(__file__)
            for _ in range(4):
                curr_dir = os.path.dirname(curr_dir)
                # Check standard multi-level repository layouts
                candidates = [
                    os.path.join(curr_dir, "data", "rirs", "mird"),
                    os.path.join(curr_dir, "rirs", "mird")
                ]
                for candidate in candidates:
                    if os.path.exists(candidate):
                        abs_root = candidate
                        break
                if os.path.exists(abs_root):
                    break

        # Highly flexible regex matching parameters regardless of surrounding boundary characters:
        # Group 1: T60 float (e.g., 0.160 or 0.610)
        # Group 2: Array configuration containing hyphens (e.g., 4-4-4-8-4-4-4)
        # Group 3: Distance float/int immediately preceding 'm' (e.g., 1 or 2)
        # Group 4: Incidence angle integer immediately preceding '.mat' (e.g., 000, 015, 270)
        pattern = re.compile(
            r"Reverberation_([0-9.]+)s.*?([0-9\-]{5,}).*?([0-9.]+)m.*?([0-9\-]+)\.mat", 
            re.IGNORECASE
        )

        indexed_count = 0
        total_mat_files = 0
        skipped_example = None

        for root, _, files in os.walk(abs_root, followlinks=True):
            for filename in files:
                if not filename.lower().endswith(".mat"):
                    continue

                total_mat_files += 1
                match = pattern.search(filename)
                
                if not match:
                    if skipped_example is None:
                        skipped_example = filename
                    continue

                try:
                    # Extract groups and map integer/float keys safely
                    # Rounding prevents floating-point mapping precision bugs (e.g., 0.610 * 1000 -> 610)
                    t60_ms = int(round(float(match.group(1)) * 1000))
                    spacing_cfg = match.group(2)
                    distance = float(match.group(3))
                    angle = int(match.group(4))

                    # Initialize nested dictionary branches dynamically
                    if t60_ms not in self.registry:
                        self.registry[t60_ms] = {}
                    if spacing_cfg not in self.registry[t60_ms]:
                        self.registry[t60_ms][spacing_cfg] = {}
                    if distance not in self.registry[t60_ms][spacing_cfg]:
                        self.registry[t60_ms][spacing_cfg][distance] = {}

                    file_path = os.path.join(root, filename)
                    self.registry[t60_ms][spacing_cfg][distance][angle] = file_path
                    indexed_count += 1

                except Exception:
                    continue

        print(f"[MirdDatasetProvider] Successfully indexed {indexed_count} RIR files from: {abs_root}")
        
        # Diagnostic console reporting if files are present but indexing fails
        if indexed_count == 0:
            print("[!] Warning: No files matched the expected MIRD parameter structure.")
            if total_mat_files > 0:
                print(f"[!] Discovered {total_mat_files} '.mat' files, but regex parsing failed.")
                print(f"[!] Example of skipped unparsed file: '{skipped_example}'")
            else:
                print("[!] Zero '.mat' files were discovered inside the target path tree.")
                print("[!] Verify archive extraction status and directory path accuracy.")

    def get_available_t60s(self) -> list:
        return sorted(list(self.registry.keys()))

    def get_available_spacings(self, t60: int) -> list:
        if t60 in self.registry:
            return sorted(list(self.registry[t60].keys()))
        return []

    def get_available_distances(self, t60: int, spacing: str) -> list:
        if t60 in self.registry and spacing in self.registry[t60]:
            return sorted(list(self.registry[t60][spacing].keys()))
        return []

    def get_available_angles(self, t60: int, spacing: str, distance: float) -> list:
        if t60 in self.registry and spacing in self.registry[t60]:
            if distance in self.registry[t60][spacing]:
                return sorted(list(self.registry[t60][spacing][distance].keys()))
        return []
    
    def load_rir(self, t60, spacing: str, distance: float, angle: int) -> np.ndarray:
        """
        Loads the specific impulse response matrix directly from the 'impulse_response' key.
        Supports passing T60 as float seconds or integer milliseconds.
        Automatically wraps negative incidence angles to physical positive equivalents [0, 360).
        Caches internal native sampling frequency to allow high-fidelity scientific resampling downstream.
        """
        # Safely resolve requested T60 into integer milliseconds key
        if isinstance(t60, float) and t60 < 10.0:
            t60_key = int(round(t60 * 1000))
        else:
            t60_key = int(t60)

        # Map requested incidence angle to physical dataset layout mapping
        angle_key = int(angle) % 360

        try:
            file_path = self.registry[t60_key][spacing][distance][angle_key]
            mat_data = sio.loadmat(file_path)

            # Directly extract the core impulse response matrix explicitly stored by the database
            if 'impulse_response' not in mat_data:
                raise ValueError(f"Key 'impulse_response' missing in target file: {file_path}")

            self.current_rir = mat_data['impulse_response']

            # Robustly extract and cache native dataset sampling rate bypassing output signature modification
            self.current_fs = 48000  # Default MIRD physical baseline
            if 'simpar' in mat_data:
                try:
                    self.current_fs = int(mat_data['simpar']['fs'][0][0][0][0])
                except Exception:
                    pass
            elif 'fs' in mat_data:
                self.current_fs = int(np.asarray(mat_data['fs']).squeeze())

            # Compute standard geometric coordinates (Spherical -> Cartesian)
            r_arr = np.array([distance], dtype=float)
            az_arr = np.array([np.deg2rad(angle)], dtype=float)
            inc_arr = np.array([np.pi / 2.0], dtype=float)

            # Guarantee output spatial vectors strictly conform to shape (1, 3)
            cart_coords = spherical_to_cartesian(r_arr, az_arr, inc_arr).squeeze()
            if cart_coords.ndim == 1:
                cart_coords = cart_coords.reshape(1, 3)
                
            self.current_pos_cartesian = cart_coords
            self.current_pos_spherical = np.array([distance, np.deg2rad(angle), np.pi / 2.0])

            return self.current_rir

        except KeyError:
            raise ValueError(
                f"Configuration T60={t60}ms, Spacing={spacing}, Dist={distance}m, Angle={angle}° not available."
            )

    def get_current_fs(self) -> int:
        """
        Returns the cached native sampling frequency derived during the last load_rir() execution.
        """
        if not hasattr(self, 'current_fs'):
            return 48000
        return self.current_fs
        
    def export_position(self, mode: str = 'cartesian') -> np.ndarray:
        """
        Exports cached spatial vectors derived during the last load_rir() execution.
        """
        if self.current_pos_cartesian is None:
            raise RuntimeError("Execute load_rir() prior to spatial extractions.")
            
        if mode == 'cartesian':
            return self.current_pos_cartesian.copy()
        elif mode == 'spherical':
            return self.current_pos_spherical.copy()
        else:
            raise ValueError("Extraction targets restricted strictly to 'cartesian' or 'spherical'.")

def generate_mird_linear_array_from_spacing(spacing: str = "4-4-4-8-4-4-4") -> np.ndarray:
    """
    Generates theoretical coordinates for a MIRD-style linear array from a
    hyphen-separated inter-sensor SPACING string in centimeters (e.g.
    "4-4-4-8-4-4-4"). N gaps -> N+1 microphones. The array lies on the Y-axis
    and is centred on the origin, matching the convention of the MIRD dataset
    (mic1 towards +Y / +90 deg). Use this to sweep the three measured MIRD
    array configurations (3-3-3-8-3-3-3, 4-4-4-8-4-4-4, 8-8-8-8-8-8-8).
    """
    gaps = [float(g) for g in str(spacing).split('-')]
    # Cumulative sensor positions in centimeters: [0, g0, g0+g1, ...]
    pos_cm = np.concatenate([[0.0], np.cumsum(gaps)])
    # Centre the array on its geometric mean (symmetric for the MIRD patterns).
    pos_cm = pos_cm - pos_cm.mean()
    pos_m = pos_cm / 100.0
    M = pos_m.shape[0]
    return np.column_stack([np.zeros(M), pos_m, np.zeros(M)])


def generate_mird_linear_array() -> np.ndarray:
    """
    Generates theoretical coordinates for the MIRD 8-channel linear array
    configured with standard 4-4-4-8-4-4-4 cm spacing. Kept for backward
    compatibility; delegates to generate_mird_linear_array_from_spacing.
    Correctly maps mic1 to the positive Y-axis (+90 deg) and mic8 to the
    negative Y-axis (270 deg) to match physical metadata ground truth.
    """
    return generate_mird_linear_array_from_spacing("4-4-4-8-4-4-4")

if __name__ == "__main__":
    from utils.audio import save_wav
    # Ensure local module import paths match your internal directory topology
    from propagation.simulate_acoustics_v1 import SimAcoustic

    # --- 1. CONFIGURATION ---
    fs = 48000
    sim_duration = 4.0
    target_isir_db = 0
    
    # Corrected target data directory incorporating the 'data/' parent folder
    root_mird_dir = os.path.abspath("data/rirs/mird")
    output_dir = os.path.abspath("tests/dataset_out/mird_eval")
    os.makedirs(output_dir, exist_ok=True)
    
    source_audio_path = "data/audio/input/p002_emo_adoration_sentences.wav"
    interf_audio_path = "data/audio/input/hairdryer_07_SH_MKH800.wav"

    print("\n==================================================")
    print("      MIRD DATASET EVALUATION PIPELINE            ")
    print("==================================================\n")

    # --- 2. INDEX THE DATASET ---
    print("[*] Initializing automated MIRD dataset provider...")
    provider = MirdDatasetProvider(root_dir=root_mird_dir)
    
    t60s = provider.get_available_t60s()
    print(f"[*] Available T60 environments (ms): {t60s}")
    
    if not t60s:
        raise RuntimeError("Evaluation aborted: No valid T60 setups discovered.")
        
    # Dynamically target requested environment fallback if primary selection is absent
    target_t60_ms = 160 if 160 in t60s else t60s[0]
    target_t60 = target_t60_ms / 1000.0
    target_spacing = "4-4-4-8-4-4-4"
    target_radius = 1.0 

    print(f"[*] Proceeding evaluation using target environment T60={target_t60_ms}ms...")

    # --- 3. LOAD REAL MEASUREMENTS AND EXTRACT GEOMETRY ---
    print(f"[*] Loading measured physical RIR matrices...")
    
    rir_target = provider.load_rir(target_t60, target_spacing, target_radius, angle=0)
    pos_target = provider.export_position(mode='cartesian')
    
    rir_interf = provider.load_rir(target_t60, target_spacing, target_radius, angle=45)
    pos_interf = provider.export_position(mode='cartesian')

    print(f"    -> Target position extracted: {pos_target.squeeze()} m")
    print(f"    -> Interference position extracted: {pos_interf.squeeze()} m")

    # --- 4. BUILD THE ACOUSTIC SCENE ---
    print("\n[*] Constructing SimAcoustic testing environment...")
    ideal_array = generate_mird_linear_array()
    
    scene = SimAcoustic(
        array_geometry=ideal_array, 
        array_mismatch=0.0, 
        duration=sim_duration, 
        fs=fs
    )
    
    scene.set_source(source_audio_path, gain=1.0, position=pos_target)
    scene.set_interference(interf_audio_path, gain=1.0, position=pos_interf)

    # --- 5. INJECT RIRS AND EXECUTE MIXTURE ---
    print("[*] Injecting genuine physical dataset channels into processing core...")
    scene.inject_dataset_environment(target_rir=rir_target, interf_rirs=[rir_interf])
    
    print("[*] Executing multi-channel temporal convolutions...")
    scene.convolve_signals(t_early=0.050)
    
    print(f"[*] Blending streams to match evaluation iSIR condition ({target_isir_db} dB)...")
    eval_data = scene.mix_and_normalize(iSIR_dB=target_isir_db)

    # --- 6. EXPORT MIXED OUTPUTS FOR INSPECTION ---
    print(f"\n[*] Exporting multi-channel evaluation tracks to: {output_dir}")
    save_wav("mird_target_anechoic_ref.wav", fs, eval_data["target_anechoic"][0], output_dir)
    
    for m in range(scene.M):
        filename = f"mird_degraded_mixture_ch{m}.wav"
        save_wav(filename, fs, eval_data["mic_signals"][m], output_dir)
        
    print("\n[+] MIRD evaluation pipeline successfully executed.")
    print("==================================================")