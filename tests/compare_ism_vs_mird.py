"""
ISM vs MIRD metric comparison under GEOMETRIC and ACOUSTIC coincidence.

The exact same SimAcoustic scene (MIRD room 6x6x2.4, MIRD 8-mic linear array,
same source/interference positions, same audio, same iSIR) is evaluated two ways:
  - MIRD : real measured RIRs   (scene.import_rirs)
  - ISM  : calibrated pure-ISM  (scene.compute_rirs) targeted to the RT60
           MEASURED on the MIRD RIRs with our own Schroeder estimator, so both
           sides share the same acoustic yardstick (acoustic coincidence).

Runs the given processors on both and compares the Delta metrics side by side.

Run:  PYTHONPATH=src python tests/compare_ism_vs_mird.py
"""
import os
import time
import numpy as np
import pandas as pd

from propagation.simulate_acoustics_v1 import SimAcoustic
from propagation.mird_loader import MirdDatasetProvider, generate_mird_linear_array
from propagation.acoustic_descriptors import measure_rt60_schroeder
from evaluation.metrics import evaluate_full_pipeline
from evaluation.bf_wrappers import (
    NM_MVDR,
    DTLN_MB_MVDR_SOUDEN_SLOW,
    NM_MVDR_PF,
    ORACLE_MB_MVDR_SOUDEN,
)

# ------------------------- Config -------------------------
REPO = "/home/matias/Documents/Tesis/Vision-Aided-Beamformer"
MIRD_ROOT = os.path.join(REPO, "tools/data/rirs/mird")
SOURCE_WAV = os.path.join(REPO, "tools/data/signals/p002_emo_adoration_sentences.wav")
INTERF_WAV = os.path.join(REPO, "tools/data/signals/techno_gated commune.wav")
DTLN_MODEL = os.path.join(REPO, "src/dnn_denoise/models/model_quant_1.tflite")
OUT_DIR = os.path.join(REPO, "tests/dataset_out/ism_vs_mird_compare")

FS = 16000
DURATION = 8
ROOM_DIMS = np.array([6.0, 6.0, 2.4])
ARRAY_CENTER = np.array([3.0, 3.0, 1.2])
MIRD_SPACING = "4-4-4-8-4-4-4"
T_EARLY = 0.050
ISIR_DB = 5.0
EVAL_REFS = ['anechoic', 'early', 'reverberant']

# Geometric coincidence: (rt60_label, target_angle, target_dist, [(interf_angle, interf_dist), ...])
SCENARIOS = [
    (0.160, 0, 1.0, [(45, 1.0)]),
    (0.360, 0, 1.0, [(45, 1.0)]),
    (0.610, 0, 1.0, [(45, 1.0)]),
]

# Metrics we line up in the printed comparison (evaluated against 'early')
REPORT = [("Delta_tot_PESQ_early", "dPESQ"),
          ("Delta_tot_SIR_early", "dSIR"),
          ("Delta_tot_STOI_early", "dSTOI")]


def build_processors():
    return {
        "NM-MVDR_alpha_1_ref":         NM_MVDR(min_loading=1e-6, alpha=1),
        "NM-MVDR_alpha_0.99_ref":      NM_MVDR(min_loading=1e-6, alpha=0.99),
        "Oracle-MVDR_alpha_1":         ORACLE_MB_MVDR_SOUDEN(min_loading=1e-6, alpha=1, sharpen_exp=1.0),
        "Oracle-MVDR_alpha_0.99":      ORACLE_MB_MVDR_SOUDEN(min_loading=1e-6, alpha=0.99, sharpen_exp=1.0),
        "Oracle-MVDR_hard_alpha_1":    ORACLE_MB_MVDR_SOUDEN(min_loading=1e-6, alpha=1, sharpen_exp=4.0),
        "Oracle-MVDR_hard_alpha_0.99": ORACLE_MB_MVDR_SOUDEN(min_loading=1e-6, alpha=0.99, sharpen_exp=4.0),
        "Slow":                        DTLN_MB_MVDR_SOUDEN_SLOW(),
        "Specsub":                     NM_MVDR_PF(smooth=1.0, min_loading=1e-6),
    }


def make_scene(provider, rt60_label, target_angle, target_dist, interf_cfgs):
    """Common scene (array + source/interf positions + audio). RIRs added later."""
    mics = generate_mird_linear_array() + ARRAY_CENTER
    scene = SimAcoustic(mics, array_mismatch=0.0, duration=DURATION, fs=FS, seed=0)

    provider.load_rir(rt60_label, MIRD_SPACING, target_dist, target_angle)
    src = ARRAY_CENTER + provider.export_position('cartesian').squeeze()
    scene.set_source(SOURCE_WAV, gain=1.0, position=src.reshape(1, 3))

    for (i_ang, i_dist) in interf_cfgs:
        provider.load_rir(rt60_label, MIRD_SPACING, i_dist, i_ang)
        ip = ARRAY_CENTER + provider.export_position('cartesian').squeeze()
        scene.set_interference(INTERF_WAV, gain=1.0, position=ip.reshape(1, 3))
    return scene


def mean_rt60(scene):
    vals = [measure_rt60_schroeder(scene.rirs[m][0], FS, method='T20') for m in range(scene.M)]
    vals = np.array(vals, float)
    return float(np.nanmean(vals))


def build_proc_config(scene, scene_data):
    return {
        'fs': FS,
        'stft_window': 512,
        'stft_overlap': 384,
        'dtln_model_path': DTLN_MODEL,
        'mic_coords': scene.real_array,
        'source_pos': np.asarray(scene.audio_sources[0]["position"]).reshape(1, 3),
        'VAD': scene_data["VAD"],
        # Oracle references (agnostic upper bound): full reverberant target vs interference
        'oracle_target': scene_data["target_early"] + scene_data["target_late"],
        'oracle_noise': scene_data["interference_early"] + scene_data["interference_late"],
    }


def eval_all_refs(refs, deg_sig, scene_data, eval_start_s):
    out = {}
    for name, ref in refs.items():
        m = evaluate_full_pipeline(
            ref_sig=ref, deg_sig=deg_sig, fs=FS,
            interf_early=scene_data["interference_early"][0],
            interf_late=scene_data["interference_late"][0],
            target_late=scene_data["target_late"][0],
            eval_start_s=eval_start_s,
        )
        for k, v in m.items():
            out[f"{k}_{name}"] = v
    return out


def run_mode(scene, processors):
    """Mix, run baseline + every processor, return {proc_name: {Delta_tot_<metric>_<ref>: val}}."""
    scene_data = scene.mix_and_normalize(iSIR_dB=ISIR_DB)
    refs = {}
    if 'anechoic' in EVAL_REFS:
        refs['anechoic'] = scene_data["target_anechoic"][0]
    if 'early' in EVAL_REFS:
        refs['early'] = scene_data["target_early"][0]
    if 'reverberant' in EVAL_REFS:
        refs['reverberant'] = scene_data["target_early"][0] + scene_data["target_late"][0]

    eval_start_s = min(5.0, DURATION * 0.3)
    proc_config = build_proc_config(scene, scene_data)

    base = eval_all_refs(refs, scene_data["mic_signals"][0], scene_data, eval_start_s)

    per_proc = {}
    for name, proc in processors.items():
        try:
            y, _ = proc.process(scene_data["mic_signals"], proc_config)
            pm = eval_all_refs(refs, y, scene_data, eval_start_s)
            deltas = {f"Delta_tot_{k}": pm[k] - base[k] for k in pm}
        except Exception as e:
            print(f"      [!] processor {name} failed: {e}")
            deltas = {f"Delta_tot_{k}": np.nan for k in base}
        per_proc[name] = deltas
    return per_proc


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    provider = MirdDatasetProvider(MIRD_ROOT)
    processors = build_processors()

    rows = []
    t0 = time.time()
    for (rt_label, t_ang, t_dist, interf) in SCENARIOS:
        print(f"\n{'='*72}\nSCENARIO rt60={rt_label} angle={t_ang} dist={t_dist} interf={interf}\n{'='*72}")

        # --- MIRD (real RIRs) ---
        scene_m = make_scene(provider, rt_label, t_ang, t_dist, interf)
        scene_m.import_rirs(provider, target_t60=rt_label,
                            array_center=ARRAY_CENTER, spacing_cfg=MIRD_SPACING)
        scene_m.convolve_signals(t_early=T_EARLY)
        rt_mird = mean_rt60(scene_m)

        # --- ISM (calibrated, targeted to MIRD-measured RT for acoustic coincidence) ---
        scene_i = make_scene(provider, rt_label, t_ang, t_dist, interf)
        scene_i.compute_rirs(ROOM_DIMS, desire_RT=rt_mird)
        scene_i.convolve_signals(t_early=T_EARLY)
        rt_ism = mean_rt60(scene_i)

        print(f"[acoustic coincidence] RT60 mird={rt_mird:.3f}s  ->  ISM target -> realised={rt_ism:.3f}s")

        print("[*] Running processors on MIRD...")
        res_m = run_mode(scene_m, processors)
        print("[*] Running processors on ISM...")
        res_i = run_mode(scene_i, processors)

        for name in processors:
            row = {"rt60": rt_label, "rt60_mird": round(rt_mird, 3), "rt60_ism": round(rt_ism, 3),
                   "processor": name}
            for key, short in REPORT:
                vm = res_m[name].get(key, np.nan)
                vi = res_i[name].get(key, np.nan)
                row[f"{short}_mird"] = vm
                row[f"{short}_ism"] = vi
                row[f"{short}_gap"] = vi - vm
            rows.append(row)

    df = pd.DataFrame(rows)
    csv = os.path.join(OUT_DIR, "ism_vs_mird_compare.csv")
    df.to_csv(csv, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)
    print(f"\n{'#'*72}\nCOMPARISON (Delta vs baseline, evaluated against 'early' reference)\n{'#'*72}")
    for rt_label, _, _, _ in SCENARIOS:
        sub = df[df["rt60"] == rt_label]
        if sub.empty:
            continue
        print(f"\n--- rt60={rt_label} (mird={sub['rt60_mird'].iloc[0]}s / ism={sub['rt60_ism'].iloc[0]}s) ---")
        cols = ["processor"] + [c for c in df.columns if any(c.startswith(s) for _, s in REPORT)]
        print(sub[cols].to_string(index=False, float_format=lambda x: f"{x:7.3f}"))

    print(f"\n[*] CSV: {csv}")
    print(f"[*] Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
