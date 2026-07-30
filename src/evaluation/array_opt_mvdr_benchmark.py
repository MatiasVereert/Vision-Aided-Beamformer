"""Does the Konforti-2022 optimized array help data-dependent MVDR?

Detour experiment (Step 1.5): compare the *optimized* geometry against the
paper's own baselines (ULA and dense-DMA) under the ISM acoustic simulator,
scoring real speech-enhancement metrics for two processors:

  * ``SOUDEN_ORACLE_SCM`` -- data-dependent MVDR with ORACLE spatial covariances
    (cleanest upper bound; isolates the geometry effect from mask errors).
  * ``DS``               -- delay-and-sum (fixed spatial filter) baseline.

The array lies on the x-axis, so **endfire = ±x**. The target is placed inside
the optimized ROI (|phi| <= 30 deg around endfire); interferers are placed at
other azimuths (outside the ROI). A linear array cannot resolve mirror angles
(cos phi ambiguity), so target and interferers are kept at well-separated
|cos phi|.

Reverberation (RT60=0.5 s, calibrated pure-ISM) provides the diffuse late field
that the directivity optimization is designed to reject, and moderate sensor
noise (SNR=50 dB) makes the White-Noise-Gain robustness constraint matter --
these are the two axes on which the optimized geometry is expected to win.

Reuses the ISM pipeline (:class:`SimAcoustic`), the hardware emulation
(:class:`Microphone`) and the metric helper ``evaluate_all_references`` from the
main benchmark ``full_benchmark_test_dtln.py``.

Run:
    cd src && python -m evaluation.array_opt_mvdr_benchmark
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from propagation.simulate_acoustics_v1 import SimAcoustic
from beamforming.array.microphone import Microphone
from beamforming.array.optimization import baselines as opt_baselines
from dereverberation.nara_wrappers import process_wpe_online
from evaluation.bf_wrappers import DS, SOUDEN_ORACLE_SCM
from evaluation.full_benchmark_test_dtln import evaluate_all_references

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # .../src
REPO_ROOT = os.path.dirname(SRC_DIR)
SIG = os.path.join(REPO_ROOT, "tools", "data", "signals")
OPT_POS_NPY = os.path.join(SRC_DIR, "beamforming", "array", "optimization",
                           "konforti2022_M6_positions.npy")


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def embed_x_axis(positions_1d: np.ndarray, room_center: np.ndarray) -> np.ndarray:
    """Embed 1-D positions on the x-axis, centred at ``room_center``. -> (M,3)."""
    x = positions_1d - positions_1d.mean()
    coords = np.zeros((positions_1d.shape[0], 3))
    coords[:, 0] = x
    return coords + room_center


def build_geometries(A: float, M: int, d_c: float, room_center: np.ndarray) -> dict:
    """The paper's three geometries (all M mics), embedded on the x-axis."""
    opt_1d = np.load(OPT_POS_NPY)                      # MOSEK solve, A=17.5cm, M=6
    assert opt_1d.shape[0] == M, "saved optimized geometry has a different M"
    return {
        "opt":   embed_x_axis(opt_1d, room_center),
        "ula":   embed_x_axis(opt_baselines.ula(M, A), room_center),
        "dense": embed_x_axis(opt_baselines.dense(M, d_c), room_center),
    }


def polar_position(room_center: np.ndarray, radius: float, phi_deg: float) -> np.ndarray:
    """Point at azimuth ``phi_deg`` from +x (endfire), radius ``radius``, in-plane."""
    phi = np.deg2rad(phi_deg)
    return (room_center + radius * np.array([np.cos(phi), np.sin(phi), 0.0])).reshape(1, 3)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
CFG = dict(
    fs=16000,
    duration=10.0,
    rt60_list=[0.1, 0.3, 0.5, 0.7, 0.9],  # reverberation sweep (extended)
    isir_list=[0.0, 5.0, 10.0],           # target/interference ratio sweep [dB]
    use_wpe_list=[False, True],           # WPE dereverberation on/off
    room_dims=np.array([4.0, 5.0, 3.0]),
    A=0.175, M=6, d_c=0.005,            # paper geometry parameters
    radius=1.0,                         # source/interferer distance [m]
    target_phi_deg=15.0,                # target inside endfire ROI (|phi|<=30)
    snr_db=50.0,                        # sensor self-noise (WNG matters)
    n_noise_seeds=2,                    # average metrics over sensor-noise realizations
    scene_seed=12345,                   # reproducible ISM (rand_ism) per scene
    noise_seed_base=1000,               # reproducible sensor-noise draws
    mismatch_gain=0.0, mismatch_phase=0.0, mismatch_pos=0.0,
    t_early=0.032,                      # early/late split (matches ~WPE t_early)
    stft_window=512, stft_overlap=384,
    # WPE (online) params -- same as the main benchmark driver.
    wpe_taps=7, wpe_delay=3, wpe_alpha=0.9999, wpe_stft_size=512, wpe_stft_shift=128,
    source_path=os.path.join(SIG, "p002_emo_adoration_sentences.wav"),
)

# Interferer scenes (competing point-source talkers at azimuths OUTSIDE the ROI).
# Two representative cases: the single interferer closest to the ROI edge (hardest
# for angular resolution) and a two-interferer case.
INTERFERER_CONFIGS = [
    {"name": "1int@60",     "phis": [60.0],         "paths": ["p011_emo_anger_sentences.wav"]},
    {"name": "2int@60,120", "phis": [60.0, 120.0],  "paths": ["p011_emo_anger_sentences.wav",
                                                              "p008_emo_contentment_sentences.wav"]},
]


def make_processors():
    return {
        "SOUDEN_ORACLE_SCM": SOUDEN_ORACLE_SCM(min_loading=1e-6, alpha=0.99),
        "DS":                DS(),
    }


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def run(output_dir="tests/array_opt_mvdr_out"):
    os.makedirs(output_dir, exist_ok=True)
    room_center = CFG["room_dims"] / 2.0
    geometries = build_geometries(CFG["A"], CFG["M"], CFG["d_c"], room_center)
    target_pos = polar_position(room_center, CFG["radius"], CFG["target_phi_deg"])
    eval_start_s = min(5.0, CFG["duration"] * 0.3)
    mic_sim = Microphone(fs=CFG["fs"])
    mic_sim.set_custom_errors(std_gain_dB=CFG["mismatch_gain"],
                              std_phase_deg=CFG["mismatch_phase"], snr_dB=CFG["snr_db"])
    processors = make_processors()

    rows = []
    for rt60 in CFG["rt60_list"]:
        for iconf in INTERFERER_CONFIGS:
            interf_pos = [polar_position(room_center, CFG["radius"], p) for p in iconf["phis"]]
            for geo_name, mic_coords in geometries.items():
                print(f"\n=== RT60={rt60}s  interferers={iconf['name']}  geometry={geo_name} "
                      f"(aperture={np.ptp(mic_coords[:,0])*100:.1f}cm) ===")

                # ----- ISM physics (depends on RT60+geometry+interferers, NOT iSIR) -----
                np.random.seed(CFG["scene_seed"])  # reproducible rand_ism draws
                scene = SimAcoustic(array_geometry=mic_coords, array_mismatch=CFG["mismatch_pos"],
                                    duration=CFG["duration"], fs=CFG["fs"], seed=0)
                scene.set_source(CFG["source_path"], gain=1.0, position=target_pos)
                for pth, pos in zip(iconf["paths"], interf_pos):
                    scene.set_interference(os.path.join(SIG, pth), gain=1.0, position=pos)
                scene.compute_rirs(room_dimensions=CFG["room_dims"], desire_RT=rt60,
                                   ray_tracing=False)
                scene.convolve_signals(t_early=CFG["t_early"])

                # ----- iSIR sweep: mix_and_normalize is cheap, reuses the RIRs -----
                for isir in CFG["isir_list"]:
                    scene_data = scene.mix_and_normalize(iSIR_dB=isir)
                    refs_dict = {"early": scene_data["target_early"][0]}
                    oracle_target = scene_data["target_early"] + scene_data["target_late"]
                    oracle_noise = scene_data["interference_early"] + scene_data["interference_late"]
                    proc_config = dict(fs=CFG["fs"], mic_coords=mic_coords, source_pos=target_pos,
                                       stft_window=CFG["stft_window"], stft_overlap=CFG["stft_overlap"],
                                       oracle_target=oracle_target, oracle_noise=oracle_noise)

                    # average over independent sensor-noise realizations, for each
                    # WPE setting. Delta is ALWAYS vs the raw unprocessed mic
                    # (pre-WPE, pre-BF), so Delta captures total (WPE+BF) gain.
                    accum = {(p, w): {} for p in processors for w in CFG["use_wpe_list"]}
                    for rep in range(CFG["n_noise_seeds"]):
                        np.random.seed(CFG["noise_seed_base"] + rep)  # same draw for all geometries
                        mic_ready = mic_sim.emulate(scene_data["mic_signals"])
                        base_metrics = evaluate_all_references(
                            refs_dict=refs_dict, deg_sig=mic_ready[0], fs=CFG["fs"],
                            interf_early=scene_data["interference_early"][0],
                            interf_late=scene_data["interference_late"][0],
                            target_late=scene_data["target_late"][0],
                            eval_start_s=eval_start_s,
                            prefix_name=f"base_{geo_name}_{iconf['name']}_rt{rt60}_i{isir}_s{rep}")

                        for use_wpe in CFG["use_wpe_list"]:
                            if use_wpe:
                                obs = process_wpe_online(
                                    u=mic_ready, taps=CFG["wpe_taps"], delay=CFG["wpe_delay"],
                                    alpha=CFG["wpe_alpha"], stft_size=CFG["wpe_stft_size"],
                                    stft_shift=CFG["wpe_stft_shift"])
                            else:
                                obs = mic_ready

                            for proc_name, proc in processors.items():
                                y, _ = proc.process(obs, proc_config)
                                pm = evaluate_all_references(
                                    refs_dict=refs_dict, deg_sig=y, fs=CFG["fs"],
                                    interf_early=scene_data["interference_early"][0],
                                    interf_late=scene_data["interference_late"][0],
                                    target_late=scene_data["target_late"][0],
                                    eval_start_s=eval_start_s,
                                    prefix_name=f"{proc_name}_{geo_name}_{iconf['name']}_rt{rt60}_i{isir}_wpe{int(use_wpe)}_s{rep}")
                                for key in pm:
                                    accum[(proc_name, use_wpe)].setdefault(f"proc_{key}", []).append(pm[key])
                                    accum[(proc_name, use_wpe)].setdefault(f"delta_{key}", []).append(
                                        pm[key] - base_metrics.get(key, np.nan))

                    for (proc_name, use_wpe), acc in accum.items():
                        row = {"rt60": rt60, "isir_db": isir, "interferers": iconf["name"],
                               "geometry": geo_name, "processor": proc_name, "use_wpe": use_wpe,
                               "n_seeds": CFG["n_noise_seeds"]}
                        for key, vals in acc.items():
                            row[key] = float(np.nanmean(vals))
                            row[f"{key}_std"] = float(np.nanstd(vals))
                        rows.append(row)

    df = pd.DataFrame(rows)
    csv = os.path.join(output_dir, "array_opt_mvdr_metrics.csv")
    df.to_csv(csv, index=False)
    print(f"\nSaved full metrics to {csv}")
    _print_summary(df)
    return df


def _print_summary(df: pd.DataFrame):
    """Summarize the RT60 x iSIR x WPE sweep (deltas vs the 'early' reference)."""
    metrics = ["delta_PESQ_early", "delta_STOI_early", "delta_SIR_early", "delta_SDR_early"]
    order = ["opt", "ula", "dense"]
    for proc in df["processor"].unique():
        for use_wpe in sorted(df["use_wpe"].unique()):
            sub = df[(df["processor"] == proc) & (df["use_wpe"] == use_wpe)]
            tag = "WPE" if use_wpe else "no-WPE"
            print(f"\n################  {proc}  [{tag}]  (Delta vs raw mic, 'early' ref)  ################")

            # opt - ula gap resolved over RT60 (averaged over iSIR + interferers).
            for m in ["delta_PESQ_early", "delta_SIR_early"]:
                g_opt = sub[sub.geometry == "opt"].groupby("rt60")[m].mean()
                g_ula = sub[sub.geometry == "ula"].groupby("rt60")[m].mean()
                gap = (g_opt - g_ula)
                print(f"  {m}: (opt-ula) by RT60 -> "
                      + "  ".join(f"{rt}:{gap[rt]:+.3f}" for rt in gap.index))

            mean_by_geo = sub.groupby("geometry")[metrics].mean().reindex(order)
            print(f"  --- MEAN over sweep by geometry [{tag}] ---")
            print("  " + mean_by_geo.round(3).to_string().replace("\n", "\n  "))

    # WPE effect: does dereverberation shrink the opt-ula advantage at high RT?
    print("\n################  WPE effect on opt-ula gap (MVDR, delta_PESQ_early)  ################")
    mv = df[df["processor"] == "SOUDEN_ORACLE_SCM"]
    for use_wpe in sorted(df["use_wpe"].unique()):
        s = mv[mv.use_wpe == use_wpe]
        go = s[s.geometry == "opt"].groupby("rt60")["delta_PESQ_early"].mean()
        gu = s[s.geometry == "ula"].groupby("rt60")["delta_PESQ_early"].mean()
        gap = go - gu
        tag = "WPE   " if use_wpe else "no-WPE"
        print(f"  {tag}: " + "  ".join(f"RT{rt}:{gap[rt]:+.3f}" for rt in gap.index))


if __name__ == "__main__":
    run()
