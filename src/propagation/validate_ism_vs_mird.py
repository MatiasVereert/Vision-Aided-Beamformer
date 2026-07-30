"""
Level-2 validation: is the calibrated ISM simulator ACOUSTICALLY REALISTIC?

Not a fit and not a perfect MIRD replica. Goal: confirm that, once the RT is set
by the closed-loop calibration, the simulated reflection order / diffuse field
are SUFFICIENT and land in a realistic regime. We compare, on the same
geometry (MIRD 4-4-4-8-4-4-4 array, r=1 m, all angles) and RT conditions:

  - RT60           (should match: it is what we calibrated)
  - DRR            (expect a systematic offset: MIRD source is directional,
                    pra is omni -> we watch the trend, not the absolute value)
  - diffuse MSC    (cleanest test: source-directivity independent; compares the
                    simulated reverberant field vs measured vs the ideal sinc^2)

Run:  PYTHONPATH=src python src/propagation/validate_ism_vs_mird.py
"""
import os
import numpy as np
import scipy.signal as dsp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyroomacoustics as pra

from utils.geometry import spherical_to_cartesian
from propagation.mird_loader import MirdDatasetProvider, generate_mird_linear_array
from propagation.rt60_calibration import sweep_alpha, build_alpha_to_rt60, simulate_singlechannel_rir
from propagation.acoustic_descriptors import (
    measure_rt60_schroeder, compute_drr, averaged_msc, theoretical_diffuse_msc, C_SOUND,
)

# ---- Config ----
MIRD_ROOT = "tools/data/rirs/mird"
ROOM_DIMS = np.array([6.0, 6.0, 2.4])          # Hadad et al. 2014
ARRAY_CENTER = np.array([3.0, 3.0, 1.2])       # centred; height not given (RT/coh robust to it)
SPACING = "4-4-4-8-4-4-4"
DISTANCE = 1.0
FS = 16000
MAX_ORDER = 50                                 # from Level-1 convergence (saturated)
RT_CONDS_MS = [160, 360, 610]
BAND = (200.0, 7500.0)                          # comparison band at fs=16 kHz
MSC_DISTS = [0.08, 0.16, 0.32]                  # inter-mic distances to validate (m)
OUT_FIG = os.path.join(
    os.environ.get("SCRATCH", "/tmp/claude-1000/-home-matias-pdm-mic-interface/"
                   "a7748981-ceb9-4ca9-b46f-227311c7e123/scratchpad"),
    "ism_vs_mird_coherence.png",
)


def resample_to_fs(x, fs_in, fs_out):
    if fs_in == fs_out:
        return x
    g = np.gcd(int(fs_in), int(fs_out))
    return dsp.resample_poly(x, fs_out // g, fs_in // g, axis=0)


def load_mird_banks(provider, t60_ms, angles):
    """Return list over angles of (M, L) RIR banks resampled to FS."""
    banks = []
    for ang in angles:
        ir = provider.load_rir(t60_ms, SPACING, DISTANCE, ang)   # (N, 8) @ native fs
        native = provider.get_current_fs()
        ir_rs = resample_to_fs(np.asarray(ir, float), native, FS)  # (L, 8)
        banks.append(ir_rs.T)                                       # (8, L)
    return banks


def simulate_banks(alpha, angles):
    """Simulate the MIRD geometry in pra for each angle -> list of (M, L) banks."""
    mics_rel = generate_mird_linear_array()          # (8,3), along Y
    mics_abs = (ARRAY_CENTER + mics_rel).T           # (3, 8)
    banks = []
    for ang in angles:
        rel = spherical_to_cartesian(np.array([DISTANCE]),
                                     np.array([np.deg2rad(ang)]),
                                     np.array([np.pi / 2.0])).squeeze()
        src = ARRAY_CENTER + rel
        room = pra.ShoeBox(ROOM_DIMS, fs=FS, materials=pra.Material(float(alpha)),
                           max_order=MAX_ORDER, air_absorption=True,
                           use_rand_ism=True, max_rand_disp=0.08)
        room.add_source(src)
        room.add_microphone_array(mics_abs)
        room.compute_rir()
        M = mics_abs.shape[1]
        banks.append([np.asarray(room.rir[m][0], float) for m in range(M)])
    return banks


def pairs_for_distance(mics_rel, d, tol=0.005):
    """All (i, j) microphone pairs whose axial separation equals d (pooled)."""
    ys = mics_rel[:, 1]
    M = len(ys)
    return [(i, j) for i in range(M) for j in range(i + 1, M)
            if abs(abs(ys[i] - ys[j]) - d) < tol]


def band_rms(freqs, a, b):
    m = (freqs >= BAND[0]) & (freqs <= BAND[1])
    return np.sqrt(np.mean((a[m] - b[m]) ** 2))


def mean_rt60(banks, ch=0):
    vals = [measure_rt60_schroeder(bank[ch], FS, method='T20') for bank in banks]
    vals = np.array(vals, float)
    vals = vals[~np.isnan(vals)]
    return float(np.mean(vals)), float(np.std(vals))


def mean_drr(banks, ch=0):
    vals = np.array([compute_drr(bank[ch], FS) for bank in banks], float)
    return float(np.mean(vals)), float(np.std(vals))


def main():
    provider = MirdDatasetProvider(MIRD_ROOT)

    # Build the closed-loop alpha map at the validation fs (single-channel, cheap)
    print("[*] Building alpha->RT60 map (single-channel sweep)...")
    alphas = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90])
    sweep = sweep_alpha(ROOM_DIMS, FS, alphas, MAX_ORDER)
    _, alpha_of_rt60 = build_alpha_to_rt60(sweep)

    mics_rel = generate_mird_linear_array()
    dist_pairs = {d: pairs_for_distance(mics_rel, d) for d in MSC_DISTS}

    fig, axes = plt.subplots(len(RT_CONDS_MS), len(MSC_DISTS),
                             figsize=(4 * len(MSC_DISTS), 3 * len(RT_CONDS_MS)),
                             squeeze=False)

    print("\n" + "=" * 80)
    print(f"{'RT[ms]':>7} {'alpha':>6} | {'RT60 mird':>10} {'RT60 sim':>10} | "
          f"{'DRR mird':>9} {'DRR sim':>9} | {'MSC err(sim-mird)':>17}")
    print("(alpha calibrado al RT60 MEDIDO de MIRD, no a la etiqueta nominal)")
    print("=" * 80)

    for r, t60_ms in enumerate(RT_CONDS_MS):
        angles = provider.get_available_angles(t60_ms, SPACING, DISTANCE)

        mird_banks = load_mird_banks(provider, t60_ms, angles)

        # Closed-loop target = RT60 MEASURED on MIRD with our own estimator
        # (same yardstick for both sides; the nominal label used a different band/method).
        rt_m, rt_m_sd = mean_rt60(mird_banks)
        alpha = float(alpha_of_rt60(rt_m))

        sim_banks = simulate_banks(alpha, angles)
        rt_s, rt_s_sd = mean_rt60(sim_banks)
        drr_m, _ = mean_drr(mird_banks)
        drr_s, _ = mean_drr(sim_banks)

        msc_errs = []
        for c, d in enumerate(MSC_DISTS):
            pairs = dist_pairs[d]
            f, msc_m = averaged_msc(mird_banks, FS, pairs)
            _, msc_s = averaged_msc(sim_banks, FS, pairs)
            msc_th = theoretical_diffuse_msc(f, d)
            msc_errs.append(band_rms(f, msc_s, msc_m))

            ax = axes[r][c]
            ax.plot(f, msc_m, label="MIRD", lw=1.6)
            ax.plot(f, msc_s, label="ISM sim", lw=1.6)
            ax.plot(f, msc_th, "k--", label="difuso ideal", lw=1.0, alpha=0.7)
            ax.set_xlim(0, BAND[1])
            ax.set_ylim(0, 1.05)
            ax.set_title(f"RT{t60_ms} | d={d*100:.0f}cm ({len(pairs)} pares)", fontsize=9)
            if r == len(RT_CONDS_MS) - 1:
                ax.set_xlabel("Hz")
            if c == 0:
                ax.set_ylabel("MSC")
            if r == 0 and c == 0:
                ax.legend(fontsize=7)

        print(f"{t60_ms:7d} {alpha:6.3f} | {rt_m:10.3f} {rt_s:10.3f} | "
              f"{drr_m:9.2f} {drr_s:9.2f} | {np.mean(msc_errs):17.3f}")

    fig.suptitle("Coherencia difusa inter-microfono: MIRD vs ISM calibrado vs ideal", y=1.0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    fig.savefig(OUT_FIG, dpi=110, bbox_inches="tight")
    print("=" * 78)
    print(f"[*] Figura de coherencia guardada en: {OUT_FIG}")


if __name__ == "__main__":
    main()
