"""Reproduce Figs. 1-2 of Konforti, Cohen & Berdugo (IWAENC 2022).

Runs the geometry optimizer, applies the superdirective coefficient
post-processing (Sec. 4) to the optimized / ULA / dense geometries, and plots:

  * Fig. 1  -- the optimized microphone layout on the aperture.
  * Fig. 2  -- broadband directivity index vs. look angle for the three
               geometries (the optimized array should dominate over the ROI).

The paper's full configuration (``--full``) is ``N=40, M=6, Q=15, P=15`` and is
only practical with MOSEK. The default configuration is reduced so that the
open-source SCIP backend finishes in a reasonable time while still exhibiting
the paper's qualitative result (mics clustered near the edges, optimized DI
above ULA/dense across the ROI).

Usage::

    python -m beamforming.array.optimization.reproduce_paper           # reduced, SCIP
    python -m beamforming.array.optimization.reproduce_paper --full --solver MOSEK
"""
from __future__ import annotations

import argparse

import numpy as np

from . import baselines, metrics, superdirective
from .geometry_opt import optimize_geometry


def directivity_curve(x, omegas, thetas_deg, delta):
    """Broadband DI [dB] vs look angle for geometry ``x`` (post-processed)."""
    di = np.empty(thetas_deg.shape[0])
    for i, th_deg in enumerate(thetas_deg):
        th = np.deg2rad(abs(th_deg))  # symmetric about endfire
        H = np.array(
            [superdirective.robust_superdirective(x, w, th, delta)[0] for w in omegas]
        )
        di[i] = metrics.lin2db(metrics.broadband_directivity_index(x, omegas, th, H))
    return di


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true",
                    help="Use the paper's full N=40,M=6,Q=15,P=15 config (needs MOSEK).")
    ap.add_argument("--solver", default="SCIP")
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--out", default="konforti2022_repro.png")
    args = ap.parse_args()

    # Common physical setup (Fig. 1 caption).
    A, M, d_c = 0.175, 6, 0.005
    fL, fH, theta_H, delta_db = 2000.0, 6000.0, 30.0, -10.0
    delta = metrics.db2lin(delta_db)

    if args.full:
        N, Q, P = 40, 15, 15
    else:
        N, Q, P = 24, 8, 8

    print(f"Optimizing: A={A*100:.1f}cm N={N} M={M} d_c={d_c*100:.1f}cm "
          f"f=[{fL:.0f},{fH:.0f}]Hz theta_H={theta_H:.0f}deg delta={delta_db}dB "
          f"Q={Q} P={P} solver={args.solver}")
    res = optimize_geometry(A=A, N=N, M=M, d_c=d_c, fL=fL, fH=fH,
                            theta_H_deg=theta_H, delta_db=delta_db, Q=Q, P=P,
                            solver=args.solver, time_limit=args.time_limit, verbose=True)
    print(f"  status={res.status}  time={res.solve_time:.1f}s")
    print(f"  optimal positions [cm]: {np.round(res.positions*100, 2)}")
    print(f"  worst-case broadband DI over ROI: {res.worstcase_di_db:.2f} dB")

    geometries = {
        "Proposed (opt.)": res.positions,
        "ULA": baselines.ula(M, A),
        "Dense (DMA)": baselines.dense(M, d_c),
    }
    thetas_deg = np.linspace(-theta_H, theta_H, 61)
    omegas = res.omegas

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7))

    # Fig. 1: geometry.
    ax1.add_patch(plt.Rectangle((0, -0.5), A * 100, 1.0, fill=False, ec="k"))
    ax1.scatter(res.positions * 100, np.zeros(M), c="tab:blue", zorder=3)
    ax1.set_xlim(-0.5, A * 100 + 0.5)
    ax1.set_ylim(-2, 2)
    ax1.set_yticks([])
    ax1.set_xlabel("x [cm]")
    ax1.set_title(f"Fig. 1 - Optimized geometry (M={M}, worst-case DI="
                  f"{res.worstcase_di_db:.1f} dB)")

    # Fig. 2: DI vs theta.
    for name, x in geometries.items():
        ax2.plot(thetas_deg, directivity_curve(x, omegas, thetas_deg, delta),
                 label=f"{name}")
    ax2.set_xlabel(r"$\theta$ [deg]")
    ax2.set_ylabel("Broadband DI [dB]")
    ax2.set_title("Fig. 2 - Directivity index vs. look direction")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()
