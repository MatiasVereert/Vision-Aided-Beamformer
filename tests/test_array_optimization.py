"""Fast sanity tests for the array-geometry optimization module.

Run with:  pytest tests/test_array_optimization.py -q
The solver test uses a tiny instance that SCIP solves to optimality in a few
seconds; the rest are solver-free unit checks.
"""
import numpy as np
import pytest

from beamforming.array.optimization import (
    baselines,
    farfield,
    metrics,
    optimize_geometry,
    robust_superdirective,
)


# --------------------------------------------------------------------------- #
# Far-field model
# --------------------------------------------------------------------------- #
def test_diffuse_coherence_is_psd_with_unit_diagonal():
    x = farfield.grid_positions(0.175, 12)
    G = farfield.diffuse_coherence(x, 2 * np.pi * 3000, 343.0)
    assert np.allclose(np.diag(G), 1.0)
    assert np.allclose(G, G.T)
    assert np.min(np.linalg.eigvalsh(G)) > -1e-9  # PSD


def test_psd_sqrt_factorization():
    x = farfield.grid_positions(0.1, 8)
    G = farfield.diffuse_coherence(x, 2 * np.pi * 4000, 343.0)
    B = farfield.psd_sqrt(G)
    assert np.allclose(B.T @ B, G, atol=1e-8)


def test_steering_endfire_is_all_ones_phase():
    x = farfield.grid_positions(0.1, 5)
    d = farfield.steering_vector(x, 2 * np.pi * 3000, theta=0.0)
    # At endfire the reference (x=0) has zero phase; magnitudes are unit.
    assert np.allclose(np.abs(d), 1.0)
    assert d[0] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_wng_of_delay_and_sum_equals_M():
    # Distortionless delay-and-sum h = d/M gives WNG = M (max possible).
    x = farfield.grid_positions(0.1, 6)
    d = farfield.steering_vector(x, 2 * np.pi * 3000, np.deg2rad(15))
    h = d / 6
    assert metrics.white_noise_gain(d, h) == pytest.approx(6.0)


def test_db_roundtrip():
    assert metrics.lin2db(metrics.db2lin(-10.0)) == pytest.approx(-10.0)


# --------------------------------------------------------------------------- #
# Superdirective post-processing (Sec. 4)
# --------------------------------------------------------------------------- #
def test_superdirective_is_distortionless_and_meets_wng():
    x = baselines.ula(6, 0.05)
    delta = metrics.db2lin(-10.0)
    omega, theta = 2 * np.pi * 3000, np.deg2rad(10)
    h, wng, df = robust_superdirective(x, omega, theta, delta)
    d = farfield.steering_vector(x, omega, theta)
    assert np.vdot(d, h) == pytest.approx(1.0, abs=1e-6)   # distortionless
    assert wng >= delta - 1e-6                              # WNG constraint met
    assert df >= wng - 1e-6                                 # DF >= WNG for diffuse


# --------------------------------------------------------------------------- #
# Optimizer (tiny instance, solves to optimality quickly)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_optimizer_respects_constraints_and_beats_ula():
    res = optimize_geometry(
        A=0.10, N=12, M=3, d_c=0.01, fL=2000, fH=6000,
        theta_H_deg=30, delta_db=-10, Q=3, P=3, solver="SCIP", time_limit=120,
    )
    assert res.status in ("optimal", "optimal_inaccurate")
    # C1: exactly M mics.
    assert res.selection.sum() == 3
    # C2: minimum spacing respected.
    diffs = np.diff(np.sort(res.positions))
    assert np.all(diffs >= 0.01 - 1e-9)
    # Optimized worst-case DI should not be worse than a ULA's worst-case DI.
    delta = metrics.db2lin(-10.0)
    thetas = res.thetas
    from beamforming.array.optimization import superdirective, broadband_directivity_index

    def worstcase_di(x):
        vals = []
        for th in thetas:
            H = np.array([superdirective.robust_superdirective(x, w, th, delta)[0]
                          for w in res.omegas])
            vals.append(broadband_directivity_index(x, res.omegas, th, H))
        return metrics.lin2db(min(vals))

    assert worstcase_di(res.positions) >= worstcase_di(baselines.ula(3, 0.10)) - 0.2


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-m", "slow or not slow"]))
