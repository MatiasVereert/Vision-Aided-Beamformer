"""Robust superdirective coefficient post-processing (Konforti et al. 2022, Sec. 4).

Once a geometry ``x*`` is fixed, the beamformer coefficients are recomputed per
look direction / frequency so that the directivity factor is maximised subject
to a minimum White Noise Gain (WNG >= delta). This is the classical
regularised (diagonally-loaded) superdirective / MVDR-against-diffuse-noise
beamformer

    h_eps = Gamma_eps^{-1} d / (d^H Gamma_eps^{-1} d),   Gamma_eps = Gamma + eps I

with ``eps`` chosen by bisection. The paper applies this same post-processing to
**every** geometry (optimized, ULA, dense) before comparing them, so the same
routine is reused for the baselines.
"""
from __future__ import annotations

import numpy as np

from . import farfield, metrics


def superdirective_beamformer(
    x: np.ndarray, omega: float, theta: float, eps: float, c: float = farfield.C_SOUND
) -> np.ndarray:
    """Diagonally-loaded superdirective beamformer for a single (omega, theta).

    ``h = Gamma_eps^{-1} d / (d^H Gamma_eps^{-1} d)``, Eq. (28)-(29).
    Satisfies the distortionless constraint ``d^H h = 1`` by construction.
    """
    M = x.shape[0]
    d = farfield.steering_vector(x, omega, theta, c)
    Gamma = farfield.diffuse_coherence(x, omega, c)
    Gamma_eps = Gamma + eps * np.eye(M)
    w = np.linalg.solve(Gamma_eps, d)
    return w / np.vdot(d, w)


def _eps_upper_bound(Gamma: np.ndarray, delta: float) -> float:
    """Analytic upper bound on eps that still guarantees WNG >= delta, Eq. (31).

    ``0 <= eps <= (lambda_1 - sqrt(M/delta) lambda_M) / (sqrt(M/delta) - 1)``.
    Returns ``0`` when the bound is non-positive (i.e. eps = 0 already meets WNG).
    """
    M = Gamma.shape[0]
    lam = np.linalg.eigvalsh(Gamma)  # ascending
    lam_min, lam_max = lam[0], lam[-1]
    r = np.sqrt(M / delta)
    if r <= 1.0:
        return 0.0
    ub = (lam_max - r * lam_min) / (r - 1.0)
    return float(max(ub, 0.0))


def robust_superdirective(
    x: np.ndarray,
    omega: float,
    theta: float,
    delta: float,
    c: float = farfield.C_SOUND,
    n_bisect: int = 40,
) -> tuple[np.ndarray, float, float]:
    """Robust superdirective beamformer meeting WNG >= delta with maximal DF.

    WNG is monotonically non-decreasing in the diagonal loading ``eps``; DF is
    non-increasing. We therefore bisect for the *smallest* ``eps`` in
    ``[0, eps_max]`` (Eq. 31) whose WNG reaches ``delta``, which maximises DF
    under the robustness constraint.

    Args:
        x: ``(M,)`` geometry.
        omega, theta: frequency [rad/s] and look angle [rad].
        delta: minimum WNG (linear, e.g. ``db2lin(-10)``).
        n_bisect: bisection iterations.

    Returns:
        ``(h, wng, df)`` with the achieved (linear) WNG and DF.
    """
    d = farfield.steering_vector(x, omega, theta, c)
    Gamma = farfield.diffuse_coherence(x, omega, c)

    def make(eps):
        h = superdirective_beamformer(x, omega, theta, eps, c)
        return h, metrics.white_noise_gain(d, h)

    # eps = 0 (max DF, i.e. plain MVDR-against-diffuse): may already be robust.
    h0, wng0 = make(0.0)
    if wng0 >= delta:
        return h0, wng0, metrics.directivity_factor(d, Gamma, h0)

    lo, hi = 0.0, _eps_upper_bound(Gamma, delta)
    if hi <= 0.0:
        # Fall back: grow hi until WNG is satisfied (numerical safety net).
        hi = max(np.trace(Gamma).real / Gamma.shape[0], 1e-6)
        while make(hi)[1] < delta:
            hi *= 2.0
            if hi > 1e12:
                break

    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        if make(mid)[1] >= delta:
            hi = mid
        else:
            lo = mid
    h, wng = make(hi)
    return h, wng, metrics.directivity_factor(d, Gamma, h)


def superdirective_over_grid(
    x: np.ndarray,
    omegas: np.ndarray,
    thetas: np.ndarray,
    delta: float,
    c: float = farfield.C_SOUND,
) -> np.ndarray:
    """Post-processed coefficients for every (theta_p, omega_q).

    Returns an array of shape ``(P, Q, M)`` (complex).
    """
    P, Q, M = thetas.shape[0], omegas.shape[0], x.shape[0]
    H = np.empty((P, Q, M), dtype=complex)
    for p, theta in enumerate(thetas):
        for q, omega in enumerate(omegas):
            H[p, q], _, _ = robust_superdirective(x, omega, theta, delta, c)
    return H
