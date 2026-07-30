"""Far-field acoustic model for linear-array geometry optimization.

Implements the signal model of

    Y. Konforti, I. Cohen, B. Berdugo,
    "Array Geometry Optimization for Region-of-Interest Broadband Beamforming,"
    IWAENC 2022.

The array lies on the x-axis. The angle ``theta`` is measured with respect to
the **endfire** direction (theta = 0 points along the array axis), consistent
with Eq. (2) of the paper. Because a linear array is symmetric about the endfire
axis, only ``theta >= 0`` needs to be sampled during design.

All positions are in metres, frequencies (``omega``) in rad/s.

This module is intentionally self-contained and does **not** reuse the
near-field ``signal_model`` of the main package: the paper works in the
far-field endfire convention and mixing sign/normalisation conventions is the
usual source of bugs when reproducing its figures.
"""
from __future__ import annotations

import numpy as np
from scipy.constants import speed_of_sound

C_SOUND = float(speed_of_sound)


def grid_positions(A: float, N: int) -> np.ndarray:
    """Candidate microphone positions on the search grid ``[0, A]``.

    Eq. (11): ``Delta_x = A / (N - 1)`` so there are ``N`` grid points from
    ``0`` (the endfire reference) to ``A``.

    Args:
        A: Aperture length [m].
        N: Number of candidate grid points.

    Returns:
        ``(N,)`` array of x-coordinates.
    """
    if N < 2:
        raise ValueError("N must be >= 2.")
    return np.linspace(0.0, A, N)


def steering_vector(x: np.ndarray, omega, theta, c: float = C_SOUND) -> np.ndarray:
    """Far-field steering vector d(x, omega, theta), Eq. (2)/(20).

    ``d_i = exp(-j * omega/c * x_i * cos(theta))``.

    Args:
        x: ``(N,)`` microphone positions [m].
        omega: scalar angular frequency [rad/s].
        theta: scalar look angle w.r.t. endfire [rad].
        c: speed of sound [m/s].

    Returns:
        ``(N,)`` complex steering vector.
    """
    x = np.asarray(x, dtype=float)
    return np.exp(-1j * omega / c * x * np.cos(theta))


def diffuse_coherence(x: np.ndarray, omega: float, c: float = C_SOUND) -> np.ndarray:
    """Spherically-isotropic (diffuse) noise coherence matrix, Eq. (8)/(25).

    ``Gamma_ij = sinc( omega * (x_i - x_j) / c )`` with ``sinc(0) = 1``.

    This is the coherence of an ideal diffuse field and is real, symmetric and
    positive semidefinite. It is the noise model against which broadband
    directivity is maximised.

    Args:
        x: ``(N,)`` positions [m].
        omega: angular frequency [rad/s].
        c: speed of sound [m/s].

    Returns:
        ``(N, N)`` real PSD coherence matrix.
    """
    x = np.asarray(x, dtype=float)
    dx = x[:, None] - x[None, :]
    # np.sinc(y) = sin(pi y)/(pi y); we need sin(z)/z with z = omega*dx/c
    return np.sinc(omega * dx / (c * np.pi))


def freq_grid(fL: float, fH: float, Q: int) -> np.ndarray:
    """Angular-frequency samples omega_q, Eq. (17)."""
    f = np.linspace(fL, fH, Q)
    return 2.0 * np.pi * f


def angle_grid(theta_H_deg: float, P: int) -> np.ndarray:
    """Look-direction samples theta_p in [0, theta_H], Eq. (18) [rad]."""
    return np.deg2rad(np.linspace(0.0, theta_H_deg, P))


def psd_sqrt(Gamma: np.ndarray, eps_clip: float = 1e-12) -> np.ndarray:
    """Real matrix square-root factor ``B`` such that ``B.T @ B = Gamma``.

    Uses a symmetric eigendecomposition with eigenvalue clipping so that the
    (numerically) PSD diffuse-coherence matrix yields a valid factor usable
    inside a CVXPY ``sum_squares``: ``h^H Gamma h = || B h ||_2^2``.
    """
    w, V = np.linalg.eigh((Gamma + Gamma.T) / 2.0)
    w = np.clip(w, eps_clip, None)
    return (V * np.sqrt(w)).T
