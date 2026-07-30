"""Beamformer robustness / directivity metrics (Konforti et al. 2022, Sec. 2).

All beamformers are column vectors ``h`` of the same length as the geometry
``x``; steering vectors ``d`` and coherence matrices ``Gamma`` come from
:mod:`farfield`.
"""
from __future__ import annotations

import numpy as np

from . import farfield


def db2lin(x_db: float) -> float:
    """dB -> linear power ratio."""
    return 10.0 ** (x_db / 10.0)


def lin2db(x: float | np.ndarray) -> float | np.ndarray:
    """linear power ratio -> dB."""
    return 10.0 * np.log10(x)


def white_noise_gain(d: np.ndarray, h: np.ndarray) -> float:
    """White Noise Gain, Eq. (6): ``|d^H h|^2 / (h^H h)``."""
    num = np.abs(np.vdot(d, h)) ** 2
    den = np.real(np.vdot(h, h))
    return float(num / den)


def directivity_factor(d: np.ndarray, Gamma: np.ndarray, h: np.ndarray) -> float:
    """Narrowband directivity factor, Eq. (7): ``|d^H h|^2 / (h^H Gamma h)``."""
    num = np.abs(np.vdot(d, h)) ** 2
    den = np.real(np.vdot(h, Gamma @ h))
    return float(num / den)


def broadband_directivity_index(
    x: np.ndarray,
    omegas: np.ndarray,
    theta: float,
    coeffs: np.ndarray,
    c: float = farfield.C_SOUND,
) -> float:
    """Broadband directivity index over ``omegas`` at look angle ``theta``, Eq. (9).

    ``DI = sum_q |d_q^H h_q|^2 / sum_q (h_q^H Gamma_q h_q)`` (rectangular
    approximation of the integral; the common ``d_omega`` cancels).

    Args:
        x: ``(M,)`` geometry.
        omegas: ``(Q,)`` angular-frequency samples.
        theta: look angle [rad].
        coeffs: ``(Q, M)`` beamformer per frequency for this look direction.
        c: speed of sound.

    Returns:
        Broadband DI (linear). Use :func:`lin2db` for dB.
    """
    num = 0.0
    den = 0.0
    for q, omega in enumerate(omegas):
        d = farfield.steering_vector(x, omega, theta, c)
        Gamma = farfield.diffuse_coherence(x, omega, c)
        h = coeffs[q]
        num += np.abs(np.vdot(d, h)) ** 2
        den += np.real(np.vdot(h, Gamma @ h))
    return float(num / den)
