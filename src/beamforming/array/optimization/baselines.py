"""Reference linear-array geometries used for comparison (Konforti et al. 2022, Sec. 5).

All geometries are 1-D (x-coordinates only), anchored at ``x = 0`` so they are
directly comparable to the optimizer output.
"""
from __future__ import annotations

import numpy as np


def ula(M: int, A: float) -> np.ndarray:
    """Uniform linear array spread over the full aperture ``[0, A]``.

    Matches the paper's ULA baseline ("ULA geometry spread on all A").
    """
    return np.linspace(0.0, A, M)


def dense(M: int, d_c: float) -> np.ndarray:
    """Densest feasible ULA: uniform spacing at the minimum distance ``d_c``.

    This is the differential-microphone-array (DMA) baseline of the paper.
    """
    return np.arange(M) * d_c
