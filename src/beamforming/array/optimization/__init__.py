"""Region-of-interest broadband array-geometry optimization.

Python port of Konforti, Cohen & Berdugo, "Array Geometry Optimization for
Region-of-Interest Broadband Beamforming" (IWAENC 2022).

Typical use::

    from beamforming.array.optimization import optimize_geometry, robust_superdirective

    res = optimize_geometry(
        A=0.175, N=40, M=6, d_c=0.005,
        fL=2000, fH=6000, theta_H_deg=30, delta_db=-10, Q=15, P=15,
    )
    print(res.positions)          # optimal microphone x-coordinates [m]
"""
from . import baselines, farfield, metrics, superdirective  # noqa: F401
from .geometry_opt import GeometryOptResult, optimize_geometry  # noqa: F401
from .metrics import (  # noqa: F401
    broadband_directivity_index,
    db2lin,
    directivity_factor,
    lin2db,
    white_noise_gain,
)
from .superdirective import robust_superdirective, superdirective_over_grid  # noqa: F401

__all__ = [
    "optimize_geometry",
    "GeometryOptResult",
    "robust_superdirective",
    "superdirective_over_grid",
    "broadband_directivity_index",
    "directivity_factor",
    "white_noise_gain",
    "db2lin",
    "lin2db",
    "farfield",
    "metrics",
    "superdirective",
    "baselines",
]
