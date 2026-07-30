"""Mixed-integer convex geometry optimization (Konforti et al. 2022, Sec. 3).

Finds the placement of ``M`` microphones on a grid of ``N`` candidate positions
over an aperture ``[0, A]`` that maximises the worst-case broadband directivity
index over a region-of-interest ``|theta| <= theta_H`` around endfire, subject
to a minimum White Noise Gain and a minimum inter-microphone distance.

The problem is the mixed-integer second-order-cone program (27):

    min_{s, h_tot}   R
    s.t.  C1: sum(s) = M                          (exactly M mics)
          C2: at most one mic per d_c window       (min. distance)
          C3: d^H h = 1     for all (q, p)         (distortionless)
          C4: ||h||^2 <= 1/delta                   (WNG >= delta)
          C5: |h_i| <= s_i / sqrt(delta)           (couple selection & coeffs)
          R  >= sum_q h(q,p)^H Gamma(q) h(q,p)     for all p   (epigraph of Eq. 26)

Since the distortionless constraint makes the DI numerator constant, maximising
the worst-case DI is equivalent to minimising the worst-case (over look
directions) diffuse-noise output power -- a convex objective (Eq. 23-26).

Solved with CVXPY. The default backend is SCIP (open-source, ships with
``pyscipopt``); MOSEK is used by the original authors and is far faster for the
full-scale problem -- pass ``solver="MOSEK"`` if a licence is available.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

try:
    import cvxpy as cp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "geometry_opt requires cvxpy. Install with `pip install cvxpy pyscipopt`."
    ) from exc

from . import farfield
from .metrics import db2lin


@dataclass
class GeometryOptResult:
    """Result of :func:`optimize_geometry`."""

    positions: np.ndarray            # (M,) selected microphone x-coordinates [m]
    selection: np.ndarray            # (N,) binary selection vector s
    grid: np.ndarray                 # (N,) candidate grid positions [m]
    objective: float                 # worst-case diffuse output power R (linear)
    worstcase_di_db: float           # 10 log10(Q / R): worst-case broadband DI [dB]
    omegas: np.ndarray               # (Q,) angular-frequency samples used
    thetas: np.ndarray               # (P,) look-angle samples used [rad]
    status: str
    solve_time: float
    params: dict = field(default_factory=dict)


def _min_distance_constraints(s, N: int, min_sep: int):
    """C2 via sliding-window: at most one selected point per ``min_sep`` window.

    ``min_sep = ceil(d_c / Delta_x)`` grid steps guarantees every selected pair
    is separated by at least ``d_c``.
    """
    cons = []
    if min_sep <= 1:
        return cons
    for i in range(N - 1):
        hi = min(i + min_sep, N)
        cons.append(cp.sum(s[i:hi]) <= 1)
    return cons


def optimize_geometry(
    A: float,
    N: int,
    M: int,
    d_c: float,
    fL: float,
    fH: float,
    theta_H_deg: float,
    delta_db: float,
    Q: int,
    P: int,
    c: float = farfield.C_SOUND,
    anchor_reference: bool = True,
    solver: str = "SCIP",
    time_limit: float | None = None,
    verbose: bool = False,
) -> GeometryOptResult:
    """Solve the MISOCP (27) for the optimal linear-array geometry.

    Args:
        A: aperture length [m].
        N: number of candidate grid positions.
        M: number of microphones to place.
        d_c: minimum distance between adjacent microphones [m].
        fL, fH: broadband frequency range [Hz].
        theta_H_deg: ROI half-width around endfire [deg].
        delta_db: minimum WNG [dB] (e.g. -10).
        Q, P: number of frequency / look-direction samples.
        c: speed of sound [m/s].
        anchor_reference: fix ``s[0] = 1`` to remove the on-grid translation
            symmetry of the beampattern (WLOG; speeds up branch-and-bound).
        solver: CVXPY MI-conic solver name ("SCIP" or "MOSEK").
        time_limit: solver time limit [s], if supported.
        verbose: pass through to the solver.

    Returns:
        :class:`GeometryOptResult`.
    """
    delta = db2lin(delta_db)
    x = farfield.grid_positions(A, N)
    omegas = farfield.freq_grid(fL, fH, Q)
    thetas = farfield.angle_grid(theta_H_deg, P)
    dx = A / (N - 1)
    min_sep = math.ceil(d_c / dx)

    if (M - 1) * min_sep > (N - 1):
        raise ValueError(
            f"Infeasible: {M} mics with min spacing {min_sep} steps do not fit in "
            f"{N} grid points. Increase N/A or decrease d_c/M."
        )

    inv_sqrt_delta = 1.0 / math.sqrt(delta)

    # Pre-factor the diffuse coherence per frequency: h^H Gamma_q h = ||B_q h||^2.
    B = [farfield.psd_sqrt(farfield.diffuse_coherence(x, w, c)) for w in omegas]
    # Steering vectors d(q, p) as constants.
    D = np.array(
        [[farfield.steering_vector(x, w, th, c) for th in thetas] for w in omegas]
    )  # (Q, P, N)

    # --- Variables ---
    s = cp.Variable(N, boolean=True)
    # Coefficients h[q][p] in C^N, one beamformer per (frequency, look direction).
    h = [[cp.Variable(N, complex=True) for _ in range(P)] for _ in range(Q)]
    R = cp.Variable(nonneg=True)  # epigraph of the worst-case objective (Eq. 26)

    cons = [cp.sum(s) == M]  # C1
    if anchor_reference:
        cons.append(s[0] == 1)
    cons += _min_distance_constraints(s, N, min_sep)  # C2

    for p in range(P):
        power_p = 0
        for q in range(Q):
            hqp = h[q][p]
            cons.append(cp.conj(D[q, p]) @ hqp == 1)          # C3 distortionless
            cons.append(cp.norm(hqp, 2) <= inv_sqrt_delta)     # C4 WNG
            cons.append(cp.abs(hqp) <= s * inv_sqrt_delta)     # C5 coupling
            power_p = power_p + cp.sum_squares(B[q] @ hqp)     # h^H Gamma_q h
        cons.append(power_p <= R)                              # Eq. (26)

    prob = cp.Problem(cp.Minimize(R), cons)

    solver_kwargs = {"verbose": verbose}
    if time_limit is not None:
        if solver.upper() == "SCIP":
            solver_kwargs["scip_params"] = {"limits/time": float(time_limit)}
        elif solver.upper() == "MOSEK":
            solver_kwargs["mosek_params"] = {"MSK_DPAR_OPTIMIZER_MAX_TIME": float(time_limit)}
    prob.solve(solver=getattr(cp, solver.upper()), **solver_kwargs)

    if s.value is None:
        raise RuntimeError(f"Solver returned no solution (status={prob.status}).")

    sel = (np.asarray(s.value).ravel() > 0.5).astype(int)
    positions = x[sel == 1]
    R_val = float(R.value)
    worstcase_di_db = 10.0 * math.log10(Q / R_val)

    return GeometryOptResult(
        positions=positions,
        selection=sel,
        grid=x,
        objective=R_val,
        worstcase_di_db=worstcase_di_db,
        omegas=omegas,
        thetas=thetas,
        status=prob.status,
        solve_time=float(prob.solver_stats.solve_time or 0.0)
        if prob.solver_stats else 0.0,
        params=dict(
            A=A, N=N, M=M, d_c=d_c, fL=fL, fH=fH, theta_H_deg=theta_H_deg,
            delta_db=delta_db, Q=Q, P=P, min_sep=min_sep, solver=solver,
        ),
    )
