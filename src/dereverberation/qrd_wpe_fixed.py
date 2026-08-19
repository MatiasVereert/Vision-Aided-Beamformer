"""
Inverse-QRD-RLS (square-root) Online-WPE, fixed-point FPGA feasibility study.
=============================================================================

This module is the *square-root* companion of ``nara_wrappers_fixed.py``.  It
computes exactly the same recursive (online) WPE filter as
``nara_wpe.wpe.online_wpe_step`` / ``nara_wrappers.process_wpe_online``, but it
propagates a **triangular square-root factor** of the inverse correlation matrix
instead of the inverse correlation matrix itself.

Why bother (the whole point of the study):
  * The covariance form stores ``P = R^-1`` directly.  ``P`` must stay
    positive-definite; finite-precision rounding corrupts that positivity and
    the RLS diverges.  The covariance emulation in ``nara_wrappers_fixed.py``
    shows a HARD cliff: faithful at 24 bit, divergent at <=20 bit.
  * The inverse-QRD form stores a lower-triangular factor ``L`` such that
    ``P = L . L^H``.  Then ``P`` is positive-*semidefinite* BY CONSTRUCTION for
    ANY ``L`` -- quantise ``L`` however coarsely you like, ``L L^H`` is still a
    valid (PSD) covariance.  The recursion cannot lose positivity, so it
    tolerates roughly half the word length.  This module measures how far down
    it really goes (target: stable/faithful at ~16 bit).

Algorithm (derivation kept exact so float QRD == float covariance == nara):
  The nara online step realises the weighted RLS recursion
      R_n = alpha * R_{n-1} + (1/power_n) w_n w_n^H ,     R_0 = I ,   P = R^-1 .
  With  S = L_{n-1} / sqrt(alpha)   (so  S S^H = P_{n-1}/alpha = (alpha R_{n-1})^-1)
  and   a = w_n / sqrt(power_n),   v = S^H a,   s = ||v||^2,
  the inverse-QRD pre-array
      [ 1   v^H ]                         [ beta   0^H ]
      [ 0   S   ]  --(unitary Theta)-->   [  x     B   ]
  (Theta = a sequence of complex Givens rotations that annihilate the top row,
   applied right-to-left so B stays lower-triangular) yields
      B B^H = P_n            (new triangular factor, PSD by construction)
      |beta|^2 = 1 + s       (beta = gamma^{-1/2}, the RLS conversion factor)
      k = x / (beta * sqrt(power_n))   == nara's kalman_gain  (phases cancel).
  Everything else (pred = Y - G^H w, G += k conj(pred)) is identical to nara.

Fixed-point philosophy (identical to nara_wrappers_fixed.py):
  QUANTISED (the things an FPGA STORES in BRAM/URAM -> memory + precision gate):
    - L  : the triangular square-root factor        [L x L lower-tri]  (was P)
    - G  : the prediction filter                    [L x M]
    - in : the STFT input window / current frame
    - pred: the dereverberated output
  BLOCK-FLOAT / float (transient datapath, enormous dynamic range, NOT stored):
    - power / weighting, the scaled input a, the projection v, the Givens
      rotations, beta, x, and the Kalman gain k.  On real HW these are guard-bit
      / block-floating-point (CORDIC) quantities, cheap because they never touch
      the big L/G arrays.

Author: (generated with Claude) for the Vision-Aided-Beamformer thesis.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Optional

import numpy as np

from nara_wpe.utils import stft, istft
from nara_wpe.wpe import get_power_online

# Re-use the exact same fixed-point primitive as the covariance study so the two
# emulations are strictly comparable (same rounding / saturation / bit-true I/Q).
# Works both as a package (dereverberation.qrd_wpe_fixed) and as a bare script.
try:
    from .nara_wrappers_fixed import Fx, FxStats
except ImportError:
    from nara_wrappers_fixed import Fx, FxStats


# ---------------------------------------------------------------------------
#  Per-signal integer-bit budget (headroom) for the STORED state.
# ---------------------------------------------------------------------------
# Measured maxima on real speech in the normalised domain (max|Y|=0.5):
#   win<=0.5, pred<=0.42, G<=3.9  (same as the covariance study), and the
#   triangular factor is provably bounded: P_ii = sum_k |L_ik|^2 ~ 1  =>  |L| <= 1
#   (measured max|L| = 1.000 at EVERY word length -- it never grows, unlike P
#   which explodes to ~22 when the covariance form diverges).  So L needs only
#   1 integer bit (max 2), where P needed 4 for divergence headroom; QRD spends
#   those saved bits on the fraction.  ``l`` replaces the cov study's ``p`` slot.
_DEFAULT_INT_BITS: Dict[str, int] = {
    "in":   1,   # STFT window / current frame   (|.| <= 0.5)   -> max 2
    "pred": 1,   # prediction error / output     (|.| <= 0.42)  -> max 2
    "g":    4,   # filter taps                    (|g| <= 3.9)   -> max 16
    "l":    1,   # triangular sqrt factor L       (|L| <= 1.0)   -> max 2
}

_SWEPT_SIGNALS = list(_DEFAULT_INT_BITS.keys())


# ---------------------------------------------------------------------------
#  Configuration object (mirrors FixedPointConfig, minus the P-specific knobs)
# ---------------------------------------------------------------------------
@dataclass
class QRDFixedPointConfig:
    """Fixed-point datapath description for the inverse-QRD Online-WPE."""
    formats: Dict[str, Fx]
    rounding: str = "nearest"          # "nearest" | "floor"
    saturate: bool = True              # saturate vs wrap on overflow
    normalize_target: float = 0.5      # pre-scale so max|Y| ~= this (0 disables)
    power_floor_ratio: float = 1e-12   # relative floor on power (avoids /0 in a=w/sqrt(power))
    # Transient datapath formats.  Default = float / block-float (Fx None), i.e.
    # not stored in the big arrays.  Set them to a real Fx to study a fully-fixed
    # datapath (rotations / gain), analogous to nom_fx / k_fx in the cov study.
    a_fx: Fx = field(default_factory=Fx)       # scaled input a = w / sqrt(power)
    v_fx: Fx = field(default_factory=Fx)       # projection v = S^H a
    k_fx: Fx = field(default_factory=Fx)       # Kalman gain

    # ---- factory helpers ---------------------------------------------------
    @classmethod
    def _from_intbits(cls, bits: Optional[int], int_bits: Dict[str, int], **kw) -> "QRDFixedPointConfig":
        fmts = {}
        for name, ib in int_bits.items():
            if bits is None:
                fmts[name] = Fx(None, 0)
            else:
                frac = max(0, bits - 1 - ib)
                fmts[name] = Fx(bits, frac)
        return cls(formats=fmts, **kw)

    @classmethod
    def float_ref(cls, **kw) -> "QRDFixedPointConfig":
        """Infinite precision -- must reproduce nara float output (sanity check)."""
        kw.setdefault("normalize_target", 0.0)
        return cls._from_intbits(None, _DEFAULT_INT_BITS, **kw)

    @classmethod
    def wordlength(cls, bits: int, int_bits: Optional[Dict[str, int]] = None, **kw) -> "QRDFixedPointConfig":
        """Uniform ``bits``-wide stored state with the default per-signal int budget.
        This is the headline knob for the word-length sweep (16 / 18 / 24 / 32)."""
        ib = dict(_DEFAULT_INT_BITS)
        if int_bits:
            ib.update(int_bits)
        return cls._from_intbits(bits, ib, **kw)

    def with_bits(self, bits: int) -> "QRDFixedPointConfig":
        """Return a copy where every swept signal uses ``bits`` total width."""
        new = {n: (Fx(bits, max(0, bits - 1 - _DEFAULT_INT_BITS[n]))
                   if n in _SWEPT_SIGNALS else f)
               for n, f in self.formats.items()}
        return replace(self, formats=new)

    def f(self, name: str) -> Fx:
        return self.formats[name]

    def summary(self) -> str:
        parts = [f"{n}=Q{self.f(n).bits}.{self.f(n).frac}" if self.f(n).bits else f"{n}=float"
                 for n in _SWEPT_SIGNALS]
        return ("QRDFixedPointConfig(" + ", ".join(parts) +
                f", round={self.rounding}, sat={self.saturate}, "
                f"norm={self.normalize_target})")


# ---------------------------------------------------------------------------
#  Vectorised complex Givens annihilation of the pre-array top row.
# ---------------------------------------------------------------------------
def _inverse_qrd_rank1(S: np.ndarray, v: np.ndarray,
                       cfg: "QRDFixedPointConfig",
                       stats: Optional[FxStats]):
    """Apply the inverse-QRD unitary rotation to the pre-array, per frequency bin.

    Parameters
    ----------
    S : (F, L, L) complex   -- lower-triangular  L_{n-1} / sqrt(alpha).
    v : (F, L)    complex   -- projection  S^H a.

    Returns
    -------
    B    : (F, L, L) complex -- new lower-triangular factor  (B B^H = P_n).
    x    : (F, L)    complex -- post-array bottom-left column (normalised gain).
    beta : (F,)      complex -- post-array (0,0) entry, |beta|^2 = 1 + ||v||^2.

    The rotations are ordinary (energy-preserving) complex Givens rotations, so
    the result is exact regardless of how coarsely S was quantised; B B^H is a
    valid PSD covariance by construction.  Column phases are immaterial: they
    cancel in B B^H and in the Kalman gain (see module docstring).
    """
    F, L, _ = S.shape
    # Pre-array A = [[1, v^H], [0, S]]  of shape (F, L+1, L+1).
    A = np.zeros((F, L + 1, L + 1), dtype=np.complex128)
    A[:, 0, 0] = 1.0
    A[:, 0, 1:] = np.conjugate(v)              # v^H
    A[:, 1:, 1:] = S
    # Optional fixed-point on the projection (transient; float by default).
    A[:, 0, 1:] = cfg.v_fx.q(A[:, 0, 1:], cfg.rounding, cfg.saturate, stats)

    # Annihilate the top-row entries right-to-left (keeps B lower-triangular).
    for j in range(L, 0, -1):
        p = A[:, 0, 0].copy()
        q = A[:, 0, j].copy()
        abs_p = np.abs(p)
        rho = np.hypot(abs_p, np.abs(q))
        safe = rho > 0.0
        # c real >= 0 ;  s = c * conj(q) / conj(p)  (zeros q against pivot p)
        c = np.where(safe, abs_p / np.where(safe, rho, 1.0), 1.0)
        pnz = abs_p > 0.0
        s = np.zeros(F, dtype=np.complex128)
        np.divide(c * np.conjugate(q), np.conjugate(p), out=s, where=pnz)
        # Degenerate pivot p==0 (should not happen: pivot starts at 1): pure swap.
        swap = (~pnz) & (np.abs(q) > 0.0)
        if np.any(swap):
            c = c.copy()
            c[swap] = 0.0
            s[swap] = 1.0
        # Apply the 2x2 unitary to columns 0 and j across ALL rows.
        col0 = A[:, :, 0].copy()
        colj = A[:, :, j].copy()
        A[:, :, 0] = c[:, None] * col0 + s[:, None] * colj
        A[:, :, j] = -np.conjugate(s)[:, None] * col0 + c[:, None] * colj

    beta = A[:, 0, 0]
    x = A[:, 1:, 0]
    B = A[:, 1:, 1:]
    return B, x, beta


# ---------------------------------------------------------------------------
#  Fixed-point inverse-QRD Online-WPE step  (drop-in for online_wpe_step_fixed)
# ---------------------------------------------------------------------------
def qrd_wpe_step_fixed(input_buffer, power_estimate, L_factor, filter_taps,
                       alpha, taps, delay, cfg: QRDFixedPointConfig,
                       stats: Optional[FxStats] = None, n_iter: int = 1,
                       refine_floor: float = 0.0):
    """One fixed-point inverse-QRD Online-WPE step.

    ``L_factor`` is the stored lower-triangular square-root of P (= inv_cov);
    ``P = L_factor . L_factor^H`` is never formed.  With ``cfg.float_ref()`` this
    reproduces nara_wpe's ``online_wpe_step`` exactly.

    ``n_iter``/``refine_floor`` are accepted for signature compatibility with
    ``online_wpe_step_fixed`` but only ``n_iter == 1`` is supported (the variance
    refinement was shown to hurt the beamformer and is discarded); values > 1 are
    treated as 1.
    """
    rnd, sat = cfg.rounding, cfg.saturate
    F, D = input_buffer.shape[-2:]
    L = taps * D
    Y_t = input_buffer[-1]

    # ---- build the (causal) prediction window (stored / fixed-point) --------
    window = input_buffer[:-delay - 1][::-1]
    window = window.transpose(1, 2, 0).reshape((F, L))
    window = cfg.f("in").q(window, rnd, sat, stats)

    # ---- transient datapath (block-float): scaled input & projection --------
    power = np.asarray(power_estimate, dtype=np.float64).real
    pfloor = max(cfg.power_floor_ratio * (float(np.max(power)) if power.size else 1.0), 1e-30)
    sqrt_power = np.sqrt(np.maximum(power, pfloor))
    inv_sqrt_power = 1.0 / sqrt_power
    a = window * inv_sqrt_power[:, None]                     # a = w / sqrt(power)
    a = cfg.a_fx.q(a, rnd, sat, stats)

    # S = L_{n-1} / sqrt(alpha)  (the forgetting factor scales the stored root).
    S = L_factor * (1.0 / np.sqrt(alpha))
    # v = S^H a  (project the new sample onto the current inverse-root).
    v = np.einsum('fji,fj->fi', np.conjugate(S), a)          # (S^H a)_i = sum_j conj(S_ji) a_j

    # ---- inverse-QRD unitary update: new triangular factor + gain -----------
    B, x, beta = _inverse_qrd_rank1(S, v, cfg, stats)

    # Kalman gain  k = x / (beta * sqrt(power))  == nara's kalman_gain (phases cancel).
    kalman_gain = x / (beta[:, None] * sqrt_power[:, None])
    kalman_gain = cfg.k_fx.q(kalman_gain, rnd, sat, stats)

    # ---- prediction with the pre-update filter, then filter update ----------
    pred = Y_t - np.einsum('fid,fi->fd', np.conjugate(filter_taps), window)
    filter_taps_k = filter_taps + np.einsum('fi,fm->fim', kalman_gain, np.conjugate(pred))

    # ---- commit the stored state in fixed point -----------------------------
    L_new = cfg.f("l").q(B, rnd, sat, stats)                 # new triangular factor
    filter_taps_k = cfg.f("g").q(filter_taps_k, rnd, sat, stats)
    pred_out = cfg.f("pred").q(pred, rnd, sat, stats)

    if stats is not None:
        stats.max_absP = max(stats.max_absP, float(np.max(np.abs(L_new))))   # max|L| (reuse field)
        stats.max_absG = max(stats.max_absG, float(np.max(np.abs(filter_taps_k))))

    return pred_out, L_new, filter_taps_k


# ---------------------------------------------------------------------------
#  Drop-in wrapper (same signature as process_wpe_online_fixed)
# ---------------------------------------------------------------------------
def process_qrd_wpe_online_fixed(u, taps=5, delay=1, alpha=0.9999,
                                 stft_size=256, stft_shift=64,
                                 fp_cfg: Optional[QRDFixedPointConfig] = None,
                                 return_stats: bool = False, n_iter: int = 1,
                                 refine_floor: float = 0.0):
    """Fixed-point inverse-QRD Online-WPE dereverberation.

    Drop-in replacement for ``process_wpe_online`` /
    ``process_wpe_online_fixed`` (same call signature plus ``fp_cfg``).

    Parameters
    ----------
    u : (channels, samples) real array   -- multichannel time-domain input.
    taps, delay, alpha, stft_size, stft_shift : WPE / STFT parameters.
    fp_cfg : QRDFixedPointConfig
        Datapath description.  Defaults to a 16-bit stored state (the target).
        Use ``QRDFixedPointConfig.wordlength(bits)`` to sweep, or
        ``QRDFixedPointConfig.float_ref()`` to reproduce the nara float baseline.
    return_stats : bool
        If True, also return an ``FxStats`` (overflow / max|L| / max|G| / diverged).
    """
    if fp_cfg is None:
        fp_cfg = QRDFixedPointConfig.wordlength(16)
    stats = FxStats() if return_stats else None

    # 1. STFT -> (frames, bins, channels)
    Y = stft(u, size=stft_size, shift=stft_shift)
    Y = Y.transpose(1, 2, 0)
    T, F, M = Y.shape

    if T < taps + delay + 1:
        print("Warning: Signal is too short for WPE with given taps and delay.")
        return (u, stats) if return_stats else u

    # 1b. Input normalisation (fixed ADC/front-end gain -> transferable ranges).
    gnorm = 1.0
    if fp_cfg.normalize_target and fp_cfg.normalize_target > 0:
        peak = float(np.max(np.abs(Y))) + 1e-12
        gnorm = fp_cfg.normalize_target / peak
        Y = Y * gnorm
    Y = fp_cfg.f("in").q(Y, fp_cfg.rounding, fp_cfg.saturate, stats)

    # 2. Initialise L (sqrt of P) = I and G = 0, per bin.  P_0 = L L^H = I.
    L_state = np.stack([np.identity(M * taps) for _ in range(F)]).astype(np.complex128)
    G = np.zeros((F, M * taps, M), dtype=np.complex128)

    Z_list = []

    # 3. Bypass the first (taps+delay) frames to keep temporal alignment.
    for i in range(taps + delay):
        Z_list.append(Y[i, :, :])
    buffer = list(Y[:taps + delay, :, :])

    # 4. Frame-by-frame causal processing.
    for t in range(taps + delay, T):
        buffer.append(Y[t, :, :])
        Y_step = np.array(buffer)

        power = get_power_online(Y_step.transpose(1, 2, 0))

        Z_frame, L_state, G = qrd_wpe_step_fixed(
            Y_step, power, L_state, G,
            alpha=alpha, taps=taps, delay=delay, cfg=fp_cfg, stats=stats,
            n_iter=n_iter, refine_floor=refine_floor,
        )
        Z_list.append(Z_frame)
        buffer.pop(0)

    # 5. Reconstruct.
    Z_stacked = np.stack(Z_list)

    if stats is not None and not np.all(np.isfinite(Z_stacked)):
        stats.diverged = True

    Z_out = Z_stacked.transpose(2, 0, 1)
    z_time = istft(Z_out, size=stft_size, shift=stft_shift)
    z_time = z_time[:, :u.shape[1]]

    if gnorm != 1.0:
        z_time = z_time / gnorm

    return (z_time, stats) if return_stats else z_time


# ---------------------------------------------------------------------------
#  Self-test: prove the emulation is faithful and find the QRD cliff.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import sys
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)

    from nara_wrappers import process_wpe_online                  # float reference
    from nara_wrappers_fixed import (process_wpe_online_fixed,
                                      FixedPointConfig)            # covariance study

    def rel_err(a, b):
        a = a[:, :b.shape[1]] if a.shape[1] > b.shape[1] else a
        b = b[:, :a.shape[1]]
        return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))

    def make_synth(M=4, dur=4.0, fs=16000, seed=0):
        rng = np.random.default_rng(seed)
        n = int(fs * dur)
        src = rng.standard_normal(n)
        src = np.cumsum(src) - np.cumsum(np.concatenate([[0], src[:-1]]))
        src = src / (np.std(src) + 1e-9)
        x = np.zeros((M, n))
        for m in range(M):
            h = np.zeros(1600)
            h[10 + m * 3] = 1.0
            tail = rng.standard_normal(1600) * np.exp(-np.arange(1600) / 300.0) * 0.3
            h += tail
            x[m] = np.convolve(src, h)[:n]
        return x / (np.max(np.abs(x)) + 1e-9)

    P = dict(taps=7, delay=3, alpha=0.9999, stft_size=512, stft_shift=128)
    x = make_synth(M=4)
    ref = process_wpe_online(x.copy(), **P)

    print("=" * 72)
    print("Inverse-QRD-RLS WPE -- fixed-point feasibility self-test")
    print("=" * 72)
    print(f"Signal: synthetic 4-ch, {P}")
    print()

    # (1) ALGORITHM CORRECTNESS: QRD float == nara float == covariance float.
    print("--- (1) Float-precision correctness (must be ~1e-10) ---")
    z_qrd_f, _ = process_qrd_wpe_online_fixed(
        x.copy(), **P, fp_cfg=QRDFixedPointConfig.float_ref(), return_stats=True)
    z_cov_f, _ = process_wpe_online_fixed(
        x.copy(), **P, fp_cfg=FixedPointConfig.float_ref(), return_stats=True)
    print(f"  QRD  float_ref  vs nara       rel_err = {rel_err(z_qrd_f, ref):.3e}")
    print(f"  cov  float_ref  vs nara       rel_err = {rel_err(z_cov_f, ref):.3e}")
    print(f"  QRD  float_ref  vs cov float  rel_err = {rel_err(z_qrd_f, z_cov_f):.3e}")
    print()

    # (2) WORD-LENGTH SWEEP: where is the QRD cliff?
    print("--- (2) Word-length sweep (QRD stored state L + G) ---")
    print("  bits | rel_err vs float |  overflow | max|L|   | max|G|   | diverged")
    print("  -----|------------------|-----------|----------|----------|---------")
    for bits in (32, 24, 20, 18, 16, 14, 12):
        cfg = QRDFixedPointConfig.wordlength(bits)
        z, st = process_qrd_wpe_online_fixed(x.copy(), **P, fp_cfg=cfg, return_stats=True)
        print(f"   {bits:2d}  |    {rel_err(z, ref):.3e}   | {st.overflow:>9d} | "
              f"{st.max_absP:.2e} | {st.max_absG:.2e} | {st.diverged}")
    print()

    # (3) Side-by-side with the covariance study on the same signal.
    print("--- (3) covariance-form sweep on the SAME signal (for contrast) ---")
    print("  bits | rel_err vs float |  overflow | max|P|   | max|G|   | diverged")
    print("  -----|------------------|-----------|----------|----------|---------")
    for bits in (32, 24, 20, 18, 16, 14, 12):
        cfg = FixedPointConfig.wordlength(bits)
        z, st = process_wpe_online_fixed(x.copy(), **P, fp_cfg=cfg, return_stats=True)
        print(f"   {bits:2d}  |    {rel_err(z, ref):.3e}   | {st.overflow:>9d} | "
              f"{st.max_absP:.2e} | {st.max_absG:.2e} | {st.diverged}")
