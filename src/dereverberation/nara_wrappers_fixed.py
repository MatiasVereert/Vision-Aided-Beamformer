"""
Fixed-point emulation of nara_wpe Online-WPE (RLS) for FPGA feasibility studies.
=============================================================================

This module mirrors, bit-for-bit at the algorithm level, the recursive
(online) WPE step used in ``nara_wrappers.process_wpe_online`` -- i.e.
``nara_wpe.wpe.online_wpe_step`` -- but replaces every stored quantity and
arithmetic result with a **fixed-point** representation, so we can measure how
the causal Online-WPE dereverberator would behave on a Zynq/KV260-class FPGA
*before* writing any RTL.

What is emulated (the things that actually cost precision on an FPGA):
  * The inverse-correlation matrix  P (= inv_cov)  stored in URAM/BRAM.
  * The prediction filter           G (= filter_taps).
  * The STFT input window / current frame (ADC+FFT output word length).
  * The MAC accumulator results (nominator P*w, prediction, updates).
  * The scalar denominator and its reciprocal (real reciprocal unit).
  * The Kalman gain.
  * Saturation on overflow + rounding (nearest / truncate).
  * Optional Hermitian symmetrisation of P (what you get "for free" if you
    only store the upper triangle -- it also stabilises the recursion).

What is NOT re-derived here: the STFT/ISTFT themselves are kept in float
(they are well-behaved FFTs; on the FPGA they would be a Xilinx FFT IP whose
output word length is captured by the ``in`` field below). The numerically
dangerous part of WPE is the RLS recursion, and that is fully quantised.

The whole thing is exposed with the *same call signature* as
``process_wpe_online`` (plus a ``fp_cfg``), so it drops straight into the
benchmark's NODE 4.

Author: (generated with Claude) for the Vision-Aided-Beamformer thesis.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Optional

import numpy as np

from nara_wpe.utils import stft, istft
from nara_wpe.wpe import get_power_online


# ---------------------------------------------------------------------------
#  Low-level fixed-point primitive
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Fx:
    """A signed two's-complement fixed-point format: ``bits`` total, ``frac``
    fractional bits => 1 sign bit + (bits-1-frac) integer bits + frac frac bits.

    ``bits=None`` disables quantisation (pass-through = infinite precision),
    used for the float reference and for signals you want to leave untouched.
    """
    bits: Optional[int] = None
    frac: int = 0

    def q_real(self, x: np.ndarray, rounding: str, saturate: bool,
               stats: Optional["FxStats"] = None) -> np.ndarray:
        if self.bits is None:
            return x
        scale = 2.0 ** self.frac
        xs = x * scale
        if rounding == "floor":       # truncate toward -inf (cheapest in HW)
            xi = np.floor(xs)
        elif rounding == "nearest":   # round-half-up-ish (np banker's rounding)
            xi = np.round(xs)
        else:
            raise ValueError(f"unknown rounding {rounding!r}")
        hi = 2.0 ** (self.bits - 1) - 1.0
        lo = -(2.0 ** (self.bits - 1))
        if saturate:
            if stats is not None:
                n_ovf = int(np.count_nonzero((xi > hi) | (xi < lo)))
                if n_ovf:
                    stats.overflow += n_ovf
            xi = np.clip(xi, lo, hi)
        else:                          # wrap-around (2's complement modulo)
            m = 2.0 ** self.bits
            xi = ((xi - lo) % m) + lo
        return xi / scale

    def q(self, x: np.ndarray, rounding: str, saturate: bool,
          stats: Optional["FxStats"] = None) -> np.ndarray:
        """Quantise real or complex arrays (I and Q handled independently, as
        an FPGA stores them in two separate fixed-point words)."""
        if self.bits is None:
            return x
        if np.iscomplexobj(x):
            re = self.q_real(x.real, rounding, saturate, stats)
            im = self.q_real(x.imag, rounding, saturate, stats)
            return re + 1j * im
        return self.q_real(x, rounding, saturate, stats)

    def resolution(self) -> float:
        return float("inf") if self.bits is None else 2.0 ** (-self.frac)

    def max_abs(self) -> float:
        return float("inf") if self.bits is None else 2.0 ** (self.bits - 1 - self.frac)


@dataclass
class FxStats:
    """Diagnostics accumulated over a run (helps spot saturation / divergence)."""
    overflow: int = 0
    max_absP: float = 0.0
    max_absG: float = 0.0
    diverged: bool = False


# ---------------------------------------------------------------------------
#  The configuration object exposed to the benchmark
# ---------------------------------------------------------------------------
# Per-signal integer-bit budget (headroom) assuming the STFT input has been
# pre-normalised so that max|Y| ~= `normalize_target` (<= 1). `frac` is derived
# as bits - 1 - int_bits. These are the quantities an FPGA stores in BRAM/URAM
# or carries on the datapath -- this is where storage precision actually bites.
# Tuned (see __main__ self-test) so the float reference is reproduced at high
# word length and saturation is not an artefact there.
# The FIXED-POINT / SWEPT quantities are exactly the ones an FPGA STORES in
# BRAM/URAM (the inverse-correlation P and the filter G) plus the fixed-point
# I/O (STFT window in, prediction/output pred). This is the real precision +
# memory gate. Values are measured maxima (real speech, normalised max|Y|=0.5)
# plus guard bits: win<=0.5, pred<=0.42, G<=3.9, P<=1.05.
_DEFAULT_INT_BITS: Dict[str, int] = {
    "in":    1,   # STFT window / current frame   (|.| <= 0.5)   -> max 2
    "pred":  1,   # prediction error / output     (|.| <= 0.42)  -> max 2
    "g":     4,   # filter taps                    (|g| <= 3.9)   -> max 16
    "p":     4,   # inverse-correlation matrix P   (|P| <= 1.05)  -> max 16 (headroom)
}

# Signals that share the swept "word length" knob.
_SWEPT_SIGNALS = list(_DEFAULT_INT_BITS.keys())

# NOT fixed by default -- transient datapath values with ENORMOUS dynamic range
# (loud vs silent frames): the power/weighting (spans >7 decades), the quadratic
# -form accumulator nom=P*w (whose consistency with the window keeps denom=w^H P w
# non-negative -- quantising it breaks positivity and blows up the reciprocal),
# the scalar denominator, its reciprocal, and the Kalman gain. On a real FPGA
# these are guard-bit / block-floating-point (exponent-bearing) values, cheap
# because they are not stored in the big P/G arrays. Modelled as float (Fx None)
# by default; set the *_fx fields to a real Fx to study a fully-fixed datapath.


@dataclass
class FixedPointConfig:
    """Fixed-point datapath description for the Online-WPE emulation."""
    formats: Dict[str, Fx]
    rounding: str = "nearest"          # "nearest" | "floor"
    saturate: bool = True              # saturate vs wrap on overflow
    force_hermitian: bool = True       # symmetrise P each step (store-half + stability)
    normalize_target: float = 0.5      # pre-scale so max|Y| ~= this (0 disables)
    denom_floor_ratio: float = 1e-10   # relative reciprocal floor (nara-style: eps = ratio*max(denom))
    reg_load: float = 0.0              # absolute diagonal loading added to denominator (regularisation)
    nom_fx: Fx = field(default_factory=Fx)     # nominator P*w (quadratic-form acc; default float / block-float)
    pow_fx: Fx = field(default_factory=Fx)     # power / weighting     (default float / block-float)
    denom_fx: Fx = field(default_factory=Fx)   # scalar denominator format (default float / block-float)
    recip_fx: Fx = field(default_factory=Fx)   # reciprocal format      (default float / block-float)
    k_fx: Fx = field(default_factory=Fx)       # Kalman gain format     (default float / block-float)

    # ---- factory helpers ---------------------------------------------------
    @classmethod
    def _from_intbits(cls, bits: Optional[int], int_bits: Dict[str, int], **kw) -> "FixedPointConfig":
        fmts = {}
        for name, ib in int_bits.items():
            if bits is None:
                fmts[name] = Fx(None, 0)
            else:
                frac = max(0, bits - 1 - ib)
                fmts[name] = Fx(bits, frac)
        return cls(formats=fmts, **kw)

    @classmethod
    def float_ref(cls, **kw) -> "FixedPointConfig":
        """Infinite precision -- must reproduce nara float output (sanity check)."""
        kw.setdefault("force_hermitian", False)
        kw.setdefault("normalize_target", 0.0)
        return cls._from_intbits(None, _DEFAULT_INT_BITS, **kw)

    @classmethod
    def wordlength(cls, bits: int, int_bits: Optional[Dict[str, int]] = None, **kw) -> "FixedPointConfig":
        """Uniform ``bits``-wide datapath with the default per-signal int budget.
        This is the headline knob for the word-length sweep (16 / 18 / 24 / 32)."""
        ib = dict(_DEFAULT_INT_BITS)
        if int_bits:
            ib.update(int_bits)
        return cls._from_intbits(bits, ib, **kw)

    def with_bits(self, bits: int) -> "FixedPointConfig":
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
        recip = "float" if self.k_fx.bits is None else f"Q{self.k_fx.bits}.{self.k_fx.frac}"
        return ("FixedPointConfig(" + ", ".join(parts) +
                f", k/recip={recip}, round={self.rounding}, sat={self.saturate}, "
                f"herm={self.force_hermitian}, norm={self.normalize_target}, "
                f"reg_load={self.reg_load})")


# ---------------------------------------------------------------------------
#  Fixed-point Online-WPE step  (mirrors nara_wpe.wpe.online_wpe_step)
# ---------------------------------------------------------------------------
def _stable_positive_inverse_fixed(power: np.ndarray, cfg: FixedPointConfig,
                                   stats: Optional[FxStats]) -> np.ndarray:
    """Reciprocal of a positive scalar with an FPGA-representable floor.

    Mirrors nara's relative floor (eps = ratio*max(denom)) so the block-float
    reciprocal reproduces float behaviour, plus an optional absolute diagonal
    loading (reg_load) for studying regularised / bounded-gain variants.
    """
    denom = cfg.denom_fx.q(power, cfg.rounding, cfg.saturate, stats)
    denom = denom + cfg.reg_load
    eps = max(cfg.denom_floor_ratio * (np.max(denom) if denom.size else 1.0), 1e-20)
    inv = 1.0 / np.maximum(denom, eps)
    inv = cfg.recip_fx.q(inv, cfg.rounding, cfg.saturate, stats)
    return inv


def online_wpe_step_fixed(input_buffer, power_estimate, inv_cov, filter_taps,
                          alpha, taps, delay, cfg: FixedPointConfig,
                          stats: Optional[FxStats] = None):
    """One fixed-point Online-WPE step.

    Index/einsum conventions are copied verbatim from nara_wpe so the float
    reference (cfg.float_ref) reproduces nara exactly.
    """
    rnd, sat = cfg.rounding, cfg.saturate
    F, D = input_buffer.shape[-2:]

    # ---- build the (causal) prediction window --------------------------------
    window = input_buffer[:-delay - 1][::-1]
    window = window.transpose(1, 2, 0).reshape((F, taps * D))
    window = cfg.f("in").q(window, rnd, sat, stats)

    # ---- prediction error  pred = Y_t - conj(G)^T . window -------------------
    g_contrib = np.einsum('fid,fi->fd', np.conjugate(filter_taps), window)
    pred = input_buffer[-1] - g_contrib
    pred = cfg.f("pred").q(pred, rnd, sat, stats)

    # ---- nominator = P . window (block-float: keeps w^H P w >= 0) ------------
    nominator = np.einsum('fij,fj->fi', inv_cov, window)
    nominator = cfg.nom_fx.q(nominator, rnd, sat, stats)

    # ---- denominator = alpha*power + window^H . nominator   (real scalar) ----
    denom = (alpha * power_estimate).astype(window.dtype)
    denom = denom + np.einsum('fi,fi->f', np.conjugate(window), nominator)
    denom = denom.real                       # Hermitian form => real
    inv_denom = _stable_positive_inverse_fixed(denom, cfg, stats)

    # ---- Kalman gain (block-float datapath by default) -----------------------
    kalman_gain = nominator * inv_denom[:, None]
    kalman_gain = cfg.k_fx.q(kalman_gain, rnd, sat, stats)

    # ---- inv_cov update:  P <- (P - k .(window^H P)) / alpha -----------------
    wH_P = np.einsum('fj,fjm->fm', np.conjugate(window), inv_cov)
    update = np.einsum('fi,fm->fim', kalman_gain, wH_P)
    inv_cov_k = (inv_cov - update) / alpha
    if cfg.force_hermitian:
        inv_cov_k = 0.5 * (inv_cov_k + np.conjugate(np.swapaxes(inv_cov_k, -1, -2)))
    inv_cov_k = cfg.f("p").q(inv_cov_k, rnd, sat, stats)

    # ---- filter update:  G <- G + k . conj(pred) -----------------------------
    filter_taps_k = filter_taps + np.einsum('fi,fm->fim', kalman_gain, np.conjugate(pred))
    filter_taps_k = cfg.f("g").q(filter_taps_k, rnd, sat, stats)

    if stats is not None:
        stats.max_absP = max(stats.max_absP, float(np.max(np.abs(inv_cov_k))))
        stats.max_absG = max(stats.max_absG, float(np.max(np.abs(filter_taps_k))))

    return pred, inv_cov_k, filter_taps_k


# ---------------------------------------------------------------------------
#  Drop-in wrapper (same signature as process_wpe_online)
# ---------------------------------------------------------------------------
def process_wpe_online_fixed(u, taps=5, delay=1, alpha=0.9999,
                             stft_size=256, stft_shift=64,
                             fp_cfg: Optional[FixedPointConfig] = None,
                             return_stats: bool = False):
    """Fixed-point Online-WPE dereverberation, drop-in for ``process_wpe_online``.

    Parameters
    ----------
    u : (channels, samples) real array   -- multichannel time-domain input.
    taps, delay, alpha, stft_size, stft_shift : WPE / STFT parameters
        (use the same values as the benchmark: 7, 3, 0.9999, 512, 128).
    fp_cfg : FixedPointConfig
        Datapath description. Defaults to a 24-bit datapath. Use
        ``FixedPointConfig.wordlength(16|18|24|32)`` to sweep, or
        ``FixedPointConfig.float_ref()`` to reproduce the nara float baseline.
    return_stats : bool
        If True, also return an ``FxStats`` with overflow counts / max|P| / etc.
    """
    if fp_cfg is None:
        fp_cfg = FixedPointConfig.wordlength(24)
    stats = FxStats() if return_stats else None

    # 1. STFT
    Y = stft(u, size=stft_size, shift=stft_shift)
    Y = Y.transpose(1, 2, 0)          # (frames, bins, channels)
    T, F, M = Y.shape

    buffer_target_size = taps + delay + 1
    if T < buffer_target_size:
        print("Warning: Signal is too short for WPE with given taps and delay.")
        return (u, stats) if return_stats else u

    # 1b. Input normalisation (models a fixed ADC/front-end gain so that the
    #     fixed-point ranges below are meaningful and transferable).
    gnorm = 1.0
    if fp_cfg.normalize_target and fp_cfg.normalize_target > 0:
        peak = float(np.max(np.abs(Y))) + 1e-12
        gnorm = fp_cfg.normalize_target / peak
        Y = Y * gnorm
    # Quantise the STFT coefficients to the input word (ADC/FFT output).
    Y = fp_cfg.f("in").q(Y, fp_cfg.rounding, fp_cfg.saturate, stats)

    # 2. Initialise P (inv_cov) = I and G (filter_taps) = 0, per bin.
    Q = np.stack([np.identity(M * taps) for _ in range(F)]).astype(np.complex128)
    G = np.zeros((F, M * taps, M), dtype=np.complex128)

    Z_list = []

    # 3. Bypass the first (taps+delay) frames to keep temporal alignment.
    for i in range(taps + delay):
        Z_list.append(Y[i, :, :])
    buffer = list(Y[:taps + delay, :, :])

    # 4. Frame-by-frame causal processing.
    for t in range(taps + delay, T):
        buffer.append(Y[t, :, :])
        Y_step = np.array(buffer)                       # (buf, F, M)

        power = get_power_online(Y_step.transpose(1, 2, 0))
        power = fp_cfg.pow_fx.q(power, fp_cfg.rounding, fp_cfg.saturate, stats)

        Z_frame, Q, G = online_wpe_step_fixed(
            Y_step, power, Q, G,
            alpha=alpha, taps=taps, delay=delay, cfg=fp_cfg, stats=stats,
        )
        Z_list.append(Z_frame)
        buffer.pop(0)

    # 5. Reconstruct.
    Z_stacked = np.stack(Z_list)                        # (frames, F, M)

    if stats is not None and not np.all(np.isfinite(Z_stacked)):
        stats.diverged = True

    Z_out = Z_stacked.transpose(2, 0, 1)                # (channels, frames, bins)
    z_time = istft(Z_out, size=stft_size, shift=stft_shift)
    z_time = z_time[:, :u.shape[1]]

    # Undo input normalisation to return to the original signal scale.
    if gnorm != 1.0:
        z_time = z_time / gnorm

    return (z_time, stats) if return_stats else z_time


# ---------------------------------------------------------------------------
#  Self-test: prove the emulation is faithful (run this file directly).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from nara_wrappers import process_wpe_online   # float reference

    rng = np.random.default_rng(0)
    fs = 16000
    dur = 4.0
    M = 4
    n = int(fs * dur)

    # Synthetic multichannel "reverberant-ish" signal: white speech-like source
    # convolved with short random per-channel impulse responses + a late tail.
    src = rng.standard_normal(n)
    # crude 1/f-ish colouring so it is not flat white
    src = np.cumsum(src) - np.cumsum(np.concatenate([[0], src[:-1]]))
    src = src / (np.std(src) + 1e-9)
    x = np.zeros((M, n))
    for m in range(M):
        h = np.zeros(1600)
        h[10 + m * 3] = 1.0                                   # direct path (per-mic delay)
        tail = rng.standard_normal(1600) * np.exp(-np.arange(1600) / 300.0) * 0.3
        h += tail
        x[m] = np.convolve(src, h)[:n]
    x = x / (np.max(np.abs(x)) + 1e-9)

    P = dict(taps=7, delay=3, alpha=0.9999, stft_size=512, stft_shift=128)

    ref = process_wpe_online(x.copy(), **P)

    def rel_err(a, b):
        a = a[:, :b.shape[1]] if a.shape[1] > b.shape[1] else a
        b = b[:, :a.shape[1]]
        return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))

    print("=== Faithfulness / word-length sweep on synthetic 4-ch signal ===")
    # (a) float_ref must match nara float almost exactly.
    z_float, st = process_wpe_online_fixed(x.copy(), **P,
                                           fp_cfg=FixedPointConfig.float_ref(),
                                           return_stats=True)
    print(f"float_ref   rel_err vs nara = {rel_err(z_float, ref):.2e}   "
          f"(should be ~1e-12; validates the emulation math)")

    # (b) word-length sweep.
    for bits in (32, 24, 20, 18, 16, 14, 12):
        cfg = FixedPointConfig.wordlength(bits)
        z, st = process_wpe_online_fixed(x.copy(), **P, fp_cfg=cfg, return_stats=True)
        print(f"  {bits:2d}-bit  rel_err={rel_err(z, ref):.3e}  "
              f"overflow={st.overflow:>8d}  max|P|={st.max_absP:.2e}  "
              f"max|G|={st.max_absG:.2e}  diverged={st.diverged}")
