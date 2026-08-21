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
from nara_wpe.wpe import (
    get_power_online, build_y_tilde, get_power_inverse,
    get_correlations_v6, perform_filter_operation_v5, hermite,
)
# Solve float del block (Hermitianiza + carga diagonal + LAPACK); lo reusamos
# como nucleo del solve fixed (la parte de storage-precision la modelamos afuera).
from dereverberation.nara_wrappers import _block_cholesky_solve, _block_load, block_wpe_warmup


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

    # ---- BLOCK-FLOAT del BUFFER (path block-online / HLS) --------------------
    # El buffer de L tramas es lo que domina la memoria on-chip. En vez de
    # fixed uniforme (f("in")), que desperdicia bits en headroom por el enorme
    # rango dinamico del STFT, se guarda block-float: un exponente COMPARTIDO
    # entre los M microfonos de cada (t,f) + mantisa baja. Recupera el rango
    # dinamico "gratis" y todos los bits de mantisa son precision util.
    buffer_blockfloat: bool = False            # si True, el buffer se guarda block-float
    buffer_mant_bits: Optional[int] = None     # bits de mantisa (signed) por componente I/Q
    buffer_exp_bits: Optional[int] = None      # ancho del exponente compartido (None=ilimitado)

    # ---- BLOCK-FLOAT del FILTRO G (otro storage persistente: se aplica cada frame) --
    # G es (F, taps*M, M). Exponente compartido por (bin, canal de salida) sobre los
    # taps*M coeficientes -> axis=1. OJO: G no es promediado como R y esta mal-
    # condicionado (cond(R)^2), asi que tolera MENOS bajada de mantisa que el buffer.
    g_blockfloat: bool = False
    g_mant_bits: Optional[int] = None
    g_exp_bits: Optional[int] = None

    # ---- DATAPATH del SOLVE (factorizacion Cholesky + sustitucion) -----------
    # Ancho de palabra INTERNO del solver (donde muerde cond(R)^2). None = float
    # (usa el solve LAPACK del path float). Con un Fx real -> _block_cholesky_solve_fixed.
    solve_fx: Fx = field(default_factory=Fx)
    # Algoritmo del solve: "cholesky" (sobre R=A^H A, cond^2) o "qr" (Householder
    # sobre la matriz de datos A, cond(A)=sqrt(cond(R)) -> ~la mitad de bits).
    solve_method: str = "cholesky"

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
                          stats: Optional[FxStats] = None, n_iter: int = 1,
                          refine_floor: float = 0.0):
    """One fixed-point Online-WPE step, with optional per-frame variance
    refinement (``n_iter`` > 1).

    n_iter == 1 reproduces the standard online WPE (power estimated from the
    observed signal) and, with cfg.float_ref(), matches nara_wpe exactly.

    n_iter > 1 emulates the batch WPE outer loop *locally, within a frame*:
    it alternates  predict Z (with the current tentative filter) -> re-estimate
    the variance from the DEREVERBERATED output Z -> re-derive the filter from
    the (unchanged) previous covariance P_prev with that better variance.
    The covariance update is committed ONCE at the end (P is a running estimate;
    re-applying it per inner iteration would corrupt the recursion). This costs
    only extra COMPUTE per frame (no look-ahead / no buffering) -> zero added
    algorithmic latency, which is exactly what a fast FPGA clock can absorb.
    """
    rnd, sat = cfg.rounding, cfg.saturate
    F, D = input_buffer.shape[-2:]
    Y_t = input_buffer[-1]

    # ---- build the (causal) prediction window --------------------------------
    window = input_buffer[:-delay - 1][::-1]
    window = window.transpose(1, 2, 0).reshape((F, taps * D))
    window = cfg.f("in").q(window, rnd, sat, stats)

    # ---- nominator = P . window : depends only on P & window (compute once) --
    nominator = np.einsum('fij,fj->fi', inv_cov, window)
    nominator = cfg.nom_fx.q(nominator, rnd, sat, stats)
    wHn = np.einsum('fi,fi->f', np.conjugate(window), nominator).real  # w^H P w >= 0

    # ---- per-frame refinement loop (transient datapath, block-float) ---------
    g_cur = filter_taps            # tentative filter, re-derived from P_prev each iter
    kalman_gain = None
    pred = None
    for it in range(max(1, n_iter)):
        # prediction with the current tentative filter
        pred = Y_t - np.einsum('fid,fi->fd', np.conjugate(g_cur), window)
        # variance: iter 0 uses the observed-signal estimate (baseline);
        # later iters re-estimate it from the dereverberated output Z.
        if it == 0:
            power = power_estimate
        else:
            power = np.mean(np.abs(pred) ** 2, axis=-1)     # mean over channels of |Z|^2
            if refine_floor > 0.0:                          # stabiliser: floor at ratio*baseline
                power = np.maximum(power, refine_floor * power_estimate)
            power = cfg.pow_fx.q(power, rnd, sat, stats)
        denom = (alpha * power).astype(window.dtype).real + wHn
        inv_denom = _stable_positive_inverse_fixed(denom, cfg, stats)
        kalman_gain = nominator * inv_denom[:, None]
        kalman_gain = cfg.k_fx.q(kalman_gain, rnd, sat, stats)
        # re-derive the filter from the ORIGINAL committed filter (not compounding)
        g_cur = filter_taps + np.einsum('fi,fm->fim', kalman_gain, np.conjugate(pred))

    # ---- commit inv_cov update ONCE:  P <- (P - k .(w^H P)) / alpha ----------
    wH_P = np.einsum('fj,fjm->fm', np.conjugate(window), inv_cov)
    update = np.einsum('fi,fm->fim', kalman_gain, wH_P)
    inv_cov_k = (inv_cov - update) / alpha
    if cfg.force_hermitian:
        inv_cov_k = 0.5 * (inv_cov_k + np.conjugate(np.swapaxes(inv_cov_k, -1, -2)))
    inv_cov_k = cfg.f("p").q(inv_cov_k, rnd, sat, stats)

    # ---- commit filter; output = last in-loop prediction ---------------------
    # (with n_iter=1 this is exactly nara's pred, computed with the pre-update
    #  filter; with n_iter>1 it used the most-refined filter available.)
    filter_taps_k = cfg.f("g").q(g_cur, rnd, sat, stats)
    pred_out = cfg.f("pred").q(pred, rnd, sat, stats)

    if stats is not None:
        stats.max_absP = max(stats.max_absP, float(np.max(np.abs(inv_cov_k))))
        stats.max_absG = max(stats.max_absG, float(np.max(np.abs(filter_taps_k))))

    return pred_out, inv_cov_k, filter_taps_k


# ---------------------------------------------------------------------------
#  Drop-in wrapper (same signature as process_wpe_online)
# ---------------------------------------------------------------------------
def process_wpe_online_fixed(u, taps=5, delay=1, alpha=0.9999,
                             stft_size=256, stft_shift=64,
                             fp_cfg: Optional[FixedPointConfig] = None,
                             return_stats: bool = False, n_iter: int = 1,
                             refine_floor: float = 0.0):
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
            n_iter=n_iter, refine_floor=refine_floor,
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


# ===========================================================================
#  FIXED-POINT BLOCK-ONLINE WPE  (Opcion B: Cholesky por bloque)
# ---------------------------------------------------------------------------
# A diferencia del RLS, el block NO tiene estado recursivo (P). Lo que un FPGA
# ALMACENA on-chip y define la precision/memoria es:
#   * el BUFFER de ventana (STFT observado, L frames) -> formato cfg.f("in")
#   * el filtro G (se aplica cada frame)               -> formato cfg.f("g")
#   * (R es SCRATCH: se rearma por re-solve; su precision de datapath -> f("p"))
#   * la salida/pred                                    -> formato cfg.f("pred")
# Reusa la MISMA FixedPointConfig que el path online. Como el buffer se promedia
# sobre L frames al formar R (~sqrt(L) de atenuacion de ruido) y el solve es
# independiente + carga diagonal, el buffer tolera muchos MENOS bits que el audio
# -> se puede poner cfg.f("in") a 8-10 bits y cfg.f("g") mas alto, y asi meter L
# grande on-chip sin DDR. El solve/Cholesky se hace en float (modelamos la
# precision de ALMACENAMIENTO, que es la que muerde memoria; no el datapath del
# sqrt, que es guard-bit/block-float como en el online).
# ===========================================================================


def blockfloat_quantize(x, mant_bits, axis, exp_bits=None, stats=None):
    """Cuantizacion block-floating-point de un arreglo complejo ``x``.

    Un UNICO exponente (peso del LSB) se comparte sobre el grupo que se obtiene
    reduciendo ``axis`` (p.ej. el eje de microfonos), independiente para cada
    indice restante -> por cada (t,f) los M mics comparten exponente. Cada
    componente real (I y Q) se guarda con ``mant_bits`` de mantisa signed en
    complemento a 2; el exponente compartido corre la coma.

    Modela EXACTAMENTE lo que se almacena on-chip: 2*M mantisas de ``mant_bits``
    + 1 exponente por grupo. El aritmetico (outer products para R) sigue afuera
    en el scratch ancho -> esto solo cambia el STORAGE, transparente al rebuild
    de R y a las iteraciones.

    Parameters
    ----------
    x : complex ndarray
    mant_bits : int      -- bits de mantisa signed por componente (None = passthrough).
    axis : int           -- eje a reducir para el exponente compartido (mics).
    exp_bits : int|None  -- ancho del exponente almacenado; acota el rango de
                            exponentes a 2**exp_bits pasos anclados al maximo
                            global (modela la memoria del exponente). None = ilimitado.
    stats : FxStats|None  -- cuenta saturaciones de mantisa (deberian ser ~0).
    """
    if mant_bits is None:
        return x
    re, im = x.real, x.imag
    mag = np.maximum(np.abs(re), np.abs(im))
    amax = np.max(mag, axis=axis, keepdims=True)                 # pico por grupo (t,f)
    # Peso del LSB por grupo: el pico mapea a fondo de escala 2**(mant_bits-1).
    with np.errstate(divide="ignore"):
        e = np.ceil(np.log2(amax)) - (mant_bits - 1)
    e = np.where(amax > 0, e, 0.0)
    if exp_bits is not None:                                     # acota el rango del exponente
        emax = float(np.max(e))
        emin = emax - (2 ** exp_bits - 1)
        e = np.clip(e, emin, emax)
    scale = 2.0 ** (-e)
    hi = 2.0 ** (mant_bits - 1) - 1.0
    lo = -(2.0 ** (mant_bits - 1))

    def _q(v):
        vi = np.round(v * scale)
        if stats is not None:
            n = int(np.count_nonzero((vi > hi) | (vi < lo)))
            if n:
                stats.overflow += n
        return np.clip(vi, lo, hi) / scale

    return _q(re) + 1j * _q(im)


def blockfloat_bits_per_complex(mant_bits, group_size, exp_bits):
    """Bits/muestra-compleja del storage block-float: 2 mantisas + exp/grupo."""
    return 2 * mant_bits + (exp_bits or 0) / float(group_size)


def _block_cholesky_solve_fixed(R, P, reg, fx, rounding="nearest", saturate=True, stats=None):
    """Resuelve R G = P por Cholesky con el DATAPATH INTERNO en punto fijo ``fx``.

    Modela el ancho de palabra de la FACTORIZACION + SUSTITUCION (donde muerde
    cond(R)^2), NO el storage de G (eso es aparte, en _estimate_block_filter_fixed).

    Estructura fiel a HLS:
      * Carga diagonal (reg) + Hermitianizacion en float (pre-datapath).
      * ESCALA block-float por bin (potencia de 2 sobre diag_mean) -> mete R,P en
        rango O(1). G es invariante al escalado (s.R)G=(s.P), asi que el escalado
        NO cambia la solucion: solo separa RANGO (exponente) de SIGNIFICANCIA. El
        `frac` de ``fx`` es entonces el knob puro de significancia -> revela el
        piso por cond^2 (lo que block-float NO puede rescatar).
      * k-sums acumuladas en float ancho (acumulador DSP); se cuantiza cada
        L/z/G escrito (lo que iria a registro/BRAM ap_fixed) + las entradas R,P.
      * sqrt/division cuantizadas al mismo formato (unidad CORDIC/recip del solve).

    R:(F,N,N) Herm, P:(F,N,M). Devuelve G:(F,N,M).
    """
    F, N, _ = R.shape
    M = P.shape[-1]
    R = _block_load(R, reg, N)                              # Herm + carga diagonal (float)

    # --- escala block-float por bin: diag_mean -> 2^-e para llevar R,P a O(1) ---
    diag_mean = np.einsum('fii->f', R).real / N
    e = np.where(diag_mean > 0, np.round(np.log2(np.maximum(diag_mean, 1e-30))), 0.0)
    s = (2.0 ** (-e))[:, None, None]
    Rs = fx.q(R * s, rounding, saturate, stats)            # R escalado y cuantizado al datapath
    Ps = fx.q(P * s, rounding, saturate, stats)

    def q(x):  return fx.q(x, rounding, saturate, stats)
    def qr(x): return fx.q_real(x, rounding, saturate, stats)

    # --- Cholesky (columna a columna, vectorizado sobre F) : Rs = L L^H ---
    L = np.zeros((F, N, N), dtype=Rs.dtype)
    for j in range(N):
        if j > 0:
            sd = Rs[:, j, j].real - (np.abs(L[:, j, :j]) ** 2).sum(axis=1)
        else:
            sd = Rs[:, j, j].real
        res = fx.resolution()                              # 1 LSB del datapath
        Ljj = qr(np.sqrt(np.maximum(sd, res * res)))       # cancelacion cond^2 -> aca duele
        Ljj = np.maximum(Ljj, res)                         # piso de pivote (HW no divide por 0)
        L[:, j, j] = Ljj
        if j + 1 < N:
            if j > 0:
                acc = np.einsum('fik,fk->fi', L[:, j + 1:, :j], np.conjugate(L[:, j, :j]))
            else:
                acc = 0.0
            L[:, j + 1:, j] = q((Rs[:, j + 1:, j] - acc) / Ljj[:, None])

    # --- sustitucion adelante  L z = Ps ---
    Z = np.zeros((F, N, M), dtype=Rs.dtype)
    for i in range(N):
        acc = np.einsum('fk,fkm->fm', L[:, i, :i], Z[:, :i, :]) if i > 0 else 0.0
        Z[:, i, :] = q((Ps[:, i, :] - acc) / L[:, i, i][:, None])

    # --- sustitucion atras  L^H G = z   (L^H[i,k] = conj(L[k,i])) ---
    G = np.zeros((F, N, M), dtype=Rs.dtype)
    for i in range(N - 1, -1, -1):
        if i + 1 < N:
            acc = np.einsum('fk,fkm->fm', np.conjugate(L[:, i + 1:, i]), G[:, i + 1:, :])
        else:
            acc = 0.0
        G[:, i, :] = q((Z[:, i, :] - acc) / np.conjugate(L[:, i, i])[:, None])
    # G resuelve (s.R)G=(s.P) == R G = P  -> NO hay que des-escalar.
    return G


def _block_qr_solve_fixed(A, B, reg, fx, rounding="nearest", saturate=True, stats=None):
    """Resuelve min ||A G - B|| (== R G = P con R=A^H A, P=A^H B) por QR de
    HOUSEHOLDER con datapath fijo ``fx``, en vez de Cholesky sobre R.

    Ventaja clave: nunca forma R=A^H A, asi que trabaja en cond(A)=sqrt(cond(R))
    -> ~la mitad de los bits que el Cholesky para el mismo reg. Las reflexiones
    son ortogonales (preservan norma) => rango dinamico acotado, amigable a fixed.

    A:(F,T,N) alto (T = ventana [+ N filas de regularizacion]), B:(F,T,M).
    Devuelve G:(F,N,M).

    Regularizacion: se AUMENTA A con N filas ``sqrt(load).I`` (y B con ceros),
    load = reg * mean_k ||A[:,:,k]||^2 -> resuelve el mismo sistema Tikhonov que
    el ``_block_load`` del Cholesky (R+load.I), pero sin elevar cond al cuadrado.

    Fidelidad HLS: se cuantiza el vector de Householder ``v``, la fila triangular
    finalizada de R y la fila transformada de B (lo que la QRD systolica carga en
    fixed). Los acumuladores de la reflexion van en float ancho (celda DSP).
    """
    F, T, N = A.shape
    Mm = B.shape[2]
    on = fx.bits is not None
    def q(x):  return fx.q(x, rounding, saturate, stats) if on else x
    def qr_(x): return fx.q_real(x, rounding, saturate, stats) if on else x

    # --- escala block-float por bin (rango O(1); G invariante al escalado) ---
    rms = np.sqrt(np.mean(np.abs(A) ** 2, axis=(1, 2)))                 # (F,)
    e = np.where(rms > 0, np.round(np.log2(np.maximum(rms, 1e-30))), 0.0)
    s = (2.0 ** (-e))[:, None, None]
    A = A * s
    B = B * s

    # --- filas de regularizacion sqrt(load).I (Tikhonov via QR) ---
    load = reg * (np.abs(A) ** 2).sum(axis=1).mean(axis=1)              # (F,)
    Ireg = np.sqrt(load)[:, None, None] * np.eye(N, dtype=A.dtype)[None]
    R = q(np.concatenate([A, Ireg], axis=1))                           # (F, T+N, N)
    Bt = q(np.concatenate([B, np.zeros((F, N, Mm), dtype=B.dtype)], axis=1))
    Tt = T + N

    # --- QR de Householder (por columna, vectorizado sobre F) ---
    for k in range(N):
        x = R[:, k:, k]                                                # (F, Tt-k)
        nrm = np.sqrt((np.abs(x) ** 2).sum(axis=1))                    # (F,)
        x0 = x[:, 0]
        phase = np.where(np.abs(x0) > 1e-30, x0 / np.maximum(np.abs(x0), 1e-30), 1.0 + 0j)
        alpha = -phase * nrm                                           # (F,) : |alpha|=||x||
        v = x.copy()
        v[:, 0] = x0 - alpha
        vn2 = (np.abs(v) ** 2).sum(axis=1)                            # (F,)
        beta = np.where(vn2 > 0, 2.0 / np.where(vn2 > 0, vn2, 1.0), 0.0)
        v = q(v)
        # reflexion H = I - beta v v^H aplicada al bloque restante y a B
        w = np.einsum('ft,ftn->fn', np.conjugate(v), R[:, k:, k:])
        R[:, k:, k:] -= beta[:, None, None] * v[:, :, None] * w[:, None, :]
        wb = np.einsum('ft,ftm->fm', np.conjugate(v), Bt[:, k:, :])
        Bt[:, k:, :] -= beta[:, None, None] * v[:, :, None] * wb[:, None, :]
        # finalizar + cuantizar la fila triangular de R y la fila de B
        R[:, k, k] = q(alpha)
        if k + 1 < N:
            R[:, k, k + 1:] = q(R[:, k, k + 1:])
        R[:, k + 1:, k] = 0.0
        Bt[:, k, :] = q(Bt[:, k, :])

    # --- sustitucion atras: R[:N,:N] G = Bt[:N] ---
    G = np.zeros((F, N, Mm), dtype=R.dtype)
    for i in range(N - 1, -1, -1):
        acc = np.einsum('fk,fkm->fm', R[:, i, i + 1:N], G[:, i + 1:, :]) if i + 1 < N else 0.0
        G[:, i, :] = q((Bt[:, i, :] - acc) / R[:, i, i][:, None])
    return G


def _build_weighted_AB(Y_win, Y_tilde_win, inverse_power):
    """Matriz de datos ponderada para el QR: A(F,T,N), B(F,T,M) tal que
    A^H A = R y A^H B = P (mismos R,P que get_correlations_v6)."""
    w = np.sqrt(np.maximum(inverse_power, 0.0))                        # (F, T)
    A = (w[:, None, :] * np.conjugate(Y_tilde_win)).transpose(0, 2, 1)  # (F, T, KM)
    B = (w[:, None, :] * np.conjugate(Y_win)).transpose(0, 2, 1)        # (F, T, D)
    return np.ascontiguousarray(A), np.ascontiguousarray(B)


def _estimate_block_filter_fixed(Y_win, Y_tilde_win, iterations, reg, cfg, stats):
    rnd, sat = cfg.rounding, cfg.saturate
    X = Y_win
    G_float = None
    
    # We must calculate a noise floor relative to the LSB of the input format
    # to prevent amplifying quantization noise during the inverse power weighting.
    in_res = cfg.f("in").resolution()
    power_floor = (in_res ** 2) * 10.0  # Guard margin above pure quantization noise
    
    use_qr = (cfg.solve_method == "qr")
    for it in range(iterations):
        # Calculate raw power, then apply the noise floor stabilization
        power = np.mean(np.abs(X) ** 2, axis=-2)
        power = np.maximum(power, power_floor)
        inverse_power = 1.0 / power

        if use_qr:
            # QR sobre la matriz de datos ponderada (no forma R=A^H A -> cond(A)=sqrt(cond(R))).
            A, B = _build_weighted_AB(Y_win, Y_tilde_win, inverse_power)
            G_float = _block_qr_solve_fixed(A, B, reg, cfg.solve_fx,
                                            rounding=rnd, saturate=sat, stats=stats)
        else:
            R, P = get_correlations_v6(Y_win, Y_tilde_win, inverse_power)
            if stats is not None:
                stats.max_absP = max(stats.max_absP, float(np.max(np.abs(R))) if R.size else 0.0)
            # Solve: float (LAPACK) o datapath fijo (factorizacion Cholesky en fx).
            if cfg.solve_fx.bits is not None:
                G_float = _block_cholesky_solve_fixed(R, P, reg, cfg.solve_fx,
                                                      rounding=rnd, saturate=sat, stats=stats)
            else:
                G_float = _block_cholesky_solve(R, P, reg=reg)
        
        # Keep G in float (or block-float datapath) for the intermediate target signal 
        # to allow statistical convergence. Only the final G gets hard-quantized for BRAM.
        if it < iterations - 1:
            X = perform_filter_operation_v5(Y=Y_win, Y_tilde=Y_tilde_win, filter_matrix=G_float)
            
    # Apply storage precision quantization only once the filter has converged.
    # G es storage persistente (se aplica cada frame) -> block-float opcional, con
    # exponente compartido por (bin, canal de salida) sobre los taps*M coeficientes.
    if cfg.g_blockfloat:
        G_quantized = blockfloat_quantize(G_float, cfg.g_mant_bits, axis=1,
                                          exp_bits=cfg.g_exp_bits, stats=stats)
    else:
        G_quantized = cfg.f("g").q(G_float, rnd, sat, stats)
    
    if stats is not None:
        stats.max_absG = max(stats.max_absG, float(np.max(np.abs(G_quantized))) if G_quantized.size else 0.0)
        
    return G_quantized

def process_wpe_block_online_fixed(u, taps=5, delay=1, L=256, block_shift=32,
                                   iterations=3, reg=1e-6, stft_size=512, stft_shift=128,
                                   fp_cfg: Optional[FixedPointConfig] = None,
                                   return_stats: bool = False):
    """Block-online WPE en punto fijo, drop-in de ``process_wpe_block_online``.

    fp_cfg : precision de las cantidades ALMACENADAS (buffer f("in"), filtro
    f("g"), correlacion f("p"), salida f("pred")). Usar por ej.
    ``FixedPointConfig.wordlength(16)`` (uniforme) o construir una config con
    f("in") mas baja que f("g") para estudiar buffer a 8-10 bits (la idea que
    mete L=512 on-chip).
    """
    if fp_cfg is None:
        fp_cfg = FixedPointConfig.wordlength(16)
    stats = FxStats() if return_stats else None
    rnd, sat = fp_cfg.rounding, fp_cfg.saturate

    if u.ndim == 1:
        u = u[np.newaxis, :]

    Y = np.ascontiguousarray(stft(u, size=stft_size, shift=stft_shift).transpose(2, 0, 1))
    F, D, T = Y.shape
    warmup = taps + delay
    if T < warmup + 1:
        print("Warning: Signal is too short for block WPE with given taps and delay.")
        return (u, stats) if return_stats else u

    # Normalizacion de front-end (rangos fixed transferibles) + cuantizacion del
    # BUFFER a f("in"): esta es la precision que domina la memoria on-chip.
    gnorm = 1.0
    if fp_cfg.normalize_target and fp_cfg.normalize_target > 0:
        peak = float(np.max(np.abs(Y))) + 1e-12
        gnorm = fp_cfg.normalize_target / peak
        Y = Y * gnorm
    # Cuantizacion del BUFFER (lo que domina la memoria on-chip). Y es (F, D, T),
    # asi que axis=1 comparte exponente ENTRE MICROFONOS por cada (t,f).
    if fp_cfg.buffer_blockfloat:
        Y = blockfloat_quantize(Y, fp_cfg.buffer_mant_bits, axis=1,
                                exp_bits=fp_cfg.buffer_exp_bits, stats=stats)
    else:
        Y = fp_cfg.f("in").q(Y, rnd, sat, stats)      # fixed uniforme (baseline)

    Y_tilde = build_y_tilde(Y, taps, delay)
    X = Y.copy()

    min_window = max(warmup + 8, 4 * taps)
    G_current = None
    for t_r in range(warmup, T, block_shift):
        lo = max(0, t_r - L)
        if (t_r - lo) >= min_window:
            G_current = _estimate_block_filter_fixed(
                Y[:, :, lo:t_r], Y_tilde[:, :, lo:t_r], iterations, reg, fp_cfg, stats)
        hi = min(t_r + block_shift, T)
        if G_current is not None:
            Xb = perform_filter_operation_v5(
                Y=Y[:, :, t_r:hi], Y_tilde=Y_tilde[:, :, t_r:hi], filter_matrix=G_current)
            X[:, :, t_r:hi] = fp_cfg.f("pred").q(Xb, rnd, sat, stats)

    if stats is not None and not np.all(np.isfinite(X)):
        stats.diverged = True

    z_time = istft(X.transpose(1, 2, 0), size=stft_size, shift=stft_shift)
    z_time = z_time[:, :u.shape[1]]
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
