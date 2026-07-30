"""
Acoustic descriptors for room impulse responses (RIRs).

Single source of truth shared by the simulator calibration and the MIRD
validation. Currently implements the Schroeder-integration RT60 estimator
(T20/T30), matching the method used to characterise the MIRD database
(Hadad et al., 2014). DRR and inter-microphone coherence will be added here too.
"""
import numpy as np
from scipy.signal import csd

C_SOUND = 343.0


def _to_1d(rir):
    return np.asarray(rir, dtype=float).flatten()


def _find_noise_floor_end(energy, fs, floor_offset_db=10.0, tail_frac=0.10):
    """
    Estimate the sample index at which the decay meets the noise floor.
    Everything past this point is noise and must be excluded from the Schroeder
    integral, otherwise the EDC flattens out (the classic bias with *measured*
    RIRs). Simulated RIRs are essentially noise-free, so this returns the full
    length in practice.

    Rule: noise power = mean energy over the last `tail_frac` of the response;
    the decay ends at the last sample where a ~10 ms smoothed energy envelope
    still sits `floor_offset_db` above that noise power.
    """
    n = len(energy)
    if n == 0:
        return 0
    tail = energy[int((1.0 - tail_frac) * n):]
    noise_pow = float(np.mean(tail)) if len(tail) else 0.0
    if noise_pow <= 0:
        return n

    win = max(1, int(0.010 * fs))
    env = np.convolve(energy, np.ones(win) / win, mode='same')

    thresh = noise_pow * 10 ** (floor_offset_db / 10.0)
    above = np.where(env > thresh)[0]
    if len(above) == 0:
        return n
    return int(above[-1]) + 1


def energy_decay_curve(rir, fs, truncate='auto', floor_offset_db=10.0):
    """
    Schroeder energy decay curve (EDC) in dB, normalised to 0 dB at the direct
    sound. Integration starts at the direct-path peak so the propagation
    pre-delay does not add a flat plateau at the top of the curve.

    Returns (t, edc_db); t in seconds with t=0 at the direct peak.
    """
    h = _to_1d(rir)
    if h.size == 0:
        return np.array([]), np.array([])

    peak = int(np.argmax(np.abs(h)))
    energy = h[peak:] ** 2

    if truncate == 'auto':
        end = _find_noise_floor_end(energy, fs, floor_offset_db=floor_offset_db)
        energy = energy[:max(end, 2)]

    # Backward (Schroeder) integration -> monotonically decreasing curve.
    sch = np.cumsum(energy[::-1])[::-1]
    sch = sch / (sch[0] + 1e-20)
    edc_db = 10.0 * np.log10(sch + 1e-20)
    t = np.arange(len(edc_db)) / fs
    return t, edc_db


def measure_rt60_schroeder(rir, fs, method='T20', truncate='auto',
                           upper_db=-5.0, return_diagnostics=False):
    """
    Estimate RT60 from a RIR via Schroeder integration and a least-squares fit
    of the EDC decay slope, extrapolated to a 60 dB drop.

    method : 'T20' fits between -5 and -25 dB, 'T30' between -5 and -35 dB.
             T20 is more robust when the usable dynamic range is limited.

    Returns RT60 in seconds (NaN if the EDC never reaches the fit window, e.g.
    too much noise / too short a decay). With return_diagnostics=True also
    returns a dict (fit range, R^2, achieved dynamic range, EDC) for QC.
    """
    if method == 'T20':
        lower_db = upper_db - 20.0
    elif method == 'T30':
        lower_db = upper_db - 30.0
    else:
        raise ValueError("method must be 'T20' or 'T30'")

    t, edc = energy_decay_curve(rir, fs, truncate=truncate)

    rt60, r2, slope, valid = np.nan, np.nan, np.nan, False
    dyn_range = float(edc[-1]) if edc.size else np.nan

    if edc.size:
        i_up = int(np.argmax(edc <= upper_db))
        i_lo = int(np.argmax(edc <= lower_db))
        valid = (edc[i_up] <= upper_db) and (edc[i_lo] <= lower_db) and (i_lo > i_up)

        if valid:
            seg_t = t[i_up:i_lo]
            seg_e = edc[i_up:i_lo]
            A = np.vstack([seg_t, np.ones_like(seg_t)]).T
            (slope, intercept), *_ = np.linalg.lstsq(A, seg_e, rcond=None)
            if slope < 0:
                rt60 = -60.0 / slope
            pred = A @ np.array([slope, intercept])
            ss_res = float(np.sum((seg_e - pred) ** 2))
            ss_tot = float(np.sum((seg_e - np.mean(seg_e)) ** 2)) + 1e-20
            r2 = 1.0 - ss_res / ss_tot

    if return_diagnostics:
        diag = {
            'method': method,
            'fit_range_db': (upper_db, lower_db),
            'dynamic_range_db': dyn_range,
            'r2': float(r2),
            'valid': bool(valid),
            't': t,
            'edc_db': edc,
        }
        return rt60, diag
    return rt60


def compute_drr(rir, fs, direct_pre_ms=0.5, direct_post_ms=2.5):
    """
    Direct-to-Reverberant Ratio in dB. The direct window spans
    [peak - direct_pre_ms, peak + direct_post_ms]; everything after is
    reverberant. Sets the ceiling of achievable beamformer enhancement.

    NOTE: DRR depends on source directivity (MIRD's Fostex is directional, pra
    is omni), so a systematic offset between measured and simulated DRR is
    EXPECTED and not a defect — we look at the trend with distance/RT, not an
    exact match.
    """
    h = _to_1d(rir)
    peak = int(np.argmax(np.abs(h)))
    pre = int(direct_pre_ms * fs / 1000.0)
    post = int(direct_post_ms * fs / 1000.0)

    d0 = max(0, peak - pre)
    d1 = min(len(h), peak + post + 1)

    e_direct = float(np.sum(h[d0:d1] ** 2))
    e_reverb = float(np.sum(h[d1:] ** 2))
    return 10.0 * np.log10(e_direct / (e_reverb + 1e-20))


def theoretical_diffuse_msc(freqs, d, c=C_SOUND):
    """
    Magnitude-squared coherence of an ideal spherically-isotropic diffuse field
    between two omni sensors a distance d apart: |sinc(2 f d / c)|^2, with
    np.sinc(x) = sin(pi x)/(pi x). Source-directivity independent, which is why
    it is the cleanest target to validate the simulated diffuse field.
    """
    return np.sinc(2.0 * np.asarray(freqs) * d / c) ** 2


def averaged_msc(rir_sets, fs, pairs, t_late_ms=8.0, nperseg=256):
    """
    Magnitude-squared coherence of the LATE (diffuse) field, pooled over several
    source positions AND over several equal-distance microphone pairs for low
    variance. Pooling is done at the cross/auto spectral-density level
    (accumulate Pxy, Pxx, Pyy, then form MSC), the statistically correct way to
    average coherence.

    rir_sets : iterable of per-position RIR banks, each indexable as
               rir_sets[p][mic] -> 1-D RIR.
    pairs    : a single (i, j) pair, or a list of (i, j) pairs that share the
               same inter-microphone distance (pooled together).

    The late window starts t_late_ms after the (later) direct peak and ENDS at
    the noise floor of each channel (via _find_noise_floor_end). Truncating the
    noise tail is essential for MEASURED RIRs (e.g. MIRD RIRs are ~10 s long but
    only the first fraction is reverberation; the rest is incoherent measurement
    noise that would otherwise collapse the coherence). Simulated RIRs are
    noise-free, so the truncation leaves them essentially unchanged.

    Returns (freqs, msc), or (None, None) if no segment was long enough.
    """
    if isinstance(pairs[0], (int, np.integer)):
        pairs = [tuple(pairs)]

    Pxy = Pxx = Pyy = None
    freqs = None
    late_off = int(t_late_ms * fs / 1000.0)

    for bank in rir_sets:
        for (mic_i, mic_j) in pairs:
            hi = _to_1d(bank[mic_i])
            hj = _to_1d(bank[mic_j])
            L = min(len(hi), len(hj))
            hi, hj = hi[:L], hj[:L]

            start = max(int(np.argmax(np.abs(hi))), int(np.argmax(np.abs(hj)))) + late_off
            end = min(_find_noise_floor_end(hi ** 2, fs),
                      _find_noise_floor_end(hj ** 2, fs))
            if end - start < nperseg:
                continue

            hi_l, hj_l = hi[start:end], hj[start:end]
            freqs, pxy = csd(hi_l, hj_l, fs=fs, nperseg=nperseg)
            _, pxx = csd(hi_l, hi_l, fs=fs, nperseg=nperseg)
            _, pyy = csd(hj_l, hj_l, fs=fs, nperseg=nperseg)

            Pxy = pxy if Pxy is None else Pxy + pxy
            Pxx = pxx if Pxx is None else Pxx + pxx
            Pyy = pyy if Pyy is None else Pyy + pyy

    if Pxy is None:
        return None, None
    msc = np.abs(Pxy) ** 2 / (np.real(Pxx) * np.real(Pyy) + 1e-20)
    return freqs, msc


if __name__ == "__main__":
    # Self-test: synthetic exponentially-decaying noise with a known RT60.
    # Energy decays 60 dB over RT60 -> h(t) = noise * exp(-a t), a = 6.9078/RT60.
    rng = np.random.default_rng(0)
    fs = 16000
    print("Schroeder RT60 estimator self-test (synthetic decays):")
    print(f"{'target':>8} {'T20':>8} {'T30':>8} {'R2(T20)':>9}")
    for rt60_true in (0.16, 0.36, 0.61, 1.0):
        n = int(2.0 * rt60_true * fs)
        t = np.arange(n) / fs
        a = 6.9078 / rt60_true
        h = rng.standard_normal(n) * np.exp(-a * t)
        h[0] += 5.0  # explicit direct-path peak at t=0
        est20, d = measure_rt60_schroeder(h, fs, method='T20', return_diagnostics=True)
        est30 = measure_rt60_schroeder(h, fs, method='T30')
        print(f"{rt60_true:8.3f} {est20:8.3f} {est30:8.3f} {d['r2']:9.4f}")
