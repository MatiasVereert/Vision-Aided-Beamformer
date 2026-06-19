import numpy as np
from scipy import signal
import pb_bss
from pb_bss.evaluation import OutputMetrics

def precise_slice_alignment(ref_sig: np.ndarray, deg_sig: np.ndarray, fs: int, max_shift_s: float = 0.5) -> tuple:
    """
    Aligns signals by slicing out the unaligned portions using cross-correlation.
    Preserved exactly to handle temporal processing delays before metric evaluation.
    """
    # Emphasize transients for better phase-alignment using a high-pass filter
    b, a = signal.butter(4, 300 / (fs / 2), btype='high')
    filt_ref = signal.filtfilt(b, a, ref_sig)
    filt_deg = signal.filtfilt(b, a, deg_sig)

    max_shift_samples = int(max_shift_s * fs)

    # Calculate cross-correlation on the actual waveforms
    corr = signal.correlate(filt_deg, filt_ref, mode='full', method='fft')
    lags = signal.correlation_lags(len(filt_deg), len(filt_ref), mode='full')

    valid_idx = np.where(np.abs(lags) <= max_shift_samples)[0]
    if len(valid_idx) == 0:
        shift = 0
    else:
        best_idx = valid_idx[np.argmax(corr[valid_idx])]
        shift = lags[best_idx]

    # Slice arrays to perfectly match without introducing zero-padding artifacts
    if shift > 0:
        aligned_deg = deg_sig[shift:]
        aligned_ref = ref_sig[:-shift]
    elif shift < 0:
        shift_abs = abs(shift)
        aligned_ref = ref_sig[shift_abs:]
        aligned_deg = deg_sig[:-shift_abs]
    else:
        aligned_ref = ref_sig.copy()
        aligned_deg = deg_sig.copy()

    min_len = min(len(aligned_ref), len(aligned_deg))
    return aligned_ref[:min_len], aligned_deg[:min_len], shift


def compute_si_sar(target: np.ndarray, estimate: np.ndarray, noise: np.ndarray) -> float:
    """
    Computes Scale-Invariant SAR (SI-SAR) according to Le Roux et al. (2019).
    Requires the target reference and the composite noise reference to project
    the residual error and isolate algorithmic artifacts.
    """
    target = target.flatten()
    estimate = estimate.flatten()
    noise = noise.flatten()

    # 1. Compute scaling factor and isolate the target component
    alpha = np.dot(estimate, target) / (np.dot(target, target) + 1e-15)
    e_target = alpha * target

    # 2. Compute the residual error (interference + artifacts)
    e_res = estimate - e_target

    # 3. Orthogonalize the noise reference with respect to the target
    beta = np.dot(noise, target) / (np.dot(target, target) + 1e-15)
    n_orth = noise - beta * target

    # 4. Project the residual error onto the orthogonalized noise subspace to find interference
    gamma = np.dot(e_res, n_orth) / (np.dot(n_orth, n_orth) + 1e-15)
    e_interf = gamma * n_orth

    # 5. Isolate artifacts (what cannot be explained by target or noise)
    e_artif = e_res - e_interf

    # 6. Calculate final SI-SAR ratio
    den = np.sum(e_artif**2)
    if den < 1e-15:
        return np.inf

    return float(10 * np.log10(np.sum(e_target**2) / den))


def evaluate_full_pipeline(ref_sig: np.ndarray, deg_sig: np.ndarray, fs: int,
                           interf_early: np.ndarray = None,
                           interf_late: np.ndarray = None,
                           target_late: np.ndarray = None,
                           eval_start_s: float = 5.0,
                           **kwargs) -> dict:
    """
    Master evaluation function.
    Uses pb_bss for PESQ, STOI, and SI-SDR.
    Calculates SI-SAR analytically using the noise subspace.
    Absorbs extra kwargs to maintain benchmark compatibility.
    """
    ref_sig = np.squeeze(ref_sig)
    deg_sig = np.squeeze(deg_sig)
    results = {}

    # 1. Strict physical slice alignment
    aligned_ref, aligned_deg, shift = precise_slice_alignment(ref_sig, deg_sig, fs)

    # 2. Extract steady-state signals to avoid penalizing algorithm convergence
    start_idx = int(eval_start_s * fs)
    min_len = len(aligned_ref)

    if start_idx < min_len:
        ref_crop = aligned_ref[start_idx:]
        deg_crop = aligned_deg[start_idx:]
    else:
        # Fallback just in case the signal is shorter than the crop time
        ref_crop = aligned_ref
        deg_crop = aligned_deg

    # 3. Core single-channel Metrics via pb_bss OutputMetrics Facade
    try:
        metrics_facade = OutputMetrics(
            speech_source=ref_crop[np.newaxis, :],
            speech_prediction=deg_crop[np.newaxis, :],
            sample_rate=fs,
            enable_si_sdr=True,
            compute_permutation=False
        )

        results['PESQ'] = float(metrics_facade.pesq[0])
        results['STOI'] = float(metrics_facade.stoi[0])
        results['SI-SDR'] = float(metrics_facade.si_sdr[0])

        # We retain the classic SDR and SAR just so the benchmark's tracked_metrics
        # dictionary doesn't throw KeyErrors, but we focus analysis on SI metrics.
        results['SDR'] = float(metrics_facade.mir_eval_sdr[0])
        results['SAR'] = float(metrics_facade.mir_eval_sar[0])

    except Exception:
        for key in ['PESQ', 'STOI', 'SI-SDR', 'SDR', 'SAR']:
            results[key] = np.nan

    # 4. Modern SI-SAR computation
    def align_secondary(sig):
        if sig is None: return None
        sig_sq = np.squeeze(sig)
        if shift > 0:
            arr = sig_sq[shift:]
        elif shift < 0:
            arr = sig_sq[:-abs(shift)]
        else:
            arr = sig_sq.copy()
        return arr[:min_len]

    aligned_ie = align_secondary(interf_early)
    aligned_il = align_secondary(interf_late)
    aligned_tl = align_secondary(target_late)

    # Calculate SI-SAR only if all noise components are available
    if start_idx < min_len and all(s is not None for s in [aligned_ie, aligned_il, aligned_tl]):
        ie_crop = aligned_ie[start_idx:]
        il_crop = aligned_il[start_idx:]
        tl_crop = aligned_tl[start_idx:]

        # Composite noise is everything spatial/reverberant we don't want
        noise_total_crop = ie_crop + il_crop + tl_crop

        try:
            results['SI-SAR'] = compute_si_sar(ref_crop, deg_crop, noise_total_crop)
        except Exception:
            results['SI-SAR'] = np.nan
    else:
        results['SI-SAR'] = np.nan

    # Fill deprecated keys with NaN to prevent benchmark DataFrame from shifting
    results['SIR'] = np.nan
    results['SINR'] = np.nan

    return results