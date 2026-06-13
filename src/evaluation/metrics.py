
import numpy as np
import mir_eval
import pysepm
from scipy import signal
import os
import scipy.io.wavfile as wav
# pb_bss official wrappers
from pb_bss.evaluation import pesq as pb_pesq
from pb_bss.evaluation import stoi as pb_stoi

def precise_slice_alignment(ref_sig: np.ndarray, deg_sig: np.ndarray, fs: int, max_shift_s: float = 0.5) -> tuple:
    """
    Aligns signals by slicing out the unaligned portions.
    Used for metrics that do not have internal alignment (SDR, SIR, CD).
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


def evaluate_bss_metrics(ref_sig: np.ndarray, deg_sig: np.ndarray, interf_sig: np.ndarray = None) -> tuple:
    """
    Evaluates BSS Eval metrics safely using mir_eval.
    """
    ref_sig = np.squeeze(ref_sig)
    deg_sig = np.squeeze(deg_sig)

    if interf_sig is not None:
        interf_sig = np.squeeze(interf_sig)
        ref_sources = np.vstack((ref_sig, interf_sig))
        est_sources = np.vstack((deg_sig, interf_sig))
        calc_permutation = False
    else:
        ref_sources = ref_sig[np.newaxis, :]
        est_sources = deg_sig[np.newaxis, :]
        calc_permutation = True

    try:
        sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(
            ref_sources, est_sources, compute_permutation=calc_permutation
        )
        return float(sdr[0]), float(sir[0]), float(sar[0])
    except Exception:
        return np.nan, np.nan, np.nan

def evaluate_full_pipeline(ref_sig: np.ndarray, deg_sig: np.ndarray, fs: int,
                           interf_early: np.ndarray = None,
                           interf_late: np.ndarray = None,
                           target_late: np.ndarray = None,
                           compute_pesq: bool = True,
                           compute_cd: bool = True,
                           eval_start_s: float = 5.0,
                           inspection_name: str = "eval") -> dict:
    """
    Master evaluation function. Updated to compute both spatial SIR
    and global acoustic SINR by incorporating late reverberation tails.
    """
    ref_sig = np.squeeze(ref_sig)
    deg_sig = np.squeeze(deg_sig)
    results = {}

    # Calculate the starting sample index to discard the convergence period
    start_idx_raw = int(eval_start_s * fs)

    # --- PATH A: Perceptual Metrics (PESQ / STOI) ---
    min_len_raw = min(len(ref_sig), len(deg_sig))

    # Apply the crop to avoid penalizing the algorithm's transient state
    if start_idx_raw < min_len_raw:
        ref_perceptual = ref_sig[start_idx_raw:min_len_raw]
        deg_perceptual = deg_sig[start_idx_raw:min_len_raw]
    else:
        # Fallback just in case the signal is shorter than the crop time
        ref_perceptual = ref_sig[:min_len_raw]
        deg_perceptual = deg_sig[:min_len_raw]

    if compute_pesq:
        try:
            pesq_score = pb_pesq(ref_perceptual, deg_perceptual, sample_rate=fs)
            results['PESQ'] = float(np.mean(pesq_score))
        except Exception:
            results['PESQ'] = np.nan
    else:
        results['PESQ'] = np.nan

    try:
        results['STOI'] = float(np.mean(pb_stoi(ref_perceptual, deg_perceptual, sample_rate=fs)))
    except Exception:
        results['STOI'] = np.nan

    # --- PATH B: Spatial & Distortion Metrics ---
    # Perform strict physical slice alignment
    aligned_ref, aligned_deg, shift = precise_slice_alignment(ref_sig, deg_sig, fs)

    # Helper function to align individual secondary sources using the identical shift
    def align_secondary(sig):
        if sig is None: return None
        sig_sq = np.squeeze(sig)
        if shift > 0:
            arr = sig_sq[shift:]
        elif shift < 0:
            arr = sig_sq[:-abs(shift)]
        else:
            arr = sig_sq.copy()
        return arr[:len(aligned_ref)]

    aligned_interf_early = align_secondary(interf_early)
    aligned_interf_late = align_secondary(interf_late)
    aligned_target_late = align_secondary(target_late)

    # Cepstrum Distance
    if compute_cd:
        try:
            results['CD'] = float(pysepm.cepstrum_distance(aligned_ref, aligned_deg, fs))
        except Exception:
            results['CD'] = np.nan
    else:
        results['CD'] = np.nan

    # Spatial metrics with steady-state crop
    start_idx = int(eval_start_s * fs)
    if start_idx < len(aligned_ref):
        ref_crop = aligned_ref[start_idx:]
        deg_crop = aligned_deg[start_idx:]

        # 1. Compute strict Spatial SIR and global SDR
        if aligned_interf_early is not None:
            ie_crop = aligned_interf_early[start_idx:]
            sdr, sir, sar = evaluate_bss_metrics(ref_crop, deg_crop, interf_sig=ie_crop)
            results['SDR'], results['SIR'], results['SAR'] = sdr, sir, sar
        else:
            results['SDR'] = results['SIR'] = results['SAR'] = np.nan

        # 2. Compute true SINR by creating the composite unwanted noise subspace
        if all(s is not None for s in [aligned_interf_early, aligned_interf_late, aligned_target_late]):
            noise_total_crop = (aligned_interf_early[start_idx:] +
                                aligned_interf_late[start_idx:] +
                                aligned_target_late[start_idx:])

            # The SIR returned against the total noise composite is mathematically the SINR
            _, sinr, _ = evaluate_bss_metrics(ref_crop, deg_crop, interf_sig=noise_total_crop)
            results['SINR'] = sinr
        else:
            results['SINR'] = np.nan
    else:
        results['SDR'] = results['SIR'] = results['SAR'] = results['SINR'] = np.nan

    return results