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


def evaluate_full_pipeline(ref_sig: np.ndarray, deg_sig: np.ndarray, fs: int,
                           interf_early: np.ndarray = None,
                           interf_late: np.ndarray = None,
                           target_late: np.ndarray = None,
                           eval_start_s: float = 5.0,
                           **kwargs) -> dict:
    """
    Master evaluation function.
    Uses pb_bss for PESQ, STOI, SI-SDR, and traditional SDR/SIR/SAR via multi-source evaluation.
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

    # 3. Core multi-source Metrics via pb_bss OutputMetrics Facade
    try:
        # Check if secondary noise components are available to build the full subspace
        if interf_early is not None and interf_late is not None and target_late is not None:

            # Align secondary paths using the same shift parameter
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

            if start_idx < min_len:
                ie_crop = aligned_ie[start_idx:]
                il_crop = aligned_il[start_idx:]
                tl_crop = aligned_tl[start_idx:]
                # Composite noise is everything spatial/reverberant we do not want
                noise_total_crop = ie_crop + il_crop + tl_crop
            else:
                noise_total_crop = np.zeros_like(ref_crop)

            # Stack target and noise as separate reference sources: shape (2, samples)
            speech_source = np.stack([ref_crop, noise_total_crop], axis=0)

            # Stack the degraded signal and a low-power dummy noise floor to match shapes: shape (2, samples)
            # This prevents mathematical degeneracy in the underlying projections
            dummy_noise_floor = np.random.randn(len(deg_crop)) * 1e-10
            speech_prediction = np.stack([deg_crop, dummy_noise_floor], axis=0)

            metrics_facade = OutputMetrics(
                speech_source=speech_source,
                speech_prediction=speech_prediction,
                sample_rate=fs,
                enable_si_sdr=True,
                compute_permutation=False
            )

            # Extract index 0 metrics which correspond strictly to the target speech channel
            results['PESQ'] = float(metrics_facade.pesq[0])
            results['STOI'] = float(metrics_facade.stoi[0])
            results['SI-SDR'] = float(metrics_facade.si_sdr[0])
            results['SDR'] = float(metrics_facade.mir_eval_sdr[0])
            results['SIR'] = float(metrics_facade.mir_eval_sir[0])
            results['SAR'] = float(metrics_facade.mir_eval_sar[0])

        else:
            # Fallback to single-source evaluation if secondary components are missing
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
            results['SDR'] = float(metrics_facade.mir_eval_sdr[0])
            results['SAR'] = float(metrics_facade.mir_eval_sar[0])
            results['SIR'] = np.nan

    except Exception:
        for key in ['PESQ', 'STOI', 'SI-SDR', 'SDR', 'SIR', 'SAR']:
            results[key] = np.nan

    # Fill deprecated keys with NaN to prevent benchmark DataFrame from shifting
    results['SINR'] = np.nan

    return results