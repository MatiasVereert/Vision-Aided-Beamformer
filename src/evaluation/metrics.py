import numpy as np
import mir_eval
import pysepm
from scipy import signal
import matplotlib.pyplot as plt
import os
import scipy.io.wavfile as wav
# pb_bss official wrappers
from pb_bss.evaluation import pesq as pb_pesq
from pb_bss.evaluation import stoi as pb_stoi


def robust_envelope_alignment(ref_sig: np.ndarray, deg_sig: np.ndarray, fs: int, 
                              max_shift_s: float = 0.5, inspect: bool = False, 
                              inspect_name: str = "alignment") -> tuple:
    """
    Aligns ref_sig to deg_sig using envelope correlation to avoid phase-distortion jitter.
    If inspect=True, generates a plot to visually verify the alignment.
    """
    # 1. Extract energy envelopes to ignore high-frequency phase shifts caused by beamformers
    b, a = signal.butter(3, 50 / (fs / 2), btype='low')
    env_ref = signal.filtfilt(b, a, np.abs(ref_sig))
    env_deg = signal.filtfilt(b, a, np.abs(deg_sig))
    
    # 2. Restrict correlation window to prevent absurd shifts
    max_shift_samples = int(max_shift_s * fs)
    
    # 3. Compute cross-correlation of envelopes
    corr = signal.correlate(env_deg, env_ref, mode='full', method='fft')
    lags = signal.correlation_lags(len(env_deg), len(env_ref), mode='full')
    
    # Restrict search space
    valid_idx = np.where(np.abs(lags) <= max_shift_samples)[0]
    if len(valid_idx) == 0:
        shift = 0
    else:
        best_idx = valid_idx[np.argmax(corr[valid_idx])]
        shift = lags[best_idx]
    
    # 4. Apply shift to reference signal (padding with zeros to maintain length)
    if shift > 0:
        # Degraded is delayed relative to reference, shift reference forward
        aligned_ref = np.pad(ref_sig, (shift, 0))[:-shift]
    elif shift < 0:
        # Reference is delayed relative to degraded, shift reference backward
        aligned_ref = np.pad(ref_sig[abs(shift):], (0, abs(shift)))
    else:
        aligned_ref = ref_sig.copy()

    # 5. Visual Inspection Generation
    if inspect:
        plot_samples = int(fs * 1.5) # Plot only first 1.5 seconds for clarity
        plt.figure(figsize=(12, 6))
        
        plt.subplot(2, 1, 1)
        plt.title(f"[{inspect_name}] Before Alignment (PESQ Killer)")
        plt.plot(ref_sig[:plot_samples], label='Reference', alpha=0.7)
        plt.plot(deg_sig[:plot_samples], label='Degraded', alpha=0.7)
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.title(f"[{inspect_name}] After Envelope Alignment (Shift: {shift} samples / {shift/fs*1000:.1f} ms)")
        plt.plot(aligned_ref[:plot_samples], label='Aligned Reference', alpha=0.7)
        plt.plot(deg_sig[:plot_samples], label='Degraded', alpha=0.7)
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        os.makedirs("inspections", exist_ok=True)
        plt.savefig(f"inspections/{inspect_name}.png", dpi=150)
        plt.close()

    return aligned_ref, shift


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

def precise_slice_alignment(ref_sig: np.ndarray, deg_sig: np.ndarray, fs: int, max_shift_s: float = 0.5) -> tuple:
    """
    Aligns signals by slicing out the unaligned portions rather than padding with zeros.
    Uses transient-emphasized cross-correlation for higher temporal precision.
    """
    # Emphasize transients for better phase-alignment using a high-pass filter
    b, a = signal.butter(4, 300 / (fs / 2), btype='high')
    filt_ref = signal.filtfilt(b, a, ref_sig)
    filt_deg = signal.filtfilt(b, a, deg_sig)
    
    max_shift_samples = int(max_shift_s * fs)
    
    # Calculate cross-correlation on the actual waveforms, not just the envelope
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
        # Degraded is delayed, drop the beginning of degraded and the end of reference
        aligned_deg = deg_sig[shift:]
        aligned_ref = ref_sig[:-shift]
    elif shift < 0:
        # Reference is delayed, drop the beginning of reference and the end of degraded
        shift_abs = abs(shift)
        aligned_ref = ref_sig[shift_abs:]
        aligned_deg = deg_sig[:-shift_abs]
    else:
        aligned_ref = ref_sig.copy()
        aligned_deg = deg_sig.copy()

    # Ensure both arrays have the exact same length for PESQ
    min_len = min(len(aligned_ref), len(aligned_deg))
    return aligned_ref[:min_len], aligned_deg[:min_len], shift


def dump_for_pesq_inspection(ref_sig: np.ndarray, deg_sig: np.ndarray, fs: int, name: str):
    """
    Exports the exact arrays passed to PESQ as WAV files for bit-accurate debugging.
    Uses absolute paths to prevent files from being lost in relative directories.
    """
    # Get the absolute path of the directory where this metrics.py file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate up to the project root (assuming metrics.py is inside evaluation/)
    project_root = os.path.dirname(current_dir)
    
    # Define the exact absolute path for the dump folder
    dump_dir = os.path.join(project_root, "tests", "data", "pesq_inspection")
    os.makedirs(dump_dir, exist_ok=True)
    
    # Debug print: If you don't see this in the terminal, Python is reading an old file
    print(f" -> [DEBUG] Exporting audio for PESQ inspection to: {dump_dir}")
    
    # Normalize to 16-bit PCM standard for safe WAV export
    max_val = max(np.max(np.abs(ref_sig)), np.max(np.abs(deg_sig))) + 1e-10
    ref_norm = np.int16((ref_sig / max_val) * 32767)
    deg_norm = np.int16((deg_sig / max_val) * 32767)
    
    # Write the files
    wav.write(os.path.join(dump_dir, f"{name}_REF.wav"), fs, ref_norm)
    wav.write(os.path.join(dump_dir, f"{name}_DEG.wav"), fs, deg_norm)

def match_rms_and_prevent_clipping(ref_sig: np.ndarray, target_sig: np.ndarray) -> np.ndarray:
    """
    Scales target_sig to match ref_sig RMS, strictly capping peaks at 0.99.
    """
    rms_ref = np.sqrt(np.mean(ref_sig**2)) + 1e-10
    rms_target = np.sqrt(np.mean(target_sig**2)) + 1e-10
    
    if rms_target < 1e-10:
        return target_sig
        
    scaling_factor = rms_ref / rms_target
    matched_sig = target_sig * scaling_factor
    
    max_peak = np.max(np.abs(matched_sig))
    if max_peak > 0.99:
        matched_sig = matched_sig * (0.99 / max_peak)
        
    return matched_sig

def evaluate_full_pipeline(ref_sig: np.ndarray, deg_sig: np.ndarray, fs: int, 
                           interf_sig: np.ndarray = None,
                           compute_pesq: bool = True, 
                           compute_cd: bool = True,
                           eval_start_s: float = 5.0,
                           inspection_name: str = "eval") -> dict:
    # Master evaluation function integrating precise slicing and visual alignment
    ref_sig = np.squeeze(ref_sig)
    deg_sig = np.squeeze(deg_sig)
    
    # Determine minimum initial length before alignment
    if interf_sig is not None:
        interf_sig = np.squeeze(interf_sig)
        min_len = min(len(ref_sig), len(deg_sig), len(interf_sig))
        interf_sig_full = interf_sig[:min_len]
    else:
        min_len = min(len(ref_sig), len(deg_sig))
        interf_sig_full = None
        
    ref_sig_full = ref_sig[:min_len]
    deg_sig_full = deg_sig[:min_len]
    
    # 1. PRECISE SLICE ALIGNMENT
    aligned_ref, aligned_deg, shift = precise_slice_alignment(
        ref_sig_full, deg_sig_full, fs, max_shift_s=0.5
    )
    
    # Align interference signal if present using the identical slice logic applied to deg_sig
    if interf_sig_full is not None:
        if shift > 0:
            aligned_interf = interf_sig_full[shift:]
        elif shift < 0:
            shift_abs = abs(shift)
            aligned_interf = interf_sig_full[:-shift_abs]
        else:
            aligned_interf = interf_sig_full.copy()
            
        # Match length exactly to the aligned reference
        aligned_interf = aligned_interf[:len(aligned_ref)]
    else:
        aligned_interf = None

    # 2. SAFETY SCALING
    aligned_deg = match_rms_and_prevent_clipping(aligned_ref, aligned_deg)
    
    # 3. INSPECTION DUMP
    # Export audio files directly to the inspection folder
    dump_for_pesq_inspection(aligned_ref, aligned_deg, fs, inspection_name)
    
    results = {}

    # 4. PERCEPTUAL METRICS
    try:
        stoi_score = pb_stoi(aligned_ref, aligned_deg, sample_rate=fs)
        results['STOI'] = float(np.mean(stoi_score))
    except Exception:
        results['STOI'] = np.nan
    
    if compute_pesq:
        try:
            pesq_score = pb_pesq(aligned_ref, aligned_deg, sample_rate=fs)
            results['PESQ'] = float(np.mean(pesq_score))
        except Exception:
            results['PESQ'] = np.nan
    else:
        results['PESQ'] = np.nan
        
    if compute_cd:
        try:
            results['CD'] = float(pysepm.cepstrum_distance(aligned_ref, aligned_deg, fs))
        except Exception:
            results['CD'] = np.nan
            
    # 5. SPATIAL METRICS (Steady-state cropped)
    start_idx = int(eval_start_s * fs)
    if start_idx < len(aligned_ref):
        ref_sig_crop = aligned_ref[start_idx:]
        deg_sig_crop = aligned_deg[start_idx:]
        
        if aligned_interf is not None:
            interf_sig_crop = aligned_interf[start_idx:]
            interf_sig_crop = match_rms_and_prevent_clipping(ref_sig_crop, interf_sig_crop)
        else:
            interf_sig_crop = None
            
        sdr, sir, sar = evaluate_bss_metrics(ref_sig_crop, deg_sig_crop, interf_sig=interf_sig_crop)
        results['SDR'] = sdr
        results['SIR'] = sir
        results['SAR'] = sar
    else:
        results['SDR'] = results['SIR'] = results['SAR'] = np.nan
        
    return results