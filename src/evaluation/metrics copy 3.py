import warnings
import numpy as np
import librosa
import pysepm
import mir_eval
from scipy import signal
from pystoi import stoi
from pesq import pesq

# The align_signals function using cross-correlation was completely removed.
# Pre-aligning the signals destructively interferes with PESQ's internal 
# variable delay estimator and perceptual time alignment mapping.


def evaluate_bss_metrics(ref_sig: np.ndarray, deg_sig: np.ndarray, interf_sig: np.ndarray = None) -> tuple:
    """
    Evaluates SDR, SIR, and SAR using the mir_eval library (BSS Eval protocol).
    If interf_sig is provided, it calculates the true spatial SIR.
    """
    ref_sig = np.squeeze(ref_sig)
    deg_sig = np.squeeze(deg_sig)
    
    if ref_sig.ndim != 1 or deg_sig.ndim != 1:
        raise ValueError("Signals must be 1-dimensional arrays.")
        
    if interf_sig is not None:
        interf_sig = np.squeeze(interf_sig)
        
        # mir_eval requires the number of references to match the number of estimates.
        # We stack the target and the interference as references.
        ref_sources = np.vstack((ref_sig, interf_sig))
        
        # We pass a dummy interference estimate to satisfy shape requirements
        est_sources = np.vstack((deg_sig, interf_sig))
        
        # Disable permutation to evaluate strictly index by index
        calc_permutation = False
    else:
        # Fallback to standard 1v1 evaluation
        ref_sources = ref_sig[np.newaxis, :]
        est_sources = deg_sig[np.newaxis, :]
        calc_permutation = True
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(
                ref_sources, est_sources, compute_permutation=calc_permutation
            )
        return float(sdr[0]), float(sir[0]), float(sar[0])
        
    except Exception as e:
        print(f"Error calculating BSS metrics: {e}")
        return np.nan, np.nan, np.nan


def evaluate_pesq(ref_sig: np.ndarray, deg_sig: np.ndarray, fs: int, mode: str = 'wb') -> float:
    """
    Evaluates the Perceptual Evaluation of Speech Quality (PESQ) score.
    """
    ref_sig = np.squeeze(ref_sig)
    deg_sig = np.squeeze(deg_sig)
    
    target_fs = 16000 if mode == 'wb' else 8000
    
    if fs != target_fs:
        ref_sig = librosa.resample(ref_sig, orig_sr=fs, target_sr=target_fs)
        deg_sig = librosa.resample(deg_sig, orig_sr=fs, target_sr=target_fs)
    
    try:
        score = pesq(target_fs, ref_sig, deg_sig, mode)
        return float(score)
    except Exception as e:
        error_msg = str(e).lower()
        if "no utterances" not in error_msg:
            print(f"Error calculating PESQ: {e}")
        return np.nan


def evaluate_stoi(ref_sig: np.ndarray, deg_sig: np.ndarray, fs: int, extended: bool = False) -> float:
    """
    Evaluates the Short-Time Objective Intelligibility (STOI) score.
    """
    ref_sig = np.squeeze(ref_sig)
    deg_sig = np.squeeze(deg_sig)
    
    try:
        score = stoi(ref_sig, deg_sig, fs, extended=extended)
        return float(score)
    except Exception as e:
        print(f"Error calculating STOI: {e}")
        return np.nan


def evaluate_cd(ref_sig: np.ndarray, deg_sig: np.ndarray, fs: int) -> float:
    """
    Evaluates the Cepstral Distance (CD).
    """
    ref_sig = np.squeeze(ref_sig)
    deg_sig = np.squeeze(deg_sig)
    
    try:
        score = pysepm.cepstrum_distance(ref_sig, deg_sig, fs)
        return float(score)
    except Exception as e:
        print(f"Error calculating Cepstral Distance: {e}")
        return np.nan
    

def match_rms_and_prevent_clipping(ref_sig: np.ndarray, target_sig: np.ndarray) -> np.ndarray:
    """
    Scales target_sig so its RMS matches the RMS of ref_sig.
    Critically, it enforces a hard peak limit to prevent the integer 
    overflow clipping that destroys PESQ and STOI evaluations.
    """
    rms_ref = np.sqrt(np.mean(ref_sig**2)) + 1e-10
    rms_target = np.sqrt(np.mean(target_sig**2)) + 1e-10
    
    if rms_target < 1e-10:
        return target_sig
        
    # Scale to match reference energy
    scaling_factor = rms_ref / rms_target
    matched_sig = target_sig * scaling_factor
    
    # Anti-clipping safety net
    max_peak = np.max(np.abs(matched_sig))
    if max_peak > 0.99:
        # Downscale proportionally so the maximum peak rests exactly at 0.99
        # This keeps the signal strictly within the bounds expected by the C-code.
        matched_sig = matched_sig * (0.99 / max_peak)
        
    return matched_sig


def evaluate_full_pipeline(ref_sig: np.ndarray, deg_sig: np.ndarray, fs: int, 
                           interf_sig: np.ndarray = None,
                           compute_pesq: bool = True, 
                           compute_cd: bool = True,
                           eval_start_s: float = 5.0) -> dict:
    """
    Master function to evaluate all acoustic metrics efficiently.
    PESQ and STOI are evaluated on the FULL signal length.
    Spatial BSS metrics are evaluated on the CROPPED signal (steady-state).
    """
    
    # 1. STRICT 1D ENFORCEMENT PREPROCESSING
    ref_sig = np.squeeze(ref_sig)
    deg_sig = np.squeeze(deg_sig)
    
    if ref_sig.ndim != 1 or deg_sig.ndim != 1:
        raise ValueError("Reference and degraded signals must be strictly 1-dimensional.")
        
    if interf_sig is not None:
        interf_sig = np.squeeze(interf_sig)
        if interf_sig.ndim != 1:
            raise ValueError("Interference signal must be 1-dimensional.")
            
    # 2. ENFORCE LENGTH MATCHING (Without destructive cross-correlation)
    if interf_sig is not None:
        min_len = min(len(ref_sig), len(deg_sig), len(interf_sig))
        interf_sig_full = interf_sig[:min_len]
    else:
        min_len = min(len(ref_sig), len(deg_sig))
        interf_sig_full = None
        
    ref_sig_full = ref_sig[:min_len]
    deg_sig_full = deg_sig[:min_len]
    
    # 3. SAFETY SCALING
    # Match energy but strictly avoid clipping
    deg_sig_full = match_rms_and_prevent_clipping(ref_sig_full, deg_sig_full)
    
    # 4. METRICS EVALUATION
    results = {}

    # --- A. Perceptual Metrics (Evaluated on FULL signal) ---
    # PESQ and STOI need the natural silence and attack of the speech to work.
    results['STOI'] = evaluate_stoi(ref_sig_full, deg_sig_full, fs, extended=False)
    
    if compute_pesq:
        results['PESQ'] = evaluate_pesq(ref_sig_full, deg_sig_full, fs, mode='wb')
    else:
        results['PESQ'] = np.nan
        
    if compute_cd:
        results['CD'] = evaluate_cd(ref_sig_full, deg_sig_full, fs)
    else:
        results['CD'] = np.nan
        
    # --- B. Spatial / BSS Metrics (Evaluated on CROPPED signal) ---
    # We crop the first 'eval_start_s' seconds to measure pure separation 
    # performance after the beamformer has fully converged.
    start_idx = int(eval_start_s * fs)
    
    if start_idx < min_len:
        ref_sig_crop = ref_sig_full[start_idx:]
        deg_sig_crop = deg_sig_full[start_idx:]
        
        if interf_sig_full is not None:
            interf_sig_crop = interf_sig_full[start_idx:]
            # Must also be scaled safely to keep relations valid
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