import warnings
import numpy as np
import librosa
import pysepm
import mir_eval
from pystoi import stoi
from pesq import pesq

def evaluate_bss_metrics(ref_sig: np.ndarray, deg_sig: np.ndarray, interf_sig: np.ndarray = None) -> tuple:
    """
    Evaluates SDR, SIR, and SAR using the mir_eval library (BSS Eval protocol).
    If interf_sig is provided, it calculates the true spatial SIR.
    """
    
    # Ensure signals are strictly 1-dimensional
    ref_sig = np.squeeze(ref_sig)
    deg_sig = np.squeeze(deg_sig)
    
    if ref_sig.ndim != 1 or deg_sig.ndim != 1:
        raise ValueError("Signals must be 1-dimensional arrays.")
        
    # Set up the reference and estimated sources for mir_eval
    if interf_sig is not None:
        interf_sig = np.squeeze(interf_sig)
        
        # mir_eval requires the number of references to match the number of estimates.
        # We stack the target and the interference as references.
        ref_sources = np.vstack((ref_sig, interf_sig))
        
        # We pass a dummy interference estimate (the interference itself) to satisfy 
        # the shape requirement without affecting the target projection math.
        est_sources = np.vstack((deg_sig, interf_sig))
        
        # Disable permutation so index 0 (deg_sig) is strictly evaluated against index 0 (ref_sig)
        calc_permutation = False
    else:
        # Fallback to standard 1v1 evaluation (SIR will be NaN)
        ref_sources = ref_sig[np.newaxis, :]
        est_sources = deg_sig[np.newaxis, :]
        calc_permutation = True
    
    try:
        # Suppress the deprecation warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(
                ref_sources, est_sources, compute_permutation=calc_permutation
            )
        
        # We only care about the metrics of the first source (the target)
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
    

def evaluate_full_pipeline(ref_sig: np.ndarray, deg_sig: np.ndarray, fs: int, 
                           interf_sig: np.ndarray = None,
                           compute_pesq: bool = True, 
                           compute_cd: bool = True) -> dict:
    """
    Master function to evaluate all acoustic metrics efficiently.
    Includes explicit interference signals for true SIR calculation.
    """
    
    # 1. CENTRALIZED PREPROCESSING
    ref_sig = np.squeeze(ref_sig)
    deg_sig = np.squeeze(deg_sig)
    
    if ref_sig.ndim != 1 or deg_sig.ndim != 1:
        raise ValueError("Signals must be 1-dimensional arrays.")
        
    # Temporal alignment across all provided signals
    if interf_sig is not None:
        interf_sig = np.squeeze(interf_sig)
        min_len = min(len(ref_sig), len(deg_sig), len(interf_sig))
        interf_sig = interf_sig[:min_len]
    else:
        min_len = min(len(ref_sig), len(deg_sig))
        
    ref_sig = ref_sig[:min_len]
    deg_sig = deg_sig[:min_len]
    
    results = {}

    # 2. FAST METRICS
    sdr, sir, sar = evaluate_bss_metrics(ref_sig, deg_sig, interf_sig=interf_sig)
    results['SDR'] = sdr
    results['SIR'] = sir
    results['SAR'] = sar
    
    results['STOI'] = evaluate_stoi(ref_sig, deg_sig, fs, extended=False)
    
    # 3. HEAVY METRICS
    if compute_pesq:
        results['PESQ'] = evaluate_pesq(ref_sig, deg_sig, fs, mode='wb')
    else:
        results['PESQ'] = np.nan
        
    if compute_cd:
        results['CD'] = evaluate_cd(ref_sig, deg_sig, fs)
    else:
        results['CD'] = np.nan
        
    return results