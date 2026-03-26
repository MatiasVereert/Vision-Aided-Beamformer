import warnings
import numpy as np
import librosa
import pysepm
import mir_eval
from scipy import signal
from pystoi import stoi
from pesq import pesq

def align_signals(ref_sig: np.ndarray, deg_sig: np.ndarray, interf_sig: np.ndarray = None) -> tuple:
    """
    Aligns deg_sig to ref_sig using FFT-based cross-correlation to find the bulk delay.
    Maintains synchronization with interf_sig if provided.
    """
    # Compute cross-correlation using FFT for optimal performance
    corr = signal.correlate(ref_sig, deg_sig, mode='full', method='fft')
    
    # Find the delay index that maximizes cross-correlation magnitude
    delay = np.argmax(np.abs(corr)) - (len(deg_sig) - 1)
    
    # Initialize aligned interference safely
    interf_sig_aligned = None
    
    # Shift signals to compensate for algorithmic latency
    if delay > 0:
        # deg_sig is delayed relative to ref_sig
        deg_sig_aligned = deg_sig[delay:]
        ref_sig_aligned = ref_sig[:-delay] if delay < len(ref_sig) else ref_sig
        if interf_sig is not None:
            interf_sig_aligned = interf_sig[:-delay] if delay < len(interf_sig) else interf_sig
            
    elif delay < 0:
        # ref_sig is delayed relative to deg_sig
        ref_sig_aligned = ref_sig[-delay:]
        deg_sig_aligned = deg_sig[:delay] if -delay < len(deg_sig) else deg_sig
        if interf_sig is not None:
            interf_sig_aligned = interf_sig[-delay:]
            
    else:
        # Signals are perfectly aligned
        ref_sig_aligned = ref_sig
        deg_sig_aligned = deg_sig
        interf_sig_aligned = interf_sig
        
    return ref_sig_aligned, deg_sig_aligned, interf_sig_aligned


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
    
def match_rms_energy(ref_sig: np.ndarray, target_sig: np.ndarray) -> np.ndarray:
    """
    Scales target_sig so its RMS energy matches the RMS energy of ref_sig.
    This prevents amplitude-sensitive metrics (like PESQ) from failing 
    due to low volume levels after discarding initial transient peaks.
    """
    # Calculate RMS for both signals to avoid division by zero errors
    rms_ref = np.sqrt(np.mean(ref_sig**2))
    rms_target = np.sqrt(np.mean(target_sig**2))
    
    # Prevent scaling if the target signal is digital silence
    if rms_target < 1e-10:
        return target_sig
        
    # Apply scaling factor
    scaling_factor = rms_ref / rms_target
    matched_sig = target_sig * scaling_factor
    
    return matched_sig

def evaluate_full_pipeline(ref_sig: np.ndarray, deg_sig: np.ndarray, fs: int, 
                           interf_sig: np.ndarray = None,
                           compute_pesq: bool = True, 
                           compute_cd: bool = True,
                           eval_start_s: float = 5.0) -> dict:
    """
    Master function to evaluate all acoustic metrics efficiently.
    Includes FFT-based alignment to handle algorithmic latency and 
    convergence cropping to evaluate only the steady-state performance.
    """
    
    # 1. STRICT 1D ENFORCEMENT PREPROCESSING
    ref_sig = np.squeeze(ref_sig)
    deg_sig = np.squeeze(deg_sig)
    
    if ref_sig.ndim != 1 or deg_sig.ndim != 1:
        raise ValueError("Reference and degraded signals must be strictly 1-dimensional.")
        
    if interf_sig is not None:
        interf_sig = np.squeeze(interf_sig)
        if interf_sig.ndim != 1:
            raise ValueError(
                f"Interference signal must be 1-dimensional, got {interf_sig.ndim}D. "
                "Make sure to sum all interferences at the reference microphone."
            )
            
    # 2. TEMPORAL ALIGNMENT
    # Align signals to compensate for any latency introduced by the algorithm
    ref_sig, deg_sig, interf_sig = align_signals(ref_sig, deg_sig, interf_sig)
    
    # Enforce strict length matching across all signals after alignment
    if interf_sig is not None:
        min_len = min(len(ref_sig), len(deg_sig), len(interf_sig))
        interf_sig = interf_sig[:min_len]
    else:
        min_len = min(len(ref_sig), len(deg_sig))
        
    ref_sig = ref_sig[:min_len]
    deg_sig = deg_sig[:min_len]
    
    # 3. CONVERGENCE CROPPING
    start_idx = int(eval_start_s * fs)
    
    if start_idx >= len(ref_sig):
        raise ValueError(f"Evaluation start time ({eval_start_s}s) exceeds or equals signal duration.")
        
    # Crop the arrays to discard the initial adaptation phase (apply ONLY ONCE)
    ref_sig = ref_sig[start_idx:]
    deg_sig = deg_sig[start_idx:]
    
    # Robust Energy Normalization for the target signal
    # Match energy of the degraded signal to the reference signal 
    # to evaluate purely on signal quality, ignoring global volume drops.
    deg_sig = match_rms_energy(ref_sig, deg_sig)
    
    if interf_sig is not None:
        # Crucial: Ensure this slicing happens exactly one time in the whole script
        interf_sig = interf_sig[start_idx:]
        interf_sig = match_rms_energy(ref_sig, interf_sig)
    
    # 4. METRICS EVALUATION
    results = {}

    sdr, sir, sar = evaluate_bss_metrics(ref_sig, deg_sig, interf_sig=interf_sig)
    results['SDR'] = sdr
    results['SIR'] = sir
    results['SAR'] = sar
    
    results['STOI'] = evaluate_stoi(ref_sig, deg_sig, fs, extended=False)
    
    if compute_pesq:
        results['PESQ'] = evaluate_pesq(ref_sig, deg_sig, fs, mode='wb')
    else:
        results['PESQ'] = np.nan
        
    if compute_cd:
        results['CD'] = evaluate_cd(ref_sig, deg_sig, fs)
    else:
        results['CD'] = np.nan
        
    return results