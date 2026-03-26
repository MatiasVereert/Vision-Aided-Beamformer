import numpy as np
import librosa
import os

# Import your own pipeline to test the exact code you are using
from evaluation.metrics import evaluate_full_pipeline

def run_extended_pesq_sanity_check():
    # File paths
    target_path = r"tools\data\signals\FA01_09.wav"
    interf_path = r"tools\data\signals\MC15_03.wav"
    
    print("=== EXTENDED PESQ & BSS SANITY CHECK ===")
    
    # 1. Load signals
    print(" -> Loading audio files (16kHz)...")
    target_sig, fs = librosa.load(target_path, sr=16000)
    interf_sig, _ = librosa.load(interf_path, sr=16000)
    
    # 2. Force equal length
    min_len = min(len(target_sig), len(interf_sig))
    target_sig = target_sig[:min_len]
    interf_sig = interf_sig[:min_len]
    
    # 3. Base scaling for exactly 0 dB SINR
    # We normalize the interference so its RMS equals the target's RMS
    rms_target = np.sqrt(np.mean(target_sig**2)) + 1e-10
    rms_interf = np.sqrt(np.mean(interf_sig**2)) + 1e-10
    interf_scaled_0dB = interf_sig * (rms_target / rms_interf)
    
    results = {}
    
    # 4. Evaluate Perfect Identity Baseline
    print("\n[Test 0] Evaluating Perfect Identity (Clean Target vs Clean Target)...")
    res_perfect = evaluate_full_pipeline(
        ref_sig=target_sig, 
        deg_sig=target_sig, 
        fs=fs, 
        eval_start_s=0.0, 
        inspection_name="sanity_perfect"
    )
    results["Perfect"] = res_perfect
    
    # 5. Loop through progressive SINR improvements
    # 0dB is equal energy. +3dB means target is 2x power of interference, etc.
    sinr_levels_db = [0, 3, 6, 9, 12, 15]
    
    for sinr in sinr_levels_db:
        print(f"[Test] Evaluating Mixture at +{sinr} dB SINR...")
        
        # Calculate linear attenuation factor for the interference
        # factor = 10^(-SINR_dB / 20)
        factor = 10 ** (-sinr / 20.0)
        
        # Create the mixture
        sig_mix = target_sig + factor * interf_scaled_0dB
        
        # Evaluate
        res_mix = evaluate_full_pipeline(
            ref_sig=target_sig, 
            deg_sig=sig_mix, 
            fs=fs, 
            eval_start_s=0.0, 
            inspection_name=f"sanity_sinr_{sinr}dB"
        )
        results[f"+{sinr}dB"] = res_mix
        
    # 6. Print Formatted Summary Table
    print("\n" + "="*70)
    print(f"{'Test Scenario':<20} | {'PESQ':<6} | {'SDR (dB)':<9} | {'SIR (dB)':<9} | {'SAR (dB)':<9}")
    print("-" * 70)
    
    rp = results['Perfect']
    print(f"{'Perfect Identity':<20} | {rp['PESQ']:<6.3f} | {rp['SDR']:>9.2f} | {rp['SIR']:>9.2f} | {rp['SAR']:>9.2f}")
    
    for sinr in sinr_levels_db:
        r = results[f"+{sinr}dB"]
        print(f"{f'Mixture +{sinr} dB':<20} | {r['PESQ']:<6.3f} | {r['SDR']:>9.2f} | {r['SIR']:>9.2f} | {r['SAR']:>9.2f}")
    print("="*70)

if __name__ == "__main__":
    run_extended_pesq_sanity_check()