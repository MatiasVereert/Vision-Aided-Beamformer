import h5py
import numpy as np

def diagnose_weights_variance(h5_path: str):
    """
    Reads the benchmark HDF5 file and thoroughly checks if the adaptive filter 
    weights (complex) actually moved over time.
    """
    with h5py.File(h5_path, "r") as f:
        # Load the raw weights tensor directly
        w = f["weights"]["processed_MPDR_WPE_Test"][:]
        
        print(f"Weight Tensor Shape (F, T, M): {w.shape}")
        
        # ---------------------------------------------------------
        # Check 1: Global Variance (Complex -> Magnitude & Phase)
        # ---------------------------------------------------------
        w_mag_var = np.var(np.abs(w), axis=1)
        w_phase_var = np.var(np.angle(w), axis=1)
        
        print(f"Max Magnitude Variance: {np.max(w_mag_var):.2e}")
        print(f"Max Phase Variance:   {np.max(w_phase_var):.2e}")
        
        # ---------------------------------------------------------
        # Check 2: Difference from Frame 0 (Init) to Last Frame
        # ---------------------------------------------------------
        # w shape is assumed to be (Frequencies, TimeFrames, Mics)
        w_frame_0 = w[:, 0, :]
        w_frame_last = w[:, -1, :]
        
        diff_0_last = np.abs(w_frame_last - w_frame_0)
        max_diff = np.max(diff_0_last)
        mean_diff = np.mean(diff_0_last)
        
        print(f"Max absolute difference (Frame 0 vs Last): {max_diff:.2e}")
        print(f"Mean absolute difference (Frame 0 vs Last): {mean_diff:.2e}")
        
        # ---------------------------------------------------------
        # Check 3: Frame-to-Frame consecutive differences
        # ---------------------------------------------------------
        # np.diff computes w[:, t, :] - w[:, t-1, :]
        delta_w = np.abs(np.diff(w, axis=1))
        max_step = np.max(delta_w)
        mean_step = np.mean(delta_w)
        
        print(f"Max frame-to-frame step size:  {max_step:.2e}")
        print(f"Mean frame-to-frame step size: {mean_step:.2e}")
        
        # ---------------------------------------------------------
        # Check 4: Exactly how many frames are mathematically frozen?
        # ---------------------------------------------------------
        frames_identical_to_0 = 0
        T = w.shape[1]
        
        # We consider them identical if the difference is smaller than 1e-10
        for t in range(1, T):
            if np.allclose(w[:, t, :], w_frame_0, atol=1e-10):
                frames_identical_to_0 += 1
                
        print(f"Frames mathematically identical to Frame 0: {frames_identical_to_0} out of {T-1}")

        # =========================================================
        # FINAL VERDICT
        # =========================================================
        print("\n--- DIAGNOSIS ---")
        if max_diff < 1e-6:
            print("🚨 FROZEN: The filter weights are practically identical across all frames.")
            print("The algorithm is NOT adapting to the signal. Check initialization or Kalman gain.")
        elif frames_identical_to_0 == (T - 1):
            print("🚨 FROZEN: Every single frame is an exact clone of Frame 0.")
        elif frames_identical_to_0 > 0:
            print(f"⚠️ PARTIALLY FROZEN: The filter stayed stuck on Frame 0 for {frames_identical_to_0} frames, then moved.")
        else:
            print("✅ ADAPTING: The filter is actively changing over time.")
            print("If the dashboard looks static, the changes are too subtle or the acoustic scene is static.")

if __name__ == "__main__":
    diagnose_weights_variance(r"tests\data\benchmark_results\test_scene_001.h5")