"""
Equivalencia de la re-implementacion por frames de `blind_feedback.py` contra las
funciones batch que ya estaban validadas. No toca el DTLN: inyecta mascaras
conocidas y compara numero por numero.

    RTFEstimator      vs estimate_rtf_recursive
    SoudenSubtractCore vs MVDR_Souden_recursive_mask_subtract
    blind_feedback_stft(mask_fixed=...) vs el nucleo batch

Uso: python tests/window_mismatch/test_blind_feedback_equiv.py
"""
import os
import sys

import numpy as np

ROOT = "/home/matias/Documents/Tesis/Vision-Aided-Beamformer"
sys.path.insert(0, os.path.join(ROOT, "src"))

from beamforming.mask.ds_mask import estimate_rtf_recursive              # noqa: E402
from beamforming.mask.souden_mvdr import MVDR_Souden_recursive_mask_subtract  # noqa: E402
from beamforming.mask.blind_feedback import (                            # noqa: E402
    RTFEstimator, SoudenSubtractCore, blind_feedback_stft)


def main():
    rng = np.random.default_rng(7)
    K, T, M, ref = 24, 40, 8, 4
    X = (rng.normal(size=(K, T, M)) + 1j * rng.normal(size=(K, T, M)))
    m_s = rng.random((K, T)) ** 8
    m_n = (1.0 - rng.random((K, T))) ** 8

    ok = True

    # ---- 1. estimador de RTF, los tres w_mode ----------------------------
    for w_mode in ("ds", "mvdr"):
        W_ref, D_ref = estimate_rtf_recursive(
            X, m_s, m_n, ref_mic_idx=ref, rtf_alpha=0.999, rtf_loading=1e-2,
            rtf_mode="cs", w_mode=w_mode, bf_loading=1e-6)
        est = RTFEstimator(K, M, ref, rtf_alpha=0.999, rtf_loading=1e-2,
                           rtf_mode="cs", w_mode=w_mode, bf_loading=1e-6)
        W = np.zeros_like(W_ref)
        D = np.zeros_like(D_ref)
        for t in range(T):
            W[:, t, :], D[:, t, :] = est.step(X[:, t, :], m_s[:, t], m_n[:, t])
        ew, ed = np.abs(W - W_ref).max(), np.abs(D - D_ref).max()
        ok &= (ew == 0.0 and ed == 0.0)
        print(f"  RTF  w_mode={w_mode:5s}  max|dW| = {ew:.3e}   max|dD| = {ed:.3e}")

    # ---- 2. gate de confianza -------------------------------------------
    cb = np.zeros(K, bool); cb[3:20] = True
    W_ref, D_ref = estimate_rtf_recursive(
        X, m_s, m_n, ref_mic_idx=ref, conf_gate=0.35, conf_bins=cb,
        conf_smooth=0.9, conf_alpha=0.99)
    est = RTFEstimator(K, M, ref, conf_gate=0.35, conf_bins=cb,
                       conf_smooth=0.9, conf_alpha=0.99)
    W = np.zeros_like(W_ref)
    for t in range(T):
        W[:, t, :], _ = est.step(X[:, t, :], m_s[:, t], m_n[:, t])
    e = np.abs(W - W_ref).max()
    ok &= (e == 0.0)
    print(f"  RTF  conf_gate=0.35   max|dW| = {e:.3e}")

    # ---- 3. nucleo de Souden --------------------------------------------
    for ban in (False, True):
        Y_ref, Wt_ref = MVDR_Souden_recursive_mask_subtract(
            X, m_s, m_n, min_loading=1e-9, save_weights=True, alpha=0.99,
            mu=0.0, lambda_floor=1e-3, psd_project=True, ref_mic_idx=ref, ban=ban)
        print()
        core = SoudenSubtractCore(K, M, ref, alpha=0.99, min_loading=1e-9,
                                  mu=0.0, lambda_floor=1e-3, psd_project=True, ban=ban)
        Y = np.zeros_like(Y_ref)
        Wt = np.zeros_like(Wt_ref)
        for t in range(T):
            Y[:, t], Wt[:, t, :] = core.step(X[:, t, :], m_s[:, t], m_n[:, t])
        ey, ew = np.abs(Y - Y_ref).max(), np.abs(Wt - Wt_ref).max()
        ok &= (ey == 0.0 and ew == 0.0)
        print(f"  CORE ban={str(ban):5s}       max|dY| = {ey:.3e}   max|dW| = {ew:.3e}")

    # ---- 4. el lazo entero en modo verificacion --------------------------
    Y_ref, _ = MVDR_Souden_recursive_mask_subtract(
        X, m_s, m_n, min_loading=1e-9, save_weights=True, alpha=0.99,
        ref_mic_idx=ref)
    print()
    Y_fb, _ = blind_feedback_stft(X, model_path=None, nperseg=512, ref_mic_idx=ref,
                                  mask_fixed=(m_s, m_n), progress=False)
    e = np.abs(Y_fb - Y_ref).max()
    ok &= (e == 0.0)
    print(f"  blind_feedback_stft(mask_fixed)  max|dY| = {e:.3e}")

    # ---- 5. auto-bootstrap: d(t=0) = e_ref -------------------------------
    # El lazo arranca con las DOS mascaras en cero, que es lo que garantiza el
    # bootstrap en TODOS los bins: sin masa en ninguna rama, Phi_XX = Phi_NN = 0
    # -> Phi_SS = 0 -> R_est = 0 -> manda la carga -> d = e_ref.
    e_ref = np.zeros(M); e_ref[ref] = 1.0
    est = RTFEstimator(K, M, ref)
    _, d0 = est.step(X[:, 0, :], np.zeros(K), np.zeros(K))
    e = np.abs(d0 - e_ref).max()
    ok &= (e < 1e-9)
    print(f"  bootstrap  mascaras en 0    max|d(0) - e_ref| = {e:.3e}")

    # Con masa en las dos ramas tambien vale, porque ambas normalizan sobre la
    # MISMA matriz instantanea: Phi_XX = Phi_NN = R.
    est = RTFEstimator(K, M, ref)
    _, d0 = est.step(X[:, 0, :], np.full(K, 0.3), np.full(K, 0.7))
    e = np.abs(d0 - e_ref).max()
    ok &= (e < 1e-9)
    print(f"  bootstrap  masa en ambas    max|d(0) - e_ref| = {e:.3e}")

    # PERO NO vale si la rama de RUIDO se queda sin masa: ahi Phi_NN ~ 0, la
    # sustraccion no se cancela, el nivel de ruido que escala la carga es ~0 y
    # d se va de e_ref. Es el motivo por el que el lazo arranca en cero y no con
    # una mascara cualquiera.
    est = RTFEstimator(K, M, ref)
    _, d0 = est.step(X[:, 0, :], np.ones(K), np.zeros(K))
    e = np.abs(d0 - e_ref).max()
    ok &= (e > 1e-3)
    print(f"  contraejemplo  m_n = 0      max|d(0) - e_ref| = {e:.3e}  (debe ser >0)")

    print("\n  " + ("OK: equivalencia exacta" if ok else "FALLA: hay diferencias"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
