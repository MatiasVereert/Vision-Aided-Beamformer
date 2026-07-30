"""
souden_mvdr.py
==============
Algoritmos de beamforming MVDR mask-based, todos con la formulacion de Souden
et al. (2010, "On Optimal Frequency-Domain Multichannel Linear Filtering for
Noise Reduction", IEEE TASLP 18(2)):

    w = Phi_NN^-1 Phi_XX u / tr(Phi_NN^-1 Phi_XX)

Todas operan online (recursivo por frame) sobre X_stft (K, T, M) + mascaras de
voz/ruido (K, T) y devuelven Y_stft (K, T) [+ pesos si save_weights=True],
mismo contrato. Las mascaras se estiman aparte (ver dtln_masks.py).

Las variantes se diferencian por el CRITERIO DE OPTIMIZACION / post-filtro:

  - MVDR_Souden_recursive_mask        : MVDR base, factor de olvido alpha, ref=M//2.
  - MVDR_Souden_recursive_mask_slow   : idem base pero aplica los pesos del frame
                                        anterior (variante experimental).
  - MVDR_Souden_recursive_mask_BAN    : acumulacion estricta (alpha=1) + Blind
                                        Analytical Normalization, ref=0.
  - MVDR_Souden_recursive_mask_BAN_alpha : BAN + factor de olvido alpha, ref=0.
  - MVDR_Souden_recursive_mask_fixed  : carga diagonal RELATIVA (escala-invariante)
                                        + Hermitiana + solve, ref=M//2.
  - MVDR_Souden_recursive_mask_MWF    : fixed + post-filtro de Wiener parametrico
                                        (PMWF-beta, denominador lambda+mu).
  - MVDR_Souden_recursive_mask_BAN_MWF: fixed + MWF + BAN (OJO: la BAN es invariante
                                        a escala -> anula el efecto de mu).
  - MVDR_Souden_recursive_mask_specsub: fixed + sustraccion espectral (ganancia por
                                        bin con la mascara suave del DTLN).
  - MVDR_Souden_recursive_mask_BAN_specsub : fixed + BAN + sustraccion espectral.
"""

import numpy as np


# =====================================================================
# MVDR base (Souden) con factor de olvido
# =====================================================================
def MVDR_Souden_recursive_mask(X_stft, mask_s, mask_n, min_loading=1e-6, save_weights=False, alpha = 0.99):
    K, T, M = X_stft.shape

    Y_stft = np.zeros((K, T), dtype=np.complex128)

    # Accumulators for matrix estimation
    Num_XX = np.zeros((K, M, M), dtype=np.complex128)
    Num_NN = np.zeros((K, M, M), dtype=np.complex128)

    Den_XX = np.zeros((K, 1, 1), dtype=np.float64)
    Den_NN = np.zeros((K, 1, 1), dtype=np.float64)

    # Reference microphone index (equivalent to the one-hot vector 'r' in Souden's formula)


        # Define Reference Microphone Index as middle index
    ref_mic_idx = M // 2
    ref_mic = ref_mic_idx



    if save_weights:
        weights_rec = np.zeros((K, T, M), dtype=np.complex128)

    for m in range(T):
        print(f"\rProcessing frame {m} of {T}", end="")
        X_frame = X_stft[:, m, :]
        m_s_frame = mask_s[:, m, np.newaxis, np.newaxis]
        m_n_frame = mask_n[:, m, np.newaxis, np.newaxis]

        R_instant = np.einsum("fm,fn->fmn", X_frame, X_frame.conj())

        # Apply exponential forgetting factor (alpha) to both accumulators
        Num_XX = alpha * Num_XX + m_s_frame * R_instant
        Den_XX = alpha * Den_XX + m_s_frame

        Num_NN = alpha * Num_NN + m_n_frame * R_instant
        Den_NN = alpha * Den_NN + m_n_frame

        # Calculate matrices by dividing the accumulators
        Phi_XX = Num_XX / (Den_XX + 1e-15)
        Phi_NN = Num_NN / (Den_NN + 1e-15)

        # Diagonal loading for numerical stability before inversion
        tr_Phi = np.real(np.trace(Phi_NN, axis1=1, axis2=2))
        adaptive_load = min_loading * (tr_Phi / M)
        loading_matrix = np.eye(M)[np.newaxis, :, :] * np.maximum(adaptive_load, 1e-9)[:, np.newaxis, np.newaxis]

        Phi_NN_stable = Phi_NN + loading_matrix
        Phi_NN_inv = np.linalg.inv(Phi_NN_stable)

        # -----------------------------------------------------------------
        # SOUDEN'S MVDR FORMULATION
        # -----------------------------------------------------------------
        # 1. Compute the product of Phi_NN_inv and Phi_XX
        # Shape: (K, M, M)
        Phi_inv_Phi_X = np.einsum("fmn,fnk->fmk", Phi_NN_inv, Phi_XX)

        # 2. Calculate the normalization factor lambda (trace of the product)
        # The trace of this specific product is theoretically real.
        lambda_norm = np.real(np.trace(Phi_inv_Phi_X, axis1=1, axis2=2))

        # 3. Multiply by the reference microphone vector 'r'
        # Since 'r' is a unit vector [1, 0, 0...], multiplying Phi_inv_Phi_X by 'r'
        # is mathematically equivalent to taking the column at the 'ref_mic' index.
        weights_unnorm = Phi_inv_Phi_X[:, :, ref_mic]

        # 4. Normalize the weights using lambda
        weights = weights_unnorm / (lambda_norm[:, np.newaxis] + 1e-15)
        # -----------------------------------------------------------------

        if save_weights:
            weights_rec[:, m, :] = weights

        # Apply the weights to the current STFT frame
        Y_stft[:, m] = np.einsum("fm,fm->f", weights.conj(), X_frame)

    if save_weights:
        return Y_stft, weights_rec
    else:
        return Y_stft


# =====================================================================
# MVDR ORACLE (Souden) con covarianzas estimadas DIRECTAMENTE de las
# señales limpias multicanal (sin mascara)
# =====================================================================
def MVDR_Souden_recursive_oracle(X_stft, S_stft, N_stft, min_loading=1e-6, save_weights=False, alpha=0.99):
    """
    Variante ORACLE del Souden MVDR. Identica a MVDR_Souden_recursive_mask salvo
    en la ESTIMACION de las covarianzas: en vez de ponderar la mezcla ruidosa por
    una mascara, se estiman Phi_SS y Phi_NN directamente de los outer products de
    las señales LIMPIAS multicanal (target y ruido de referencia):

        Phi_SS = <s sᴴ>_alpha        (S_stft: target limpio, (K, T, M))
        Phi_NN = <n nᴴ>_alpha        (N_stft: ruido limpio,  (K, T, M))

    Los pesos resultantes se aplican a la MEZCLA OBSERVADA X_stft (K, T, M), igual
    que la version mask-based. Con alpha=1 tiende al oraculo global (SCM de toda la
    señal). Devuelve Y_stft (K, T) [+ pesos si save_weights=True].
    """
    K, T, M = X_stft.shape

    Y_stft = np.zeros((K, T), dtype=np.complex128)

    # Accumulators for matrix estimation
    Num_SS = np.zeros((K, M, M), dtype=np.complex128)
    Num_NN = np.zeros((K, M, M), dtype=np.complex128)

    Den = np.zeros((K, 1, 1), dtype=np.float64)

    # Reference microphone index (same convention as the mask-based version)
    ref_mic = M // 2

    if save_weights:
        weights_rec = np.zeros((K, T, M), dtype=np.complex128)

    for m in range(T):
        print(f"\rProcessing frame {m} of {T}", end="")
        X_frame = X_stft[:, m, :]
        S_frame = S_stft[:, m, :]
        N_frame = N_stft[:, m, :]

        # Outer products de las señales LIMPIAS (sin mascara)
        R_S = np.einsum("fm,fn->fmn", S_frame, S_frame.conj())
        R_N = np.einsum("fm,fn->fmn", N_frame, N_frame.conj())

        # Exponential forgetting factor. El denominador es comun a ambas SCM
        # (peso 1 por frame) y se cancela en la formula de Souden.
        Num_SS = alpha * Num_SS + R_S
        Num_NN = alpha * Num_NN + R_N
        Den = alpha * Den + 1.0

        Phi_SS = Num_SS / (Den + 1e-15)
        Phi_NN = Num_NN / (Den + 1e-15)

        # Diagonal loading for numerical stability before inversion
        tr_Phi = np.real(np.trace(Phi_NN, axis1=1, axis2=2))
        adaptive_load = min_loading * (tr_Phi / M)
        loading_matrix = np.eye(M)[np.newaxis, :, :] * np.maximum(adaptive_load, 1e-9)[:, np.newaxis, np.newaxis]

        Phi_NN_stable = Phi_NN + loading_matrix
        Phi_NN_inv = np.linalg.inv(Phi_NN_stable)

        # -----------------------------------------------------------------
        # SOUDEN'S MVDR FORMULATION (identica a la version mask-based)
        # -----------------------------------------------------------------
        Phi_inv_Phi_S = np.einsum("fmn,fnk->fmk", Phi_NN_inv, Phi_SS)
        lambda_norm = np.real(np.trace(Phi_inv_Phi_S, axis1=1, axis2=2))
        weights_unnorm = Phi_inv_Phi_S[:, :, ref_mic]
        weights = weights_unnorm / (lambda_norm[:, np.newaxis] + 1e-15)
        # -----------------------------------------------------------------

        if save_weights:
            weights_rec[:, m, :] = weights

        # Apply the weights to the OBSERVED mixture frame
        Y_stft[:, m] = np.einsum("fm,fm->f", weights.conj(), X_frame)

    if save_weights:
        return Y_stft, weights_rec
    else:
        return Y_stft


def MVDR_Souden_recursive_mask_slow(X_stft, mask_s, mask_n, min_loading=1e-6, save_weights=False, alpha = 0.99):
    K, T, M = X_stft.shape

    Y_stft = np.zeros((K, T), dtype=np.complex128)

    # Accumulators for matrix estimation
    Num_XX = np.zeros((K, M, M), dtype=np.complex128)
    Num_NN = np.zeros((K, M, M), dtype=np.complex128)

    Den_XX = np.zeros((K, 1, 1), dtype=np.float64)
    Den_NN = np.zeros((K, 1, 1), dtype=np.float64)

    # Reference microphone index (equivalent to the one-hot vector 'r' in Souden's formula)


        # Define Reference Microphone Index as middle index
    ref_mic_idx = M // 2
    ref_mic = ref_mic_idx


    if save_weights:
        weights_rec = np.zeros((K, T, M), dtype=np.complex128)

    for m in range(T):
        print(f"\rProcessing frame {m} of {T}", end="")
        X_frame = X_stft[:, m, :]
        m_s_frame = mask_s[:, m, np.newaxis, np.newaxis]
        m_n_frame = mask_n[:, m, np.newaxis, np.newaxis]

        R_instant = np.einsum("fm,fn->fmn", X_frame, X_frame.conj())

        # Apply exponential forgetting factor (alpha) to both accumulators
        Num_XX = alpha * Num_XX + m_s_frame * R_instant
        Den_XX = alpha * Den_XX + m_s_frame

        Num_NN = alpha * Num_NN + m_n_frame * R_instant
        Den_NN = alpha * Den_NN + m_n_frame

        # Calculate matrices by dividing the accumulators
        Phi_XX = Num_XX / (Den_XX + 1e-15)
        Phi_NN = Num_NN / (Den_NN + 1e-15)

        # Diagonal loading for numerical stability before inversion
        tr_Phi = np.real(np.trace(Phi_NN, axis1=1, axis2=2))
        adaptive_load = min_loading * (tr_Phi / M)
        loading_matrix = np.eye(M)[np.newaxis, :, :] * np.maximum(adaptive_load, 1e-9)[:, np.newaxis, np.newaxis]

        Phi_NN_stable = Phi_NN + loading_matrix
        Phi_NN_inv = np.linalg.inv(Phi_NN_stable)


        # -----------------------------------------------------------------
        # SOUDEN'S MVDR FORMULATION
        # -----------------------------------------------------------------
        # 1. Compute the product of Phi_NN_inv and Phi_XX
        # Shape: (K, M, M)
        Phi_inv_Phi_X = np.einsum("fmn,fnk->fmk", Phi_NN_inv, Phi_XX)

        # 2. Calculate the normalization factor lambda (trace of the product)
        # The trace of this specific product is theoretically real.
        lambda_norm = np.real(np.trace(Phi_inv_Phi_X, axis1=1, axis2=2))

        # 3. Multiply by the reference microphone vector 'r'
        # Since 'r' is a unit vector [1, 0, 0...], multiplying Phi_inv_Phi_X by 'r'
        # is mathematically equivalent to taking the column at the 'ref_mic' index.
        weights_unnorm = Phi_inv_Phi_X[:, :, ref_mic]

        # 4. Normalize the weights using lambda
        weights = weights_unnorm / (lambda_norm[:, np.newaxis] + 1e-15)
        # -----------------------------------------------------------------

        weights_rec[:, m, :] = weights


        # Apply the weights to the current STFT frame
        Y_stft[:, m] = np.einsum("fm,fm->f", weights_rec[:, m-1, :].conj(), X_frame)



    if save_weights:
        return Y_stft, weights_rec
    else:
        return Y_stft


# =====================================================================
# MVDR + Blind Analytical Normalization (BAN)
# =====================================================================
def MVDR_Souden_recursive_mask_BAN(X_stft, mask_s, mask_n, min_loading=1e-6, save_weights=False):
    K, T, M = X_stft.shape

    Y_stft = np.zeros((K, T), dtype=np.complex128)

    # Accumulators for matrix estimation
    Num_XX = np.zeros((K, M, M), dtype=np.complex128)
    Num_NN = np.zeros((K, M, M), dtype=np.complex128)

    Den_XX = np.zeros((K, 1, 1), dtype=np.float64)
    Den_NN = np.zeros((K, 1, 1), dtype=np.float64)

    # Reference microphone index (equivalent to the one-hot vector 'r' in Souden's formula)
    ref_mic = 0

    if save_weights:
        weights_rec = np.zeros((K, T, M), dtype=np.complex128)

    for m in range(T):
        print(f"\rProcessing frame {m} of {T}", end="")
        X_frame = X_stft[:, m, :]
        m_s_frame = mask_s[:, m, np.newaxis, np.newaxis]
        m_n_frame = mask_n[:, m, np.newaxis, np.newaxis]

        R_instant = np.einsum("fm,fn->fmn", X_frame, X_frame.conj())

        # Strict cumulative sum (Equation 4)
        Num_XX += m_s_frame * R_instant
        Den_XX += m_s_frame

        Num_NN += m_n_frame * R_instant
        Den_NN += m_n_frame

        # Calculate matrices by dividing the accumulators
        Phi_XX = Num_XX / (Den_XX + 1e-15)
        Phi_NN = Num_NN / (Den_NN + 1e-15)

        # Diagonal loading for numerical stability before inversion
        tr_Phi = np.real(np.trace(Phi_NN, axis1=1, axis2=2))
        adaptive_load = min_loading * (tr_Phi / M)
        loading_matrix = np.eye(M)[np.newaxis, :, :] * np.maximum(adaptive_load, 1e-9)[:, np.newaxis, np.newaxis]

        Phi_NN_stable = Phi_NN + loading_matrix
        Phi_NN_inv = np.linalg.inv(Phi_NN_stable)

        # -----------------------------------------------------------------
        # SOUDEN'S MVDR FORMULATION
        # -----------------------------------------------------------------
        # 1. Compute the product of Phi_NN_inv and Phi_XX
        Phi_inv_Phi_X = np.einsum("fmn,fnk->fmk", Phi_NN_inv, Phi_XX)

        # 2. Calculate the normalization factor lambda (trace of the product)
        lambda_norm = np.real(np.trace(Phi_inv_Phi_X, axis1=1, axis2=2))

        # 3. Extract the unnormalized weights
        weights_unnorm = Phi_inv_Phi_X[:, :, ref_mic]

        # 4. Normalize the weights using lambda
        weights = weights_unnorm / (lambda_norm[:, np.newaxis] + 1e-15)

        # -----------------------------------------------------------------
        # BLIND ANALYTICAL NORMALIZATION (BAN)
        # -----------------------------------------------------------------
        # Step A: Compute Phi_NN * w
        Phi_NN_w = np.einsum("fmn,fn->fm", Phi_NN_stable, weights)

        # Step B: Denominator = w^H * Phi_NN * w
        # Forcing real output to prevent accumulation of floating-point imaginary noise
        ban_denom = np.real(np.einsum("fm,fm->f", weights.conj(), Phi_NN_w))

        # Step C: Numerator = sqrt( (w^H * Phi_NN^2 * w) / M )
        # w^H * Phi_NN^2 * w is equivalent to the squared L2 norm of (Phi_NN * w)
        w_Phi_NN_sq = np.real(np.einsum("fm,fm->f", Phi_NN_w.conj(), Phi_NN_w))
        ban_num = np.sqrt(w_Phi_NN_sq / M)

        # Step D: Compute BAN scalar per frequency bin
        ban_factor = ban_num / (ban_denom + 1e-15)

        # Step E: Apply BAN scaling to the current MVDR weights
        weights = weights * ban_factor[:, np.newaxis]
        # -----------------------------------------------------------------

        if save_weights:
            weights_rec[:, m, :] = weights

        # Apply the weights to the current STFT frame
        Y_stft[:, m] = np.einsum("fm,fm->f", weights.conj(), X_frame)

    if save_weights:
        return Y_stft, weights_rec
    else:
        return Y_stft


def MVDR_Souden_recursive_mask_BAN_alpha(X_stft, mask_s, mask_n, min_loading=1e-6,
                                         save_weights=False, alpha=0.99):
    K, T, M = X_stft.shape

    Y_stft = np.zeros((K, T), dtype=np.complex128)

    # Accumulators for matrix estimation
    Num_XX = np.zeros((K, M, M), dtype=np.complex128)
    Num_NN = np.zeros((K, M, M), dtype=np.complex128)

    Den_XX = np.zeros((K, 1, 1), dtype=np.float64)
    Den_NN = np.zeros((K, 1, 1), dtype=np.float64)

    # Reference microphone index (igual que el original BAN)
    ref_mic = 0

    if save_weights:
        weights_rec = np.zeros((K, T, M), dtype=np.complex128)

    for m in range(T):
        print(f"\rProcessing frame {m} of {T}", end="")
        X_frame = X_stft[:, m, :]
        m_s_frame = mask_s[:, m, np.newaxis, np.newaxis]
        m_n_frame = mask_n[:, m, np.newaxis, np.newaxis]

        R_instant = np.einsum("fm,fn->fmn", X_frame, X_frame.conj())

        # UNICA diferencia vs el original: factor de olvido exponencial alpha
        # (original: Num += ... ; Den += ...  == alpha=1.0, acumulacion estricta)
        Num_XX = alpha * Num_XX + m_s_frame * R_instant
        Den_XX = alpha * Den_XX + m_s_frame

        Num_NN = alpha * Num_NN + m_n_frame * R_instant
        Den_NN = alpha * Den_NN + m_n_frame

        # Calculate matrices by dividing the accumulators
        Phi_XX = Num_XX / (Den_XX + 1e-15)
        Phi_NN = Num_NN / (Den_NN + 1e-15)

        # Diagonal loading for numerical stability before inversion
        tr_Phi = np.real(np.trace(Phi_NN, axis1=1, axis2=2))
        adaptive_load = min_loading * (tr_Phi / M)
        loading_matrix = np.eye(M)[np.newaxis, :, :] * np.maximum(adaptive_load, 1e-9)[:, np.newaxis, np.newaxis]

        Phi_NN_stable = Phi_NN + loading_matrix
        Phi_NN_inv = np.linalg.inv(Phi_NN_stable)

        # -----------------------------------------------------------------
        # SOUDEN'S MVDR FORMULATION
        # -----------------------------------------------------------------
        Phi_inv_Phi_X = np.einsum("fmn,fnk->fmk", Phi_NN_inv, Phi_XX)
        lambda_norm = np.real(np.trace(Phi_inv_Phi_X, axis1=1, axis2=2))
        weights_unnorm = Phi_inv_Phi_X[:, :, ref_mic]
        weights = weights_unnorm / (lambda_norm[:, np.newaxis] + 1e-15)

        # -----------------------------------------------------------------
        # BLIND ANALYTICAL NORMALIZATION (BAN)
        # -----------------------------------------------------------------
        Phi_NN_w = np.einsum("fmn,fn->fm", Phi_NN_stable, weights)
        ban_denom = np.real(np.einsum("fm,fm->f", weights.conj(), Phi_NN_w))
        w_Phi_NN_sq = np.real(np.einsum("fm,fm->f", Phi_NN_w.conj(), Phi_NN_w))
        ban_num = np.sqrt(w_Phi_NN_sq / M)
        ban_factor = ban_num / (ban_denom + 1e-15)
        weights = weights * ban_factor[:, np.newaxis]
        # -----------------------------------------------------------------

        if save_weights:
            weights_rec[:, m, :] = weights

        Y_stft[:, m] = np.einsum("fm,fm->f", weights.conj(), X_frame)

    if save_weights:
        return Y_stft, weights_rec
    else:
        return Y_stft


# =====================================================================
# MVDR FIXED (carga relativa + Hermitiana + solve) y post-filtros
# =====================================================================
def MVDR_Souden_recursive_mask_fixed(X_stft, mask_s, mask_n, min_loading=1e-2,
                                     save_weights=False, alpha=0.99, rank1=False):
    """
    Variante FIXED de MVDR_Souden_recursive_mask. Misma formulacion de Souden
    (w = Phi_NN^-1 Phi_XX u / tr(Phi_NN^-1 Phi_XX)), ref_mic = M//2, factor de
    olvido alpha; corrige:

      1. CARGA DIAGONAL escala-invariante. El original hace
             adaptive_load = min_loading*tr(Phi_NN)/M           (min_loading=1e-6)
             loading = max(adaptive_load, 1e-9)                  <-- PISO ABSOLUTO
         Con senales reales (~1e-3..1e-4 de potencia) adaptive_load ~1e-9..1e-12
         cae en el piso absoluto y queda ~1e-6 MAS CHICO que las entradas de
         Phi_NN -> la inversion corre casi SIN regularizar (mal condicionada con
         12 mics -> auto-cancelacion / musical noise). Aca: loading RELATIVO puro
         (min_loading*tr/M, default 1e-2), SIN piso absoluto (solo un epsilon
         1e-12 para el cold-start). Es proporcional a la potencia -> mismo efecto
         a cualquier nivel de senal.
      2. HERMITIANA: se fuerza Phi = (Phi + Phi^H)/2 antes de invertir (evita
         deriva por round-off complejo que rompe la simetria PSD).
      3. INVERSION por np.linalg.solve(Phi_NN, Phi_XX) en vez de inv()@ (mas
         estable numericamente y mas rapido).
      4. (opcional, rank1=True) Phi_XX se reemplaza por su aproximacion RANK-1
         (autovector principal): denoisea la covarianza de target y suele ayudar
         a SNR bajo, donde la Phi_XX enmascarada esta mas contaminada por ruido.
    """
    K, T, M = X_stft.shape
    Y_stft = np.zeros((K, T), dtype=np.complex128)
    Num_XX = np.zeros((K, M, M), dtype=np.complex128)
    Num_NN = np.zeros((K, M, M), dtype=np.complex128)
    Den_XX = np.zeros((K, 1, 1), dtype=np.float64)
    Den_NN = np.zeros((K, 1, 1), dtype=np.float64)
    ref_mic = M // 2                      # consistente con la Souden original
    eye = np.eye(M)[np.newaxis, :, :]

    if save_weights:
        weights_rec = np.zeros((K, T, M), dtype=np.complex128)

    for m in range(T):
        print(f"\rProcessing frame {m} of {T}", end="")
        X_frame = X_stft[:, m, :]
        m_s_frame = mask_s[:, m, np.newaxis, np.newaxis]
        m_n_frame = mask_n[:, m, np.newaxis, np.newaxis]

        R_instant = np.einsum("fm,fn->fmn", X_frame, X_frame.conj())

        Num_XX = alpha * Num_XX + m_s_frame * R_instant
        Den_XX = alpha * Den_XX + m_s_frame
        Num_NN = alpha * Num_NN + m_n_frame * R_instant
        Den_NN = alpha * Den_NN + m_n_frame

        Phi_XX = Num_XX / (Den_XX + 1e-15)
        Phi_NN = Num_NN / (Den_NN + 1e-15)

        # (2) forzar Hermitiana antes de invertir
        Phi_XX = 0.5 * (Phi_XX + np.conj(np.transpose(Phi_XX, (0, 2, 1))))
        Phi_NN = 0.5 * (Phi_NN + np.conj(np.transpose(Phi_NN, (0, 2, 1))))

        # (4) rank-1 de Phi_XX (autovector principal) -- opcional
        if rank1:
            evals, evecs = np.linalg.eigh(Phi_XX)          # ascendente
            v = evals[:, -1][:, np.newaxis] * evecs[:, :, -1]  # lambda_max * v_max (K,M)
            Phi_XX = np.einsum("fm,fn->fmn", v, evecs[:, :, -1].conj())

        # (1) carga diagonal RELATIVA, escala-invariante, sin piso absoluto
        tr_Phi = np.real(np.trace(Phi_NN, axis1=1, axis2=2))
        adaptive_load = min_loading * (tr_Phi / M)
        Phi_NN_stable = Phi_NN + eye * (adaptive_load[:, np.newaxis, np.newaxis] + 1e-12)

        # (3) w = Phi_NN^-1 Phi_XX  via solve
        Phi_inv_Phi_X = np.linalg.solve(Phi_NN_stable, Phi_XX)      # (K,M,M)
        lambda_norm = np.real(np.trace(Phi_inv_Phi_X, axis1=1, axis2=2))
        weights_unnorm = Phi_inv_Phi_X[:, :, ref_mic]
        weights = weights_unnorm / (lambda_norm[:, np.newaxis] + 1e-15)

        if save_weights:
            weights_rec[:, m, :] = weights
        Y_stft[:, m] = np.einsum("fm,fm->f", weights.conj(), X_frame)

    if save_weights:
        return Y_stft, weights_rec
    return Y_stft


def MVDR_Souden_recursive_mask_MWF(X_stft, mask_s, mask_n, min_loading=1e-2,
                                   save_weights=False, alpha=0.99, mu=1.0,
                                   rank1=False):
    """
    POST-FILTRO DE WIENER sobre el MVDR de Souden = MWF paramétrico (PMWF-beta del
    paper: Souden et al. 2010, "On Optimal Frequency-Domain Multichannel Linear
    Filtering for Noise Reduction", IEEE TASLP 18(2)). Parte de
    MVDR_Souden_recursive_mask_fixed (loading relativo + Hermitiana + solve) y
    cambia SÓLO el denominador de la normalización de Souden.

    Ecuaciones del paper (Phi_vv=Phi_NN ruido, Phi_xx=Phi_XX target enmascarado,
    u = one-hot en ref_mic, lambda = tr(Phi_NN^-1 Phi_XX) = Property 1, ec. 16):
        MVDR    (ec. 24):  w = Phi_NN^-1 Phi_XX u / lambda
        PMWF-beta(ec. 18): w = Phi_NN^-1 Phi_XX u / (beta + lambda)
    O sea el PMWF es el MVDR multiplicado por la ganancia de Wiener espectral
        G = lambda / (lambda + beta) ,
    con beta = mu aquí = trade-off distorsión-vs-supresión:
        mu = 0  -> MVDR puro  (reproduce EXACTAMENTE _fixed; ec. 24 = PMWF-0)
        mu = 1  -> MWF estándar / PMWF-1 (ec. 19)
        mu > 1  -> más supresión de ruido, más distorsión de voz
    lambda es el MISMO trace que el MVDR ya calcula (lambda_norm); el único cambio
    respecto de _fixed es sumarle mu al denominador. Loading, Hermitiana, solve,
    ref_mic=M//2 y (opcional) rank-1 de Phi_XX quedan idénticos a _fixed.
    """
    K, T, M = X_stft.shape
    Y_stft = np.zeros((K, T), dtype=np.complex128)
    Num_XX = np.zeros((K, M, M), dtype=np.complex128)
    Num_NN = np.zeros((K, M, M), dtype=np.complex128)
    Den_XX = np.zeros((K, 1, 1), dtype=np.float64)
    Den_NN = np.zeros((K, 1, 1), dtype=np.float64)
    ref_mic = M // 2                      # consistente con la Souden original
    eye = np.eye(M)[np.newaxis, :, :]

    if save_weights:
        weights_rec = np.zeros((K, T, M), dtype=np.complex128)

    for m in range(T):
        print(f"\rProcessing frame {m} of {T}", end="")
        X_frame = X_stft[:, m, :]
        m_s_frame = mask_s[:, m, np.newaxis, np.newaxis]
        m_n_frame = mask_n[:, m, np.newaxis, np.newaxis]

        R_instant = np.einsum("fm,fn->fmn", X_frame, X_frame.conj())

        Num_XX = alpha * Num_XX + m_s_frame * R_instant
        Den_XX = alpha * Den_XX + m_s_frame
        Num_NN = alpha * Num_NN + m_n_frame * R_instant
        Den_NN = alpha * Den_NN + m_n_frame

        Phi_XX = Num_XX / (Den_XX + 1e-15)
        Phi_NN = Num_NN / (Den_NN + 1e-15)

        # Hermitiana antes de invertir (igual que _fixed)
        Phi_XX = 0.5 * (Phi_XX + np.conj(np.transpose(Phi_XX, (0, 2, 1))))
        Phi_NN = 0.5 * (Phi_NN + np.conj(np.transpose(Phi_NN, (0, 2, 1))))

        # rank-1 de Phi_XX (autovector principal) -- opcional, igual que _fixed
        if rank1:
            evals, evecs = np.linalg.eigh(Phi_XX)
            v = evals[:, -1][:, np.newaxis] * evecs[:, :, -1]
            Phi_XX = np.einsum("fm,fn->fmn", v, evecs[:, :, -1].conj())

        # carga diagonal RELATIVA, escala-invariante (igual que _fixed)
        tr_Phi = np.real(np.trace(Phi_NN, axis1=1, axis2=2))
        adaptive_load = min_loading * (tr_Phi / M)
        Phi_NN_stable = Phi_NN + eye * (adaptive_load[:, np.newaxis, np.newaxis] + 1e-12)

        # w = Phi_NN^-1 Phi_XX  via solve (igual que _fixed)
        Phi_inv_Phi_X = np.linalg.solve(Phi_NN_stable, Phi_XX)      # (K,M,M)
        lambda_norm = np.real(np.trace(Phi_inv_Phi_X, axis1=1, axis2=2))
        weights_unnorm = Phi_inv_Phi_X[:, :, ref_mic]
        # UNICO cambio vs _fixed: denominador (lambda + mu) en vez de lambda.
        # mu=0 -> MVDR (_fixed); mu=1 -> MWF estándar; mu>1 -> más supresión.
        weights = weights_unnorm / (lambda_norm[:, np.newaxis] + mu + 1e-15)

        if save_weights:
            weights_rec[:, m, :] = weights
        Y_stft[:, m] = np.einsum("fm,fm->f", weights.conj(), X_frame)

    if save_weights:
        return Y_stft, weights_rec
    return Y_stft


def MVDR_Souden_recursive_mask_BAN_MWF(X_stft, mask_s, mask_n, min_loading=1e-2,
                                       save_weights=False, alpha=0.99, mu=4.0,
                                       rank1=False):
    """
    Souden FIXED + BAN + post-filtro de Wiener (MWF paramétrico, mu). Base = _fixed
    (loading relativo + Hermitiana + solve + ref_mic=M//2), denominador MWF
    (lambda+mu), y ENCIMA la Blind Analytical Normalization (BAN), pero con
    ref_mic=M//2 y Phi_NN_stable coherente (a diferencia del BAN viejo, que usaba
    ref_mic=0 y loading absoluto).

    OJO -- ADVERTENCIA MATEMATICA (verificada numéricamente): la BAN es INVARIANTE
    A ESCALA. La ganancia de Wiener del MWF es un escalar REAL por frecuencia
    c = lambda/(lambda+mu); bajo w->c*w el ban_factor escala como (1/c), así que el
    factor c (y por lo tanto mu) SE CANCELA. Consecuencia: la salida de esta función
    NO depende de mu -- BAN(MWF_mu) == BAN(MVDR) exacto. Se deja mu como parámetro
    por completitud/trazabilidad, pero combinar BAN con el post-filtro es un no-op:
    la BAN re-fija la escala del beamformer y borra la atenuación de Wiener.
    """
    K, T, M = X_stft.shape
    Y_stft = np.zeros((K, T), dtype=np.complex128)
    Num_XX = np.zeros((K, M, M), dtype=np.complex128)
    Num_NN = np.zeros((K, M, M), dtype=np.complex128)
    Den_XX = np.zeros((K, 1, 1), dtype=np.float64)
    Den_NN = np.zeros((K, 1, 1), dtype=np.float64)
    ref_mic = M // 2
    eye = np.eye(M)[np.newaxis, :, :]

    if save_weights:
        weights_rec = np.zeros((K, T, M), dtype=np.complex128)

    for m in range(T):
        print(f"\rProcessing frame {m} of {T}", end="")
        X_frame = X_stft[:, m, :]
        m_s_frame = mask_s[:, m, np.newaxis, np.newaxis]
        m_n_frame = mask_n[:, m, np.newaxis, np.newaxis]

        R_instant = np.einsum("fm,fn->fmn", X_frame, X_frame.conj())

        Num_XX = alpha * Num_XX + m_s_frame * R_instant
        Den_XX = alpha * Den_XX + m_s_frame
        Num_NN = alpha * Num_NN + m_n_frame * R_instant
        Den_NN = alpha * Den_NN + m_n_frame

        Phi_XX = Num_XX / (Den_XX + 1e-15)
        Phi_NN = Num_NN / (Den_NN + 1e-15)

        Phi_XX = 0.5 * (Phi_XX + np.conj(np.transpose(Phi_XX, (0, 2, 1))))
        Phi_NN = 0.5 * (Phi_NN + np.conj(np.transpose(Phi_NN, (0, 2, 1))))

        if rank1:
            evals, evecs = np.linalg.eigh(Phi_XX)
            v = evals[:, -1][:, np.newaxis] * evecs[:, :, -1]
            Phi_XX = np.einsum("fm,fn->fmn", v, evecs[:, :, -1].conj())

        tr_Phi = np.real(np.trace(Phi_NN, axis1=1, axis2=2))
        adaptive_load = min_loading * (tr_Phi / M)
        Phi_NN_stable = Phi_NN + eye * (adaptive_load[:, np.newaxis, np.newaxis] + 1e-12)

        Phi_inv_Phi_X = np.linalg.solve(Phi_NN_stable, Phi_XX)
        lambda_norm = np.real(np.trace(Phi_inv_Phi_X, axis1=1, axis2=2))
        weights_unnorm = Phi_inv_Phi_X[:, :, ref_mic]
        # denominador MWF (lambda + mu) -- se cancelará con BAN (ver docstring)
        weights = weights_unnorm / (lambda_norm[:, np.newaxis] + mu + 1e-15)

        # BLIND ANALYTICAL NORMALIZATION (invariante a escala -> anula mu)
        Phi_NN_w = np.einsum("fmn,fn->fm", Phi_NN_stable, weights)
        ban_denom = np.real(np.einsum("fm,fm->f", weights.conj(), Phi_NN_w))
        w_Phi_NN_sq = np.real(np.einsum("fm,fm->f", Phi_NN_w.conj(), Phi_NN_w))
        ban_num = np.sqrt(w_Phi_NN_sq / M)
        ban_factor = ban_num / (ban_denom + 1e-15)
        weights = weights * ban_factor[:, np.newaxis]

        if save_weights:
            weights_rec[:, m, :] = weights
        Y_stft[:, m] = np.einsum("fm,fm->f", weights.conj(), X_frame)

    if save_weights:
        return Y_stft, weights_rec
    return Y_stft


def MVDR_Souden_recursive_mask_specsub(X_stft, mask_s, mask_n, mask_s_soft,
                                       min_loading=1e-2, alpha=0.99, smooth=0.33,
                                       save_weights=False):
    """
    Souden FIXED + POST-FILTRO DE SUSTRACCION ESPECTRAL (mask-based) sobre la salida
    del beamformer.

    Cadena:
      1. Beamformer = MVDR_Souden_recursive_mask_fixed(X, mask_s, mask_n) con la
         mascara SHARPENED de siempre (sharpen_exp=4.0) -> Y_bf(k,t) complejo.
      2. Post-filtro espectral: se aplica una GANANCIA REAL por bin
             G(k,t) = smooth + (1 - smooth) * mask_s_soft(k,t)
         y  Y_out = Y_bf * G. Solo escala magnitud, conserva fase.

    IMPORTANTE (pedido explicito): mask_s_soft es la mascara ORIGINAL del DTLN (SIN
    el realce a potencia que usa el beamformer -- ese realce agudiza los bordes y la
    vuelve casi binaria). Se pasa la mascara suave/continua tal cual sale del DTLN
    (estirada a [0,1]); tipicamente = mask_sharpen ** (1/sharpen_exp).

    smooth = factor de suavizado / piso espectral (dry-wet sobre la ganancia):
        smooth=1.0 -> G=1 -> NO filtra (reproduce el fixed exacto).
        smooth=0.0 -> G=mask -> sustraccion dura (mascara suave pura).
        smooth=0.33 (default) -> extraccion SUAVE: piso 0.33 (~-9.6 dB) donde no hay
        voz, hasta 1.0 donde hay voz -> evita gating abrupto / ruido musical.
    """
    res = MVDR_Souden_recursive_mask_fixed(X_stft, mask_s, mask_n,
                                           min_loading=min_loading, alpha=alpha,
                                           save_weights=save_weights)
    Y, W = (res if save_weights else (res, None))
    Y = Y.copy()

    Tm = min(Y.shape[1], mask_s_soft.shape[1])
    G = smooth + (1.0 - smooth) * np.clip(mask_s_soft[:, :Tm], 0.0, 1.0)  # (K,Tm) real
    Y[:, :Tm] *= G   # frames sin mascara (si sobran) quedan como beamformer crudo

    if save_weights:
        return Y, W
    return Y


def MVDR_Souden_recursive_mask_specsub_base(X_stft, mask_s, mask_n, mask_s_soft,
                                            min_loading=1e-6, alpha=0.99, smooth=0.33,
                                            save_weights=False):
    """
    Sustraccion espectral sobre el CORE BASE (MVDR_Souden_recursive_mask), el mismo
    beamformer que usa NM_MVDR (el de mejor rendimiento).

    Diferencia con MVDR_Souden_recursive_mask_specsub: aquel apoyaba el post-filtro
    sobre _fixed (carga diagonal RELATIVA pura, solve). Este lo apoya sobre el core
    original -> carga diagonal RELATIVA CON PISO ABSOLUTO (max(min_loading*tr/M, 1e-9),
    inv()). Se verifico empiricamente (barrido de iSNR) que la carga liviana del core
    base preserva mejor PESQ/SIR; esta variante combina ESE beamformer con la misma
    ganancia espectral suave:

        Beamformer = MVDR_Souden_recursive_mask(X, mask_s, mask_n)  -> Y_bf
        Post-filtro: G(k,t) = smooth + (1 - smooth) * mask_s_soft(k,t) ;  Y = Y_bf * G

    mask_s_soft = mascara ORIGINAL del DTLN (sin realce; = mask_sharpen ** (1/sharpen_exp)).
    smooth: 1.0 = sin filtro (== core base exacto); 0.33 default (piso ~-9.6 dB);
    0.0 = mascara suave pura.
    """
    res = MVDR_Souden_recursive_mask(X_stft, mask_s, mask_n,
                                     min_loading=min_loading, alpha=alpha,
                                     save_weights=save_weights)
    Y, W = (res if save_weights else (res, None))
    Y = Y.copy()

    Tm = min(Y.shape[1], mask_s_soft.shape[1])
    G = smooth + (1.0 - smooth) * np.clip(mask_s_soft[:, :Tm], 0.0, 1.0)  # (K,Tm) real
    Y[:, :Tm] *= G

    if save_weights:
        return Y, W
    return Y


def MVDR_Souden_recursive_mask_BAN_specsub_base(X_stft, mask_s, mask_n, mask_s_soft,
                                                min_loading=1e-6, alpha=0.99, smooth=0.33,
                                                save_weights=False):
    """
    BAN sobre el CORE BASE + POST-FILTRO de sustraccion espectral.

    A diferencia de MVDR_Souden_recursive_mask_BAN_specsub (que apoya la BAN sobre el
    core _fixed via BAN_MWF: carga relativa pura + solve), este usa
    MVDR_Souden_recursive_mask_BAN_alpha: Blind Analytical Normalization con factor de
    olvido alpha sobre la carga diagonal RELATIVA CON PISO ABSOLUTO
    (max(min_loading*tr/M, 1e-9), inv()), el mismo estilo de loading del ganador
    NM_MVDR. Es el analogo BAN de MVDR_Souden_recursive_mask_specsub_base.

    Cadena:
        Beamformer = MVDR_Souden_recursive_mask_BAN_alpha(X, mask_s, mask_n) -> Y_ban
        Post-filtro: G(k,t) = smooth + (1 - smooth) * mask_s_soft(k,t) ;  Y = Y_ban * G

    La BAN fija una escala de salida referida al ruido (buena fidelidad de forma de
    onda); el specsub actua DESPUES sobre la salida (ganancia por bin en magnitud),
    asi que NO se cancela con la BAN (a diferencia de combinar BAN con el MWF, donde
    la BAN invariante a escala anulaba mu). mask_s_soft = mascara original del DTLN
    (= mask_sharpen ** (1/sharpen_exp)). smooth: 1.0 = sin filtro (== BAN base exacto);
    0.33 default; 0.0 = mascara suave pura.
    """
    res = MVDR_Souden_recursive_mask_BAN_alpha(X_stft, mask_s, mask_n,
                                               min_loading=min_loading, alpha=alpha,
                                               save_weights=save_weights)
    Y, W = (res if save_weights else (res, None))
    Y = Y.copy()

    Tm = min(Y.shape[1], mask_s_soft.shape[1])
    G = smooth + (1.0 - smooth) * np.clip(mask_s_soft[:, :Tm], 0.0, 1.0)  # (K,Tm) real
    Y[:, :Tm] *= G

    if save_weights:
        return Y, W
    return Y


def MVDR_Souden_recursive_mask_BAN_specsub(X_stft, mask_s, mask_n, mask_s_soft,
                                           min_loading=1e-6, alpha=0.99, smooth=0.33,
                                           save_weights=False):
    """
    Souden FIXED + BAN + POST-FILTRO DE SUSTRACCION ESPECTRAL. Combina las dos ideas:
      1. Beamformer = fixed + BAN (Blind Analytical Normalization) -> Y_ban(k,t).
         (== MVDR_Souden_recursive_mask_BAN_MWF con mu=0; la BAN fija una escala de
          salida referida al ruido -> buena fidelidad de forma de onda a SNR alto.)
      2. Sustraccion espectral con la mascara ORIGINAL del DTLN (sin realce):
             G(k,t) = smooth + (1 - smooth) * mask_s_soft(k,t) ,  Y = Y_ban * G.

    A DIFERENCIA de BAN+MWF (donde la BAN, invariante a escala, ANULABA el post-
    filtro mu porque mu era un escalar sobre los PESOS antes de la normalizacion),
    aca el specsub se aplica DESPUES, sobre la SALIDA (ganancia por bin (k,t) en
    magnitud), asi que NO se cancela: BAN y specsub actuan en etapas distintas.

    smooth: igual que en el specsub sin BAN (1=sin filtro -> reproduce fixed+BAN;
    0=mascara suave pura; 0.33 default). mask_s_soft = mascara original (tipicamente
    mask_sharpen ** (1/sharpen_exp)).
    """
    res = MVDR_Souden_recursive_mask_BAN_MWF(X_stft, mask_s, mask_n,
                                             min_loading=min_loading, alpha=alpha,
                                             mu=0.0, save_weights=save_weights)
    Y, W = (res if save_weights else (res, None))
    Y = Y.copy()

    Tm = min(Y.shape[1], mask_s_soft.shape[1])
    G = smooth + (1.0 - smooth) * np.clip(mask_s_soft[:, :Tm], 0.0, 1.0)
    Y[:, :Tm] *= G

    if save_weights:
        return Y, W
    return Y
