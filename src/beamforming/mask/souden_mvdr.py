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

MICROFONO DE REFERENCIA (unificado): TODOS los cores exponen `ref_mic_idx` y usan
el MISMO default (None -> M//2). Es el canal sobre el que se proyecta la salida (el
one-hot `r` de la formula) y debe coincidir con (a) el canal con el que el wrapper
estima la mascara del DTLN y (b) el canal de referencia de las metricas intrusivas
(ref_mic_mode). Cuando los tres coinciden el filtro estima la voz TAL COMO LLEGA al
canal que se mide -> mejores resultados y comparacion honesta. Los wrappers del
benchmark igual pasan el indice explicito (geometry.select_reference_mic).

  - MVDR_Souden_recursive_mask        : MVDR base, factor de olvido alpha.
  - MVDR_Souden_recursive_mask_slow   : idem base pero aplica los pesos del frame
                                        anterior (variante experimental).
  - MVDR_Souden_recursive_mask_BAN    : acumulacion estricta (alpha=1) + Blind
                                        Analytical Normalization.
  - MVDR_Souden_recursive_mask_BAN_alpha : BAN + factor de olvido alpha.
  - MVDR_Souden_recursive_mask_fixed  : carga diagonal RELATIVA (escala-invariante)
                                        + Hermitiana + solve.
  - MVDR_Souden_recursive_mask_MWF    : fixed + post-filtro de Wiener parametrico
                                        (PMWF-beta, denominador lambda+mu).
  - MVDR_Souden_recursive_mask_BAN_MWF: fixed + MWF + BAN (OJO: la BAN es invariante
                                        a escala -> anula el efecto de mu).
  - MVDR_Souden_recursive_mask_specsub: fixed + sustraccion espectral (ganancia por
                                        bin con la mascara suave del DTLN).
  - MVDR_Souden_recursive_mask_BAN_specsub : fixed + BAN + sustraccion espectral.
  - MVDR_Souden_recursive_mask_subtract: SUSTRACCION DE COVARIANZA. Estima el
                                        target como Phi_SS = Phi_XX - Phi_NN y
                                        normaliza por lambda_S = lambda - M, en
                                        vez de por lambda. Corrige el COLAPSO DE
                                        ESCALA en graves de todos los cores de
                                        arriba (ver mas abajo). Opcionalmente:
                                        gate por confianza y alpha por bin.

COLAPSO DE ESCALA EN GRAVES (motivo de la variante _subtract)
------------------------------------------------------------
Todos los cores de esta familia le pasan a la formula de Souden la covarianza de
la MEZCLA enmascarada en lugar de la del TARGET, o sea Phi_XX ~= Phi_SS + Phi_NN.
Eso hace que el denominador lambda = tr(Phi_NN^-1 Phi_XX) = lambda_S + M quede
ACOTADO POR ABAJO POR M. En los bins donde la mascara no encuentra voz,
Phi_XX ~= Phi_NN y el filtro degenera a w = u/M: la banda sale -20*log10(M) dB
(M=8 -> -18 dB) con ganancia de arreglo NULA. Verificado algebraicamente
(forzando Phi_XX == Phi_NN la salida es exactamente x_ref/M) y medido sobre MIRD
en tests/lowfreq_diagnostic_run.py. PESQ (P.862) no lo ve: no evalua debajo de
~300 Hz.

FACTOR DE OLVIDO POR BIN
------------------------
MVDR_Souden_recursive_mask y MVDR_Souden_recursive_mask_subtract aceptan `alpha`
ESCALAR o como array (K,) -> factor de olvido dependiente de la frecuencia. Solo
rinde COMBINADO con la sustraccion: sobre el core base es neutro, porque arregla
la varianza del estimador mientras que el problema en graves es la normalizacion.
"""

import numpy as np


# =====================================================================
# MVDR base (Souden) con factor de olvido
# =====================================================================
def MVDR_Souden_recursive_mask(X_stft, mask_s, mask_n, min_loading=1e-6, save_weights=False, alpha = 0.99,
                               ref_mic_idx=None):
    """
    MVDR de Souden mask-based, recursivo con factor de olvido alpha.

    ref_mic_idx : int | None
        Microfono de REFERENCIA sobre el que se proyecta la salida (el vector
        one-hot `r` de la formula). Fija el "punto de escucha" del filtro: la
        salida estima la voz TAL COMO LLEGA a ese canal. None -> M//2 (default
        historico). El benchmark pasa aca el microfono mas cercano al centro
        geometrico del arreglo (geometry.select_reference_mic), que es el que
        minimiza la diferencia de camino acustico hacia el resto.
    """
    K, T, M = X_stft.shape

    Y_stft = np.zeros((K, T), dtype=np.complex128)

    # Accumulators for matrix estimation
    Num_XX = np.zeros((K, M, M), dtype=np.complex128)
    Num_NN = np.zeros((K, M, M), dtype=np.complex128)

    Den_XX = np.zeros((K, 1, 1), dtype=np.float64)
    Den_NN = np.zeros((K, 1, 1), dtype=np.float64)

    # Reference microphone index (equivalent to the one-hot vector 'r' in Souden's
    # formula). Default historico: el indice del medio (M // 2).
    ref_mic = M // 2 if ref_mic_idx is None else int(ref_mic_idx)
    if not (0 <= ref_mic < M):
        raise ValueError(f"ref_mic_idx={ref_mic_idx} fuera de rango para M={M}.")
    # alpha puede ser ESCALAR o un array POR BIN (K,) -> factor de olvido
    # DEPENDIENTE DE LA FRECUENCIA. Se lleva a (K,1,1), que difunde correctamente
    # contra Num (K,M,M) y contra Den (K,1,1) sin tocar nada mas del bucle.
    # Motivacion: el campo a baja frecuencia decorrelaciona mas lento (la
    # coherencia se sostiene sobre ~lambda/2), asi que promediar mas tiempo ahi
    # baja la varianza de las SCM casi sin costo de tracking.
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.ndim == 0:
        alpha = np.full((K,), float(alpha))
    if alpha.shape != (K,):
        raise ValueError(f"alpha debe ser escalar o de shape ({K},); es {alpha.shape}.")
    alpha = alpha[:, np.newaxis, np.newaxis]

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
def MVDR_Souden_recursive_oracle(X_stft, S_stft, N_stft, min_loading=1e-6, save_weights=False, alpha=0.99,
                                 ref_mic_idx=None):
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

    ref_mic_idx : int | None
        Microfono de referencia sobre el que se proyecta la salida (mismo rol que
        en la version mask-based). None -> M//2 (default historico).
    """
    K, T, M = X_stft.shape

    Y_stft = np.zeros((K, T), dtype=np.complex128)

    # Accumulators for matrix estimation
    Num_SS = np.zeros((K, M, M), dtype=np.complex128)
    Num_NN = np.zeros((K, M, M), dtype=np.complex128)

    Den = np.zeros((K, 1, 1), dtype=np.float64)

    # Reference microphone index (same convention as the mask-based version)
    ref_mic = M // 2 if ref_mic_idx is None else int(ref_mic_idx)
    if not (0 <= ref_mic < M):
        raise ValueError(f"ref_mic_idx={ref_mic_idx} fuera de rango para M={M}.")

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


def MVDR_Souden_recursive_mask_slow(X_stft, mask_s, mask_n, min_loading=1e-6, save_weights=False, alpha = 0.99,
                                    ref_mic_idx=None):
    K, T, M = X_stft.shape

    Y_stft = np.zeros((K, T), dtype=np.complex128)

    # Accumulators for matrix estimation
    Num_XX = np.zeros((K, M, M), dtype=np.complex128)
    Num_NN = np.zeros((K, M, M), dtype=np.complex128)

    Den_XX = np.zeros((K, 1, 1), dtype=np.float64)
    Den_NN = np.zeros((K, 1, 1), dtype=np.float64)

    # Reference microphone index (equivalent to the one-hot vector 'r' in Souden's formula).
    # ref_mic_idx=None -> M//2 (default historico de la familia Souden). El canal de
    # referencia fija el DOMINIO de la salida: el filtro estima la voz TAL COMO LLEGA
    # a ese microfono, asi que las metricas tienen que compararse contra ese canal.
    ref_mic = M // 2 if ref_mic_idx is None else int(ref_mic_idx)
    if not (0 <= ref_mic < M):
        raise ValueError(f"ref_mic_idx={ref_mic_idx} fuera de rango para M={M}.")


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
def MVDR_Souden_recursive_mask_BAN(X_stft, mask_s, mask_n, min_loading=1e-6, save_weights=False,
                                   ref_mic_idx=None):
    K, T, M = X_stft.shape

    Y_stft = np.zeros((K, T), dtype=np.complex128)

    # Accumulators for matrix estimation
    Num_XX = np.zeros((K, M, M), dtype=np.complex128)
    Num_NN = np.zeros((K, M, M), dtype=np.complex128)

    Den_XX = np.zeros((K, 1, 1), dtype=np.float64)
    Den_NN = np.zeros((K, 1, 1), dtype=np.float64)

    # Reference microphone index (equivalent to the one-hot vector 'r' in Souden's
    # formula). ref_mic_idx=None -> M//2, UNIFICADO con el resto de la familia (esta
    # variante usaba 0 historicamente). Los wrappers del benchmark pasan siempre un
    # indice explicito para que el BF proyecte sobre el MISMO canal con el que se
    # estima la mascara y sobre el que miden las metricas intrusivas.
    ref_mic = M // 2 if ref_mic_idx is None else int(ref_mic_idx)
    if not (0 <= ref_mic < M):
        raise ValueError(f"ref_mic_idx={ref_mic_idx} fuera de rango para M={M}.")

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
                                         save_weights=False, alpha=0.99, ref_mic_idx=None):
    K, T, M = X_stft.shape

    Y_stft = np.zeros((K, T), dtype=np.complex128)

    # Accumulators for matrix estimation
    Num_XX = np.zeros((K, M, M), dtype=np.complex128)
    Num_NN = np.zeros((K, M, M), dtype=np.complex128)

    Den_XX = np.zeros((K, 1, 1), dtype=np.float64)
    Den_NN = np.zeros((K, 1, 1), dtype=np.float64)

    # Reference microphone index (ref_mic_idx=None -> M//2, unificado con toda la
    # familia). Ver la nota del core BAN: el benchmark inyecta el canal explicito.
    ref_mic = M // 2 if ref_mic_idx is None else int(ref_mic_idx)
    if not (0 <= ref_mic < M):
        raise ValueError(f"ref_mic_idx={ref_mic_idx} fuera de rango para M={M}.")

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
                                     save_weights=False, alpha=0.99, rank1=False,
                                     ref_mic_idx=None):
    """
    Variante FIXED de MVDR_Souden_recursive_mask. Misma formulacion de Souden
    (w = Phi_NN^-1 Phi_XX u / tr(Phi_NN^-1 Phi_XX)), ref_mic = ref_mic_idx
    (None -> M//2, unificado con toda la familia), factor de olvido alpha; corrige:

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
    ref_mic = M // 2 if ref_mic_idx is None else int(ref_mic_idx)
    if not (0 <= ref_mic < M):
        raise ValueError(f"ref_mic_idx={ref_mic_idx} fuera de rango para M={M}.")
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


def MVDR_Souden_recursive_mask_subtract(X_stft, mask_s, mask_n, min_loading=1e-9,
                                        save_weights=False, alpha=0.99, mu=0.0,
                                        lambda_floor=1e-3, psd_project=True,
                                        rank1=False, ref_mic_idx=None,
                                        gate_thresh=None, gate_sharp=2.0,
                                        gate_kmax=None, save_gate=False):
    """
    Souden MVDR con SUSTRACCION DE COVARIANZA: Phi_SS = Phi_XX - Phi_NN.

    EL PROBLEMA QUE CORRIGE
    -----------------------
    La ec. 24 de Souden usa la covarianza del TARGET (Phi_x en el paper). Todos
    los cores mask-based de este archivo le pasan en su lugar la covarianza de la
    MEZCLA enmascarada, que es

        Phi_XX  ~=  Phi_SS + Phi_NN

    Con eso, el denominador de la normalizacion vale

        lambda = tr(Phi_NN^-1 Phi_XX) = tr(Phi_NN^-1 Phi_SS) + M = lambda_S + M

    o sea que lambda esta ACOTADO POR ABAJO POR M. En un bin donde la mascara no
    encuentra voz (lambda_S -> 0) queda Phi_XX ~= Phi_NN, y entonces

        Phi_NN^-1 Phi_XX -> I ,   lambda -> M ,   w -> u / M

    el filtro DEGENERA al microfono de referencia dividido por M: la banda sale
    atenuada 1/M^2 (M=8 -> -18 dB) con ganancia de arreglo NULA. Medido sobre
    MIRD (ver tests/lowfreq_diagnostic_run.py) eso es exactamente lo que pasa
    debajo de ~130 Hz: Phi_XX es 91-100% ruido, lambda/M = 0.96-1.16, la
    respuesta al target cae a -16 dB y AG ~= +1 dB. PESQ no lo ve porque P.862
    no evalua debajo de ~300 Hz.

    Visto de otra forma: dividir por lambda_S + M es un PMWF-beta con beta = M.
    Los cores actuales son, sin quererlo, un Wiener parametrico con una
    supresion que CRECE con el numero de microfonos.

    LA CORRECCION
    -------------
    Se estima el target restando, y se normaliza con la traza que le corresponde:

        Phi_SS_hat = Phi_XX - Phi_NN          (+ proyeccion PSD)
        lambda_S   = tr(Phi_NN^-1 Phi_SS_hat)  == lambda - M  (exacto)
        w          = Phi_NN^-1 Phi_SS_hat u / (lambda_S + mu)

    Se corrige el NUMERADOR ademas del denominador: Phi_NN^-1 Phi_XX u contiene
    un termino de paso directo +u que la sustraccion elimina. Por eso mu = M NO
    reproduce exactamente el core actual (mismo denominador, distinto numerador).

    PARAMETROS
    ----------
    mu : trade-off PMWF sobre el denominador (lambda_S + mu).
        mu = 0   -> MVDR distortionless puro. Es la correccion en su forma pura:
                   quita el colapso de escala por completo. OJO: en los bins donde
                   Phi_XX ~= Phi_NN la resta es una diferencia de dos matrices casi
                   iguales -> su direccion es ruido, y un filtro distortionless
                   apuntado a una direccion aleatoria AMPLIFICA. Esperar mejora en
                   respuesta al target y posible perdida en reduccion de ruido.
        mu = 1   -> MWF estandar.
        mu = M   -> mismo denominador que el core actual (numerador distinto).
        El optimo real esta casi seguro entre 0 y 1: barrerlo.
    lambda_floor : piso numerico sobre lambda_S (relativo, adimensional). Solo
        evita la division por cero cuando mu = 0; con mu > 0 es inocuo.
    psd_project : proyecta Phi_SS_hat a PSD (autovalores negativos -> 0) antes de
        usarla. La resta de dos SCM estimadas sobre CONJUNTOS DE FRAMES DISTINTOS
        (mascara de voz vs de ruido) no tiene por que dar PSD. Ademas garantiza
        lambda_S >= 0 (traza de producto de dos PSD). Cuesta un eigh por bin y
        frame; se puede apagar por velocidad.
    rank1 : ademas de proyectar, se queda solo con el autovector principal de
        Phi_SS_hat (target de rango 1). Implica psd_project.

    GATE POR CONFIANZA (lambda_S/M)
    -------------------------------
    gate_thresh activa un blend SUAVE hacia el PASSTHROUGH del microfono de
    referencia en los bins donde el filtro no tiene informacion espacial util:

        r = lambda_S / M                       (confianza, gratis: ya se calcula)
        g = r^p / (r^p + gate_thresh^p)        (Hill, p = gate_sharp)
        w = g * w_mvdr + (1 - g) * u

    POR QUE lambda Y NO LA FRECUENCIA. Medido en MIRD (tests/lowfreq_diagnostic_run.py),
    lambda_S/M separa limpio: mediana 0.13 debajo de 130 Hz (donde AG = 2.4 dB y
    TR = -15 dB: se paga 15 dB de voz por 2 dB de SNR) contra 0.63 en 130-200 Hz
    (donde AG = 12.9 dB: NO hay que tocar nada) y 4-9 mas arriba. Un gate por
    frecuencia fija en f_c = c/2L (660 Hz para L=26 cm) tiraria los ~13 dB de
    ganancia REAL que hay entre 130 y 300 Hz. lambda_S/M es por bin, por frame,
    ciego y se adapta a la escena; la frecuencia no.

    El blend hacia `u` (NO hacia u/M) solo es coherente porque este core es
    distortionless: con mu=0, w_mvdr cumple w^H a ~= 1, igual que u, asi que los
    dos terminos estan en la MISMA normalizacion y la mezcla no introduce un
    salto de escala. Con el core estandar (que degenera a u/M) este blend no
    tendria sentido.

    gate_kmax : ultimo indice de bin donde el gate esta activo (None = todos).
        El gate por lambda es agnostico a la frecuencia por diseno; este tope es
        una red de seguridad opcional para acotarlo a graves si en agudos se
        observara perdida de supresion durante pausas.
    save_gate : devuelve tambien g (K, T) para inspeccionar cuando se activo.

    Base numerica identica a _fixed: Hermitiana forzada, carga diagonal RELATIVA
    escala-invariante y np.linalg.solve. min_loading default 1e-9 para quedar en
    el mismo regimen de carga que NM_MVDR (el core base ganador), no en el 1e-2
    de _fixed.
    """
    K, T, M = X_stft.shape
    Y_stft = np.zeros((K, T), dtype=np.complex128)
    Num_XX = np.zeros((K, M, M), dtype=np.complex128)
    Num_NN = np.zeros((K, M, M), dtype=np.complex128)
    Den_XX = np.zeros((K, 1, 1), dtype=np.float64)
    Den_NN = np.zeros((K, 1, 1), dtype=np.float64)
    ref_mic = M // 2 if ref_mic_idx is None else int(ref_mic_idx)
    if not (0 <= ref_mic < M):
        raise ValueError(f"ref_mic_idx={ref_mic_idx} fuera de rango para M={M}.")
    eye = np.eye(M)[np.newaxis, :, :]

    # alpha puede ser ESCALAR o un array POR BIN (K,) -> factor de olvido
    # dependiente de la frecuencia. Se lleva a (K,1,1), que difunde correctamente
    # contra Num (K,M,M) y contra Den (K,1,1) sin tocar el resto del bucle.
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.ndim == 0:
        alpha = np.full((K,), float(alpha))
    if alpha.shape != (K,):
        raise ValueError(f"alpha debe ser escalar o de shape ({K},); es {alpha.shape}.")
    alpha = alpha[:, np.newaxis, np.newaxis]

    if save_weights:
        weights_rec = np.zeros((K, T, M), dtype=np.complex128)
    if save_gate:
        gate_rec = np.zeros((K, T), dtype=np.float64)

    # One-hot del microfono de referencia = destino del gate (PASSTHROUGH puro,
    # NO u/M: ese factor 1/M es justamente el bug que este core corrige).
    u_ref = np.zeros(M, dtype=np.complex128)
    u_ref[ref_mic] = 1.0
    k_hi = K if gate_kmax is None else int(gate_kmax) + 1

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

        # --- LA CORRECCION: estimar el target restando el ruido (marca) -------
        Phi_SS = Phi_XX - Phi_NN
        Phi_SS = 0.5 * (Phi_SS + np.conj(np.transpose(Phi_SS, (0, 2, 1))))

        if psd_project or rank1:
            # eigh sobre matrices Hermitianas -> autovalores ascendentes, reales.
            evals, evecs = np.linalg.eigh(Phi_SS)
            if rank1:
                # solo el autovector principal (target de rango 1)
                lam_max = np.maximum(evals[:, -1], 0.0)
                v = evecs[:, :, -1]
                Phi_SS = lam_max[:, None, None] * np.einsum("fm,fn->fmn", v, v.conj())
            else:
                # proyeccion PSD: autovalores negativos (fruto de la resta) -> 0
                evals = np.maximum(evals, 0.0)
                Phi_SS = np.einsum("fmp,fnp->fmn", evecs * evals[:, np.newaxis, :],
                                   evecs.conj())

        # carga diagonal RELATIVA, escala-invariante (igual que _fixed)
        tr_Phi = np.real(np.trace(Phi_NN, axis1=1, axis2=2))
        adaptive_load = min_loading * (tr_Phi / M)
        Phi_NN_stable = Phi_NN + eye * (adaptive_load[:, np.newaxis, np.newaxis] + 1e-12)

        # w = Phi_NN^-1 Phi_SS u / (lambda_S + mu),  lambda_S = tr(.) == lambda - M
        Phi_inv_Phi_S = np.linalg.solve(Phi_NN_stable, Phi_SS)      # (K,M,M)
        lambda_S = np.real(np.trace(Phi_inv_Phi_S, axis1=1, axis2=2))
        lambda_S = np.maximum(lambda_S, lambda_floor)
        weights_unnorm = Phi_inv_Phi_S[:, :, ref_mic]
        weights = weights_unnorm / (lambda_S[:, np.newaxis] + mu + 1e-15)

        # --- GATE POR CONFIANZA: blend suave hacia el passthrough ------------
        # g -> 1 donde el filtro tiene informacion espacial util; g -> 0 donde
        # lambda_S/M dice que no la tiene (y ahi w = u: pasa el mic de
        # referencia intacto, y la supresion queda para el post-filtro).
        if gate_thresh is not None:
            r = lambda_S / M
            g = np.ones(K, dtype=np.float64)
            rp = np.power(r[:k_hi], gate_sharp)
            g[:k_hi] = rp / (rp + gate_thresh ** gate_sharp + 1e-30)
            weights = (g[:, np.newaxis] * weights
                       + (1.0 - g[:, np.newaxis]) * u_ref[np.newaxis, :])
            if save_gate:
                gate_rec[:, m] = g
        elif save_gate:
            gate_rec[:, m] = 1.0

        if save_weights:
            weights_rec[:, m, :] = weights
        Y_stft[:, m] = np.einsum("fm,fm->f", weights.conj(), X_frame)

    out = (Y_stft,)
    if save_weights:
        out += (weights_rec,)
    if save_gate:
        out += (gate_rec,)
    return out if len(out) > 1 else Y_stft


def MVDR_Souden_recursive_mask_MWF(X_stft, mask_s, mask_n, min_loading=1e-2,
                                   save_weights=False, alpha=0.99, mu=1.0,
                                   rank1=False, ref_mic_idx=None):
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
    ref_mic (ref_mic_idx, None -> M//2) y (opcional) rank-1 de Phi_XX quedan
    idénticos a _fixed.
    """
    K, T, M = X_stft.shape
    Y_stft = np.zeros((K, T), dtype=np.complex128)
    Num_XX = np.zeros((K, M, M), dtype=np.complex128)
    Num_NN = np.zeros((K, M, M), dtype=np.complex128)
    Den_XX = np.zeros((K, 1, 1), dtype=np.float64)
    Den_NN = np.zeros((K, 1, 1), dtype=np.float64)
    ref_mic = M // 2 if ref_mic_idx is None else int(ref_mic_idx)
    if not (0 <= ref_mic < M):
        raise ValueError(f"ref_mic_idx={ref_mic_idx} fuera de rango para M={M}.")
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
                                       rank1=False, ref_mic_idx=None):
    """
    Souden FIXED + BAN + post-filtro de Wiener (MWF paramétrico, mu). Base = _fixed
    (loading relativo + Hermitiana + solve + ref_mic_idx, None -> M//2), denominador
    MWF (lambda+mu), y ENCIMA la Blind Analytical Normalization (BAN), con el MISMO
    ref_mic que el resto de la familia y Phi_NN_stable coherente (a diferencia del
    BAN viejo, que usaba ref_mic=0 y loading absoluto).

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
    ref_mic = M // 2 if ref_mic_idx is None else int(ref_mic_idx)
    if not (0 <= ref_mic < M):
        raise ValueError(f"ref_mic_idx={ref_mic_idx} fuera de rango para M={M}.")
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
                                       save_weights=False, ref_mic_idx=None):
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
                                           save_weights=save_weights,
                                           ref_mic_idx=ref_mic_idx)
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
                                            save_weights=False, ref_mic_idx=None):
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
                                     save_weights=save_weights, ref_mic_idx=ref_mic_idx)
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
                                                save_weights=False, ref_mic_idx=None):
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
                                               save_weights=save_weights, ref_mic_idx=ref_mic_idx)
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
                                           save_weights=False, ref_mic_idx=None):
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
                                             mu=0.0, save_weights=save_weights,
                                             ref_mic_idx=ref_mic_idx)
    Y, W = (res if save_weights else (res, None))
    Y = Y.copy()

    Tm = min(Y.shape[1], mask_s_soft.shape[1])
    G = smooth + (1.0 - smooth) * np.clip(mask_s_soft[:, :Tm], 0.0, 1.0)
    Y[:, :Tm] *= G

    if save_weights:
        return Y, W
    return Y
