"""
ds_mask.py
==========
FRONT-END ESPACIAL FIJO PARA LA ESTIMACION DE LA MASCARA.

LA IDEA
-------
Hoy la mascara sale de correr el DTLN sobre UN canal crudo:

    m_raw(k,t) = DTLN( x_ref(n) )

o sea que el estimador de mascara ve la escena con la SNR de un microfono
solo, mientras que el beamformer que consume esa mascara tiene M. Este modulo
mete un BEAMFORMER FIJO (delay-and-sum, o superdirectivo) ANTES del DTLN:

    m_raw(k,t) = DTLN( y_fix(n) ) ,   y_fix = w_fix^H x

`w_fix` no tiene estadistica que estimar -- sale de la GEOMETRIA y del DOA (que
en este sistema lo aporta la vision) -- asi que no puede realimentar errores de
la propia mascara. El DTLN recibe una senal con hasta 10 log10(M) dB mas de SNR
(M=8 -> 9 dB) y la mascara deberia ser mas precisa justo en las celdas de SNR
baja, que son las que mas ensucian Phi_NN (la matriz que se INVIERTE).

¿HACE FALTA UNA PROYECCION HACIA ATRAS? (la pregunta que motiva el modulo)
-------------------------------------------------------------------------
Hay que separar DOS cosas que suelen confundirse:

1. DOMINIO / ESCALA POR CANAL -- NO hace falta, y ademas seria un NO-OP.
   En los cores de `souden_mvdr.py` la mascara NO filtra ninguna senal: entra
   como PESO ESCALAR de un promedio ponderado de los outer products de la senal
   MULTICANAL observada,

       Phi_XX(k,t) = sum_tau a^(t-tau) m_s(k,tau) x x^H / sum_tau a^(t-tau) m_s(k,tau)

   Numerador y denominador llevan la MISMA mascara, asi que cualquier factor que
   dependa SOLO del bin, m_s -> c(k) m_s, se CANCELA EXACTAMENTE: Phi_XX y Phi_NN
   quedan bit a bit iguales. Una "compensacion de la ganancia del DS" entendida
   como un factor por bin no cambia absolutamente nada. Tampoco hace falta una
   prediccion POR CANAL: la mascara ya es UN escalar por (k,t) para los M canales
   -- eso no lo introduce el DS, ya estaba.

2. PUNTO DE OPERACION -- SI importa, y es lo unico que hay que corregir.
   Lo que si cambia al mirar la salida del DS es el SESGO de la mascara: el DTLN
   ve una SNR mejorada en AG(k) dB, asi que devuelve m_s sistematicamente mas
   optimista, y por lo tanto m_n = 1 - m_s mas chica. Si m_n se achica, Phi_NN se
   estima con menos frames efectivos -- y esa es la unica via por la que esta
   propuesta puede EMPEORAR las cosas. La correccion tiene que ser NO LINEAL en
   m (una monotona que cambie el peso RELATIVO entre frames del mismo bin); la
   natural es en el dominio de la SNR:

       logit(m_ref) = logit(m_ds) - beta * ln AG(k)

   con AG(k) la ganancia de arreglo del filtro fijo contra un campo difuso, que
   sale de la geometria (`array_gain`). beta = 0.5 si se lee la mascara como
   razon de AMPLITUDES (|S|/(|S|+|N|), que es como la usa el DTLN), beta = 1 si
   se la lee como ganancia de Wiener en POTENCIA; se barre.

   OJO CON LA EQUIVALENCIA: esto es EXACTAMENTE `scm_calibration.warp_mask` con
   a = 1 y b_k = -beta ln AG(k). O sea que la proyeccion hacia atras NO es una
   familia nueva: es un punto de la familia que ya ajusta el banco de
   calibracion, con la diferencia de que aca b_k lo fija la FISICA en vez de un
   ajuste. Sirve como inicializador con sentido y como ablacion.

RIESGOS A MEDIR (no son hipoteticos)
------------------------------------
  - Sesgo por banda: AG(k) ~ 0 dB en graves (kd << 1: todos los micros ven lo
    mismo) y ~10 log10 M en agudos, asi que el corrimiento del punto de
    operacion es DEPENDIENTE DE LA FRECUENCIA. Por eso b_k es por bin.
  - Campo de ruido distinto: a la salida del DS el ruido queda parcialmente
    decorrelacionado / peinado. Es un corrimiento de dominio para un DTLN
    entrenado en monocanal; leve, pero real.
  - Reverberacion: el DS alinea SOLO el camino directo, asi que la cola
    reverberante del target se atenua. La mascara pasa a indicar dominancia del
    DIRECTO mas que del target completo -- puede ayudar (Phi_SS mas rango 1) o
    perjudicar; hay que medirlo a rt60 alto.
  - Error de DOA: el filtro fijo depende del apuntamiento. El DS es tolerante
    (lobulo ancho), el superdirectivo NO tanto.

MODO "sd" (superdirectivo)
--------------------------
Si el DS ayuda, la pregunta inmediata es si ayuda mas un filtro fijo MEJOR. El
superdirectivo con carga es el mismo codigo:

    w = G^-1 d / (d^H G^-1 d) ,   G = (1 - eps) Gamma(k) + eps I

Gamma(k) es la coherencia difusa de `scm_calibration.diffuse_coherence` (pura
geometria) y eps controla el WNG (eps -> 1 devuelve el DS exacto). Sigue sin
estimar nada de la senal, asi que conserva la propiedad que hace segura a toda
la idea: no puede realimentar errores de la mascara.
"""

import numpy as np
import scipy.signal as sig

from beamforming.signal_model import compute_rtf_steering_vector
from beamforming.mask.scm_calibration import diffuse_coherence, warp_mask


# =====================================================================
# Filtro fijo: pesos, ganancia de arreglo y senal de referencia
# =====================================================================
def fixed_bf_weights(mic_coords, freqs, source_pos, ref_mic_idx=0, mode="ds",
                     loading=1e-2, field="spherical", Gamma=None):
    """
    Pesos de un beamformer FIJO apuntado al target, normalizado DISTORTIONLESS
    respecto del microfono de referencia:  w^H d = 1  con d_ref = 1.

    Esa normalizacion es la que deja la salida del filtro EN EL MISMO DOMINIO que
    el canal de referencia (la voz sale tal como llega a ese microfono), que es
    justo lo que hace comparable la mascara nueva con la actual.

    Args:
        mic_coords: (M, 3) posiciones de los microfonos [m].
        freqs: (K,) frecuencias de los bins [Hz].
        source_pos: (1, 3) posicion del target (del DOA de vision o del oraculo).
        ref_mic_idx: canal de referencia (el mismo que usa el core de Souden).
        mode: "ds" (delay-and-sum) | "sd" (superdirectivo con carga).
        loading: eps de la carga del modo "sd", relativo (eps -> 1 == DS).
        field: modelo de campo difuso para "sd" ("spherical" | "cylindrical").
        Gamma: (K, M, M) coherencia difusa ya calculada (opcional, evita
            recalcularla si el llamador ya la tiene).

    Returns:
        (K, M) complejo.
    """
    P = np.asarray(mic_coords, dtype=np.float64)
    M = P.shape[0]
    f = np.asarray(freqs, dtype=np.float64)
    K = f.size

    d = compute_rtf_steering_vector(f, np.asarray(source_pos).reshape(1, 3), P,
                                    ref_mic_idx=int(ref_mic_idx),
                                    mode="near_field", squeeze=True)
    d = np.asarray(d).reshape(K, M)

    if mode == "ds":
        # G = I  ->  w = d / (d^H d). Con d unimodular esto es d / M (el DS de
        # siempre); en campo cercano d^H d != M y esta forma mantiene w^H d = 1.
        den = np.real(np.einsum("km,km->k", d.conj(), d))
        return d / (den[:, None] + 1e-30)

    if mode != "sd":
        raise ValueError(f"mode desconocido: {mode!r} (usar 'ds' o 'sd')")

    if Gamma is None:
        Gamma = diffuse_coherence(P, f, field=field)
    eps = float(np.clip(loading, 0.0, 1.0))
    G = (1.0 - eps) * Gamma.astype(np.complex128) + eps * np.eye(M)[None, :, :]
    Gi_d = np.linalg.solve(G, d[:, :, None])[:, :, 0]                 # (K, M)
    den = np.real(np.einsum("km,km->k", d.conj(), Gi_d))
    return Gi_d / (den[:, None] + 1e-30)


def array_gain(w, mic_coords=None, freqs=None, Gamma=None, field="spherical"):
    """
    Ganancia de arreglo del filtro fijo contra un campo de ruido DIFUSO y contra
    ruido espacialmente BLANCO, por bin. Con w^H d = 1 (ver `fixed_bf_weights`):

        AG_dif(k) = 1 / (w^H Gamma(k) w)        WNG(k) = 1 / (w^H w)

    AG_dif es el factor por el que MEJORA la SNR de entrada del DTLN, y es lo que
    hay que descontar en la proyeccion hacia atras. Tiende a 1 (0 dB) en graves
    -- donde el arreglo no puede hacer nada contra un campo difuso -- y sube
    hasta ~M en agudos. WNG es el control de sanidad del modo "sd": si se dispara
    por encima de M el filtro se esta apoyando en un margen que el hardware real
    (ruido propio de los MEMS) no tiene.

    Returns:
        (AG_dif, WNG), ambos (K,) reales y positivos.
    """
    w = np.asarray(w)
    K, M = w.shape
    if Gamma is None:
        if mic_coords is None or freqs is None:
            raise ValueError("hace falta Gamma, o (mic_coords, freqs) para calcularla")
        Gamma = diffuse_coherence(mic_coords, freqs, field=field)
    q_dif = np.real(np.einsum("km,kmn,kn->k", w.conj(), Gamma.astype(np.complex128), w))
    q_wht = np.real(np.einsum("km,km->k", w.conj(), w))
    return 1.0 / (np.maximum(q_dif, 1e-30)), 1.0 / (np.maximum(q_wht, 1e-30))


def fixed_bf_signal(mic_signals, mic_coords, source_pos, fs, ref_mic_idx=0,
                    nperseg=512, noverlap=384, window="hamming", mode="ds",
                    loading=1e-2, field="spherical"):
    """
    Aplica el filtro fijo y devuelve la senal MONO en el dominio del tiempo,
    alineada muestra a muestra con `mic_signals` (misma STFT/iSTFT que usa el
    wrapper DS del benchmark), lista para entrar al DTLN.

    Returns:
        (y_time (N,), w (K, M), freqs (K,)).
    """
    freqs, _, Z = sig.stft(mic_signals, fs=fs, window=window, nperseg=nperseg,
                           noverlap=noverlap, nfft=nperseg)
    X = np.transpose(Z, (1, 2, 0))                                   # (K, T, M)
    w = fixed_bf_weights(mic_coords, freqs, source_pos, ref_mic_idx=ref_mic_idx,
                         mode=mode, loading=loading, field=field)
    Y = np.einsum("km,ktm->kt", w.conj(), X)
    _, y = sig.istft(Y, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap,
                     nfft=nperseg)
    return y[:mic_signals.shape[1]], w, freqs


# =====================================================================
# Proyeccion hacia atras y post-proceso de la mascara
# =====================================================================
def backproject_mask(m, ag, beta=0.5, ag_clip=None):
    """
    Lleva una mascara estimada sobre la salida del filtro fijo al PUNTO DE
    OPERACION del microfono de referencia, descontando la ganancia de arreglo:

        logit(m_ref) = logit(m_fix) - beta * ln AG(k)

    Es `warp_mask(m, a=1, b=-beta ln AG)`: la misma familia que ajusta el banco
    de calibracion, pero con b_k fijado por la GEOMETRIA en vez de ajustado.

    NO es una simple division: un factor por bin sobre la mascara se cancela
    exacto en la recursion (numerador y denominador llevan la misma mascara).
    Lo que corrige esta transformacion es el peso RELATIVO entre frames del
    mismo bin, que es lo unico que el estimador de SCM realmente ve.

    Args:
        m: (K, T) mascara cruda del DTLN sobre la salida del filtro fijo.
        ag: (K,) ganancia de arreglo LINEAL (de `array_gain`).
        beta: 0 -> sin correccion; 0.5 -> lectura en amplitud (|S|/(|S|+|N|),
            que es como el DTLN usa su mascara); 1.0 -> lectura en potencia
            (ganancia de Wiener). Barrerlo resuelve empiricamente la ambiguedad.
        ag_clip: (lo, hi) recorte de AG antes del log. Default (1, M) via None
            no se puede inferir aca, asi que None = sin recorte; el llamador
            deberia recortar a [1, M], que es el rango fisico del filtro fijo.

    Returns:
        (K, T) mascara en el punto de operacion del canal de referencia.
    """
    if beta == 0.0:
        return np.asarray(m, dtype=np.float64)
    ag = np.asarray(ag, dtype=np.float64)
    if ag_clip is not None:
        ag = np.clip(ag, ag_clip[0], ag_clip[1])
    b = -float(beta) * np.log(np.maximum(ag, 1e-12))
    return warp_mask(m, a=1.0, b=b)


def stretch_sharpen(m_raw, sharpen_exp=4.0):
    """
    El post-proceso ACTUAL de la mascara, aislado para poder reusarlo tal cual
    (stretch min-max GLOBAL -> complemento -> potencia). Reproduce EXACTAMENTE
    lo que hace `dtln_masks.get_dtln_masks_sharpen` despues del DTLN, para que
    la unica diferencia entre el camino actual y el nuevo sea el front-end.

    (Sus problemas -- no causal, global en (k,t) -- estan documentados en
    `scm_calibration`; aca se conserva a proposito, para comparar contra el
    sistema en produccion y no contra una variante nueva.)
    """
    m = np.asarray(m_raw, dtype=np.float64)
    m = (m - np.min(m)) / (np.max(m) - np.min(m) + 1e-12)
    return m ** sharpen_exp, (1.0 - m) ** sharpen_exp


def get_fixed_bf_dtln_masks(mic_signals, mic_coords, source_pos, fs, model_path,
                            ref_mic_idx=0, block_len=512, block_shift=128,
                            sharpen_exp=4.0, mode="ds", loading=1e-2,
                            field="spherical", beta=0.0, return_raw=False):
    """
    Camino completo: filtro fijo -> DTLN -> (proyeccion hacia atras) -> stretch +
    sharpen. Drop-in de `dtln_masks.get_dtln_masks_sharpen`: devuelve
    (mask_s, mask_n) de shape (K, T).

    beta = 0 y mode = "ds" es la propuesta en su forma pura (mascara sobre la
    salida del DS, sin tocar nada mas). beta > 0 agrega la correccion de punto de
    operacion. Con return_raw=True devuelve ademas la mascara cruda del DTLN y la
    ganancia de arreglo, que es lo que necesita el banco de diagnostico.
    """
    # Import diferido: `dtln_masks` levanta ai_edge_litert al importarse y este
    # modulo tiene que poder usarse (pesos, AG, warp) sin el runtime del TFLite.
    from beamforming.mask.dtln_masks import get_dtln_masks_soft

    y_fix, w, freqs = fixed_bf_signal(
        mic_signals, mic_coords, source_pos, fs, ref_mic_idx=ref_mic_idx,
        nperseg=block_len, noverlap=block_len - block_shift, mode=mode,
        loading=loading, field=field)

    M = mic_signals.shape[0]
    ag, _ = array_gain(w, mic_coords, freqs, field=field)
    ag = np.clip(ag, 1.0, float(M))

    # get_dtln_masks_soft espera (M, N) y un indice de canal; se le pasa la senal
    # mono del filtro fijo como un "arreglo" de un solo canal.
    m_raw, _ = get_dtln_masks_soft(y_fix[None, :], 0, model_path,
                                   block_len=block_len, block_shift=block_shift)
    m_bp = backproject_mask(m_raw, ag, beta=beta)
    mask_s, mask_n = stretch_sharpen(m_bp, sharpen_exp=sharpen_exp)

    if return_raw:
        return mask_s, mask_n, m_raw, ag, freqs
    return mask_s, mask_n


# =====================================================================
# FRONT-END CIEGO: la RTF sale de la PROPIA SCM de senal estimada
# =====================================================================
# EL PROBLEMA QUE RESUELVE
# -----------------------
# El front-end fijo de arriba (`fixed_bf_weights`) apunta con la GEOMETRIA + el
# DOA. Eso es lo que lo hace seguro (no puede realimentar errores de la mascara)
# pero tambien lo que rompe la propiedad de que el estimador de mascara sea
# CIEGO: sin la posicion de la fuente no hay w_fix.
#
# La salida es REALIMENTAR: la cadena mask-based ya estima, por bin y por frame,
#
#     Phi_SS = Phi_XX - Phi_NN        (la misma sustraccion del core `subtract`)
#
# y de ahi sale la RTF sin mirar ni la geometria ni el DOA:
#
#     d = Phi_SS[:, ref] / Phi_SS[ref, ref]              (covariance subtraction)
#     d = v / v_ref,  v = autovector principal            (EVD / prewhitened EVD)
#
# Con esa d se arma el MISMO front-end distortionless de antes (w^H d = 1, con
# d_ref = 1: la voz sale como llega al microfono de referencia), se lo aplica y
# el DTLN vuelve a estimar la mascara sobre esa senal. El lazo es
#
#     mascara(1) -> Phi_SS -> RTF -> BF -> mascara(2) -> beamformer final
#
# ESTABILIDAD: CARGA DIAGONAL + SUAVIZADO TEMPORAL
# ------------------------------------------------
# Realimentar es exactamente lo que el front-end geometrico evitaba, asi que la
# matriz que se usa para ESTIMAR (no la que se invierte en el beamformer) lleva
# sus propios seguros, con variables propias:
#
#   rtf_loading (eps)  carga diagonal RELATIVA AL NIVEL DE RUIDO:
#
#                          Psi = R_est + eps * (tr(Phi_NN)/M) * I
#                          d   = Psi[:, ref] / Psi[ref, ref]
#
#       No es solo un seguro numerico: define el PUNTO DE FALLA. Donde la senal
#       estimada es debil frente al ruido (R_est -> 0) queda Psi -> eps c I y
#       entonces d -> e_ref, o sea w -> u_ref: el front-end se degrada CONTINUA
#       y automaticamente al canal de referencia crudo, que es el sistema
#       actual. Un bin/frame sin informacion no puede inventar un apuntamiento:
#       en el peor caso el DTLN ve lo que ve hoy. eps grande = mas conservador.
#
#   rtf_alpha          factor de olvido de la recursion de Phi_XX / Phi_NN que
#       alimenta la ESTIMACION, independiente del alpha del beamformer. La RTF es
#       una propiedad del CUARTO (posicion de la fuente + reflexiones), no de la
#       actividad de voz: cambia mucho mas lento que las estadisticas que el
#       beamformer necesita para seguir al ruido. Promediar mas tiempo (0.999
#       contra 0.99) baja la varianza del estimador -- que es la fuente real de
#       inestabilidad del lazo -- casi sin costo de tracking.
#
# MODOS DE ESTIMACION
# -------------------
#   "cs"  covariance subtraction: la columna ref de Phi_SS, tal cual. El mas
#         barato y el mas transparente; sesgado si Phi_SS no es de rango 1
#         (reverberacion, dos hablantes).
#   "evd" autovector principal de Phi_SS (target de rango 1). Filtra el error de
#         la resta que no cae en la direccion dominante.
#   "cw"  covariance whitening: autovector principal de L^-1 Phi_SS L^-H con
#         L = chol(Phi_NN), des-blanqueado. Es el estimador de RTF con mejor
#         reputacion en la literatura porque el error se mide en la metrica del
#         RUIDO y no en la euclidea. Cuesta un Cholesky + un eigh por frame.
# En "evd" y "cw" la carga NO puede entrar antes del autovector (desplaza los
# autovalores pero no los autovectores: no haria nada), asi que se reconstruye
# R_est = lam * v v^H y se carga eso -- con lo cual el mecanismo de degradacion
# a e_ref se conserva identico en los tres modos.


def _rtf_from_loaded(R_est, load, ref_mic):
    """Columna de referencia de R_est + load*I, normalizada a d_ref = 1."""
    K, M = R_est.shape[0], R_est.shape[1]
    num = R_est[:, :, ref_mic].copy()
    num[:, ref_mic] += load
    den = np.real(R_est[:, ref_mic, ref_mic]) + load
    return num / (den[:, None] + 1e-30)


def estimate_rtf_recursive(X_stft, mask_s, mask_n, ref_mic_idx=None,
                           rtf_alpha=0.999, rtf_loading=1e-2, rtf_mode="cs",
                           w_mode="ds", bf_loading=1e-6):
    """
    RTF CIEGA por bin y por frame a partir de la SCM de senal estimada, y los
    pesos del front-end que se arma con ella.

    Recursion identica a la de los cores de Souden (mismo promedio ponderado por
    la mascara), pero con SU PROPIO factor de olvido y su propia carga: ver la
    nota de arriba.

    Args:
        X_stft: (K, T, M) STFT multicanal de la mezcla.
        mask_s, mask_n: (K, T) mascaras de la PRIMERA pasada (las que hoy salen
            del DTLN sobre el canal de referencia crudo).
        ref_mic_idx: canal que normaliza la RTF (d_ref = 1) y al que degrada el
            front-end cuando no hay senal. Default M // 2.
        rtf_alpha: factor de olvido de la recursion de ESTIMACION.
        rtf_loading: carga diagonal relativa al nivel de ruido, eps.
        rtf_mode: "cs" | "evd" | "cw".
        w_mode: "ds" -> w = d / (d^H d) (fijo salvo por d; sin estadistica de
            ruido, el analogo directo del DS geometrico).
            "mvdr" -> w = Phi_NN^-1 d / (d^H Phi_NN^-1 d): mejor SNR de entrada
            para el DTLN, pero ya realimenta tambien la SCM de ruido.
        bf_loading: carga relativa de Phi_NN para w_mode="mvdr" y para el
            blanqueo del modo "cw" (no toca la carga de la ESTIMACION).

    Returns:
        (W, D): ambos (K, T, M) complejos. W cumple W^H D = 1 frame a frame.
    """
    X_stft = np.asarray(X_stft)
    K, T, M = X_stft.shape
    ref_mic = M // 2 if ref_mic_idx is None else int(ref_mic_idx)
    if not (0 <= ref_mic < M):
        raise ValueError(f"ref_mic_idx={ref_mic_idx} fuera de rango para M={M}.")
    if rtf_mode not in ("cs", "evd", "cw"):
        raise ValueError(f"rtf_mode desconocido: {rtf_mode!r} ('cs'|'evd'|'cw')")
    if w_mode not in ("ds", "mvdr"):
        raise ValueError(f"w_mode desconocido: {w_mode!r} ('ds'|'mvdr')")

    a = np.asarray(rtf_alpha, dtype=np.float64)
    if a.ndim == 0:
        a = np.full((K,), float(a))
    if a.shape != (K,):
        raise ValueError(f"rtf_alpha debe ser escalar o de shape ({K},); es {a.shape}.")
    a = a[:, None, None]

    Num_XX = np.zeros((K, M, M), dtype=np.complex128)
    Num_NN = np.zeros((K, M, M), dtype=np.complex128)
    Den_XX = np.zeros((K, 1, 1), dtype=np.float64)
    Den_NN = np.zeros((K, 1, 1), dtype=np.float64)
    eye = np.eye(M)[None, :, :]

    D = np.zeros((K, T, M), dtype=np.complex128)
    W = np.zeros((K, T, M), dtype=np.complex128)

    for m in range(T):
        X_frame = X_stft[:, m, :]
        R_inst = np.einsum("fm,fn->fmn", X_frame, X_frame.conj())
        m_s = mask_s[:, m, None, None]
        m_n = mask_n[:, m, None, None]

        Num_XX = a * Num_XX + m_s * R_inst
        Den_XX = a * Den_XX + m_s
        Num_NN = a * Num_NN + m_n * R_inst
        Den_NN = a * Den_NN + m_n

        Phi_XX = Num_XX / (Den_XX + 1e-15)
        Phi_NN = Num_NN / (Den_NN + 1e-15)
        Phi_XX = 0.5 * (Phi_XX + np.conj(np.transpose(Phi_XX, (0, 2, 1))))
        Phi_NN = 0.5 * (Phi_NN + np.conj(np.transpose(Phi_NN, (0, 2, 1))))

        # SCM de senal estimada: la MISMA sustraccion del core `subtract`.
        Phi_SS = Phi_XX - Phi_NN
        Phi_SS = 0.5 * (Phi_SS + np.conj(np.transpose(Phi_SS, (0, 2, 1))))

        # Nivel de ruido por bin: fija la escala de la carga de la estimacion, y
        # con ella el umbral de confianza a partir del cual la RTF deja de ser
        # e_ref. Piso absoluto para los primeros frames (Phi_NN ~ 0).
        nlev = np.real(np.trace(Phi_NN, axis1=1, axis2=2)) / M
        load_rtf = rtf_loading * nlev + 1e-20

        if rtf_mode == "cs":
            # Proyeccion PSD: la resta de dos SCM estimadas sobre conjuntos de
            # frames distintos no tiene por que dar PSD, y un autovalor negativo
            # en la columna de referencia es error puro.
            evals, evecs = np.linalg.eigh(Phi_SS)
            evals = np.maximum(evals, 0.0)
            R_est = np.einsum("fmp,fnp->fmn", evecs * evals[:, None, :], evecs.conj())
        else:
            if rtf_mode == "evd":
                _, evecs = np.linalg.eigh(Phi_SS)
                v = evecs[:, :, -1]
            else:  # "cw": autovector principal en la metrica del ruido
                load_nn = bf_loading * nlev + 1e-20
                Phi_NN_l = Phi_NN + eye * load_nn[:, None, None]
                L = np.linalg.cholesky(Phi_NN_l)                      # (K,M,M)
                Li_S = np.linalg.solve(L, Phi_SS)
                # L^-1 Phi_SS L^-H  =  (L^-1 (L^-1 Phi_SS)^H)^H
                S_w = np.conj(np.transpose(
                    np.linalg.solve(L, np.conj(np.transpose(Li_S, (0, 2, 1)))),
                    (0, 2, 1)))
                S_w = 0.5 * (S_w + np.conj(np.transpose(S_w, (0, 2, 1))))
                _, evecs_w = np.linalg.eigh(S_w)
                v = np.einsum("fmn,fn->fm", L, evecs_w[:, :, -1])     # des-blanqueo
            v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-30)
            # Potencia de senal a lo largo de v: deja R_est en la MISMA escala
            # que Phi_SS, que es lo que hace comparable la carga entre modos.
            lam = np.maximum(np.real(np.einsum("fm,fmn,fn->f", v.conj(), Phi_SS, v)), 0.0)
            R_est = lam[:, None, None] * np.einsum("fm,fn->fmn", v, v.conj())

        d = _rtf_from_loaded(R_est, load_rtf, ref_mic)
        D[:, m, :] = d

        if w_mode == "ds":
            den = np.real(np.einsum("fm,fm->f", d.conj(), d))
            W[:, m, :] = d / (den[:, None] + 1e-30)
        else:
            load_nn = bf_loading * nlev + 1e-20
            Phi_NN_l = Phi_NN + eye * load_nn[:, None, None]
            Pi_d = np.linalg.solve(Phi_NN_l, d[:, :, None])[:, :, 0]
            den = np.real(np.einsum("fm,fm->f", d.conj(), Pi_d))
            W[:, m, :] = Pi_d / (den[:, None] + 1e-30)

    return W, D


def blind_bf_signal(mic_signals, mask_s, mask_n, fs, ref_mic_idx=None,
                    nperseg=512, noverlap=384, window="hamming",
                    rtf_alpha=0.999, rtf_loading=1e-2, rtf_mode="cs",
                    w_mode="ds", bf_loading=1e-6, return_stats=False):
    """
    Analogo CIEGO de `fixed_bf_signal`: en vez de apuntar con la geometria y el
    DOA, apunta con la RTF estimada de la propia SCM de senal (ver arriba).
    Devuelve la senal MONO en el tiempo, alineada con `mic_signals` y en el
    dominio del canal de referencia, lista para entrar al DTLN.

    Returns:
        y_time (N,)  -- o (y_time, W, D, freqs) con return_stats=True.
    """
    freqs, _, Z = sig.stft(mic_signals, fs=fs, window=window, nperseg=nperseg,
                           noverlap=noverlap, nfft=nperseg)
    X = np.transpose(Z, (1, 2, 0))                                   # (K, T, M)

    # El DTLN entrega ~3 frames MENOS que scipy.stft (arranca con el buffer
    # lleno). Se extiende la mascara repitiendo el ultimo frame -- mismo criterio
    # que `align_mask_frames` -- en vez de recortar la STFT: asi y_fix conserva la
    # longitud de la senal de entrada y la segunda pasada del DTLN ve exactamente
    # los mismos bloques que la primera.
    T = X.shape[1]
    def _fit_T(m):
        m = np.asarray(m, dtype=np.float64)
        if m.shape[1] >= T:
            return m[:, :T]
        return np.concatenate(
            [m, np.repeat(m[:, -1:], T - m.shape[1], axis=1)], axis=1)
    mask_s, mask_n = _fit_T(mask_s), _fit_T(mask_n)

    W, D = estimate_rtf_recursive(
        X, mask_s, mask_n, ref_mic_idx=ref_mic_idx,
        rtf_alpha=rtf_alpha, rtf_loading=rtf_loading, rtf_mode=rtf_mode,
        w_mode=w_mode, bf_loading=bf_loading)

    Y = np.einsum("ktm,ktm->kt", W.conj(), X)
    _, y = sig.istft(Y, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap,
                     nfft=nperseg)
    y = y[:mic_signals.shape[1]]
    if return_stats:
        return y, W, D, freqs
    return y
