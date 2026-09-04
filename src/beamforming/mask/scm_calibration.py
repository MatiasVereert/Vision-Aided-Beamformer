"""
scm_calibration.py
==================
BANCO DE CALIBRACION de la transformacion MASCARA -> COVARIANZA.

QUE PROBLEMA RESUELVE
---------------------
El camino mascara -> SCM -> pesos de Souden que usan todos los cores de
`souden_mvdr.py` esta hecho de decisiones que nunca se optimizaron contra nada:
el `stretch` min-max, el `** 4`, `m_n = 1 - m_s`, alpha = 0.99, la carga
diagonal. Con SCM oracle disponibles (senales limpias multicanal) se puede
plantear esa transformacion como una FAMILIA PARAMETRICA y ajustar los
parametros minimizando la perdida REAL del beamformer respecto del oracle.

Este modulo implementa la primera etapa de ese programa: los parametros que
actuan DESPUES de la acumulacion recursiva (`nu`, `gamma`, `mu`), que son los de
mejor relacion costo/beneficio y ademas los mas baratos de barrer (no hay que
rehacer la recursion por cada evaluacion del objetivo).

LA FAMILIA PARAMETRICA
----------------------
Partiendo de las SCM enmascaradas que ya calcula el sistema,

    Phi_XX(k,t) = <m_s . x x^H>_alpha      Phi_NN(k,t) = <m_n . x x^H>_alpha

se aplican, por BANDA de frecuencia:

  1. SHRINKAGE ESTRUCTURADO HACIA LA COHERENCIA DIFUSA (usa la GEOMETRIA):

        Phi_NN(gamma) = (1 - gamma_k) Phi_NN + gamma_k (tr Phi_NN / M) Gamma(k)

     donde Gamma(k) es la matriz de coherencia de un campo difuso isotropico,
     determinada SOLO por las distancias entre microfonos (ver
     `diffuse_coherence`). Es Ledoit-Wolf con un target FISICAMENTE CORRECTO en
     lugar de la identidad -- que es lo que hace la carga diagonal actual, y que
     implicitamente asume ruido espacialmente blanco (falso en una sala). En
     graves el modelo difuso es muy preciso, asi que ahi gamma -> 1 entrega una
     Phi_NN casi sin varianza de estimacion, justo en la banda donde el
     estimador esta peor condicionado.

  2. SUSTRACCION CON ESCALA CORREGIDA:

        Phi_SS(nu) = Phi_XX - nu_k Phi_NN(gamma)     (+ proyeccion PSD)

     El core `_subtract` usa nu = 1 implicito, pero Phi_XX y Phi_NN NO estan en
     la misma escala: Phi_XX es la covarianza de la mezcla CONDICIONADA A FRAMES
     DE VOZ (normalizada por Den_XX) y Phi_NN la condicionada A FRAMES DE RUIDO
     (normalizada por Den_NN). La resta con nu = 1 asume que el nivel de ruido
     durante la voz es igual al de las pausas. nu_k corrige ese sesgo.

  3. Normalizacion de Souden con el trade-off PMWF:

        w = Phi_NN^-1 Phi_SS u / (tr(Phi_NN^-1 Phi_SS) + mu)

LA PARAMETRIZACION CONTIENE A LOS CORES ACTUALES
------------------------------------------------
    (nu=0, gamma=0, mu=0)  ==  MVDR_Souden_recursive_mask_fixed   (base)
    (nu=1, gamma=0, mu=0)  ==  MVDR_Souden_recursive_mask_subtract (mu=0)

o sea que el ajuste no puede dar PEOR que el sistema actual salvo por error de
generalizacion train->test, y los dos cores existentes son puntos concretos del
mismo mapa. Esto lo verifica `tests/test_scm_calibration.py`.

LA FUNCION DE COSTO (y por que NO es Frobenius)
-----------------------------------------------
La formula de Souden es INVARIANTE A LA ESCALA GLOBAL de ambas covarianzas: si
Phi_NN -> c Phi_NN, numerador y traza se dividen por c; idem con Phi_XX. Una
loss ||Phi_hat - Phi_oracle||_F gastaria parametros ajustando una escala que el
beamformer no ve, y seria ciega a errores de FORMA que si importan.

Aca la loss se mide sobre lo que el beamformer PIERDE, en dB, usando las SCM
oracle Phi_S / Phi_N del MISMO (k,t):

    SINR(w)   = w^H Phi_S w / w^H Phi_N w
    SINR_max  = lambda_max(Phi_N^-1 Phi_S)          (cota alcanzable)
    L_sinr    = 10 log10(SINR_max / SINR(w))        >= 0, invariante a escala
    L_dist    = 10 log10(1 + |w^H a - 1|^2)         >= 0, a = RTF oracle
    L         = L_sinr + eta * L_dist

Las dos componentes se mapean sobre el benchmark: L_sinr <-> SIR / reduccion de
ruido, L_dist <-> PESQ / distorsion. Barrer eta traza la curva de trade-off, que
es mas contable que barrer mu a ciegas.

AGREGACION -- PONDERAR POR ENERGIA (leer antes de comparar numeros)
------------------------------------------------------------------
La loss vive por celda (bin, frame), pero PESQ/STOI/SDR/SIR agregan ENERGIA
sobre toda la senal. Agregar la loss por mediana SIMPLE le da el mismo peso a un
bin de 6 kHz practicamente vacio que a uno de 500 Hz que lleva casi toda la voz,
y ahi el proxy DEJA DE SEGUIR a las metricas: medido sobre MIRD, la ventaja del
ajuste sobre NM_MVDR_SUB pasa de 1.03 dB con `how="median"` a 0.05 dB con
`how="wmedian"` -- la mediana simple sobre-reportaba la ganancia ~20x.

Default recomendado: `how="wmedian"` (mediana ponderada por tr(Phi_S), la
potencia del target oracle). Mantiene la robustez a outliers de la mediana --
que es lo que motivo el criterio en el benchmark -- pero pesa cada celda por lo
que la metrica final realmente mide. `how="median"` queda disponible para
reproducir los numeros historicos. Ver `_aggregate`.

CUIDADO CON EL TECHO
--------------------
La mascara es UN ESCALAR por (k,t): es un cuello de botella de informacion.
Calibrar la transformacion recupera solo la parte del gap contra el oracle que
viene de la CALIBRACION, no la que viene de que la mascara este mal. Para
separarlas, `fit_bands` puede ajustar los parametros POR ESCENA (sobreajuste
deliberado, cota superior de la familia) y comparar contra el ajuste global: si
coinciden, la familia esta saturada y el gap restante es la mascara.
"""

import numpy as np

try:                                   # j0 solo hace falta para el campo cilindrico
    from scipy.special import j0 as _j0
except ImportError:                    # pragma: no cover
    _j0 = None

C_SOUND = 343.0


# =====================================================================
# Geometria: el target estructurado del shrinkage
# =====================================================================
def diffuse_coherence(mic_coords, freqs, c=C_SOUND, field="spherical"):
    """
    Matriz de coherencia de un campo de ruido difuso, derivada SOLO de la
    geometria del arreglo. Es el target del shrinkage de `parametric_scms`.

        spherical   : Gamma_ij(f) = sinc(2 f d_ij / c)        (isotropico 3D)
        cylindrical : Gamma_ij(f) = J0(2 pi f d_ij / c)       (isotropico en el
                      plano horizontal; mas realista para un arreglo LINEAL con
                      fuentes y reflexiones dominantes en el plano de la mesa)

    Args:
        mic_coords: (M, 3) o (M, 2), en metros. Invariante a traslacion.
        freqs: (K,) frecuencias de los bins [Hz].
        field: "spherical" | "cylindrical".

    Returns:
        (K, M, M) real, Hermitiana (aca simetrica real), con diagonal 1.
        En f -> 0 tiende a la matriz de unos (campo perfectamente coherente),
        que es exactamente el regimen donde el shrinkage tiene que actuar.
    """
    P = np.asarray(mic_coords, dtype=np.float64)
    if P.ndim != 2:
        raise ValueError(f"mic_coords debe ser (M, D); es {P.shape}.")
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)      # (M, M)
    f = np.asarray(freqs, dtype=np.float64)[:, None, None]          # (K, 1, 1)

    if field == "spherical":
        # np.sinc(x) = sin(pi x)/(pi x)  ->  sinc(2 pi f d / c) = np.sinc(2 f d / c)
        G = np.sinc(2.0 * f * d[None, :, :] / c)
    elif field == "cylindrical":
        if _j0 is None:
            raise ImportError("field='cylindrical' necesita scipy.special.j0")
        G = _j0(2.0 * np.pi * f * d[None, :, :] / c)
    else:
        raise ValueError(f"field desconocido: {field!r}")

    # La diagonal es exactamente 1 por construccion; se fuerza por round-off.
    idx = np.arange(P.shape[0])
    G[:, idx, idx] = 1.0
    return G


# =====================================================================
# Snapshots de SCM (estimadas y oracle) en frames de evaluacion
# =====================================================================
def eval_frame_indices(T, n_eval, start_frame=0):
    """
    Frames donde se congelan las SCM para calibrar: equiespaciados entre
    `start_frame` (fin del warm-up de los acumuladores) y T-1. Se toman POCOS
    (default 16) porque las SCM recursivas con alpha=0.99 tienen memoria de
    ~100 frames: frames contiguos son casi la misma muestra.
    """
    start_frame = int(np.clip(start_frame, 0, max(T - 1, 0)))
    n_eval = int(max(1, min(n_eval, T - start_frame)))
    return np.linspace(start_frame, T - 1, n_eval).astype(int)


def snapshot_scms_masked(X_stft, mask_s, mask_n, eval_frames, alpha=0.99):
    """
    Corre EXACTAMENTE la recursion de los cores mask-based y devuelve Phi_XX y
    Phi_NN congeladas en `eval_frames`.

        Num = alpha Num + m . x x^H ;  Den = alpha Den + m ;  Phi = Num / Den

    Args:
        X_stft: (K, T, M) mezcla observada.
        mask_s, mask_n: (K, T) mascaras de voz y ruido tal como las consume el core.
        eval_frames: (E,) indices de frame.
        alpha: escalar o (K,) factor de olvido (mismo contrato que el core).

    Returns:
        (Phi_XX, Phi_NN), ambas (K, E, M, M) complex128, Hermitianas.
    """
    K, T, M = X_stft.shape
    ev = np.asarray(eval_frames, dtype=int)
    pos = {int(t): i for i, t in enumerate(ev)}

    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.ndim == 0:
        alpha = np.full((K,), float(alpha))
    alpha = alpha[:, None, None]

    Num_XX = np.zeros((K, M, M), dtype=np.complex128)
    Num_NN = np.zeros((K, M, M), dtype=np.complex128)
    Den_XX = np.zeros((K, 1, 1), dtype=np.float64)
    Den_NN = np.zeros((K, 1, 1), dtype=np.float64)

    out_XX = np.zeros((K, len(ev), M, M), dtype=np.complex128)
    out_NN = np.zeros((K, len(ev), M, M), dtype=np.complex128)

    t_last = int(ev.max())
    for t in range(t_last + 1):
        Xf = X_stft[:, t, :]
        R = np.einsum("fm,fn->fmn", Xf, Xf.conj())
        ms = mask_s[:, t, None, None]
        mn = mask_n[:, t, None, None]

        Num_XX = alpha * Num_XX + ms * R
        Den_XX = alpha * Den_XX + ms
        Num_NN = alpha * Num_NN + mn * R
        Den_NN = alpha * Den_NN + mn

        i = pos.get(t)
        if i is not None:
            out_XX[:, i] = _herm(Num_XX / (Den_XX + 1e-15))
            out_NN[:, i] = _herm(Num_NN / (Den_NN + 1e-15))

    return out_XX, out_NN


def snapshot_scms_oracle(S_stft, N_stft, eval_frames, alpha=0.99):
    """
    Ídem `snapshot_scms_masked` pero con las componentes LIMPIAS y sin mascara
    (peso 1 por frame), igual que `MVDR_Souden_recursive_oracle`.

    Se usa el MISMO alpha a proposito: la referencia tiene que ser "lo mejor
    alcanzable con la MISMA ventana temporal", no un oraculo de ventana infinita.
    De lo contrario la loss mezcla error de calibracion con error de tracking.

    Returns:
        (Phi_S, Phi_N), ambas (K, E, M, M).
    """
    K, T, M = S_stft.shape
    ev = np.asarray(eval_frames, dtype=int)
    pos = {int(t): i for i, t in enumerate(ev)}

    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.ndim == 0:
        alpha = np.full((K,), float(alpha))
    alpha = alpha[:, None, None]

    Num_S = np.zeros((K, M, M), dtype=np.complex128)
    Num_N = np.zeros((K, M, M), dtype=np.complex128)
    Den = np.zeros((K, 1, 1), dtype=np.float64)

    out_S = np.zeros((K, len(ev), M, M), dtype=np.complex128)
    out_N = np.zeros((K, len(ev), M, M), dtype=np.complex128)

    t_last = int(ev.max())
    for t in range(t_last + 1):
        Sf, Nf = S_stft[:, t, :], N_stft[:, t, :]
        Num_S = alpha * Num_S + np.einsum("fm,fn->fmn", Sf, Sf.conj())
        Num_N = alpha * Num_N + np.einsum("fm,fn->fmn", Nf, Nf.conj())
        Den = alpha * Den + 1.0

        i = pos.get(t)
        if i is not None:
            out_S[:, i] = _herm(Num_S / (Den + 1e-15))
            out_N[:, i] = _herm(Num_N / (Den + 1e-15))

    return out_S, out_N


def _herm(A):
    """(A + A^H) / 2 sobre los dos ultimos ejes."""
    return 0.5 * (A + np.conj(np.swapaxes(A, -1, -2)))


# =====================================================================
# Modelo parametrico: (nu, gamma, mu) -> pesos
# =====================================================================
def _bin_param(p, K, ndim=4):
    """Escalar o (K,) -> (K, 1, ..., 1) con `ndim` ejes, para difundir sobre (K,E,M,M)."""
    p = np.asarray(p, dtype=np.float64)
    if p.ndim == 0:
        p = np.full((K,), float(p))
    if p.shape != (K,):
        raise ValueError(f"parametro debe ser escalar o de shape ({K},); es {p.shape}.")
    return p.reshape((K,) + (1,) * (ndim - 1))


def parametric_scms(Phi_XX, Phi_NN, Gamma, nu=0.0, gamma=0.0, psd_project=True):
    """
    Aplica el shrinkage geometrico y la sustraccion con escala corregida.

    Args:
        Phi_XX, Phi_NN: (K, E, M, M) SCM enmascaradas (salida de `snapshot_scms_masked`).
        Gamma: (K, M, M) coherencia difusa (`diffuse_coherence`), o None para
            desactivar el shrinkage (equivale a gamma = 0).
        nu: escalar o (K,) -- factor de escala de la sustraccion.
            nu = 0 -> Phi_SS = Phi_XX (core base); nu = 1 -> core `_subtract`.
        gamma: escalar o (K,) en [0, 1] -- peso del shrinkage hacia Gamma.
        psd_project: proyecta Phi_SS a PSD (autovalores negativos -> 0). Solo se
            aplica si nu > 0; con nu = 0 la resta no existe y Phi_XX ya es PSD.

    Returns:
        (Phi_SS, Phi_NN_shrunk), ambas (K, E, M, M).
    """
    K, E, M, _ = Phi_XX.shape
    nu_b = _bin_param(nu, K)
    g_b = _bin_param(gamma, K)

    if Gamma is None or np.all(g_b == 0.0):
        Phi_NN_s = Phi_NN
    else:
        tr = np.real(np.trace(Phi_NN, axis1=-2, axis2=-1))[..., None, None] / M  # (K,E,1,1)
        target = tr * Gamma[:, None, :, :].astype(np.complex128)                 # (K,E,M,M)
        Phi_NN_s = (1.0 - g_b) * Phi_NN + g_b * target

    if np.all(nu_b == 0.0):
        return _herm(Phi_XX), _herm(Phi_NN_s)

    Phi_SS = _herm(Phi_XX - nu_b * Phi_NN_s)
    if psd_project:
        evals, evecs = np.linalg.eigh(Phi_SS)
        evals = np.maximum(evals, 0.0)
        Phi_SS = np.einsum("...mp,...np->...mn", evecs * evals[..., None, :], evecs.conj())
    return Phi_SS, _herm(Phi_NN_s)


def souden_weights(Phi_SS, Phi_NN, ref_mic, mu=0.0, min_loading=1e-9,
                   lambda_floor=1e-3):
    """
    w = Phi_NN^-1 Phi_SS u / (tr(Phi_NN^-1 Phi_SS) + mu), con la misma base
    numerica que los cores `_fixed` / `_subtract`: carga diagonal RELATIVA
    (escala-invariante, sin piso absoluto salvo un epsilon de cold-start),
    Hermitiana forzada y `np.linalg.solve`.

    Args:
        Phi_SS, Phi_NN: (K, E, M, M).
        ref_mic: indice del microfono de referencia (el one-hot `u`).
        mu: trade-off PMWF sobre el denominador.
        min_loading: carga diagonal relativa (fraccion de tr(Phi_NN)/M).
        lambda_floor: piso numerico sobre la traza; inerte cuando nu = 0
            (ahi lambda >= M) y necesario cuando nu > 0 y mu = 0.

    Returns:
        (W, lambda_S): W (K, E, M) complejo; lambda_S (K, E) real, la traza SIN
        el piso ni mu (util como indice de confianza: lambda_S / M).
    """
    K, E, M, _ = Phi_NN.shape
    tr = np.real(np.trace(Phi_NN, axis1=-2, axis2=-1))                 # (K,E)
    load = min_loading * (tr / M) + 1e-12
    eye = np.eye(M)[None, None, :, :]
    Phi_NN_st = Phi_NN + eye * load[..., None, None]

    G = np.linalg.solve(Phi_NN_st, Phi_SS)                              # (K,E,M,M)
    lam = np.real(np.trace(G, axis1=-2, axis2=-1))                      # (K,E)
    den = np.maximum(lam, lambda_floor) + mu + 1e-15
    W = G[..., :, ref_mic] / den[..., None]
    return W, lam


def parametric_weights(Phi_XX, Phi_NN, Gamma, ref_mic, nu=0.0, gamma=0.0, mu=0.0,
                       min_loading=1e-9, lambda_floor=1e-3, psd_project=True):
    """Atajo: `parametric_scms` + `souden_weights`. Devuelve (W, lambda_S)."""
    Phi_SS, Phi_NN_s = parametric_scms(Phi_XX, Phi_NN, Gamma, nu=nu, gamma=gamma,
                                       psd_project=psd_project)
    return souden_weights(Phi_SS, Phi_NN_s, ref_mic, mu=mu, min_loading=min_loading,
                          lambda_floor=lambda_floor)


# =====================================================================
# Referencias oracle y funcion de costo
# =====================================================================
def oracle_references(Phi_S, Phi_N, ref_mic, min_loading=1e-9, snr_floor_db=-20.0):
    """
    Precomputa, por (bin, frame de evaluacion), todo lo que la loss necesita del
    oracle. Se hace UNA VEZ por escena; el optimizador no lo vuelve a tocar.

    Returns dict con:
        sinr_max : (K, E) lambda_max(Phi_N^-1 Phi_S), el SINR alcanzable.
        a_rtf    : (K, E, M) autovector principal de Phi_S normalizado a 1 en
                   ref_mic == la RTF oracle del target.
        snr_loc  : (K, E) tr(Phi_S)/tr(Phi_N), SNR local oracle.
        valid    : (K, E) bool. Descarta las celdas donde no hay target que
                   preservar (snr_loc por debajo de `snr_floor_db`) o donde el
                   oracle es numericamente degenerado: ahi SINR_max es ruido
                   contra ruido y la loss no significa nada.

    El whitening se hace por Cholesky de Phi_N (mas estable que invertir):
        A = L^-1 Phi_S L^-H  ->  lambda_max(A) = lambda_max(Phi_N^-1 Phi_S).
    """
    K, E, M, _ = Phi_S.shape
    tr_N = np.real(np.trace(Phi_N, axis1=-2, axis2=-1))
    tr_S = np.real(np.trace(Phi_S, axis1=-2, axis2=-1))
    load = min_loading * (tr_N / M) + 1e-12
    eye = np.eye(M)[None, None, :, :]
    Phi_N_st = _herm(Phi_N) + eye * load[..., None, None]

    # El whitening por Cholesky exige Phi_N_st definida positiva. Con la carga
    # relativa lo esta salvo round-off en bins casi vacios; ahi se sube la carga
    # antes de tirar la escena entera.
    for extra in (0.0, 1e-6, 1e-3):
        try:
            L = np.linalg.cholesky(Phi_N_st + eye * (extra * (tr_N / M) + 1e-30)[..., None, None])
            break
        except np.linalg.LinAlgError:
            continue
    else:                                                      # pragma: no cover
        raise np.linalg.LinAlgError("Phi_N no es definida positiva ni con carga 1e-3")
    B = np.linalg.solve(L, _herm(Phi_S))                       # L^-1 Phi_S
    A = np.conj(np.swapaxes(np.linalg.solve(L, np.conj(np.swapaxes(B, -1, -2))), -1, -2))
    A = _herm(A)
    ev = np.linalg.eigvalsh(A)                                 # ascendente
    sinr_max = np.maximum(ev[..., -1], 0.0)                    # (K, E)

    evals, evecs = np.linalg.eigh(_herm(Phi_S))
    v = evecs[..., :, -1]                                      # (K, E, M)
    v_ref = v[..., ref_mic]
    ok_ref = np.abs(v_ref) > 1e-12
    a_rtf = np.where(ok_ref[..., None], v / np.where(ok_ref, v_ref, 1.0)[..., None], 0.0)

    snr_loc = tr_S / (tr_N + 1e-30)
    valid = (10.0 * np.log10(snr_loc + 1e-30) > snr_floor_db) & ok_ref & (sinr_max > 1e-9)

    # pow_S = tr(Phi_S) = potencia del TARGET oracle en la celda. Es el PESO con
    # el que hay que agregar la loss para que siga a las metricas globales
    # (PESQ/SDR/SIR agregan energia, no celdas): sin el, un bin de 6 kHz casi
    # vacio pesa lo mismo que uno de 500 Hz que lleva casi toda la voz.
    return {"sinr_max": sinr_max, "a_rtf": a_rtf, "snr_loc": snr_loc,
            "valid": valid, "pow_S": tr_S}


def weight_loss(W, Phi_S, Phi_N, refs, eta=1.0):
    """
    Perdida por (bin, frame) del beamformer W respecto del oracle.

        L_sinr = 10 log10(SINR_max / SINR(w))     >= 0
        L_dist = 10 log10(1 + |w^H a - 1|^2)      >= 0
        L      = L_sinr + eta * L_dist

    Ambas en dB y ambas >= 0, asi que eta pondera cantidades comparables.
    L_sinr es invariante a la escala de w (que es justamente la invariancia de
    Souden); L_dist es la que la fija, y es la que se corresponde con PESQ.

    Returns dict con "L", "L_sinr", "L_dist", todas (K, E). Las celdas no
    validas quedan en NaN (las funciones de agregacion usan nanmedian).
    """
    quad_S = np.real(np.einsum("...m,...mn,...n->...", W.conj(), Phi_S, W))
    quad_N = np.real(np.einsum("...m,...mn,...n->...", W.conj(), Phi_N, W))
    sinr = np.maximum(quad_S, 0.0) / (np.maximum(quad_N, 0.0) + 1e-30)

    L_sinr = 10.0 * np.log10((refs["sinr_max"] + 1e-30) / (sinr + 1e-30))
    L_sinr = np.maximum(L_sinr, 0.0)          # el oracle es cota: negativo = round-off

    resp = np.einsum("...m,...m->...", W.conj(), refs["a_rtf"])
    L_dist = 10.0 * np.log10(1.0 + np.abs(resp - 1.0) ** 2)

    bad = ~refs["valid"] | ~np.isfinite(L_sinr) | ~np.isfinite(L_dist)
    L_sinr = np.where(bad, np.nan, L_sinr)
    L_dist = np.where(bad, np.nan, L_dist)
    return {"L": L_sinr + eta * L_dist, "L_sinr": L_sinr, "L_dist": L_dist}


def scm_fidelity(Phi_hat, Phi_ref):
    """
    DIAGNOSTICO de cuanto se parece una SCM estimada a su oracle. NO es la loss
    del banco (para AJUSTAR sirve `weight_loss`: mide lo que el beamformer
    pierde, que es lo que se traduce en las metricas). Esta funcion contesta la
    pregunta previa y mas directa -- "¿mejoro la estimacion de la matriz de
    correlacion?" -- y por eso las dos medidas que devuelve son INVARIANTES A LA
    ESCALA, igual que la formula de Souden:

        cmd  = 1 - |<Phi_hat, Phi_ref>_F| / (||Phi_hat||_F ||Phi_ref||_F)
               Correlation Matrix Distance (Herdin et al. 2005). 0 = identicas
               salvo escala; 1 = ortogonales. Mide error de FORMA, que es lo
               unico que el filtro ve.
        evec_deg = angulo (grados) entre los autovectores principales de una y
               otra. Sobre Phi_SS es el error de la RTF estimada -> apunta
               directo a la distorsion del target; sobre Phi_NN dice si la
               direccion de ruido dominante (la que el filtro va a anular) esta
               bien identificada.

    Args:
        Phi_hat, Phi_ref: (..., M, M) Hermitianas.

    Returns:
        dict con "cmd" y "evec_deg", ambos (...).
    """
    A, B = _herm(Phi_hat), _herm(Phi_ref)
    ip = np.abs(np.einsum("...mn,...mn->...", A.conj(), B))
    na = np.sqrt(np.real(np.einsum("...mn,...mn->...", A.conj(), A)))
    nb = np.sqrt(np.real(np.einsum("...mn,...mn->...", B.conj(), B)))
    cmd = 1.0 - ip / (na * nb + 1e-30)

    va = np.linalg.eigh(A)[1][..., :, -1]
    vb = np.linalg.eigh(B)[1][..., :, -1]
    c = np.abs(np.einsum("...m,...m->...", va.conj(), vb))
    c = c / (np.linalg.norm(va, axis=-1) * np.linalg.norm(vb, axis=-1) + 1e-30)
    deg = np.degrees(np.arccos(np.clip(c, 0.0, 1.0)))

    bad = (na < 1e-30) | (nb < 1e-30)
    return {"cmd": np.where(bad, np.nan, cmd), "evec_deg": np.where(bad, np.nan, deg)}


def mask_separation_db(mask, S_stft, N_stft, ref_mic, return_parts=False):
    """
    SNR (dB) de las celdas que una mascara SELECCIONA, medida con las senales
    limpias en el canal de referencia:

        10 log10( sum_t m(k,t) |S_ref|^2  /  sum_t m(k,t) |N_ref|^2 )

    Es la lectura mas directa de para que sirve la mascara en estos cores: la
    mascara no filtra nada, PONDERA un promedio de outer products, asi que lo
    unico que importa es que celdas pesa. Sobre mask_s, cuanto mas ALTO mejor
    (Phi_XX menos contaminada por ruido); sobre mask_n, cuanto mas BAJO mejor
    (menos fuga de voz dentro de Phi_NN -> menos auto-cancelacion del target).

    Returns:
        (K,) en dB.
    """
    m = np.asarray(mask, dtype=np.float64)
    T = min(m.shape[1], S_stft.shape[1], N_stft.shape[1])
    ps = np.abs(S_stft[:, :T, ref_mic]) ** 2
    pn = np.abs(N_stft[:, :T, ref_mic]) ** 2
    num = np.sum(m[:, :T] * ps, axis=1)
    den = np.sum(m[:, :T] * pn, axis=1)
    if return_parts:
        # Numerador y denominador SIN dividir: el que agrega por banda tiene que
        # sumar energias y recien despues dividir (promediar los dB por bin le
        # daria el mismo peso a un bin vacio que a uno que lleva toda la voz).
        return num, den
    return 10.0 * np.log10((num + 1e-30) / (den + 1e-30))


# =====================================================================
# Empaquetado de una escena
# =====================================================================
def prepare_scene(X_stft, S_stft, N_stft, mask_s, mask_n, mic_coords, freqs,
                  ref_mic, alpha=0.99, n_eval=16, start_frame=0,
                  min_loading=1e-9, snr_floor_db=-20.0, field="spherical",
                  use_geometry=True, name=""):
    """
    Congela TODO lo que el optimizador necesita de una escena: las SCM
    enmascaradas, las oracle, la coherencia difusa y las referencias del oracle.

    Es el paso caro (recorre los frames) y se hace UNA sola vez: los parametros
    (nu, gamma, mu) actuan DESPUES de la acumulacion, asi que el objetivo se
    evalua sin volver a tocar la recursion. Por eso el ajuste corre en segundos
    y no en horas.

    X_stft / S_stft / N_stft: (K, T, M) mezcla observada y componentes oracle
    (target y ruido) EN EL MISMO DOMINIO -- o sea, despues de la emulacion de
    hardware, con hw_target + hw_noise == mezcla exacto.

    Returns dict con Phi_XX, Phi_NN, Phi_S, Phi_N (K,E,M,M), Gamma (K,M,M),
    refs, ref_mic, eval_frames, name.
    """
    K, T, M = X_stft.shape
    Tm = min(T, S_stft.shape[1], N_stft.shape[1], mask_s.shape[1], mask_n.shape[1])
    X_stft, S_stft, N_stft = X_stft[:, :Tm], S_stft[:, :Tm], N_stft[:, :Tm]
    mask_s, mask_n = mask_s[:, :Tm], mask_n[:, :Tm]

    ev = eval_frame_indices(Tm, n_eval, start_frame=start_frame)
    Phi_XX, Phi_NN = snapshot_scms_masked(X_stft, mask_s, mask_n, ev, alpha=alpha)
    Phi_S, Phi_N = snapshot_scms_oracle(S_stft, N_stft, ev, alpha=alpha)
    Gamma = diffuse_coherence(mic_coords, freqs, field=field) if use_geometry else None
    refs = oracle_references(Phi_S, Phi_N, ref_mic, min_loading=min_loading,
                             snr_floor_db=snr_floor_db)

    return {"name": name, "Phi_XX": Phi_XX, "Phi_NN": Phi_NN,
            "Phi_S": Phi_S, "Phi_N": Phi_N, "Gamma": Gamma,
            "refs": refs, "ref_mic": int(ref_mic), "eval_frames": ev,
            "freqs": np.asarray(freqs, dtype=np.float64)}


def oracle_bound(scenes, bin_idx=None, eta=1.0, mu=0.0, min_loading=1e-9,
                 how="median"):
    """
    Piso de la loss: los pesos de Souden calculados con las SCM ORACLE. Es el
    control de sanidad del banco -- si esto no da ~0 dB de L_sinr, la loss o las
    referencias estan mal, y cualquier numero de ajuste es basura.
    """
    Ls, Ls_sinr, Ls_dist, Ws = [], [], [], []
    for sc in scenes:
        sl = slice(None) if bin_idx is None else bin_idx
        W, _ = souden_weights(sc["Phi_S"][sl], sc["Phi_N"][sl], sc["ref_mic"],
                              mu=mu, min_loading=min_loading)
        refs = {k: v[sl] for k, v in sc["refs"].items()}
        out = weight_loss(W, sc["Phi_S"][sl], sc["Phi_N"][sl], refs, eta=eta)
        Ls.append(out["L"]); Ls_sinr.append(out["L_sinr"]); Ls_dist.append(out["L_dist"])
        Ws.append(refs["pow_S"])
    return {"L": _aggregate(Ls, how, Ws), "L_sinr": _aggregate(Ls_sinr, how, Ws),
            "L_dist": _aggregate(Ls_dist, how, Ws)}


# =====================================================================
# Bandas de frecuencia
# =====================================================================
def make_bands(freqs, n_bands=20, f_min=60.0, f_max=7000.0):
    """
    Particion LOG de los bins en `n_bands` grupos que cubren TODO el eje: los
    bins por debajo de f_min caen en la banda 0 y los de arriba de f_max en la
    ultima. Se ajusta por banda (y no por bin) para no sobreajustar: son ~2
    parametros por banda contra ~2 por bin.

    Returns:
        (edges, band_of_bin, bands) con edges (n_bands+1,), band_of_bin (K,) y
        bands = lista de arrays de indices de bin (puede haber bandas vacias en
        graves si la resolucion de la STFT no alcanza; se filtran).
    """
    f = np.asarray(freqs, dtype=np.float64)
    edges = np.logspace(np.log10(f_min), np.log10(f_max), int(n_bands) + 1)
    band_of_bin = np.digitize(f, edges[1:-1])            # 0 .. n_bands-1
    bands = [np.flatnonzero(band_of_bin == b) for b in range(int(n_bands))]
    return edges, band_of_bin, bands


# =====================================================================
# Ajuste
# =====================================================================
AGG_MODES = ("median", "mean", "wmedian", "wmean")


def _aggregate(vals, how="median", weights=None):
    """
    Agrega un conjunto de perdidas (K,E) ignorando NaN.

    POR QUE IMPORTA EL MODO. La loss vive por celda (bin, frame), pero las
    metricas del benchmark (PESQ, STOI, SDR, SIR) agregan ENERGIA sobre toda la
    senal. Una mediana simple sobre celdas le da el mismo peso a un bin de 6 kHz
    practicamente vacio que a uno de 500 Hz que lleva casi toda la voz -- y ahi
    el proxy deja de seguir a las metricas. Medido sobre MIRD, la ventaja del
    ajuste sobre NM_MVDR_SUB pasa de 1.03 dB con "median" a 0.05 dB con
    "wmedian": la mediana simple SOBRE-REPORTABA la ganancia ~20x.

        median  : mediana simple (historico; robusta pero ciega a la energia)
        mean    : media simple
        wmedian : mediana PONDERADA por la potencia del target oracle (pow_S).
                  Recomendada: robusta a outliers Y alineada con lo que miden
                  las metricas globales.
        wmean   : media ponderada por pow_S. La mas cercana a un cociente de
                  energias, tambien la mas sensible a celdas atipicas.
    """
    if how not in AGG_MODES:
        raise ValueError(f"how debe ser uno de {AGG_MODES}; es {how!r}.")

    def _cat(x):
        return np.concatenate([np.asarray(a).ravel() for a in x]) \
            if isinstance(x, list) else np.asarray(x).ravel()

    v = _cat(vals)
    if how in ("median", "mean") or weights is None:
        v = v[np.isfinite(v)]
        if v.size == 0:
            return np.nan
        return float(np.median(v) if how in ("median", "wmedian") else np.mean(v))

    w = _cat(weights)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[ok], w[ok]
    if v.size == 0:
        return np.nan
    if how == "wmean":
        return float(np.sum(v * w) / np.sum(w))
    # mediana ponderada: el valor donde el peso acumulado cruza el 50 %
    o = np.argsort(v)
    v, w = v[o], w[o]
    c = np.cumsum(w) / np.sum(w)
    return float(v[min(int(np.searchsorted(c, 0.5)), v.size - 1)])


def band_objective(scenes, bin_idx, nu, gamma, mu=0.0, eta=1.0,
                   min_loading=1e-9, lambda_floor=1e-3, psd_project=True,
                   how="median", detail=False):
    """
    Valor del objetivo para UNA banda, agregando sobre todas las escenas.

    `scenes` es una lista de dicts con las claves que produce
    `prepare_scene`: Phi_XX, Phi_NN, Phi_S, Phi_N, Gamma, refs, ref_mic.
    Solo se tocan los bins de `bin_idx`, que es lo que hace barato el barrido.
    """
    Ls, Ls_sinr, Ls_dist, Ws = [], [], [], []
    for sc in scenes:
        W, _ = parametric_weights(
            sc["Phi_XX"][bin_idx], sc["Phi_NN"][bin_idx],
            None if sc["Gamma"] is None else sc["Gamma"][bin_idx],
            sc["ref_mic"], nu=nu, gamma=gamma, mu=mu,
            min_loading=min_loading, lambda_floor=lambda_floor,
            psd_project=psd_project)
        refs = {k: v[bin_idx] for k, v in sc["refs"].items()}
        out = weight_loss(W, sc["Phi_S"][bin_idx], sc["Phi_N"][bin_idx], refs, eta=eta)
        Ls.append(out["L"]); Ls_sinr.append(out["L_sinr"]); Ls_dist.append(out["L_dist"])
        Ws.append(refs["pow_S"])

    if not detail:
        return _aggregate(Ls, how, Ws)
    return {"L": _aggregate(Ls, how, Ws),
            "L_sinr": _aggregate(Ls_sinr, how, Ws),
            "L_dist": _aggregate(Ls_dist, how, Ws)}


def fit_band(scenes, bin_idx, nu_grid, gamma_grid, mu=0.0, eta=1.0, refine=True,
             ref_point=None, sinr_tol_db=0.0, **kw):
    """
    Ajuste de UNA banda: grilla gruesa sobre (nu, gamma) y, opcionalmente,
    refinamiento Nelder-Mead desde el mejor punto de la grilla.

    Se hace grilla ANTES del refinamiento a proposito: el paisaje completo es el
    resultado interesante (¿hay un minimo definido o es una meseta?), y ademas
    protege al ajuste de minimos locales, que en `gamma` los hay cuando el
    modelo difuso no aplica.

    AJUSTE CON RESTRICCION (ref_point)
    ----------------------------------
    Sin restriccion, el optimo minimiza L = L_sinr + eta*L_dist y por lo tanto
    ACEPTA perder SINR si gana suficiente en distorsion. Medido sobre MIRD, con
    eta = 1 eso cuesta ~1.7 dB de SIR contra NM_MVDR_SUB (a cambio de ganar SAR,
    STOI y SDR). Cuando lo que se quiere no es un compromiso sino DOMINAR a un
    core de referencia, se pasa `ref_point`:

        ref_point = (nu_ref, gamma_ref)   p.ej. (1.0, 0.0) == NM_MVDR_SUB
                    (0.0, 0.0) == NM_MVDR base

    y el ajuste queda restringido a los puntos que NO empeoran el termino de
    SINR de la referencia:

        minimizar  L   sujeto a   L_sinr <= L_sinr(ref) + sinr_tol_db

    Entre los candidatos que cumplen, gana el de menor L -- o sea que la mejora
    se cobra en distorsion, que es la unica direccion libre. Si NINGUN punto de
    la grilla cumple la restriccion (puede pasar si la referencia ya es el
    minimo de L_sinr en esa banda), se devuelve el de MENOR L_sinr: la banda se
    queda con el mejor SINR disponible en vez de romper la garantia.

    sinr_tol_db : holgura en dB sobre la restriccion. 0.0 = dominacion estricta;
        un valor chico (0.1-0.3) afloja la banda a cambio de mas margen en
        distorsion, util si la restriccion deja el ajuste pegado a la referencia.

    Returns dict con nu, gamma, L, L_sinr, L_dist, la grilla de L en "grid", la
    de L_sinr en "grid_sinr", y "constrained" (bool: si la restriccion estaba
    activa y se pudo satisfacer).
    """
    n_i, n_j = len(nu_grid), len(gamma_grid)
    grid = np.full((n_i, n_j), np.nan)
    grid_sinr = np.full((n_i, n_j), np.nan)
    grid_dist = np.full((n_i, n_j), np.nan)
    for i, nu in enumerate(nu_grid):
        for j, g in enumerate(gamma_grid):
            d = band_objective(scenes, bin_idx, nu, g, mu=mu, eta=eta,
                               detail=True, **kw)
            grid[i, j], grid_sinr[i, j], grid_dist[i, j] = d["L"], d["L_sinr"], d["L_dist"]

    empty = {"nu": np.nan, "gamma": np.nan, "L": np.nan, "L_sinr": np.nan,
             "L_dist": np.nan, "grid": grid, "grid_sinr": grid_sinr,
             "constrained": False}
    if np.all(~np.isfinite(grid)):
        return empty

    # --- restriccion opcional contra un core de referencia -------------------
    cap, feasible = None, np.isfinite(grid)
    constrained = False
    if ref_point is not None:
        ref = band_objective(scenes, bin_idx, ref_point[0], ref_point[1], mu=mu,
                             eta=eta, detail=True, **kw)
        if np.isfinite(ref["L_sinr"]):
            cap = ref["L_sinr"] + float(sinr_tol_db)
            ok = feasible & (grid_sinr <= cap)
            if ok.any():
                feasible, constrained = ok, True
            else:
                # Nada cumple: quedarse con el mejor SINR posible en la banda.
                feasible = feasible & (grid_sinr <= np.nanmin(grid_sinr) + 1e-9)

    masked = np.where(feasible, grid, np.inf)
    i0, j0 = np.unravel_index(np.argmin(masked), masked.shape)
    best = {"nu": float(nu_grid[i0]), "gamma": float(gamma_grid[j0]),
            "L": float(grid[i0, j0]), "L_sinr": float(grid_sinr[i0, j0]),
            "L_dist": float(grid_dist[i0, j0]), "grid": grid,
            "grid_sinr": grid_sinr, "constrained": constrained}

    if refine:
        try:
            from scipy.optimize import minimize

            def obj(th):
                nu = float(np.clip(th[0], nu_grid[0], nu_grid[-1]))
                g = float(np.clip(th[1], 0.0, 0.999))
                d = band_objective(scenes, bin_idx, nu, g, mu=mu, eta=eta,
                                   detail=True, **kw)
                if not np.isfinite(d["L"]):
                    return 1e6
                # La restriccion entra como penalidad: Nelder-Mead no admite
                # restricciones duras, y una penalidad lineal fuerte alcanza
                # porque el punto de partida ya es factible.
                pen = 0.0
                if cap is not None and np.isfinite(d["L_sinr"]):
                    pen = 1e3 * max(0.0, d["L_sinr"] - cap)
                return d["L"] + pen

            x0 = [best["nu"], best["gamma"]]
            res = minimize(obj, x0=x0, method="Nelder-Mead",
                           options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 200})
            if np.isfinite(res.fun) and res.fun < obj(x0):
                nu_r = float(np.clip(res.x[0], nu_grid[0], nu_grid[-1]))
                g_r = float(np.clip(res.x[1], 0.0, 0.999))
                d = band_objective(scenes, bin_idx, nu_r, g_r, mu=mu, eta=eta,
                                   detail=True, **kw)
                best.update(nu=nu_r, gamma=g_r, L=d["L"], L_sinr=d["L_sinr"],
                            L_dist=d["L_dist"])
        except ImportError:                                    # pragma: no cover
            pass

    return best


def fit_bands(scenes, freqs, bands, nu_grid, gamma_grid, mu=0.0, eta=1.0,
              refine=True, verbose=True, ref_point=None, sinr_tol_db=0.0, **kw):
    """
    Ajusta todas las bandas de forma INDEPENDIENTE (la loss es separable por bin,
    y los parametros de una banda no afectan a otra). Devuelve una lista de dicts
    por banda con el optimo, la grilla, y los valores de referencia de los dos
    cores existentes:

        L_base = objetivo en (nu=0, gamma=0)  == core actual NM_MVDR
        L_sub  = objetivo en (nu=1, gamma=0)  == core NM_MVDR_SUB (mu=0)

    ref_point / sinr_tol_db: ver `fit_band`. Con ref_point=(1.0, 0.0) el ajuste
    queda restringido a NO perder SINR contra NM_MVDR_SUB en ninguna banda.
    """
    rows = []
    for b, bin_idx in enumerate(bands):
        if bin_idx.size == 0:
            continue
        res = fit_band(scenes, bin_idx, nu_grid, gamma_grid, mu=mu, eta=eta,
                       refine=refine, ref_point=ref_point,
                       sinr_tol_db=sinr_tol_db, **kw)
        base = band_objective(scenes, bin_idx, 0.0, 0.0, mu=mu, eta=eta, detail=True, **kw)
        sub = band_objective(scenes, bin_idx, 1.0, 0.0, mu=mu, eta=eta, detail=True, **kw)
        fit = band_objective(scenes, bin_idx, res["nu"], res["gamma"], mu=mu, eta=eta,
                             detail=True, **kw)
        row = {
            "band": b,
            "f_lo": float(freqs[bin_idx[0]]), "f_hi": float(freqs[bin_idx[-1]]),
            # centro geometrico, usable en eje log: la banda 0 arranca en DC y
            # un semilogx la descartaria entera.
            "f_c": float(np.sqrt(max(freqs[bin_idx[0]], 0.5 * freqs[1])
                                 * max(freqs[bin_idx[-1]], freqs[1]))),
            "n_bins": int(bin_idx.size),
            "nu": res["nu"], "gamma": res["gamma"],
            "L_base": base["L"], "L_sub": sub["L"], "L_fit": fit["L"],
            "Lsinr_base": base["L_sinr"], "Lsinr_sub": sub["L_sinr"], "Lsinr_fit": fit["L_sinr"],
            "Ldist_base": base["L_dist"], "Ldist_sub": sub["L_dist"], "Ldist_fit": fit["L_dist"],
            "gain_vs_base": base["L"] - fit["L"],
            "gain_vs_sub": sub["L"] - fit["L"],
            "constrained": bool(res.get("constrained", False)),
            "_grid": res["grid"], "_bins": bin_idx,
        }
        rows.append(row)
        if verbose:
            print(f"  banda {b:2d}  {row['f_lo']:6.0f}-{row['f_hi']:6.0f} Hz  "
                  f"nu={row['nu']:5.2f} gamma={row['gamma']:5.2f}  "
                  f"L: base {row['L_base']:6.2f} | sub {row['L_sub']:6.2f} | "
                  f"fit {row['L_fit']:6.2f}  (gana {row['gain_vs_base']:+5.2f} dB)")
    return rows


def bands_to_bin_params(rows, K):
    """
    Expande el ajuste por banda a vectores POR BIN (K,), listos para pasarle a
    un core como `nu_k` / `gamma_k`. Los bins de bandas vacias quedan en el
    valor de la banda no vacia mas cercana.
    """
    nu = np.zeros(K, dtype=np.float64)
    gam = np.zeros(K, dtype=np.float64)
    filled = np.zeros(K, dtype=bool)
    for r in rows:
        nu[r["_bins"]] = r["nu"]
        gam[r["_bins"]] = r["gamma"]
        filled[r["_bins"]] = True
    if not filled.all() and filled.any():
        idx = np.flatnonzero(filled)
        near = idx[np.abs(np.arange(K)[:, None] - idx[None, :]).argmin(axis=1)]
        nu[~filled] = nu[near][~filled]
        gam[~filled] = gam[near][~filled]
    return nu, gam


# =====================================================================
# ETAPA 2: calibracion de los parametros de la MASCARA
# =====================================================================
# Los parametros de arriba (nu, gamma) actuan DESPUES de la acumulacion, asi que
# el objetivo se evalua sin rehacer la recursion. Los de esta seccion actuan
# ANTES: cambian que se acumula, asi que cada evaluacion exige recorrer los
# frames de nuevo. Es ~100x mas caro, y por eso se hace SOLO despues de haber
# agotado la familia post-hoc (que resulto estar saturada).
#
# QUE SE REEMPLAZA
# ----------------
# El camino actual (dtln_masks.get_dtln_masks_sharpen) es:
#
#     m_raw = DTLN(x_ref)                  # ya en [0,1] por construccion
#     m     = (m_raw - min(m_raw)) / (max(m_raw) - min(m_raw))     <-- STRETCH
#     mask_s = m ** 4  ;  mask_n = (1 - m) ** 4                    <-- SHARPEN
#
# El STRETCH es GLOBAL sobre (K, T): usa el min/max de TODO el archivo y de
# TODAS las frecuencias. Eso es (a) NO CAUSAL -- para el frame 5 necesita el
# futuro, asi que no es implementable en el sistema online que se lleva a HLS;
# (b) dependiente del archivo -- el mismo segmento recibe distinta mascara segun
# que mas haya en la grabacion; (c) acoplado en frecuencia -- la normalizacion de
# un bin de 200 Hz depende del pico en 6 kHz. Y no normaliza nada que lo
# necesite: la salida del DTLN ya vive en [0,1].
#
# Se reemplaza por un WARP FIJO POR BANDA, causal y sin estado global, ajustado
# contra las SCM oracle:
#
#     mask_s = sigma(a_s * logit(m_raw) + b_s)
#     mask_n = sigma(a_n * logit(1 - m_raw) + b_n)
#
# Con (a=1, b=0) el warp es la IDENTIDAD. `a` controla el contraste (a > 1
# agudiza, a < 1 suaviza) y `b` corre el umbral -- o sea que la familia contiene
# tanto al sharpening como a una recalibracion de punto de operacion. La
# diferencia de fondo con lo que hay hoy: las dos ramas tienen parametros
# INDEPENDIENTES. Hoy mask_n = (1 - m)**4 esta atada a mask_s = m**4, pero
# Phi_NN es la que se INVIERTE, y lo que necesita no es "1 menos probabilidad de
# voz" sino un detector de DOMINANCIA DE RUIDO, que optimamente es mucho mas
# conservador.


def warp_mask(m, a=1.0, b=0.0, eps=1e-4):
    """
    Warp logit-afin de una mascara en [0,1]:  sigma(a * logit(m) + b).

    Monotono, mapea [0,1] -> [0,1], y (a=1, b=0) es la identidad EXACTA.
    a > 1 agudiza la transicion (mas binaria), a < 1 la suaviza, b corre el
    umbral. Contiene al sharpening por potencia como caso aproximado y ademas
    permite recalibrar el punto de operacion, que es lo que hoy hace de forma
    implicita (y no causal) el stretch min-max.

    Args:
        m: (K, T) mascara cruda del DTLN, en [0,1].
        a, b: escalares o arrays (K,) -> parametro POR BIN.
        eps: recorte antes del logit (m exactamente 0 o 1 lo haria infinito).
    """
    m = np.clip(np.asarray(m, dtype=np.float64), eps, 1.0 - eps)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    z = a * np.log(m / (1.0 - m)) + b
    return 1.0 / (1.0 + np.exp(-z))


def masks_from_raw(m_raw, a_s=1.0, b_s=0.0, a_n=1.0, b_n=0.0, eps=1e-4):
    """
    Mascara de voz y de ruido desde la salida CRUDA del DTLN, con las dos ramas
    parametrizadas por SEPARADO. Devuelve (mask_s, mask_n), ambas (K, T).

    (1, 0, 1, 0) -> (m_raw, 1 - m_raw): el complemento lineal sin realce.
    """
    mask_s = warp_mask(m_raw, a_s, b_s, eps=eps)
    mask_n = warp_mask(1.0 - np.asarray(m_raw, dtype=np.float64), a_n, b_n, eps=eps)
    return mask_s, mask_n


MASK_THETA0 = (1.0, 0.0, 1.0, 0.0)          # identidad: (a_s, b_s, a_n, b_n)
# El limite inferior de b es -12 y NO -4 por una razon medida: con el tope en -4
# el ajuste queda CORTADO antes del optimo (en test daba +0.64 dB contra la
# mascara actual; ampliando la cota, +1.35 dB).
#
# QUE FORMA TOMA LA RAMA DE RUIDO. Con b_n negativo, sigma(a_n z + b_n) crece
# como C*((1-m)/m)^{a_n} en el grueso de las celdas y SATURA EN 1 para las de
# ruido mas confiable. La saturacion NO es un detalle: es lo que hace que
# funcione. Medido sobre MIRD (rt60=0.61, no visto), rama de voz en identidad:
#     odds-ratio PURO ((1-m)/m)^2   L = 4.042 dB
#     sigmoide a_n=2, b_n=-8        L = 3.585 dB     <- 0.46 dB mejor
#     sigmoide a_n=2, b_n=-12       L = 3.661 dB
# Solo el 2.5 % de las celdas cae en la zona saturada, pero son las de mayor
# peso: sin recorte un punado de celdas domina Phi_NN. O sea que b_n es un
# parametro REAL (fija donde recorta), no una direccion plana -- la meseta que
# se observa barriendo b_n en UNA banda no sobrevive al agregado sobre todos los
# bins. Ver tests/test_scm_calibration.py.
MASK_BOUNDS = ((0.25, 8.0), (-12.0, 4.0), (0.25, 8.0), (-12.0, 4.0))


def _clip_theta(theta):
    return tuple(float(np.clip(t, lo, hi)) for t, (lo, hi) in zip(theta, MASK_BOUNDS))


def band_objective_mask(scenes, bin_idx, theta, nu=1.0, gamma=0.0, mu=0.0, eta=1.0,
                        min_loading=1e-9, lambda_floor=1e-3, psd_project=True,
                        how="wmedian", detail=False, masks=None):
    """
    Objetivo de UNA banda cuando los parametros estan del lado de la MASCARA.

    A diferencia de `band_objective`, aca hay que REHACER LA RECURSION en cada
    evaluacion (los parametros cambian que se acumula). El costo se acota
    corriendo la recursion SOLO sobre los bins de la banda: la acumulacion es
    separable por bin, asi que una banda de ~10 bins cuesta ~K/10 menos que la
    corrida completa.

    `scenes` necesita, ademas de lo que produce `prepare_scene`, las claves
    "X_stft" (K, T, M) y "mask_raw" (K, T) -- ver `prepare_scene_full`.

    masks : opcional, (mask_s, mask_n) YA calculadas para esta banda. Sirve para
        evaluar el camino ACTUAL (stretch + **4) dentro de la misma maquinaria,
        que es el baseline honesto contra el que hay que comparar.
    """
    Ls, Ls_sinr, Ls_dist, Ws = [], [], [], []
    for i, sc in enumerate(scenes):
        if masks is not None:
            ms, mn = masks[i]
        else:
            ms, mn = masks_from_raw(sc["mask_raw"][bin_idx], *theta)

        Pxx, Pnn = snapshot_scms_masked(sc["X_stft"][bin_idx], ms, mn,
                                        sc["eval_frames"], alpha=sc["alpha"])
        W, _ = parametric_weights(
            Pxx, Pnn, None if sc["Gamma"] is None else sc["Gamma"][bin_idx],
            sc["ref_mic"], nu=nu, gamma=gamma, mu=mu, min_loading=min_loading,
            lambda_floor=lambda_floor, psd_project=psd_project)
        refs = {k: v[bin_idx] for k, v in sc["refs"].items()}
        out = weight_loss(W, sc["Phi_S"][bin_idx], sc["Phi_N"][bin_idx], refs, eta=eta)
        Ls.append(out["L"]); Ls_sinr.append(out["L_sinr"]); Ls_dist.append(out["L_dist"])
        Ws.append(refs["pow_S"])

    if not detail:
        return _aggregate(Ls, how, Ws)
    return {"L": _aggregate(Ls, how, Ws),
            "L_sinr": _aggregate(Ls_sinr, how, Ws),
            "L_dist": _aggregate(Ls_dist, how, Ws)}


def fit_band_mask(scenes, bin_idx, starts=None, maxiter=120, **kw):
    """
    Ajuste de los 4 parametros de mascara de UNA banda, por Nelder-Mead desde
    varios puntos de arranque (el paisaje no tiene por que ser convexo y la
    grilla 4-D seria carisima).

    starts : lista de tuplas (a_s, b_s, a_n, b_n). Default: la identidad y dos
        puntos de contraste alto, uno simetrico y otro con la rama de RUIDO mas
        conservadora que la de voz -- que es la hipotesis que motiva desacoplar
        las ramas.

    Returns dict con theta, L, L_sinr, L_dist y n_eval.
    """
    if starts is None:
        starts = [MASK_THETA0, (4.0, 0.0, 4.0, 0.0), (1.0, 0.0, 2.0, -6.0)]

    n_eval = [0]

    def obj(th):
        n_eval[0] += 1
        v = band_objective_mask(scenes, bin_idx, _clip_theta(th), **kw)
        return 1e6 if not np.isfinite(v) else v

    best = None
    try:
        from scipy.optimize import minimize
        for x0 in starts:
            res = minimize(obj, x0=np.array(x0, dtype=float), method="Nelder-Mead",
                           options={"xatol": 1e-2, "fatol": 1e-3, "maxiter": maxiter})
            if np.isfinite(res.fun) and (best is None or res.fun < best[1]):
                best = (_clip_theta(res.x), float(res.fun))
    except ImportError:                                        # pragma: no cover
        pass

    if best is None:                                           # pragma: no cover
        vals = [(s, obj(s)) for s in starts]
        best = min(vals, key=lambda t: t[1])

    d = band_objective_mask(scenes, bin_idx, best[0], detail=True, **kw)
    return {"theta": best[0], "L": d["L"], "L_sinr": d["L_sinr"],
            "L_dist": d["L_dist"], "n_eval": n_eval[0]}
