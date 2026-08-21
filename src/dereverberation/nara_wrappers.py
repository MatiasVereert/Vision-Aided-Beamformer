
from nara_wpe.wpe import OnlineWPE
from nara_wpe.utils import stft, istft
from nara_wpe.wpe import online_wpe_step, get_power_online, OnlineWPE
from nara_wpe.wpe import wpe # Importamos la versión Batch/Offline
# Primitivas batcheadas del path offline de nara (v7): construccion del regresor
# apilado y_tilde, potencia inversa (1/lambda), ecuaciones normales R,P y la
# aplicacion del filtro. Se reutilizan tal cual para el BLOCK-ONLINE (Opcion B).
from nara_wpe.wpe import (
    build_y_tilde,
    get_power_inverse,
    get_correlations_v6,
    perform_filter_operation_v5,
    hermite,
    _stable_solve,
)
from nara_wpe.utils import stft, istft
import numpy as np
from numba import njit, prange
# Asumo que importas stft, istft, online_wpe_step y get_power de nara_wpe

# --- Control de threads BLAS para el block-online ---------------------------
# Gancho opcional para limitar threads BLAS alrededor del loop pesado. Por
# defecto None = NO tocar (se midio que en esta maquina el multithreading de
# OpenBLAS ayuda para F=257 bins; forzar 1 thread perjudica). Se deja el gancho
# por si en otra maquina/entorno la contencion invierte el balance.
BLOCK_BLAS_THREADS = None
try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except Exception:  # pragma: no cover
    _threadpool_limits = None

import contextlib


@contextlib.contextmanager
def _blas_threads(n):
    """Limita threads BLAS a n dentro del bloque (no-op si n es None o falta lib)."""
    if n is None or _threadpool_limits is None:
        yield
    else:
        with _threadpool_limits(limits=n):
            yield


# =====================================================================
# CORE RECURSIVO (Numba). Reimplementa online_wpe_step frame-a-frame.
# ---------------------------------------------------------------------
# Estos njit reemplazan el loop Python cuadro-a-cuadro que era el cuello
# de botella. Mantienen la STFT/ISTFT afuera (numpy vectorizado).
#
# Notas de equivalencia numerica con la implementacion de referencia:
#   * window[f, m*taps+k] = Y[t-delay-1-k, f, m]   (mismo reshape/reverse
#     que online_wpe_step, indexado directo del buffer deslizante).
#   * power[f] = media de |Y|^2 sobre canales y sobre los taps+delay+1
#     frames del buffer (== get_power_online).
#   * _stable_positive_inverse: eps = 1e-10 * max(denom) acopla TODOS los
#     bins; el max/maximum son lexicograficos sobre complejos (real, luego
#     imag), replicado abajo bit a bit.
#   * fastmath=False para no reasociar sumas y quedar lo mas cerca posible
#     del numpy original (no bit-identico por orden de reduccion, si a
#     tolerancia estrecha).
# =====================================================================


@njit(cache=True, fastmath=False, parallel=True)
def _wpe_core_njit(Y, taps, delay, alpha):
    """Core recursivo del WPE online para la mezcla. Y: (T, F, M) complex128.

    Devuelve Z (T, F, M) complex128 con los primeros taps+delay frames
    copiados sin procesar (alineacion temporal estricta).

    Los bins de frecuencia son independientes dentro de un frame (cada f solo
    toca Q[f]/G[f]), asi que las dos pasadas por f corren en paralelo (prange).
    Esto NO cambia la numerica: cada f escribe memoria disjunta y la reduccion
    del max (eps) queda serial.
    """
    T, F, M = Y.shape
    K = taps * M
    Z = np.empty((T, F, M), dtype=np.complex128)

    # Bypass de los primeros taps+delay frames.
    for i in range(taps + delay):
        for f in range(F):
            for m in range(M):
                Z[i, f, m] = Y[i, f, m]

    # Estado del filtro: Q (inv. correlacion) = identidad por bin, G = 0.
    Q = np.zeros((F, K, K), dtype=np.complex128)
    for f in range(F):
        for i in range(K):
            Q[f, i, i] = 1.0 + 0.0j
    G = np.zeros((F, K, M), dtype=np.complex128)

    # Buffers preasignados (evita realloc por frame).
    window = np.empty((F, K), dtype=np.complex128)
    nominator = np.empty((F, K), dtype=np.complex128)
    denom = np.empty(F, dtype=np.complex128)
    inv = np.empty(F, dtype=np.complex128)
    kalman = np.empty((F, K), dtype=np.complex128)
    pred = np.empty((F, M), dtype=np.complex128)
    temp = np.empty((F, K), dtype=np.complex128)

    Tp1 = taps + delay + 1

    for t in range(taps + delay, T):
        # ---- Pasada 1: window, power, nominator, denom (por bin) ----
        for f in prange(F):
            # window[f, m*taps+k] = Y[t-delay-1-k, f, m]
            for m in range(M):
                base = m * taps
                for k in range(taps):
                    window[f, base + k] = Y[t - delay - 1 - k, f, m]

            # power[f] = mean_frames( mean_channels(|Y_step|^2) )
            psum = 0.0
            for j in range(Tp1):
                tt = t - (taps + delay) + j
                cs = 0.0
                for m in range(M):
                    v = Y[tt, f, m]
                    cs += v.real * v.real + v.imag * v.imag
                psum += cs / M
            power = psum / Tp1

            # nominator[f,i] = sum_j Q[f,i,j] * window[f,j]
            for i in range(K):
                acc = 0.0 + 0.0j
                for j in range(K):
                    acc += Q[f, i, j] * window[f, j]
                nominator[f, i] = acc

            # denom = alpha*power + sum_i conj(window_i) * nominator_i
            d = complex(alpha * power, 0.0)
            for i in range(K):
                w = window[f, i]
                d += complex(w.real, -w.imag) * nominator[f, i]
            denom[f] = d

        # ---- _stable_positive_inverse sobre denom (F,) ----
        # eps = 1e-10 * max(denom)   (max lexicografico complejo)
        maxd = denom[0]
        for f in range(1, F):
            df = denom[f]
            if df.real > maxd.real or (df.real == maxd.real and df.imag > maxd.imag):
                maxd = df
        eps = 1e-10 * maxd
        if eps.real == 0.0 and eps.imag == 0.0:
            for f in range(F):
                inv[f] = 1.0 + 0.0j
        else:
            for f in range(F):
                df = denom[f]
                # maximum(df, eps): df si df>=eps (lexicografico) si no eps
                if df.real > eps.real or (df.real == eps.real and df.imag >= eps.imag):
                    mval = df
                else:
                    mval = eps
                inv[f] = 1.0 / mval

        # ---- Pasada 2: kalman, pred, updates de Q y G (por bin) ----
        for f in prange(F):
            invf = inv[f]
            for i in range(K):
                kalman[f, i] = nominator[f, i] * invf

            # pred[f,m] = Y[t,f,m] - sum_i conj(G[f,i,m]) * window[f,i]  (G viejo)
            for m in range(M):
                acc = Y[t, f, m]
                for i in range(K):
                    g = G[f, i, m]
                    acc -= complex(g.real, -g.imag) * window[f, i]
                pred[f, m] = acc
                Z[t, f, m] = acc

            # temp[f,m2] = sum_j conj(window_j) * Q[f,j,m2]   (Q viejo)
            for m2 in range(K):
                acc = 0.0 + 0.0j
                for j in range(K):
                    w = window[f, j]
                    acc += complex(w.real, -w.imag) * Q[f, j, m2]
                temp[f, m2] = acc

            # Q update: Q = (Q - kalman (x) temp) / alpha
            for i in range(K):
                ki = kalman[f, i]
                for m2 in range(K):
                    Q[f, i, m2] = (Q[f, i, m2] - ki * temp[f, m2]) / alpha

            # G update: G[f,i,m] += kalman[f,i] * conj(pred[f,m])
            for i in range(K):
                ki = kalman[f, i]
                for m in range(M):
                    p = pred[f, m]
                    G[f, i, m] = G[f, i, m] + ki * complex(p.real, -p.imag)

    return Z


@njit(cache=True, fastmath=False, parallel=True)
def _wpe_core_components_njit(Y, C, taps, delay, alpha):
    """Core recursivo con filtrado de componentes por el MISMO G que la mezcla.

    Y: (T, F, M) mezcla. C: (ncomp, T, F, M) componentes.
    El tramo de la mezcla ejecuta EXACTAMENTE las mismas operaciones que
    ``_wpe_core_njit`` -> Z resultante bit-identico. Cada componente se filtra
    con el G pre-update del frame (misma ventana/orden), garantizando
    WPE(target)+WPE(ruido) == WPE(mezcla) algebraicamente.

    Devuelve (Z (T,F,M), Zc (ncomp,T,F,M)).
    """
    T, F, M = Y.shape
    ncomp = C.shape[0]
    K = taps * M
    Z = np.empty((T, F, M), dtype=np.complex128)
    Zc = np.empty((ncomp, T, F, M), dtype=np.complex128)

    # Bypass de los primeros taps+delay frames (mezcla y componentes).
    for i in range(taps + delay):
        for f in range(F):
            for m in range(M):
                Z[i, f, m] = Y[i, f, m]
                for c in range(ncomp):
                    Zc[c, i, f, m] = C[c, i, f, m]

    Q = np.zeros((F, K, K), dtype=np.complex128)
    for f in range(F):
        for i in range(K):
            Q[f, i, i] = 1.0 + 0.0j
    G = np.zeros((F, K, M), dtype=np.complex128)

    window = np.empty((F, K), dtype=np.complex128)
    window_c = np.empty((F, K), dtype=np.complex128)
    nominator = np.empty((F, K), dtype=np.complex128)
    denom = np.empty(F, dtype=np.complex128)
    inv = np.empty(F, dtype=np.complex128)
    kalman = np.empty((F, K), dtype=np.complex128)
    pred = np.empty((F, M), dtype=np.complex128)
    temp = np.empty((F, K), dtype=np.complex128)

    Tp1 = taps + delay + 1

    for t in range(taps + delay, T):
        # ---- Pasada 1 (identica a _wpe_core_njit): mezcla ----
        for f in prange(F):
            for m in range(M):
                base = m * taps
                for k in range(taps):
                    window[f, base + k] = Y[t - delay - 1 - k, f, m]

            psum = 0.0
            for j in range(Tp1):
                tt = t - (taps + delay) + j
                cs = 0.0
                for m in range(M):
                    v = Y[tt, f, m]
                    cs += v.real * v.real + v.imag * v.imag
                psum += cs / M
            power = psum / Tp1

            for i in range(K):
                acc = 0.0 + 0.0j
                for j in range(K):
                    acc += Q[f, i, j] * window[f, j]
                nominator[f, i] = acc

            d = complex(alpha * power, 0.0)
            for i in range(K):
                w = window[f, i]
                d += complex(w.real, -w.imag) * nominator[f, i]
            denom[f] = d

        maxd = denom[0]
        for f in range(1, F):
            df = denom[f]
            if df.real > maxd.real or (df.real == maxd.real and df.imag > maxd.imag):
                maxd = df
        eps = 1e-10 * maxd
        if eps.real == 0.0 and eps.imag == 0.0:
            for f in range(F):
                inv[f] = 1.0 + 0.0j
        else:
            for f in range(F):
                df = denom[f]
                if df.real > eps.real or (df.real == eps.real and df.imag >= eps.imag):
                    mval = df
                else:
                    mval = eps
                inv[f] = 1.0 / mval

        # ---- Pasada 2: mezcla + componentes (G pre-update por bin) ----
        for f in prange(F):
            invf = inv[f]
            for i in range(K):
                kalman[f, i] = nominator[f, i] * invf

            for m in range(M):
                acc = Y[t, f, m]
                for i in range(K):
                    g = G[f, i, m]
                    acc -= complex(g.real, -g.imag) * window[f, i]
                pred[f, m] = acc
                Z[t, f, m] = acc

            # Componentes: mismo G (pre-update) y misma construccion de ventana.
            for c in range(ncomp):
                for m in range(M):
                    base = m * taps
                    for k in range(taps):
                        window_c[f, base + k] = C[c, t - delay - 1 - k, f, m]
                for m in range(M):
                    acc = C[c, t, f, m]
                    for i in range(K):
                        g = G[f, i, m]
                        acc -= complex(g.real, -g.imag) * window_c[f, i]
                    Zc[c, t, f, m] = acc

            # temp y updates de Q, G (despues de usar G viejo para las comps).
            for m2 in range(K):
                acc = 0.0 + 0.0j
                for j in range(K):
                    w = window[f, j]
                    acc += complex(w.real, -w.imag) * Q[f, j, m2]
                temp[f, m2] = acc

            for i in range(K):
                ki = kalman[f, i]
                for m2 in range(K):
                    Q[f, i, m2] = (Q[f, i, m2] - ki * temp[f, m2]) / alpha

            for i in range(K):
                ki = kalman[f, i]
                for m in range(M):
                    p = pred[f, m]
                    G[f, i, m] = G[f, i, m] + ki * complex(p.real, -p.imag)

    return Z, Zc


def process_wpe_online(u, taps=5, delay=1, alpha=0.9999, stft_size=256, stft_shift=64):
    """
    Online WPE wrapper (Functional Approach, acelerado con Numba).
    Processes a multichannel time-domain signal frame by frame to simulate
    online dereverberation. STFT/ISTFT en numpy; el loop recursivo del filtro
    (Q, G) esta jiteado en ``_wpe_core_njit`` (equivalente numerico a la
    implementacion de referencia ``_process_wpe_online_ref``).
    """
    # 1. STFT -> (frames, bins, channels)
    Y = stft(u, size=stft_size, shift=stft_shift).transpose(1, 2, 0)
    T, F, M = Y.shape

    if T < taps + delay + 1:
        print("Warning: Signal is too short for WPE with given taps and delay.")
        return u

    # 2. Core recursivo jiteado (buffer contiguo complex128).
    Y = np.ascontiguousarray(Y, dtype=np.complex128)
    Z_stacked = _wpe_core_njit(Y, int(taps), int(delay), float(alpha))

    # 3. ISTFT -> (channels, frames, bins) -> tiempo, recortado a la entrada.
    Z_out = Z_stacked.transpose(2, 0, 1)
    z_time = istft(Z_out, size=stft_size, shift=stft_shift)
    z_time = z_time[:, :u.shape[1]]
    return z_time


def process_wpe_online_with_components(u, components, taps=5, delay=1, alpha=0.9999,
                                       stft_size=256, stft_shift=64):
    """Online WPE sobre ``u`` que ademas filtra cada senal en ``components`` con la
    MISMA trayectoria del filtro G estimada desde ``u``.

    El paso online calcula el frame dereverberado como
    ``pred = Y(t) - G^H . window`` donde G se estima UNICAMENTE de ``u`` (la mezcla).
    Como esa operacion es lineal en la entrada dado G, aplicar el mismo G (frame a
    frame, ANTES de su update) al target y al ruido da una descomposicion EXACTA:
    ``WPE(target) + WPE(ruido) == WPE(mezcla)`` (salvo los primeros taps+delay frames
    que se copian sin procesar, igual que en ``process_wpe_online``).

    Version acelerada con Numba (``_wpe_core_components_njit``); numericamente
    equivalente a la referencia ``_process_wpe_online_with_components_ref``.

    Parameters
    ----------
    u : (M, N) real            -- mezcla multicanal en el dominio del tiempo.
    components : list[(M, N)]   -- senales (target, ruido, ...) a filtrar con el G de u.
    taps, delay, alpha, stft_size, stft_shift : parametros WPE/STFT (mismos que la mezcla).

    Returns
    -------
    (z_u (M, N), [z_comp (M, N), ...])   -- mezcla y componentes dereverberadas.
    """
    # 1. STFT de la mezcla y de cada componente (mismos parametros -> mismos T, F)
    Y = stft(u, size=stft_size, shift=stft_shift).transpose(1, 2, 0)  # (T, F, M)
    Cs = [stft(c, size=stft_size, shift=stft_shift).transpose(1, 2, 0) for c in components]
    T, F, M = Y.shape
    T = min([T] + [C.shape[0] for C in Cs])
    Y = Y[:T]
    Cs = [C[:T] for C in Cs]

    if T < taps + delay + 1:
        print("Warning: Signal is too short for WPE with given taps and delay.")
        return u, list(components)

    # Sin componentes: reduce al caso mezcla-sola.
    if len(Cs) == 0:
        z_u = process_wpe_online(u, taps=taps, delay=delay, alpha=alpha,
                                 stft_size=stft_size, stft_shift=stft_shift)
        return z_u, []

    # 2. Core recursivo jiteado (mezcla + componentes en un solo pase).
    Y = np.ascontiguousarray(Y, dtype=np.complex128)
    C = np.ascontiguousarray(np.stack(Cs), dtype=np.complex128)  # (ncomp, T, F, M)
    Z_stacked, Zc_stacked = _wpe_core_components_njit(
        Y, C, int(taps), int(delay), float(alpha)
    )

    # 3. ISTFT (mezcla y cada componente), recortado a la entrada.
    def _to_time(Z_arr):
        z_time = istft(Z_arr.transpose(2, 0, 1), size=stft_size, shift=stft_shift)
        return z_time[:, :u.shape[1]]

    z_u = _to_time(Z_stacked)
    z_components = [_to_time(Zc_stacked[c]) for c in range(Zc_stacked.shape[0])]
    return z_u, z_components


# =====================================================================
# IMPLEMENTACION DE REFERENCIA (Python puro, loop por frame con nara_wpe).
# Se conserva para el test de equivalencia numerica. NO usar en produccion
# (es el cuello de botella que motivo la version Numba de arriba).
# =====================================================================


def _process_wpe_online_ref(u, taps=5, delay=1, alpha=0.9999, stft_size=256, stft_shift=64):
    """Referencia original de ``process_wpe_online`` (loop Python + nara_wpe)."""
    # 1. Transform to STFT domain
    Y = stft(u, size=stft_size, shift=stft_shift)
    Y = Y.transpose(1, 2, 0)  # Shape: (frames, bins, channels)
    T, F, M = Y.shape

    buffer_target_size = taps + delay + 1
    if T < buffer_target_size:
        print("Warning: Signal is too short for WPE with given taps and delay.")
        return u

    # 2. Initialize Q (Inverse Correlation) and G (Filter) matrices manually
    Q = np.stack([np.identity(M * taps) for _ in range(F)])
    G = np.zeros((F, M * taps, M))

    Z_list = []

    # 3. Bypass the first unprocessed frames to maintain strict temporal alignment
    for i in range(taps + delay):
        Z_list.append(Y[i, :, :])

    # Initialize the sliding buffer with the first history chunk
    buffer = list(Y[:taps + delay, :, :])

    # 4. Process frame by frame
    for t in range(taps + delay, T):
        buffer.append(Y[t, :, :])

        # Convert buffer to numpy array: shape (buffer_target_size, F, M)
        Y_step = np.array(buffer)

        # Compute power. get_power_online expects (bins, channels, frames)
        power = get_power_online(Y_step.transpose(1, 2, 0))

        # Perform functional online dereverberation step
        Z_frame, Q, G = online_wpe_step(
            Y_step,
            power,
            Q,
            G,
            alpha=alpha,
            taps=taps,
            delay=delay
        )

        Z_list.append(Z_frame)

        # Discard the oldest frame to slide the window forward
        buffer.pop(0)

    # 5. Reconstruct the time-domain signal
    Z_stacked = np.stack(Z_list)

    # Transpose back to (channels, frames, frequency_bins) for istft
    Z_out = Z_stacked.transpose(2, 0, 1)

    # Inverse STFT to get the time-domain audio
    z_time = istft(Z_out, size=stft_size, shift=stft_shift)

    # Ensure the output length exactly matches the original input length
    z_time = z_time[:, :u.shape[1]]

    return z_time


def _process_wpe_online_with_components_ref(u, components, taps=5, delay=1, alpha=0.9999,
                                            stft_size=256, stft_shift=64):
    """Referencia original de ``process_wpe_online_with_components`` (loop Python)."""
    # 1. STFT de la mezcla y de cada componente (mismos parametros -> mismos T, F)
    Y = stft(u, size=stft_size, shift=stft_shift).transpose(1, 2, 0)  # (T, F, M)
    Cs = [stft(c, size=stft_size, shift=stft_shift).transpose(1, 2, 0) for c in components]
    T, F, M = Y.shape
    T = min([T] + [C.shape[0] for C in Cs])
    Y = Y[:T]
    Cs = [C[:T] for C in Cs]

    buffer_target_size = taps + delay + 1
    if T < buffer_target_size:
        print("Warning: Signal is too short for WPE with given taps and delay.")
        return u, list(components)

    # 2. Estado del filtro (identico a process_wpe_online)
    Q = np.stack([np.identity(M * taps) for _ in range(F)])
    G = np.zeros((F, M * taps, M))

    Z_list = []
    Zc_lists = [[] for _ in Cs]

    # 3. Bypass de los primeros taps+delay frames (alineacion temporal estricta)
    for i in range(taps + delay):
        Z_list.append(Y[i, :, :])
        for k, C in enumerate(Cs):
            Zc_lists[k].append(C[i, :, :])

    buffer = list(Y[:taps + delay, :, :])
    buffers_c = [list(C[:taps + delay, :, :]) for C in Cs]

    # 4. Loop frame a frame
    for t in range(taps + delay, T):
        buffer.append(Y[t, :, :])
        for k, C in enumerate(Cs):
            buffers_c[k].append(C[t, :, :])

        Y_step = np.array(buffer)
        power = get_power_online(Y_step.transpose(1, 2, 0))

        # G que se usa para el pred de la mezcla en este frame (pre-update).
        G_used = G
        Z_frame, Q, G = online_wpe_step(
            Y_step, power, Q, G_used, alpha=alpha, taps=taps, delay=delay
        )
        Z_list.append(Z_frame)

        # Aplicar el MISMO filtro G_used a cada componente.
        for k in range(len(Cs)):
            C_step = np.array(buffers_c[k])
            window = C_step[:-delay - 1][::-1].transpose(1, 2, 0).reshape((F, taps * M))
            pred_c = C_step[-1] - np.einsum('fid,fi->fd', np.conjugate(G_used), window)
            Zc_lists[k].append(pred_c)
            buffers_c[k].pop(0)

        buffer.pop(0)

    # 5. Reconstruccion al dominio del tiempo (istft) de la mezcla y cada componente
    def _to_time(Z_list_):
        Z_out = np.stack(Z_list_).transpose(2, 0, 1)  # (M, T, F)
        z_time = istft(Z_out, size=stft_size, shift=stft_shift)
        return z_time[:, :u.shape[1]]

    z_u = _to_time(Z_list)
    z_components = [_to_time(zc) for zc in Zc_lists]
    return z_u, z_components


# =====================================================================
# BLOCK-ONLINE WPE (Opcion B: re-solve por bloque via Cholesky).
# ---------------------------------------------------------------------
# En vez del update recursivo por-frame (RLS de arriba), se estima el filtro
# G RESOLVIENDO las ecuaciones normales sobre una ventana TRAILING de L frames
# PASADOS, y ese G se aplica de forma CAUSAL a los frames siguientes, congelado
# hasta el proximo re-solve (cada ``block_shift`` frames). Emula la arquitectura
# FPGA "block online con inversion por Cholesky":
#
#     R = sum_{t in ventana} (1/lambda_t) . y_tilde_t . y_tilde_t^H     (KM x KM, Hermitiana)
#     P = sum_{t in ventana} (1/lambda_t) . y_tilde_t . y_t^H            (KM x M)
#     R = L L^H  (Cholesky) ;  resolver  R G = P
#     X_t = Y_t - G^H . y_tilde_t     (aplicacion causal, G congelado)
#
# R,P y el solve se hacen con las MISMAS primitivas del path offline de nara
# (get_correlations_v6 / perform_filter_operation_v5), asi que la numerica del
# filtro coincide con el WPE batch de nara pero restringido a la ventana y
# refrescado periodicamente. La latencia de salida es ~1 frame (aplicacion
# causal); el bloque NO se espera completo.
#
# Notas de fidelidad a la Opcion B del audit FPGA:
#   * El solver por defecto es Cholesky EXPLICITO (np.linalg.cholesky batcheado
#     sobre F + sustitucion adelante/atras vectorizada) -> hace literal el
#     "R = L L^H". Expone ``reg`` (carga diagonal relativa) que garantiza R
#     definida-positiva y condiciona el sistema; es el knob que despues importa
#     en punto fijo. Fallback a _stable_solve (LU/lstsq) si Cholesky falla.
#   * En float64 el resultado del Cholesky y del solve LU son identicos: la
#     eleccion de solver NO cambia el audio dereverberado en la simulacion; solo
#     importa para el modelo de punto fijo posterior.
# =====================================================================


def _block_load(R, reg, N):
    """Hermitianiza R y le suma carga diagonal relativa ``reg*mean(diag(R))``.

    Garantiza definida-positividad (para el Cholesky) y condiciona R -equivalente
    al diagonal loading del camino FPGA-. Con reg pequeno el efecto en float es
    despreciable; su rol es robustez numerica.
    """
    R = 0.5 * (R + hermite(R))
    diag_mean = np.einsum('fii->f', R).real / N
    load = (reg * np.maximum(diag_mean, 1e-30))[:, None, None]
    return R + load * np.eye(N, dtype=R.dtype)[None]


def _block_cholesky_solve(R, P, reg=1e-6, explicit=False):
    """Resuelve R G = P por bin con carga diagonal. R:(F,N,N) Herm PD, P:(F,N,M).

    explicit=False (default, RAPIDO): np.linalg.solve batcheado (LAPACK). En float
    es IDENTICO al Cholesky (R es PD por construccion tras la carga) y ~10x mas
    rapido que la sustitucion explicita en un loop de muchas ventanas.

    explicit=True (FIDELIDAD FPGA / debug): factoriza R = L L^H (Cholesky) y hace
    sustitucion adelante/atras vectorizada -> hace literal el camino de hardware.
    """
    F, N, _ = R.shape
    R = _block_load(R, reg, N)

    if not explicit:
        try:
            return np.linalg.solve(R, P)
        except np.linalg.LinAlgError:
            return _stable_solve(R, P)

    # --- Camino explicito: R = L L^H + sustitucion (mas lento) ---
    try:
        L = np.linalg.cholesky(R)                        # (F, N, N) triangular inferior
    except np.linalg.LinAlgError:
        return _stable_solve(R, P)

    # Sustitucion hacia adelante: L z = P   (L inferior)
    z = np.empty_like(P)
    for i in range(N):
        if i == 0:
            s = P[:, 0, :]
        else:
            s = P[:, i, :] - np.einsum('fj,fjm->fm', L[:, i, :i], z[:, :i, :])
        z[:, i, :] = s / L[:, i, i][:, None]

    # Sustitucion hacia atras: L^H G = z   (L^H superior; L^H[i,k]=conj(L[k,i]))
    G = np.empty_like(z)
    for i in range(N - 1, -1, -1):
        if i == N - 1:
            s = z[:, i, :]
        else:
            s = z[:, i, :] - np.einsum('fj,fjm->fm',
                                       np.conjugate(L[:, i + 1:, i]), G[:, i + 1:, :])
        G[:, i, :] = s / np.conjugate(L[:, i, i])[:, None]
    return G


def _estimate_block_filter(Y_win, Y_tilde_win, iterations, reg, solver, G_init=None):
    """Estima el filtro WPE G sobre una ventana (F, D, Tw).

    Corre ``iterations`` pasos tipo-offline sobre la ventana: en cada uno
    re-estima 1/lambda desde el X actual, arma R,P (get_correlations_v6) y
    resuelve R G = P. Devuelve G: (F, taps*D, D). ``G_init`` (opcional) hace
    warm-start de la potencia con el filtro del bloque anterior.

    solver: "cholesky"/"lu" -> solve rapido LAPACK (mismo resultado en float);
            "cholesky_explicit" -> Cholesky explicito + sustitucion (fidelidad FPGA).
    """
    explicit = (solver == "cholesky_explicit")
    # WARM-START: si viene G_init (filtro del bloque anterior), la potencia de la
    # 1ra iteracion se estima de la ventana YA dereverberada con ese filtro, en vez
    # del reverberante crudo. En escena cuasi-estacionaria esto acerca iters=1 a la
    # calidad de iters altos (la iteracion "continua" desde el bloque previo).
    if G_init is not None:
        X = perform_filter_operation_v5(Y=Y_win, Y_tilde=Y_tilde_win, filter_matrix=G_init)
    else:
        X = Y_win
    G = None
    for it in range(iterations):
        inverse_power = get_power_inverse(X)                     # (F, Tw)
        R, P = get_correlations_v6(Y_win, Y_tilde_win, inverse_power)
        G = _block_cholesky_solve(R, P, reg=reg, explicit=explicit)
        # El filtrado de la ventana solo sirve para re-estimar la potencia de la
        # SIGUIENTE iteracion. En la ultima iteracion X se descarta (devolvemos G),
        # asi que lo omitimos (ahorra ~25% del costo por-resolve, todo si iters=1).
        if it < iterations - 1:
            X = perform_filter_operation_v5(Y=Y_win, Y_tilde=Y_tilde_win, filter_matrix=G)
    return G


def _block_filters(Y, Y_tilde, taps, delay, L, block_shift, iterations, reg, solver,
                   warm_start=False):
    """Scheduling de la Opcion B: lista de bloques (t_r, hi, G) sobre el eje T.

    Para cada ancla ``t_r`` (en ``taps+delay, +block_shift, ...``) estima G a
    partir de la ventana trailing ``[max(0,t_r-L), t_r)`` (SOLO de la mezcla Y) y
    lo asocia al rango de salida ``[t_r, hi)``. Si la ventana todavia es mas corta
    que ``min_window`` (arranque en frio) devuelve ``G=None`` para ese rango ->
    bypass. El filtro se congela (reusa el ultimo G) hasta el proximo re-solve.

    warm_start: si True, la estimacion de cada bloque arranca la potencia con el
    filtro G del bloque ANTERIOR (continua la iteracion en vez de arrancar en
    frio del reverberante). Barato y suele acercar iters=1 a iters altos.

    Se factoriza aparte para que la version mezcla-sola y la version con
    componentes compartan EXACTAMENTE el mismo G por bloque (garantiza
    WPE(target)+WPE(ruido)==WPE(mezcla), igual que el par online).
    """
    F, D, T = Y.shape
    warmup = taps + delay
    min_window = max(warmup + 8, 4 * taps)
    blocks = []
    G_current = None
    with _blas_threads(BLOCK_BLAS_THREADS):
        for t_r in range(warmup, T, block_shift):
            lo = max(0, t_r - L)
            if (t_r - lo) >= min_window:
                G_current = _estimate_block_filter(
                    Y[:, :, lo:t_r], Y_tilde[:, :, lo:t_r],
                    iterations=iterations, reg=reg, solver=solver,
                    G_init=(G_current if warm_start else None),
                )
            hi = min(t_r + block_shift, T)
            blocks.append((t_r, hi, G_current))
    return blocks


def _block_filters_sliding(Y, Y_tilde, taps, delay, L, block_shift, reg, solver):
    """Version RAPIDA del scheduling: R,P por VENTANA DESLIZANTE con updates
    incrementales O(block_shift) en vez de recomputar la correlacion sobre L
    frames en cada ancla. Clave para barrer block_shift chico (a block_shift=2 y
    L=256 evita ~128x de trabajo redundante en la correlacion).

    Aproxima ``_block_filters`` con iterations=1 y potencia GLOBAL (una sola
    estimacion de 1/lambda sobre toda la senal, en vez de por-ventana). La unica
    diferencia con iterations=1 exacto es la normalizacion eps de la potencia
    (despreciable, ~1e-6). NO soporta iterations>1 (el reweight por iteracion
    rompe el sliding).

    Deriva incremental en float64: R acumula terminos O(1) sobre ~L frames, el
    add/sub por ancla introduce error relativo ~1e-16/op -> despreciable sobre
    miles de anclas.
    """
    F, D, T = Y.shape
    KM = taps * D
    warmup = taps + delay
    min_window = max(warmup + 8, 4 * taps)
    explicit = (solver == "cholesky_explicit")

    ip = get_power_inverse(Y)                     # (F, T) potencia inversa GLOBAL

    def contrib(a, b):
        """Suma de contribuciones a R,P sobre frames [a, b) (b>a)."""
        W = Y_tilde[:, :, a:b] * ip[:, None, a:b]           # (F, KM, b-a)
        R_ab = np.matmul(W, hermite(Y_tilde[:, :, a:b]))    # (F, KM, KM)
        P_ab = np.matmul(W, hermite(Y[:, :, a:b]))          # (F, KM, D)
        return R_ab, P_ab

    R = np.zeros((F, KM, KM), dtype=Y.dtype)
    P = np.zeros((F, KM, D), dtype=Y.dtype)
    blocks = []
    G_current = None
    prev_lo = prev_hi = 0
    with _blas_threads(BLOCK_BLAS_THREADS):
        for t_r in range(warmup, T, block_shift):
            lo = max(0, t_r - L)
            # Deslizar la ventana: sumar frames nuevos [prev_hi, t_r), restar
            # los que salieron [prev_lo, lo).
            if t_r > prev_hi:
                dR, dP = contrib(prev_hi, t_r); R += dR; P += dP
            if lo > prev_lo:
                dR, dP = contrib(prev_lo, lo); R -= dR; P -= dP
            prev_lo, prev_hi = lo, t_r
            if (t_r - lo) >= min_window:
                G_current = _block_cholesky_solve(R, P, reg=reg, explicit=explicit)
            hi = min(t_r + block_shift, T)
            blocks.append((t_r, hi, G_current))
    return blocks


def block_wpe_warmup(taps, delay, L, block_shift, stft_shift):
    """Frontera de WARMUP del block-online (Opcion B).

    Antes de esta frontera el filtro G esta en frio (bypass, sin ventana) o
    entrenado con ventana PARCIAL (transitorio, todavia < L frames). Esas
    muestras NO son representativas del regimen de la Opcion B y DEBEN
    descartarse al medir metricas. La frontera marca el primer frame cuya
    estimacion uso una ventana COMPLETA de L frames.

    Debe coincidir con el scheduling interno de ``process_wpe_block_online``:
    las anclas de re-solve caen en ``warmup, warmup+block_shift, ...`` con
    ``warmup = taps+delay``, y la ventana en el ancla ``t_r`` tiene
    ``min(t_r, L)`` frames -> llena cuando ``t_r >= L``.

    Returns
    -------
    (warm_frame, warm_sample) : primer frame con ventana llena y su muestra
    temporal aproximada (``warm_frame * stft_shift``). Recorta ``[warm_sample:]``
    de la salida (y de la referencia) antes de medir.
    """
    import math
    warmup = taps + delay
    if L <= warmup:
        warm_frame = warmup
    else:
        k = math.ceil((L - warmup) / block_shift)
        warm_frame = warmup + k * block_shift
    return warm_frame, warm_frame * stft_shift


def _dispatch_block_filters(Y, Y_tilde, taps, delay, L, block_shift,
                            iterations, reg, solver, mode, warm_start=False):
    """Elige el scheduler: 'resolve' (exacto, iterations>=1) o 'sliding' (rapido,
    iterations=1 con potencia global e updates incrementales de R,P)."""
    if mode == "sliding":
        # sliding usa potencia global (iters=1); el warm-start por-bloque no aplica.
        return _block_filters_sliding(Y, Y_tilde, taps, delay, L, block_shift,
                                      reg, solver)
    return _block_filters(Y, Y_tilde, taps, delay, L, block_shift,
                          iterations, reg, solver, warm_start=warm_start)


def process_wpe_block_online(u, taps=5, delay=1, L=256, block_shift=32,
                             iterations=3, reg=1e-6, solver="cholesky",
                             stft_size=512, stft_shift=128, return_warmup=False,
                             mode="resolve", warm_start=False):
    """Block-online WPE (Opcion B) sobre senal multicanal en tiempo (M, N).

    Estima G resolviendo las ecuaciones normales sobre una ventana trailing de
    ``L`` frames PASADOS, refresca cada ``block_shift`` frames, y aplica G de
    forma CAUSAL (X_t = Y_t - G^H y_tilde_t). Los primeros frames (hasta juntar
    ventana minima) se copian sin procesar (arranque en frio, como el RLS).

    Parameters
    ----------
    u : (M, N) real            -- mezcla multicanal en el dominio del tiempo.
    taps, delay                -- orden del filtro de prediccion y retardo (guard).
    L : int                    -- largo de la ventana trailing de estadistica [frames].
    block_shift : int          -- cada cuantos frames se re-estima G [frames].
    iterations : int           -- iteraciones tipo-offline por re-solve (>=1; solo mode='resolve').
    reg : float                -- carga diagonal relativa para el Cholesky.
    solver : {"cholesky","lu","cholesky_explicit"} -- solve LAPACK rapido vs Cholesky explicito (fidelidad FPGA).
    stft_size, stft_shift      -- parametros STFT (nara).
    return_warmup : bool       -- si True, devuelve tambien la muestra de warmup.
    mode : {"resolve","sliding"} -- 'resolve' recomputa la correlacion sobre L en
                                  cada ancla (exacto, permite iterations>1).
                                  'sliding' mantiene R,P por ventana deslizante
                                  con updates O(block_shift) -> MUCHO mas rapido
                                  para block_shift chico; equivale a iterations=1
                                  con potencia global (aprox ~1e-6).

    Returns
    -------
    z_time : (M, N) real       -- mezcla dereverberada, recortada a la entrada.
    warmup_sample : int        -- (solo si return_warmup) muestra en regimen.
    """
    if u.ndim == 1:
        u = u[np.newaxis, :]

    # STFT nara: (M, T, F) -> convencion batcheada (F, D=M, T).
    Y_full = stft(u, size=stft_size, shift=stft_shift)        # (M, T, F)
    Y = np.ascontiguousarray(Y_full.transpose(2, 0, 1))       # (F, M, T)
    F, D, T = Y.shape

    warmup = taps + delay
    if T < warmup + 1:
        print("Warning: Signal is too short for block WPE with given taps and delay.")
        return u

    # Regresor apilado y_tilde (F, taps*M, T): y_tilde[:, :, t] = historia causal.
    Y_tilde = build_y_tilde(Y, taps, delay)

    X = Y.copy()   # salida en STFT; arranca como copia (bypass en frio).

    # Bloques (t_r, hi, G) con G estimado sobre la ventana trailing de la mezcla.
    for t_r, hi, G in _dispatch_block_filters(Y, Y_tilde, taps, delay, L, block_shift,
                                              iterations, reg, solver, mode, warm_start):
        if G is not None:
            X[:, :, t_r:hi] = perform_filter_operation_v5(
                Y=Y[:, :, t_r:hi], Y_tilde=Y_tilde[:, :, t_r:hi], filter_matrix=G,
            )
        # si G es None (todavia sin ventana): X ya es copia de Y (bypass).

    # (F, M, T) -> (M, T, F) para istft, recortado a la entrada.
    z_time = istft(X.transpose(1, 2, 0), size=stft_size, shift=stft_shift)
    z_time = z_time[:, :u.shape[1]]
    if return_warmup:
        _, warmup_sample = block_wpe_warmup(taps, delay, L, block_shift, stft_shift)
        return z_time, min(warmup_sample, z_time.shape[1])
    return z_time


def process_wpe_block_online_with_components(u, components, taps=5, delay=1,
                                             L=256, block_shift=32, iterations=3,
                                             reg=1e-6, solver="cholesky",
                                             stft_size=512, stft_shift=128,
                                             mode="resolve", warm_start=False):
    """Block-online WPE sobre ``u`` que ademas filtra cada senal de ``components``
    con el MISMO filtro G por bloque estimado desde ``u`` (la mezcla).

    Analogo block de ``process_wpe_online_with_components``. Como el filtro se
    congela por bloque y su aplicacion (X = Y - G^H y_tilde) es LINEAL en la
    entrada dado G, aplicar el mismo G (por bloque) al target y al ruido da una
    descomposicion EXACTA: ``WPE(target)+WPE(ruido) == WPE(mezcla)`` (incluido el
    arranque en frio, que copia sin procesar en mezcla y componentes por igual).
    z_u es IDENTICO a ``process_wpe_block_online(u, ...)``.

    Parameters
    ----------
    u : (M, N) real            -- mezcla multicanal en el dominio del tiempo.
    components : list[(M, N)]   -- senales (target, ruido, ...) a filtrar con el G de u.
    (resto de parametros: idem process_wpe_block_online)

    Returns
    -------
    (z_u (M, N), [z_comp (M, N), ...])
    """
    if u.ndim == 1:
        u = u[np.newaxis, :]

    # STFT mezcla y componentes -> convencion batcheada (F, D, T).
    Y = np.ascontiguousarray(stft(u, size=stft_size, shift=stft_shift).transpose(2, 0, 1))
    Cs = [np.ascontiguousarray(stft(c, size=stft_size, shift=stft_shift).transpose(2, 0, 1))
          for c in components]
    F, D, T = Y.shape
    T = min([T] + [C.shape[2] for C in Cs])
    Y = Y[:, :, :T]
    Cs = [C[:, :, :T] for C in Cs]

    warmup = taps + delay
    if T < warmup + 1:
        print("Warning: Signal is too short for block WPE with given taps and delay.")
        return u, list(components)

    if len(Cs) == 0:
        z_u = process_wpe_block_online(u, taps=taps, delay=delay, L=L,
                                       block_shift=block_shift, iterations=iterations,
                                       reg=reg, solver=solver, stft_size=stft_size,
                                       stft_shift=stft_shift, mode=mode, warm_start=warm_start)
        return z_u, []

    # Regresores apilados (mezcla + cada componente), mismos taps/delay.
    Y_tilde = build_y_tilde(Y, taps, delay)
    Cs_tilde = [build_y_tilde(C, taps, delay) for C in Cs]

    X = Y.copy()
    Xc = [C.copy() for C in Cs]

    # G se estima SOLO de la mezcla (Y); se aplica congelado a mezcla y componentes.
    for t_r, hi, G in _dispatch_block_filters(Y, Y_tilde, taps, delay, L, block_shift,
                                              iterations, reg, solver, mode, warm_start):
        if G is None:
            continue
        X[:, :, t_r:hi] = perform_filter_operation_v5(
            Y=Y[:, :, t_r:hi], Y_tilde=Y_tilde[:, :, t_r:hi], filter_matrix=G)
        for k in range(len(Cs)):
            Xc[k][:, :, t_r:hi] = perform_filter_operation_v5(
                Y=Cs[k][:, :, t_r:hi], Y_tilde=Cs_tilde[k][:, :, t_r:hi], filter_matrix=G)

    def _to_time(Xarr):
        z = istft(Xarr.transpose(1, 2, 0), size=stft_size, shift=stft_shift)
        return z[:, :u.shape[1]]

    z_u = _to_time(X)
    z_components = [_to_time(Xc[k]) for k in range(len(Xc))]
    return z_u, z_components
