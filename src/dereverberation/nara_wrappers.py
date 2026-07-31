
from nara_wpe.wpe import OnlineWPE
from nara_wpe.utils import stft, istft
from nara_wpe.wpe import online_wpe_step, get_power_online, OnlineWPE
from nara_wpe.wpe import wpe # Importamos la versión Batch/Offline
from nara_wpe.utils import stft, istft
import numpy as np
from numba import njit, prange
# Asumo que importas stft, istft, online_wpe_step y get_power de nara_wpe


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
