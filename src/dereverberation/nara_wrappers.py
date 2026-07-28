
from nara_wpe.wpe import OnlineWPE
from nara_wpe.utils import stft, istft
from nara_wpe.wpe import online_wpe_step, get_power_online, OnlineWPE
from nara_wpe.wpe import wpe # Importamos la versión Batch/Offline
from nara_wpe.utils import stft, istft
import numpy as np
import numpy as np
# Asumo que importas stft, istft, online_wpe_step y get_power de nara_wpe

def process_wpe_online(u, taps=5, delay=1, alpha=0.9999, stft_size=256, stft_shift=64):
    """
    Online WPE wrapper (Functional Approach).
    Processes a multichannel time-domain signal frame by frame to simulate 
    online dereverberation. Bypasses the buggy OnlineWPE class state management 
    by handling the Q and G matrices directly.
    """
    # 1. Transform to STFT domain
    Y = stft(u, size=stft_size, shift=stft_shift)
    Y = Y.transpose(1, 2, 0)  # Shape: (frames, bins, channels)
    T, F, M = Y.shape
    
    buffer_target_size = taps + delay + 1
    if T < buffer_target_size:
        print("Warning: Signal is too short for WPE with given taps and delay.")
        return u
        
    # 2. Initialize Q (Inverse Correlation) and G (Filter) matrices manually
    # Q shape: (F, M*taps, M*taps) -> Identity matrices
    Q = np.stack([np.identity(M * taps) for _ in range(F)])
    # G shape: (F, M*taps, M) -> Zeros
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


def process_wpe_online_with_components(u, components, taps=5, delay=1, alpha=0.9999,
                                       stft_size=256, stft_shift=64):
    """Online WPE sobre ``u`` que ademas filtra cada senal en ``components`` con la
    MISMA trayectoria del filtro G estimada desde ``u``.

    El paso online de nara_wpe calcula el frame dereverberado como
    ``pred = Y(t) - G^H . window`` donde G se estima UNICAMENTE de ``u`` (la mezcla).
    Como esa operacion es lineal en la entrada dado G, aplicar el mismo G (frame a
    frame, ANTES de su update) al target y al ruido da una descomposicion EXACTA:
    ``WPE(target) + WPE(ruido) == WPE(mezcla)`` (salvo los primeros taps+delay frames
    que se copian sin procesar, igual que en ``process_wpe_online``).

    Esto es la referencia oracle correcta cuando la mezcla pasa por WPE: el target/
    ruido "tal como quedan en la observacion dereverberada", sin re-estimar un filtro
    propio por componente.

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
        # online_wpe_step NO muta G in place (crea filter_taps_k nuevo), asi que
        # este mismo G sirve para filtrar las componentes de forma consistente.
        G_used = G
        Z_frame, Q, G = online_wpe_step(
            Y_step, power, Q, G_used, alpha=alpha, taps=taps, delay=delay
        )
        Z_list.append(Z_frame)

        # Aplicar el MISMO filtro G_used a cada componente (misma ventana/orden que
        # online_wpe_step para que los indices de G coincidan exactamente).
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