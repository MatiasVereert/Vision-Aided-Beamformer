"""
oracle_masks.py
===============
Mascaras ORACLE (ideales) analogas a las de dtln_masks.py, pero calculadas a
partir de las señales de referencia LIMPIAS (target y ruido/interferencia) en
lugar del modelo neuronal DTLN. Sirven como cota superior / referencia
agnostica al modelo para comparar Oracle vs DTLN dentro del mismo pipeline
mask-based (Souden MVDR, etc.).

Definicion (Ideal Ratio Mask en dominio de potencia == ganancia tipo Wiener),
que es la version SUAVE estandar en la literatura de mask-based beamforming
(Heymann 2016, Erdogan 2016, Boeddeker 2018):

    M_s(f,t) = |S(f,t)|^2 / (|S(f,t)|^2 + |N(f,t)|^2)
    M_n(f,t) = 1 - M_s(f,t)

donde S es la señal target (voz) de referencia y N el ruido+interferencia de
referencia, ambos evaluados en el microfono de referencia. Las mascaras se
mantienen SUAVES (sin exponente de realce): a diferencia de las mascaras DTLN
--que se elevan a `** 4` para agudizar la transicion voz/ruido-- aca la mascara
ideal se define continua/lineal como en la literatura. El parametro
`sharpen_exp` queda expuesto para uso futuro y por defecto vale 1.0 (sin efecto),
replicando el rol del exponente de las mascaras DTLN.

El "framing" (block_len, block_shift, rfft rectangular por bloques) es IDENTICO
al de get_dtln_masks, de modo que la unica diferencia entre una corrida Oracle y
una DTLN sea el valor de la mascara y NO la alineacion temporal de los frames.
"""

import numpy as np


def _stft_mag_blocks(audio_mono, block_len, block_shift):
    """
    STFT por bloques con rfft rectangular, usando exactamente el mismo esquema de
    buffer deslizante que get_dtln_masks. Devuelve la magnitud |X| con forma
    (block_len // 2 + 1, num_blocks).
    """
    samples = audio_mono.shape[0]
    num_blocks = (samples - (block_len - block_shift)) // block_shift

    mag = np.zeros((block_len // 2 + 1, num_blocks), dtype=np.float32)
    in_buffer = np.zeros((block_len,), dtype=np.float32)

    for idx in range(num_blocks):
        # Desplazar buffer y cargar el nuevo hop de muestras
        in_buffer[:-block_shift] = in_buffer[block_shift:]
        start_idx = idx * block_shift
        in_buffer[-block_shift:] = audio_mono[start_idx: start_idx + block_shift]

        mag[:, idx] = np.abs(np.fft.rfft(in_buffer))

    return mag


def _select_ref(signal, ref_mic):
    """
    Devuelve el canal de referencia como 1D. Acepta señales 1D (canal ya
    seleccionado) o 2D (M, N_samples). Si ref_mic es None usa el canal central
    (M // 2), mismo criterio que los wrappers DTLN.
    """
    sig = np.asarray(signal)
    if sig.ndim == 1:
        return sig
    if ref_mic is None:
        ref_mic = sig.shape[0] // 2
    return sig[ref_mic, :]


def get_oracle_masks(speech_signal, noise_signal, ref_mic=None,
                     block_len=512, block_shift=128, sharpen_exp=1.0, eps=1e-10):
    """
    Genera mascaras ideales de voz/ruido (IRM de potencia) a partir de las
    señales de referencia limpias. Interfaz de salida identica a get_dtln_masks:
    devuelve (mask_s, mask_n) con forma (K, T) = (block_len // 2 + 1, num_blocks).

    Parametros
    ----------
    speech_signal : np.ndarray
        Señal target (voz) de referencia. 1D (canal ya seleccionado) o 2D (M, N).
        Tipicamente target_early[ref] o (target_early + target_late)[ref].
    noise_signal : np.ndarray
        Señal de ruido + interferencia de referencia. Misma convencion de forma
        que speech_signal. Tipicamente (interference_early + interference_late).
        IMPORTANTE: speech y noise deben venir SIN normalizar por separado; su
        nivel relativo (el iSIR) es justamente lo que define la mascara.
    ref_mic : int | None
        Indice del microfono de referencia si las entradas son 2D. None -> M // 2.
    block_len, block_shift : int
        Framing STFT, identico a get_dtln_masks (512 / 128 por defecto).
    sharpen_exp : float
        Exponente de realce. 1.0 (por defecto) = mascara SUAVE sin realce, como
        se define la mascara ideal en la literatura. Se deja parametrizado para
        uso futuro (analogo al exponente de las mascaras DTLN); valores > 1 hacen
        la transicion voz/ruido mas abrupta (mascara mas binaria).
    eps : float
        Regularizacion para evitar division por cero en bins T-F silenciosos.

    Retorna
    -------
    (mask_s, mask_n) : tuple[np.ndarray, np.ndarray]
        Mascaras de voz y ruido, cada una con forma (K, T), en float32.
    """
    speech_mono = _select_ref(speech_signal, ref_mic)
    noise_mono = _select_ref(noise_signal, ref_mic)

    # Alinear largos por si difieren en un par de muestras
    n = int(min(speech_mono.shape[0], noise_mono.shape[0]))
    speech_mono = speech_mono[:n]
    noise_mono = noise_mono[:n]

    # Magnitudes STFT (mismo framing que DTLN)
    S_pow = _stft_mag_blocks(speech_mono, block_len, block_shift) ** 2
    N_pow = _stft_mag_blocks(noise_mono, block_len, block_shift) ** 2

    # Ideal Ratio Mask (dominio de potencia / ganancia Wiener) -> mascara SUAVE
    mask_s = S_pow / (S_pow + N_pow + eps)
    mask_n = 1.0 - mask_s

    # Realce opcional. Por defecto (1.0) no tiene efecto y las mascaras quedan
    # suaves y complementarias. Si se activa, se elevan por separado igual que en
    # dtln_masks (con lo que dejan de ser estrictamente complementarias).
    if sharpen_exp != 1.0:
        mask_s = mask_s ** sharpen_exp
        mask_n = mask_n ** sharpen_exp

    return mask_s.astype(np.float32), mask_n.astype(np.float32)
