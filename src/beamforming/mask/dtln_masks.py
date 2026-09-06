"""
dtln_masks.py
=============
Estimacion de mascaras de voz/ruido a partir del modelo DTLN.

Expone dos funciones:
  - `get_dtln_masks`: mascara base.
  - `get_dtln_masks_sharpen`: variante con exponente de "sharpening" (que agudiza
    la transicion voz/ruido) como PARAMETRO `sharpen_exp` en vez del `** 4` fijo.
    Con `sharpen_exp=4.0` produce la misma mascara que la version fija; bajarlo
    (p.ej. 2 o 3) suaviza la transicion, subirlo (6, 8) la hace mas abrupta
    (mascara mas binaria).
"""

import numpy as np
from ai_edge_litert.interpreter import Interpreter

def get_dtln_masks(time_domain_input, ref_mic, model1_path, block_len=512, block_shift=128):
    """
    Processes a single reference channel offline to extract neural masks
    using the DTLN STFT-based model. This avoids computing masks for all M channels.
    """
    M, samples = time_domain_input.shape
    num_blocks = (samples - (block_len - block_shift)) // block_shift

    interpreter_1 = Interpreter(model_path=model1_path)
    interpreter_1.allocate_tensors()

    input_details_1 = interpreter_1.get_input_details()
    output_details_1 = interpreter_1.get_output_details()

    # Isolate the audio from the selected reference microphone
    audio_mono = time_domain_input[ref_mic, :]

    # Normalize channel audio to prevent DTLN saturation
    max_val = np.max(np.abs(audio_mono))
    if max_val > 0:
        audio_mono = audio_mono / max_val

    # Initialize LSTM states for the reference channel
    states_1 = np.zeros(input_details_1[1]['shape'], dtype=np.float32)
    in_buffer = np.zeros((block_len), dtype=np.float32)

    ch_mask = np.zeros((block_len // 2 + 1, num_blocks), dtype=np.float32)

    print(f"\r -> Computing DTLN mask ONLY for reference channel {ref_mic}...", end="")

    for idx in range(num_blocks):
        # Shift buffer and load new audio samples
        in_buffer[:-block_shift] = in_buffer[block_shift:]
        start_idx = idx * block_shift
        in_buffer[-block_shift:] = audio_mono[start_idx : start_idx + block_shift]

        # Compute FFT and magnitude
        in_block_fft = np.fft.rfft(in_buffer)
        in_mag = np.abs(in_block_fft)
        in_mag = np.reshape(in_mag, (1, 1, -1)).astype(np.float32)

        # Predict mask
        interpreter_1.set_tensor(input_details_1[1]['index'], states_1)
        interpreter_1.set_tensor(input_details_1[0]['index'], in_mag)
        interpreter_1.invoke()

        out_mask = interpreter_1.get_tensor(output_details_1[0]['index'])
        states_1 = interpreter_1.get_tensor(output_details_1[1]['index'])

        ch_mask[:, idx] = np.squeeze(out_mask)

    print() # New line after processing

    # Skip median pooling as we only have one channel mask now
    sp_mask_condensed = ch_mask

    # Contrast patch: stretch values between 0 and 1
    def stretch_mask(m):
        return (m - np.min(m)) / (np.max(m) - np.min(m) + 1e-12)

    sp_mask_condensed = stretch_mask(sp_mask_condensed)

    # Calculate noise mask mathematically
    n_mask_condensed = 1.0 - sp_mask_condensed

    # Sharpen the masks by squaring them
    sp_mask_condensed = sp_mask_condensed ** 4
    n_mask_condensed = n_mask_condensed ** 4

    return sp_mask_condensed, n_mask_condensed



def get_dtln_masks_sharpen(time_domain_input, ref_mic, model1_path,
                           block_len=512, block_shift=128, sharpen_exp=4.0,
                           peak_norm="global", stretch=True):
    """
    Copia de get_dtln_masks con el exponente de sharpening parametrizable.
    sharpen_exp=4.0 -> identico al original. Devuelve (mask_s, mask_n), shape (K, T).

    CAUSALIDAD (peak_norm / stretch)
    --------------------------------
    El camino historico tiene DOS etapas que miran TODA la senal, o sea que NO son
    implementables en tiempo real:

      1. `peak_norm="global"`: divide el canal por su pico sobre el archivo entero
         antes de enmarcar. Alternativa causal: pasar un FLOAT (la escala fija; el
         hardware entrega unidades de fondo de escala, asi que 1.0 es lo natural).
         Medido sobre MIRD: escala fija 1.0 vs pico global -> corr 0.999,
         MAE 0.009 en la mascara. Un pico corriente (cummax) es MUCHO peor
         (corr 0.87): tarda ~8 s en converger.
      2. `stretch=True`: reescala la mascara por su min/max global sobre (k, t).
         `stretch=False` la saltea y deja solo la potencia, que es puntual y por
         lo tanto causal.

    Los defaults conservan el comportamiento historico bit a bit.
    """
    M, samples = time_domain_input.shape
    num_blocks = (samples - (block_len - block_shift)) // block_shift

    interpreter_1 = Interpreter(model_path=model1_path)
    interpreter_1.allocate_tensors()

    input_details_1 = interpreter_1.get_input_details()
    output_details_1 = interpreter_1.get_output_details()

    # Isolate the audio from the selected reference microphone
    audio_mono = time_domain_input[ref_mic, :]

    # Normalize channel audio to prevent DTLN saturation.
    # "global" = pico de TODA la senal (no causal); un float = escala fija.
    if peak_norm == "global":
        max_val = np.max(np.abs(audio_mono))
        if max_val > 0:
            audio_mono = audio_mono / max_val
    elif peak_norm is not None:
        audio_mono = audio_mono / float(peak_norm)

    # Initialize LSTM states for the reference channel
    states_1 = np.zeros(input_details_1[1]['shape'], dtype=np.float32)
    in_buffer = np.zeros((block_len), dtype=np.float32)

    ch_mask = np.zeros((block_len // 2 + 1, num_blocks), dtype=np.float32)

    print(f"\r -> Computing DTLN mask (sharpen_exp={sharpen_exp}) for ref channel {ref_mic}...", end="")

    for idx in range(num_blocks):
        # Shift buffer and load new audio samples
        in_buffer[:-block_shift] = in_buffer[block_shift:]
        start_idx = idx * block_shift
        in_buffer[-block_shift:] = audio_mono[start_idx: start_idx + block_shift]

        # Compute FFT and magnitude
        in_block_fft = np.fft.rfft(in_buffer)
        in_mag = np.abs(in_block_fft)
        in_mag = np.reshape(in_mag, (1, 1, -1)).astype(np.float32)

        # Predict mask
        interpreter_1.set_tensor(input_details_1[1]['index'], states_1)
        interpreter_1.set_tensor(input_details_1[0]['index'], in_mag)
        interpreter_1.invoke()

        out_mask = interpreter_1.get_tensor(output_details_1[0]['index'])
        states_1 = interpreter_1.get_tensor(output_details_1[1]['index'])

        ch_mask[:, idx] = np.squeeze(out_mask)

    print()  # New line after processing

    # Skip median pooling as we only have one channel mask now
    sp_mask_condensed = ch_mask

    # Contrast patch: stretch values between 0 and 1. NO CAUSAL (min/max global):
    # stretch=False lo saltea.
    def stretch_mask(m):
        return (m - np.min(m)) / (np.max(m) - np.min(m) + 1e-12)

    if stretch:
        sp_mask_condensed = stretch_mask(sp_mask_condensed)

    # Calculate noise mask mathematically
    n_mask_condensed = 1.0 - sp_mask_condensed

    # Sharpen the masks -- UNICA diferencia vs el original: exponente parametrizado
    # (original fijo en 4). Mas alto = transicion voz/ruido mas abrupta.
    sp_mask_condensed = sp_mask_condensed ** sharpen_exp
    n_mask_condensed = n_mask_condensed ** sharpen_exp

    return sp_mask_condensed, n_mask_condensed


def get_dtln_masks_soft(time_domain_input, ref_mic, model1_path, block_len=512,
                        block_shift=128, peak_norm="global"):
    """
    Variante SUAVE (sin sharpening): procesa el canal de referencia offline y
    devuelve mascaras continuas en [0,1]. A diferencia de get_dtln_masks_sharpen,
    NO eleva la mascara a ninguna potencia -> mantiene la transicion voz/ruido
    lineal/continua (evita ceros duros en el dominio STFT y reduce artefactos).
    La mascara de ruido es el complemento lineal (1 - mask_s).
    """
    M, samples = time_domain_input.shape
    num_blocks = (samples - (block_len - block_shift)) // block_shift

    interpreter_1 = Interpreter(model_path=model1_path)
    interpreter_1.allocate_tensors()

    input_details_1 = interpreter_1.get_input_details()
    output_details_1 = interpreter_1.get_output_details()

    # Isolate the audio from the selected reference microphone
    audio_mono = time_domain_input[ref_mic, :]

    # Normalize channel audio to prevent DTLN saturation.
    # "global" = pico de TODA la senal (NO CAUSAL); un float = escala fija.
    # Ver la nota de causalidad en get_dtln_masks_sharpen.
    if peak_norm == "global":
        max_val = np.max(np.abs(audio_mono))
        if max_val > 0:
            audio_mono = audio_mono / max_val
    elif peak_norm is not None:
        audio_mono = audio_mono / float(peak_norm)

    # Initialize LSTM states for the reference channel
    states_1 = np.zeros(input_details_1[1]['shape'], dtype=np.float32)
    in_buffer = np.zeros((block_len), dtype=np.float32)

    ch_mask = np.zeros((block_len // 2 + 1, num_blocks), dtype=np.float32)

    print(f"\r -> Computing DTLN mask ONLY for reference channel {ref_mic}...", end="")

    for idx in range(num_blocks):
        # Shift buffer and load new audio samples
        in_buffer[:-block_shift] = in_buffer[block_shift:]
        start_idx = idx * block_shift
        in_buffer[-block_shift:] = audio_mono[start_idx : start_idx + block_shift]

        # Compute FFT and magnitude
        in_block_fft = np.fft.rfft(in_buffer)
        in_mag = np.abs(in_block_fft)
        in_mag = np.reshape(in_mag, (1, 1, -1)).astype(np.float32)

        # Predict mask
        interpreter_1.set_tensor(input_details_1[1]['index'], states_1)
        interpreter_1.set_tensor(input_details_1[0]['index'], in_mag)
        interpreter_1.invoke()

        out_mask = interpreter_1.get_tensor(output_details_1[0]['index'])
        states_1 = interpreter_1.get_tensor(output_details_1[1]['index'])

        ch_mask[:, idx] = np.squeeze(out_mask)

    print() # New line after processing

    # Skip median pooling as we only have one channel mask now
    sp_mask_condensed = ch_mask

    # Calculate noise mask mathematically (linear complement)
    n_mask_condensed = 1.0 - sp_mask_condensed

    # The squaring operation was removed here to keep the soft, linear continuous mask
    # This prevents harsh zeros in the STFT domain and reduces potential artifacts

    return sp_mask_condensed, n_mask_condensed


# =====================================================================
# ALINEACION DE FRAMES: mascara del DTLN vs STFT del beamformer
# =====================================================================
# El DTLN enmarca con un buffer deslizante: el bloque `i` contiene las ultimas
# `block_len` muestras que terminan en (i+1)*block_shift, o sea la ventana
#
#     bloque i  ->  [ i*hop - (block_len - hop) ,  i*hop + hop )
#
# scipy.signal.stft(..., nperseg=block_len, noverlap=block_len-hop) con el
# padding por defecto (boundary='zeros') centra el frame `t` en t*hop:
#
#     frame t   ->  [ t*hop - block_len/2 ,  t*hop + block_len/2 )
#
# Para block_len=512 y hop=128 las dos ventanas coinciden EXACTAMENTE cuando
# t = i - 1: ambas cubren [i*128 - 384, i*128 + 128). Verificado numericamente
# (correlacion 1.000 exacta en lag +1, ver tests/ds_mask_scm_run.py).
#
# Pero todos los wrappers mask-based aparean mask[:, m] con X_stft[:, m], o sea
# que a cada frame de la STFT le aplican la mascara del frame ANTERIOR: la
# mascara llega 1 frame (8 ms) tarde. `align_mask_frames` lo corrige adelantando
# la mascara un frame.
#
# NO CUESTA LATENCIA. El bloque i+1 del DTLN y el frame i de la STFT cubren las
# MISMAS 512 muestras y terminan en la MISMA muestra ((i+2)*hop), asi que la
# correccion no mira ni una muestra hacia el futuro: es un error de indexado, no
# un adelanto temporal. Por eso es implementable tal cual en el sistema online
# que se lleva a HLS.
#
# OJO AL RE-AJUSTAR: cualquier calibracion ajustada ANTES de esta correccion
# (los .npz de scm_calibration_run / scm_mask_calibration_run) se ajusto CON el
# desfasaje puesto y hay que rehacerla.
DTLN_MASK_SHIFT = 1


def align_mask_frames(mask, shift=None):
    """
    Adelanta la mascara `shift` frames para alinearla con la STFT del beamformer
    (ver la nota de arriba). Los ultimos `shift` frames se repiten.

    Args:
        mask: (K, T), o una tupla/lista de arrays (K, T).
        shift: None -> DTLN_MASK_SHIFT (1 = corregido). 0 = comportamiento
            historico, para reproducir resultados previos.

    Returns:
        lo mismo que se le paso (array o tupla), corrido.
    """
    s = DTLN_MASK_SHIFT if shift is None else int(shift)
    if isinstance(mask, (tuple, list)):
        return type(mask)(align_mask_frames(m, s) for m in mask)
    if s <= 0:
        return mask
    return np.concatenate([mask[:, s:], np.repeat(mask[:, -1:], s, axis=1)], axis=1)
