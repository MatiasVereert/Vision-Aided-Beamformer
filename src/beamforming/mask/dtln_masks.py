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
                           block_len=512, block_shift=128, sharpen_exp=4.0):
    """
    Copia de get_dtln_masks con el exponente de sharpening parametrizable.
    sharpen_exp=4.0 -> identico al original. Devuelve (mask_s, mask_n), shape (K, T).
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

    # Contrast patch: stretch values between 0 and 1
    def stretch_mask(m):
        return (m - np.min(m)) / (np.max(m) - np.min(m) + 1e-12)

    sp_mask_condensed = stretch_mask(sp_mask_condensed)

    # Calculate noise mask mathematically
    n_mask_condensed = 1.0 - sp_mask_condensed

    # Sharpen the masks -- UNICA diferencia vs el original: exponente parametrizado
    # (original fijo en 4). Mas alto = transicion voz/ruido mas abrupta.
    sp_mask_condensed = sp_mask_condensed ** sharpen_exp
    n_mask_condensed = n_mask_condensed ** sharpen_exp

    return sp_mask_condensed, n_mask_condensed


def get_dtln_masks_soft(time_domain_input, ref_mic, model1_path, block_len=512, block_shift=128):
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

    # Calculate noise mask mathematically (linear complement)
    n_mask_condensed = 1.0 - sp_mask_condensed

    # The squaring operation was removed here to keep the soft, linear continuous mask
    # This prevents harsh zeros in the STFT domain and reduces potential artifacts

    return sp_mask_condensed, n_mask_condensed
