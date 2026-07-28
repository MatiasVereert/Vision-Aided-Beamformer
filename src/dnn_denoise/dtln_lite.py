import numpy as np
import tensorflow as tf


def apply_dtln_post_tflite_realtime(interpreter_1, interpreter_2, audio_mono):
    """
    Applies the DTLN TF-Lite models using a strict real-time
    frame-by-frame loop.

    Includes critical fixes for benchmark integration:
    - Memory contiguity enforcement for TFLite raw pointer access.
    - Peak normalization so inputs of arbitrary scale (e.g. beamformer
      outputs) reach the network in the [-1, 1] range it was trained on,
      without clipping. The original scale is restored at the end.
    - Hard copies of interpreter state tensors to prevent memory
      corruption across massive grid-search loops.
    """

    # 1. Enforce shape and memory contiguity
    audio_mono = np.ascontiguousarray(np.squeeze(audio_mono), dtype=np.float32)

    # 2. Peak normalization (lossless, no clipping)
    peak = np.max(np.abs(audio_mono))
    scale_factor = peak if peak > 1e-6 else 1.0

    audio_norm = audio_mono / scale_factor

    block_len = 512
    block_shift = 128

    out_audio = np.zeros_like(audio_norm)
    in_buffer = np.zeros((block_len), dtype=np.float32)
    out_buffer = np.zeros((block_len), dtype=np.float32)

    # Get input/output indices
    input_details_1 = interpreter_1.get_input_details()
    output_details_1 = interpreter_1.get_output_details()

    input_details_2 = interpreter_2.get_input_details()
    output_details_2 = interpreter_2.get_output_details()

    # INITIALIZE LSTM STATES (Fresh zero-state for every new benchmark file)
    states_1 = np.zeros(input_details_1[1]['shape'], dtype=np.float32)
    states_2 = np.zeros(input_details_2[1]['shape'], dtype=np.float32)

    num_blocks = (len(audio_norm) - (block_len - block_shift)) // block_shift

    for idx in range(num_blocks):
        # Shift buffer and load data
        in_buffer[:-block_shift] = in_buffer[block_shift:]
        start_idx = idx * block_shift
        in_buffer[-block_shift:] = audio_norm[start_idx : start_idx + block_shift]

        # Compute FFT
        in_block_fft = np.fft.rfft(in_buffer)
        in_mag = np.abs(in_block_fft)
        in_phase = np.angle(in_block_fft)

        # Reshape safely maintaining contiguity
        in_mag = np.ascontiguousarray(np.reshape(in_mag, (1, 1, -1)), dtype=np.float32)

        # Model 1
        interpreter_1.set_tensor(input_details_1[1]['index'], states_1)
        interpreter_1.set_tensor(input_details_1[0]['index'], in_mag)
        interpreter_1.invoke()

        # Extract and explicitly .copy() to prevent memory aliasing in long loops
        out_mask = interpreter_1.get_tensor(output_details_1[0]['index']).copy()
        states_1 = interpreter_1.get_tensor(output_details_1[1]['index']).copy()

        # Reconstruct and IFFT
        estimated_complex = in_mag * out_mask * np.exp(1j * in_phase)
        estimated_block = np.fft.irfft(estimated_complex, n=block_len)
        estimated_block = np.ascontiguousarray(np.reshape(estimated_block, (1, 1, -1)), dtype=np.float32)

        # Model 2
        interpreter_2.set_tensor(input_details_2[1]['index'], states_2)
        interpreter_2.set_tensor(input_details_2[0]['index'], estimated_block)
        interpreter_2.invoke()

        # Extract and explicitly .copy()
        out_block = interpreter_2.get_tensor(output_details_2[0]['index']).copy()
        states_2 = interpreter_2.get_tensor(output_details_2[1]['index']).copy()

        # Overlap-add
        out_buffer[:-block_shift] = out_buffer[block_shift:]
        out_buffer[-block_shift:] = np.zeros((block_shift), dtype=np.float32)
        out_buffer += np.squeeze(out_block)

        out_audio[start_idx : start_idx + block_shift] = out_buffer[:block_shift]

    # Restore the acoustic simulator's physical scale for accurate metric evaluation
    out_audio = out_audio * scale_factor

    return out_audio


import numpy as np
import soundfile as sf
import tensorflow as tf

# Place the apply_dtln_post_tflite_realtime function definition here


def main():
    # Define placeholder paths for models and audio files
    model1_path = r"/home/matias/Documents/Tesis/Vision-Aided-Beamformer/src/dnn_denoise/models/model_quant_1.tflite"
    model2_path = r"/home/matias/Documents/Tesis/Vision-Aided-Beamformer/src/dnn_denoise/models/model_quant_2.tflite"

    input_path = r"/home/matias/Documents/Tesis/Vision-Aided-Beamformer/tools/data/signals/audioset_realrec_airconditioner_2TE3LoA2OUQ.wav"
    output_path = r"audio_limpio.wav"

    # Initialize and allocate interpreters for Model 1 and Model 2
    interpreter_1 = tf.lite.Interpreter(model_path=model1_path)
    interpreter_1.allocate_tensors()

    interpreter_2 = tf.lite.Interpreter(model_path=model2_path)
    interpreter_2.allocate_tensors()

    # Load the audio file in float32 format
    print(f"Loading audio from: {input_path}")
    audio, sample_rate = sf.read(input_path, dtype="float32")

    # Convert to mono if the input audio has multiple channels
    if len(audio.shape) > 1:
        # Average the channels to create a single mono track
        audio = np.mean(audio, axis=1)

    # Normalization now happens inside the wrapper (peak-based, scale restored)

    print("Processing audio frame-by-frame...")

    # Run the real-time DTLN noise suppression loop
    clean_audio = apply_dtln_post_tflite_realtime(
        interpreter_1, interpreter_2, audio
    )

    # Export the processed audio back to a WAV file
    sf.write(output_path, clean_audio, sample_rate)
    print(f"Processed audio successfully saved to: {output_path}")


if __name__ == "__main__":
    main()