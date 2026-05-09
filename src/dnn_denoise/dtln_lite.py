import numpy as np
import tensorflow as tf




def apply_dtln_post_tflite_realtime(interpreter_1, interpreter_2, audio_mono, blend_alpha=1):
    """
    Applies the Float32 DTLN TF-Lite models using a strict real-time 
    frame-by-frame loop. Handles LSTM hidden states properly to prevent 
    signal degradation. Compensates for algorithmic delay during blending.
    """
    # 1. Normalize input to prevent LSTM saturation
    max_val = np.max(np.abs(audio_mono))
    if max_val > 0.0:
        # Scale to peak at 0.9 to give the network optimal headroom
        audio_mono = audio_mono * (0.9 / max_val)

    audio_mono = np.asarray(audio_mono, dtype=np.float32)
    
    block_len = 512
    block_shift = 128
    
    out_audio = np.zeros_like(audio_mono)
    in_buffer = np.zeros((block_len), dtype=np.float32)
    out_buffer = np.zeros((block_len), dtype=np.float32)
    
    # Get input/output indices 
    input_details_1 = interpreter_1.get_input_details()
    output_details_1 = interpreter_1.get_output_details()
    
    input_details_2 = interpreter_2.get_input_details()
    output_details_2 = interpreter_2.get_output_details()

    # INITIALIZE LSTM STATES
    states_1 = np.zeros(input_details_1[1]['shape'], dtype=np.float32)
    states_2 = np.zeros(input_details_2[1]['shape'], dtype=np.float32)

    num_blocks = (len(audio_mono) - (block_len - block_shift)) // block_shift
    
    for idx in range(num_blocks):
        # Shift buffer and load data directly
        in_buffer[:-block_shift] = in_buffer[block_shift:]
        start_idx = idx * block_shift
        in_buffer[-block_shift:] = audio_mono[start_idx : start_idx + block_shift]
        
        # Compute FFT, magnitude and phase
        in_block_fft = np.fft.rfft(in_buffer)
        in_mag = np.abs(in_block_fft)
        in_phase = np.angle(in_block_fft)
        
        # Reshape magnitude
        in_mag = np.reshape(in_mag, (1, 1, -1)).astype(np.float32)
        
        # Feed magnitude AND previous states to Model 1
        interpreter_1.set_tensor(input_details_1[1]['index'], states_1)
        interpreter_1.set_tensor(input_details_1[0]['index'], in_mag)
        interpreter_1.invoke()
        
        # Extract mask and NEW states
        out_mask = interpreter_1.get_tensor(output_details_1[0]['index'])
        states_1 = interpreter_1.get_tensor(output_details_1[1]['index']) 
        
        # Reconstruct complex FFT and apply IFFT
        estimated_complex = in_mag * out_mask * np.exp(1j * in_phase)
        estimated_block = np.fft.irfft(estimated_complex, n=block_len)
        estimated_block = np.reshape(estimated_block, (1, 1, -1)).astype(np.float32)
        
        # Feed time-domain block AND previous states to Model 2
        interpreter_2.set_tensor(input_details_2[1]['index'], states_2)
        interpreter_2.set_tensor(input_details_2[0]['index'], estimated_block)
        interpreter_2.invoke()
        
        # Extract final audio block and NEW states
        out_block = interpreter_2.get_tensor(output_details_2[0]['index'])
        states_2 = interpreter_2.get_tensor(output_details_2[1]['index'])
        
        # Overlap-add
        out_buffer[:-block_shift] = out_buffer[block_shift:]
        out_buffer[-block_shift:] = np.zeros((block_shift))
        out_buffer += np.squeeze(out_block)
        
        out_audio[start_idx : start_idx + block_shift] = out_buffer[:block_shift]

    # 2. Apply Wet/Dry blending with precise delay compensation
    if blend_alpha < 1.0:
        delay = block_len - block_shift # 384 samples algorithmic delay
        
        # Shift the dry audio to align perfectly with the processed output
        audio_mono_delayed = np.zeros_like(audio_mono)
        audio_mono_delayed[delay:] = audio_mono[:-delay]
        
        out_audio = (blend_alpha * out_audio) + ((1.0 - blend_alpha) * audio_mono_delayed)

    return out_audio