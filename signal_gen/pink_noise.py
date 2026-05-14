import os
import numpy as np
from scipy.io import wavfile

def generate_pink_noise(duration_seconds, sample_rate=16000):
    # Calculate total number of samples
    num_samples = int(duration_seconds * sample_rate)
    
    # Generate standard white noise
    white_noise = np.random.randn(num_samples)
    
    # Transform to frequency domain using Real FFT
    fft_values = np.fft.rfft(white_noise)
    
    # Create frequency indices (adding 1 to avoid division by zero at DC offset)
    frequencies = np.arange(len(fft_values)) + 1
    
    # Apply 1/sqrt(f) filter to achieve pink noise characteristics
    fft_values /= np.sqrt(frequencies)
    
    # Transform back to time domain
    pink_noise = np.fft.irfft(fft_values, n=num_samples)
    
    # Normalize audio to prevent clipping (-0.9 to 0.9 range for safety headroom)
    normalized_noise = pink_noise / np.max(np.abs(pink_noise)) * 0.9
    
    # Convert to standard 16-bit PCM integer format
    audio_int16 = (normalized_noise * 32767).astype(np.int16)
    
    return audio_int16

def save_audio(output_dir, filename, audio_data, sample_rate=16000):
    # Ensure the target directory exists; create it if it does not
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct the absolute or relative file path
    file_path = os.path.join(output_dir, filename)
    
    # Write the WAV file to disk
    wavfile.write(file_path, sample_rate, audio_data)
    print(f"Success: File saved accurately at '{file_path}'")

if __name__ == "__main__":
    # Configuration parameters
    # Define your custom output directory path here
    TARGET_DIRECTORY = r"data/audio/input" 
    OUTPUT_FILENAME = "ruido_rosa_16k.wav"
    DURATION = 10  # Audio duration in seconds
    SAMPLE_RATE = 16000  # Target sample rate (16 kHz)
    
    print("Generating 16 kHz pink noise...")
    noise_data = generate_pink_noise(DURATION, SAMPLE_RATE)
    
    print("Saving audio file...")
    save_audio(TARGET_DIRECTORY, OUTPUT_FILENAME, noise_data, SAMPLE_RATE)