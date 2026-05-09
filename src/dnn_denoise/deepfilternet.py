import torch

# Import standard DeepFilterNet enhance function
from df.enhance import enhance, init_df
import torchaudio


def apply_deepfilter_post_resampled(model, df_state, audio_mono, fs_in=16000, blend_alpha=0.95):
    """
    Applies DeepFilterNet as a post-processing stage.
    Dynamically upsamples the input to the network's native sample rate, 
    processes the signal, and downsamples it back to the original input rate.
    
    Args:
        model: Loaded DeepFilterNet model.
        df_state: DeepFilterNet state configuration.
        audio_mono: 1D numpy array of the audio signal (e.g., MVDR output).
        fs_in: Sampling frequency of the input audio array.
        blend_alpha: Wet/Dry mix ratio (0.0 = only MVDR, 1.0 = Max DeepFilterNet).
        
    Returns:
        processed_np: 1D numpy array at the original fs_in sample rate.
    """
    # Get the native sample rate required by the DeepFilterNet model
    fs_net = df_state.sr() 
    
    # Ensure input is a float32 PyTorch tensor with shape [Channels, Time]
    input_tensor = torch.tensor(audio_mono, dtype=torch.float32).unsqueeze(0)
    
    # 1. Upsample from original rate to network's native rate
    if fs_in != fs_net:
        resampler_up = torchaudio.transforms.Resample(orig_freq=fs_in, new_freq=fs_net)
        audio_net_fs = resampler_up(input_tensor)
    else:
        audio_net_fs = input_tensor

    # 2. Process with DeepFilterNet
    with torch.no_grad():
        enhanced_tensor = enhance(model, df_state, audio_net_fs)
        
    # 3. Downsample back to the original input rate
    if fs_in != fs_net:
        resampler_down = torchaudio.transforms.Resample(orig_freq=fs_net, new_freq=fs_in)
        enhanced_tensor_out = resampler_down(enhanced_tensor)
    else:
        enhanced_tensor_out = enhanced_tensor

    # 4. Extract the raw NumPy arrays for blending
    processed_np = enhanced_tensor_out.squeeze(0).numpy()
    dry_np = audio_mono  # The original input is already at fs_in
    
    # Match lengths in case resampling introduced a 1-sample difference due to rounding
    min_length = min(processed_np.shape[0], dry_np.shape[0])
    processed_np = processed_np[:min_length]
    dry_np = dry_np[:min_length]

    # 5. Apply Wet/Dry blending at the original sample rate
    if blend_alpha < 1.0:
        processed_np = (blend_alpha * processed_np) + ((1.0 - blend_alpha) * dry_np)

    return processed_np