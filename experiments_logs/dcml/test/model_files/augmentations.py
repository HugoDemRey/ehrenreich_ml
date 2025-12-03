import numpy as np
import torch


def DataTransform(sample, config):
    """
    Apply augmentations appropriate for the data type.
    Handles both 1D waveforms and 2D spectrograms.
    """
    # Check if we have spectrogram data (3D: batch, mel_bands, time_frames)
    # or waveform data (3D: batch, channels, time_samples)
    if sample.shape[1] > 10:  # Likely spectrogram (e.g., 80 mel bands)
        # Use spectrogram-specific augmentations
        weak_aug = freq_mask(sample, config.augmentation.freq_mask_ratio if hasattr(config.augmentation, 'freq_mask_ratio') else 0.1)
        strong_aug = time_mask(freq_shift(sample, config.augmentation.freq_shift_ratio if hasattr(config.augmentation, 'freq_shift_ratio') else 0.1), 
                              config.augmentation.time_mask_ratio if hasattr(config.augmentation, 'time_mask_ratio') else 0.1)
    else:
        # Use original waveform augmentations
        weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)
        strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)

    return weak_aug, strong_aug


# ===== SPECTROGRAM AUGMENTATIONS =====

def freq_mask(x, mask_ratio=0.1):
    """
    Frequency masking for spectrograms (SpecAugment style)
    Masks random frequency bands by setting them to the mean value
    """
    # Convert to numpy if tensor, handle both cases
    if torch.is_tensor(x):
        x_aug = x.clone().numpy()
    else:
        x_aug = x.copy()
    
    for i in range(x_aug.shape[0]):  # For each sample in batch
        n_freq_bins = x_aug.shape[1]  # Number of frequency bins (mel bands)
        mask_size = int(n_freq_bins * mask_ratio)
        
        if mask_size > 0:
            start_freq = np.random.randint(0, n_freq_bins - mask_size + 1)
            # Set masked frequencies to mean value to avoid harsh artifacts
            mean_val = np.mean(x_aug[i])
            x_aug[i, start_freq:start_freq + mask_size, :] = mean_val
            
    return torch.from_numpy(x_aug)


def time_mask(x, mask_ratio=0.1):
    """
    Time masking for spectrograms (SpecAugment style)
    Masks random time frames by setting them to the mean value
    """
    # Convert to tensor if needed, then work with tensor operations
    if torch.is_tensor(x):
        x_aug = x.clone()
    else:
        x_aug = torch.from_numpy(x.copy())
    
    for i in range(x_aug.shape[0]):  # For each sample in batch
        n_time_frames = x_aug.shape[2]  # Number of time frames
        mask_size = int(n_time_frames * mask_ratio)
        
        if mask_size > 0:
            start_time = np.random.randint(0, n_time_frames - mask_size + 1)
            # Set masked time frames to mean value
            mean_val = torch.mean(x_aug[i])
            x_aug[i, :, start_time:start_time + mask_size] = mean_val
            
    return x_aug


def freq_shift(x, shift_ratio=0.1):
    """
    Random frequency shifting for spectrograms
    Simulates slight pitch variations
    """
    # Convert to numpy for easier indexing operations
    if torch.is_tensor(x):
        x_aug = x.clone().numpy()
    else:
        x_aug = x.copy()
    
    for i in range(x_aug.shape[0]):  # For each sample in batch
        n_freq_bins = x_aug.shape[1]
        max_shift = int(n_freq_bins * shift_ratio)
        
        if max_shift > 0:
            shift = np.random.randint(-max_shift, max_shift + 1)
            
            if shift > 0:
                # Shift up: pad bottom, truncate top
                x_aug[i, shift:, :] = x_aug[i, :-shift, :]
                x_aug[i, :shift, :] = x_aug[i, shift, :][:, np.newaxis].repeat(shift, axis=1).T
            elif shift < 0:
                # Shift down: pad top, truncate bottom  
                shift = abs(shift)
                x_aug[i, :-shift, :] = x_aug[i, shift:, :]
                x_aug[i, -shift:, :] = x_aug[i, -shift-1, :][:, np.newaxis].repeat(shift, axis=1).T
                
    return torch.from_numpy(x_aug)


# ===== ORIGINAL WAVEFORM AUGMENTATIONS =====

def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)


def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0,warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)

