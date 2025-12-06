from loader import load_signal, load_transitions_labels
import os
import torch
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import argparse

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Helper function to convert transition times to window labels (STRICT CONTAINMENT)
def transitions_to_labels(transitions, total_duration, window_duration=6.0, stride_duration=2.0):
    """
    transitions: list of transition times in seconds
    total_duration: total duration of the signal in seconds
    window_duration: duration of each window in seconds
    stride_duration: stride between windows in seconds

    Returns:
    labels: binary numpy array of shape (num_windows,)
    
    STRICT CONTAINMENT APPROACH: A window gets label=1 if a transition occurs 
    WITHIN the window boundaries (no tolerance).
    """
    # Calculate number of windows
    num_windows = int((total_duration - window_duration) / stride_duration) + 1
    labels = np.zeros(num_windows, dtype=np.float32)

    # For each window, check if any transition falls within its boundaries
    for i in range(num_windows):
        window_start = i * stride_duration
        window_end = window_start + window_duration
        
        # Check if any transition is strictly within this window
        for transition_time in transitions:
            if window_start <= transition_time <= window_end:
                labels[i] = 1.0
                break  # Found a transition within window, mark as positive
                
    return labels

# Function to extract mel-spectrogram features
def extract_mel_spectrogram(waveform, sample_rate, n_mels=80, n_fft=2048, hop_length=256, fmin=0, fmax=8000):
    """
    Extract mel-spectrogram features from waveform
    
    Args:
        waveform: audio signal
        sample_rate: sampling rate
        n_mels: number of mel bands
        n_fft: FFT window size
        hop_length: hop length for STFT
        fmin: minimum frequency
        fmax: maximum frequency (should be <= sample_rate/2)
    
    Returns:
        mel_spec: mel-spectrogram in dB scale, shape (n_mels, time_frames)
    """
    # Compute mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=min(fmax, sample_rate // 2)  # Ensure fmax doesn't exceed Nyquist
    )
    
    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db


def compute_spectrogram_worker(args):
    """
    Worker function for multiprocessing spectrogram computation
    Returns (index, mel_spec, label) to preserve order
    """
    index, window, sample_rate, n_mels, n_fft, hop_length, label = args
    
    mel_spec = extract_mel_spectrogram(
        window, sample_rate, n_mels=n_mels, 
        n_fft=n_fft, hop_length=hop_length
    )
    
    return index, mel_spec, label


# Main preprocessing function
def preprocess_signals(signals, transitions_list, window_duration=4.0, stride_duration=1.0, 
                      n_mels=80, n_fft=2048, hop_length=256, num_workers=None):
    """
    signals: list of signal objects with samples and sample_rate attributes
    transitions_list: list of lists of transition timestamps for each signal
    window_duration: duration of each window in seconds (default: 6.0)
    stride_duration: stride between windows in seconds (default: 2.0)
    n_mels: number of mel bands for spectrogram
    n_fft: FFT window size
    hop_length: hop length for STFT
    num_workers: number of worker processes (None = auto-detect)

    Returns:
    - spectrograms_np: numpy array of shape (num_windows, n_mels, time_frames)
    - labels_np: numpy array of shape (num_windows,) binary labels for each window
    
    STRICT CONTAINMENT APPROACH: Windows are labeled 1 only if transitions 
    occur within the window boundaries.
    """
    
    if num_workers is None:
        num_workers = min(cpu_count() - 1, 10)  # Leave one core free, max 10 workers
    
    print(f"Using {num_workers} worker processes for spectrogram computation")
    
    # Prepare all windows for parallel processing
    all_windows = []
    all_labels = []
    
    # First pass: extract all windows and labels
    for signal_idx, (signal, transitions) in enumerate(tqdm(zip(signals, transitions_list), 
                                                           total=len(signals), 
                                                           desc="Extracting windows")):
        waveform = signal.samples
        sample_rate = signal.sample_rate
        total_duration = len(waveform) / sample_rate
        
        print(f"Processing signal {signal_idx+1}/{len(signals)}: duration={total_duration:.2f}s, window={window_duration}s, stride={stride_duration}s")
        print(f"STRICT CONTAINMENT labeling: Windows labeled 1 only if transition occurs within window")
        
        # Generate window-level labels based on strict containment
        labels = transitions_to_labels(transitions, total_duration, window_duration, stride_duration)
        
        # Calculate window parameters in samples
        window_samples = int(window_duration * sample_rate)
        stride_samples = int(stride_duration * sample_rate)
        
        # Extract windows
        num_windows = len(labels)
        
        for i in range(num_windows):
            start_sample = i * stride_samples
            end_sample = start_sample + window_samples
            
            # Ensure we don't go beyond the signal length
            if end_sample > len(waveform):
                break
                
            window = waveform[start_sample:end_sample]
            # Add index to preserve order during multiprocessing
            all_windows.append((len(all_windows), window, sample_rate, n_mels, n_fft, hop_length, labels[i]))
            all_labels.append(labels[i])
    
    print(f"Extracted {len(all_windows)} windows, computing spectrograms in parallel...")
    
    # Parallel spectrogram computation
    if num_workers == 1:
        # Single-threaded fallback
        results = []
        for args in tqdm(all_windows, desc="Computing spectrograms (single-threaded)"):
            results.append(compute_spectrogram_worker(args))
    else:
        # Multi-threaded processing
        with Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(compute_spectrogram_worker, all_windows), 
                              total=len(all_windows), 
                              desc=f"Computing spectrograms ({num_workers} workers)"))
    
    # Sort results by index to preserve original order
    results.sort(key=lambda x: x[0])
    
    # Separate spectrograms and labels (skip index)
    spectrograms_all = [result[1] for result in results]
    labels_all = [result[2] for result in results]
    
    spectrograms_np = np.stack(spectrograms_all).astype(np.float32)
    labels_np = np.array(labels_all).astype(np.float32)

    print(f"Created {len(spectrograms_np)} spectrograms of shape {spectrograms_np.shape[1:]} each")
    print(f"Original dataset - Positive labels: {np.sum(labels_np)} / {len(labels_np)} ({np.mean(labels_np)*100:.1f}%)")
    print(f"Strict containment labeling ensures clear task definition")

    return spectrograms_np, labels_np


def balance_dataset(spectrograms, labels, target_ratio=0.5, random_seed=RANDOM_SEED):
    """
    Balance the dataset by undersampling the majority class to achieve target ratio.
    
    Args:
        spectrograms: numpy array of spectrograms
        labels: numpy array of labels (0 or 1)
        target_ratio: target ratio of positive samples (default: 0.5 for 50/50)
        random_seed: random seed for reproducibility
        
    Returns:
        balanced_spectrograms, balanced_labels: balanced dataset
    """
    np.random.seed(random_seed)
    
    # Get indices for each class
    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]
    
    n_pos = len(pos_indices)
    n_neg = len(neg_indices)
    
    print(f"Original distribution: {n_pos} positive, {n_neg} negative")
    
    if n_pos == 0 or n_neg == 0:
        print("Warning: One class has no samples. Returning original dataset.")
        return spectrograms, labels
    
    # Calculate target number of samples for balanced dataset
    if target_ratio == 0.5:
        # For 50/50 split, use the minority class count for both
        target_count = min(n_pos, n_neg)
        n_pos_target = target_count
        n_neg_target = target_count
    else:
        # For other ratios, calculate based on total desired samples
        total_target = min(n_pos / target_ratio, n_neg / (1 - target_ratio))
        n_pos_target = int(total_target * target_ratio)
        n_neg_target = int(total_target * (1 - target_ratio))
    
    print(f"Target distribution: {n_pos_target} positive, {n_neg_target} negative")
    
    # Sample indices for each class
    if n_pos_target <= n_pos:
        selected_pos_indices = np.random.choice(pos_indices, size=n_pos_target, replace=False)
    else:
        selected_pos_indices = pos_indices
        print(f"Warning: Not enough positive samples ({n_pos}), using all available")
    
    if n_neg_target <= n_neg:
        selected_neg_indices = np.random.choice(neg_indices, size=n_neg_target, replace=False)
    else:
        selected_neg_indices = neg_indices
        print(f"Warning: Not enough negative samples ({n_neg}), using all available")
    
    # Combine selected indices
    selected_indices = np.concatenate([selected_pos_indices, selected_neg_indices])
    
    # Shuffle the combined indices
    np.random.shuffle(selected_indices)
    
    # Return balanced dataset
    balanced_spectrograms = spectrograms[selected_indices]
    balanced_labels = labels[selected_indices]
    
    return balanced_spectrograms, balanced_labels

# Split train/val/test with balanced training data only
def save_splits(spectrograms_np, labels_np, output_dir):
    X_train, X_temp, y_train, y_temp = train_test_split(
        spectrograms_np, labels_np, test_size=0.2, shuffle=False
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, shuffle=False
    )

    # Shuffle training data before balancing
    train_indices = np.arange(len(y_train))
    np.random.shuffle(train_indices)
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]

    print(f"\nOriginal split distributions (no balancing):")
    print(f"Train: {np.sum(y_train)} / {len(y_train)} ({np.mean(y_train)*100:.1f}%) positive")
    print(f"Val:   {np.sum(y_val)} / {len(y_val)} ({np.mean(y_val)*100:.1f}%) positive")
    print(f"Test:  {np.sum(y_test)} / {len(y_test)} ({np.mean(y_test)*100:.1f}%) positive")

    # === NO MORE balacing the training set here ===
    # X_train_balanced, y_train_balanced = balance_dataset(...)

    # Compute normalization stats from *natural* training distribution
    train_mean = np.mean(X_train)
    train_std = np.std(X_train)
    print(f'\nTraining data statistics - Mean: {train_mean:.6f}, Std: {train_std:.6f}')

    X_train_norm = (X_train - train_mean) / (train_std + 1e-8)
    X_val_norm   = (X_val   - train_mean) / (train_std + 1e-8)
    X_test_norm  = (X_test  - train_mean) / (train_std + 1e-8)

    os.makedirs(output_dir, exist_ok=True)

    norm_stats = {'mean': train_mean, 'std': train_std}
    torch.save(norm_stats, os.path.join(output_dir, 'normalization_stats.pt'))

    for split_name, X_split, y_split in [
        ('train', X_train_norm, y_train),
        ('val',   X_val_norm,   y_val),
        ('test',  X_test_norm,  y_test),
    ]:
        data_dict = {
            'samples': torch.from_numpy(X_split),
            'labels':  torch.from_numpy(y_split)
        }
        output_path = os.path.join(output_dir, f'{split_name}.pt')
        torch.save(data_dict, output_path)
        print(f'Saved {split_name} set to {output_path} with {len(y_split)} samples.')
        print(f'  - Shape: {X_split.shape}, Positive labels: {np.sum(y_split)} / {len(y_split)} ({np.mean(y_split)*100:.1f}%)')


def load_raw_data(dataset_id):
    dataset_path = f"data_files/{dataset_id}.wav"
    labels_path = f"data_files/{dataset_id}_transitions_ts.txt"
    cuts_path = f"data_files/{dataset_id}_cuts.txt"

    # first line of cuts file as float
    cut_start_seconds = float(os.popen(f'head -n 1 {cuts_path}').read().strip())
    cut_end_seconds = float(os.popen(f'tail -n 1 {cuts_path}').read().strip())

    signal = load_signal(dataset_path, cut_start_seconds=cut_start_seconds, cut_end_seconds=cut_end_seconds)
    transitions_labels = load_transitions_labels(labels_path, cut_start_seconds=cut_start_seconds, cut_end_seconds=cut_end_seconds)
    return signal, transitions_labels

# To run preprocessing and saving:
if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description="Preprocess Ehrenreich spectrogram data")
    # parser.add_argument('--ids', type=str, required=True,
    #                     help="Comma-separated list of dataset IDs (e.g. 'bar20-t2-c2,bar103-t2-c2')")
    # parser.add_argument('--config_type', type=str, default='default',
    #                     help="Configuration type (e.g. 'default', 'strict', etc.)")
    # args = parser.parse_args()

    # ids = [id_.strip() for id_ in args.ids.split(',')]
    # config_type = args.config_type

    # print(f"Using dataset IDs: {ids}")
    # print(f"Configuration type: {config_type}")

    ids = ["bar20-t2-c2"]#, "bar103-t2-c2", "bar2-t1-c1"]
    signals = []
    transitions_labels = []

    for dataset_id in ids:
        signal, transitions = load_raw_data(dataset_id)
        signals.append(signal)
        transitions_labels.append(transitions)

    # Use strict containment labeling with 6s windows and 1s stride
    spectrograms_np, labels_np = preprocess_signals(
        signals, transitions_labels,
        window_duration=6.0,
        stride_duration=1.0,
        n_mels=80,
        n_fft=2048,
        hop_length=256,
        num_workers=None  # Auto-detect number of workers
    )

    print(f'\n=== STRICT CONTAINMENT LABELING RESULTS ===')
    print(f'Dataset overview - Total windows: {len(labels_np)}, Positive labels: {np.sum(labels_np)} ({(np.sum(labels_np)/len(labels_np))*100:.2f}%)')
    print(f'Each window is exactly 6s with no tolerance - clear task definition')

    save_splits(spectrograms_np, labels_np, output_dir='../../data/Ehrenreich_Spectrogram')