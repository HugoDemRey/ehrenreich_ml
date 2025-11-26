from scipy.ndimage import gaussian_filter1d
import numpy as np
import librosa
import src.libfmp, src.libfmp.c4, src.libfmp.b
from abc import ABC, abstractmethod
from typing import Optional, Union
from scipy import signal
from src.audio.signal import Signal
from typing import Self
from src.interfaces.feature import BaseFeature, Feature, SimilarityMatrix
class Spectrogram(BaseFeature):
    def __init__(self, S: np.ndarray, S_sr: float):
        super().__init__(S, S_sr)

    def plot(self, x_axis_type: str = 'time',
             time_annotations: Optional[list] = None,
             original_signal: Optional[Signal] = None):
        self._plot(self.data(), self.sampling_rate(), "Spectrogram", x_axis_type, time_annotations, original_signal)
        
    def to_db(self) -> 'Spectrogram':
        if not self._db_scaled:
            new_spec = self.copy()
            new_spec._data = librosa.power_to_db(self.data(), ref=np.max)
            new_spec._db_scaled = True
            return new_spec
        else:
            print("/!\\ Spectrogram is already in dB scale.")
            return self


class Chromagram(BaseFeature):
    def __init__(self, C: np.ndarray, C_sr: float):
        super().__init__(C, C_sr)

    def plot(self, x_axis_type: str = 'time',
             time_annotations: Optional[list] = None,
             original_signal: Optional[Signal] = None):
        self._plot(self.data(), self.sampling_rate(), "Chromagram", x_axis_type, time_annotations, original_signal)

class MFCC(BaseFeature):
    def __init__(self, mfcc: np.ndarray, mfcc_sr: float):
        super().__init__(mfcc, mfcc_sr)

    def plot(self, x_axis_type: str = 'time',
             time_annotations: Optional[list] = None,
             original_signal: Optional[Signal] = None):
        self._plot(self.data(), self.sampling_rate(), "MFCC", x_axis_type, time_annotations, original_signal)

class Tempogram(BaseFeature):
    def __init__(self, T: np.ndarray, T_sr: float):
        super().__init__(T, T_sr)

    def plot(self, x_axis_type: str = 'time',
             time_annotations: Optional[list] = None,
             original_signal: Optional[Signal] = None):
        self._plot(self.data(), self.sampling_rate(), "Tempogram", x_axis_type, time_annotations, original_signal)

class HRPS(BaseFeature):
    def __init__(self, H: np.ndarray, H_sr: float):
        super().__init__(H, H_sr)

    def plot(self, x_axis_type: str = 'time',
             time_annotations: Optional[list] = None,
             original_signal: Optional[Signal] = None):
        self._plot(self.data(), self.sampling_rate(), "HRPS", x_axis_type, time_annotations, original_signal)

    @property
    def harmonic_data(self) -> np.ndarray:
        return self.data()[0, :]

    @property    
    def residual_data(self) -> np.ndarray:
        return self.data()[1, :]
    
    @property
    def percussive_data(self) -> np.ndarray:
        return self.data()[2, :]
    
    @staticmethod
    def _min_max_normalize(data: np.ndarray) -> np.ndarray:
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val == min_val:
            return np.zeros_like(data)
        return (data - min_val) / (max_val - min_val)

    def find_peaks(self, threshold: float = 0.1, distance_seconds: int = 10) -> np.ndarray:
            """Find peaks in the percussive component of the HRPS feature.
             Args:
                threshold (float): Relative height threshold for peak detection (0 to 1).
                distance_seconds (int): Minimum distance between peaks in seconds.
            Returns:
                tuple[np.ndarray, np.ndarray]: Indices of detected peaks and the corresponding harmonic intensity curve.
            """
            from scipy.signal import find_peaks as scipy_find_peaks
            
            # Calculate absolute height threshold based on relative threshold
            height_threshold = np.max(self.residual_data) * threshold
            
            # Use scipy's find_peaks to identify peaks
            distance_samples = int(distance_seconds * self.sampling_rate())
            peaks, _ = scipy_find_peaks(self.residual_data, height=height_threshold, distance=distance_samples)
            return peaks

class SilenceCurve(BaseFeature):
    def __init__(self, silence_curve: np.ndarray, silence_sr: float):
        print("(SILENCE CURVE): Original silence curve dimensions: ", silence_curve.ndim)
        sc = self._min_max_normalize(silence_curve)

        # Enhance contrast Near 1
        power_factor = 25
        sc = np.power(sc, power_factor)

        if sc.ndim == 1:
            # transform to 2D array with shape (1, N)
            sc = sc[np.newaxis, :]
            print("(SILENCE CURVE): Converted 1D silence curve to 2D array: shape ", sc.shape)
        super().__init__(sc, silence_sr)

    def _min_max_normalize(self, data) -> np.ndarray:
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val == min_val:
            return np.zeros_like(data)
        return (data - min_val) / (max_val - min_val)

    def plot(self, x_axis_type: str = 'time',
             time_annotations: Optional[list] = None,
             original_signal: Optional[Signal] = None):
        self._plot(self.data(), self.sampling_rate(), "Silence Curve", x_axis_type, time_annotations, original_signal)

    def find_peaks(self, threshold: float = 0.5, distance_seconds: int = 10) -> np.ndarray:
            """Find peaks in the silence curve.
             Args:
                threshold (float): Relative height threshold for peak detection (0 to 1).
                distance_seconds (int): Minimum distance between peaks in seconds.
            Returns:
                np.ndarray: Indices of detected peaks.
            """
            from scipy.signal import find_peaks as scipy_find_peaks
            
            # Calculate absolute height threshold based on relative threshold
            height_threshold = np.max(self.data()) * threshold
            
            # Use scipy's find_peaks to identify peaks
            distance_samples = int(distance_seconds * self.sampling_rate())
            peaks, _ = scipy_find_peaks(self.data()[0, :], height=height_threshold, distance=distance_samples)
            return peaks

class SelfSimilarityMatrix(SimilarityMatrix):
    def __init__(self, ssm: np.ndarray, ssm_sr: float):
        super().__init__(ssm, ssm_sr)

    def plot(self, x_axis_type: str = 'time',
             time_annotations: Optional[list] = None,
             original_base_feature: Optional[BaseFeature] = None):
        self._plot("SelfSimilarityMatrix", x_axis_type, time_annotations, original_base_feature)

    def threshold(self, 
                  thresh: float = 0.5, 
                  strategy: str = 'relative',
                  scale: bool = True,
                  penalty: float = 0.0,
                  binarize: bool = False) -> 'SelfSimilarityMatrix':
        
        S_thresholded = src.libfmp.c4.threshold_matrix(self.data(), 
                                                   thresh=thresh, 
                                                   strategy=strategy,
                                                   scale=scale, 
                                                   penalty=penalty, 
                                                   binarize=binarize)
        
        return SelfSimilarityMatrix(S_thresholded, self.sampling_rate())

    def compute_novelty_curve(self,
                             kernel_size: int = 16,
                             variance: float = 0.5,
                             exclude_borders: bool = True) -> 'NoveltyCurve':
        novelty_curve = src.libfmp.c4.compute_novelty_ssm(self.data(),
                                                         L=kernel_size,
                                                         var=variance,
                                                         exclude=exclude_borders)
        novelty_sr = self.sampling_rate()
        return NoveltyCurve(novelty_curve, novelty_sr)
    
    def compute_novelty_curve_fast(self,
                             kernel_size: int = 16,
                             variance: float = 0.5,
                             exclude_borders: bool = True) -> 'NoveltyCurve':

        kernel = src.libfmp.c4.compute_kernel_checkerboard_gaussian(L=kernel_size, var=variance)
        N = self.data().shape[0]
        M = 2*kernel_size + 1

        # Pad S with zeros manually
        S_padded = np.pad(self.data(), pad_width=kernel_size, mode='constant')

        # Create sliding window view of shape (N, M, M)
        windows = np.lib.stride_tricks.sliding_window_view(S_padded, (M, M))
        # windows shape: (N+2L - M +1, N+2L - M +1, M, M) = (N, N, M, M) here
        # We want only the diagonal patches: windows[i, i, :, :]
        diagonal_windows = windows[np.arange(N), np.arange(N)]

        # Compute novelty by element-wise multiply and sum over kernel dims
        nov = np.einsum('ij,ij->i', diagonal_windows, kernel)

        if exclude_borders:
            right = min(kernel_size, N)
            left = max(0, N - kernel_size)
            nov[:right] = 0
            nov[left:] = 0

        novelty_sr = self.sampling_rate()
        return NoveltyCurve(nov, novelty_sr)

class TimeLagMatrix(SimilarityMatrix):
    def __init__(self, tlm: np.ndarray, tlm_sr: float):
        super().__init__(tlm, tlm_sr)

    def plot(self, x_axis_type: str = 'time',
             
             time_annotations: Optional[list] = None,
             original_base_feature: Optional[BaseFeature] = None):
        self._plot("Time-Lag Matrix", x_axis_type, time_annotations, original_base_feature)

    def compute_novelty_curve(self, padding: bool = True) -> 'NoveltyCurve':
        N = self.data().shape[0]
        if padding:
            nov = np.zeros(N)
        else:
            nov = np.zeros(N-1)
        for n in range(N-1):
            nov[n] = np.linalg.norm(self.data()[:, n+1] - self.data()[:, n])
        
        nc: NoveltyCurve = NoveltyCurve(nov, self.sampling_rate())
        return nc

class NoveltyCurve(Feature):
    def __init__(self, novelty: np.ndarray, novelty_sr: float):
        normalized_novelty = self._min_max_normalize(novelty)
        super().__init__(normalized_novelty, novelty_sr)

    def smooth(self, sigma: float=1.0) -> 'NoveltyCurve':
        data = self.data()
        if data is None:
            raise ValueError("Data cannot be None.")
        smoothed_curve = gaussian_filter1d(data, sigma=sigma)

        return NoveltyCurve(smoothed_curve, self.sampling_rate())

    def find_peaks(self, threshold: float = 0.1, distance_seconds: int = 10) -> np.ndarray:
            from scipy.signal import find_peaks as scipy_find_peaks
            
            # Calculate absolute height threshold based on relative threshold
            height_threshold = np.max(self.data()) * threshold
            
            # Use scipy's find_peaks to identify peaks
            distance_samples = int(distance_seconds * self.sampling_rate())
            peaks, _ = scipy_find_peaks(self.data(), height=height_threshold, distance=distance_samples)

            return peaks
    
    def _min_max_normalize(self, data) -> np.ndarray:
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val == min_val:
            return np.zeros_like(data)
        return (data - min_val) / (max_val - min_val)
    
    @staticmethod
    def combine(ncs: list['NoveltyCurve'], weights: list[float] = None, method: str = 'mean') -> 'NoveltyCurve':
        if not ncs or not weights or len(ncs) != len(weights):
            raise ValueError("The list of novelty curves and weights must be non-empty and of the same length.")

        # Ensure all novelty curves have the same length
        length = ncs[0].data().shape[0]
        sr = ncs[0].sampling_rate()
        for nc in ncs:
            if nc.data().shape[0] != length or nc.sampling_rate() != sr:
                raise ValueError(f"All novelty curves must have the same length and sampling rate to combine. {nc.data().shape[0]} != {length} or {nc.sampling_rate()} != {sr}")

        weighted_data = [nc.data() * weight for nc, weight in zip(ncs, weights)]

        # Combine based on the specified method
        if method == 'mean':
            combined_data = np.mean(weighted_data, axis=0)
        elif method == 'max':
            combined_data = np.max(weighted_data, axis=0)
        elif method == 'weighted':
            combined_data = np.sum(weighted_data, axis=0) / np.sum(weights)
        else:
            raise ValueError("Method must be 'mean', 'max', or 'weighted'.")

        return NoveltyCurve(combined_data, ncs[0].sampling_rate())

    def combine_with(self, novelty_curves: Union[list['NoveltyCurve'], 'NoveltyCurve'], method: str = 'mean') -> 'NoveltyCurve':
        if not novelty_curves:
            raise ValueError("The list of novelty curves is empty.")

        # If a single NoveltyCurve is provided, convert it to a list
        if isinstance(novelty_curves, NoveltyCurve):
            novelty_curves = [novelty_curves]

        # Stretch all novelty curves to match the length of self
        length = self.data().shape[0]
        
        # Stretch all novelty curves to match the length -> Modifying nc values
        nc_data_aligned = [self.data()]  # Start with self
        for nc in novelty_curves:
            if nc.data().shape[0] != length:
                # Stretching using linear interpolation
                stretched_data = np.interp(
                    np.linspace(0, nc.data().shape[0] - 1, length),
                    np.arange(nc.data().shape[0]),
                    nc.data()
                )
                nc_data_aligned.append(stretched_data)
            else:
                # Curve already has the correct length, add it directly
                nc_data_aligned.append(nc.data())

        # Stack the data from all novelty curves
        stacked_data = np.vstack([data for data in nc_data_aligned])
        
        # Combine based on the specified method
        if method == 'mean':
            combined_data = np.mean(stacked_data, axis=0)
        elif method == 'max':
            combined_data = np.max(stacked_data, axis=0)
        else:
            raise ValueError("Method must be 'mean' or 'max'.")
        
        return NoveltyCurve(combined_data, self.sampling_rate())

    def plot(self, x_axis_type: str = 'time',
            novelty_name: str = "Novelty Curve",
            time_annotations: Optional[list] = None,
            peaks: Optional[np.ndarray] = None, save_path: Optional[str] = None):
        
        """
        Plot a novelty curve with identified peaks marked as vertical dashed red lines.
        Vertical lines span the full height of the axes.
        
        Args:
            novelty_curve (np.ndarray): Input novelty curve.
            novelty_sr (float): Sampling rate of the novelty curve.
            peaks (np.ndarray): Array of peak indices.
            novelty_name (str): Name for the plot title.
            x_axis_type (str): 'time' or 'frame' for x-axis type.
            time_annotations (Optional[list]): List of time-based annotations in format [start_time, end_time, label].
                Always in time units (seconds) regardless of x_axis_type.
        """
        import matplotlib.pyplot as plt

        if (type(time_annotations) is np.ndarray):
            time_annotations = time_annotations.tolist()

        if x_axis_type not in ['time', 'frame']:
            raise ValueError("x_axis_type must be 'time' or 'frame'")

        n_frames = self.data().shape[0]
        novelty_time = np.arange(n_frames) / self.sampling_rate()

        fig, ax = plt.subplots(figsize=(10, 4))

        if peaks is None:
            peaks = np.empty((0,), dtype=int)

        # Plot novelty curve depending on x axis type
        if x_axis_type == 'time':
            x_values = peaks / self.sampling_rate()
            ax.plot(novelty_time, self.data(), label='Novelty Curve', color='black')
            ax.set_xlabel('Time (s)')
        else:
            x_values = peaks
            ax.plot(np.arange(n_frames), self.data(), label='Novelty Curve', color='black')
            ax.set_xlabel('Frames')

        # Draw vertical dashed red lines spanning the full axis height
        for i, xv in enumerate(x_values):
            # label only the first line so legend remains clean
            ax.axvline(xv, color='red', linestyle='--', linewidth=0.5, label='Predictions' if i == 0 else None)

            ax.set_title(f'Novelty Curve with Peaks: {novelty_name}')
            ax.set_ylabel('Novelty')
            max_val = np.max(self.data())
            if max_val <= 0:
                max_val = 1.0
            ax.set_ylim(0, max_val * 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot time annotations as colored rectangles on top of novelty curve
        if time_annotations is not None and len(time_annotations) > 0 and type(time_annotations[0]) == list:
            novelty_max = np.max(self.data()) * 1.1
            for i, ann in enumerate(time_annotations):
                if len(ann) >= 2:  # Expecting [start_time, end_time, label] format
                    start_time, end_time = ann[0], ann[1]
                    label = ""
                    
                    # Convert time annotations to appropriate x-coordinates
                    if x_axis_type == 'time':
                        start_coord = start_time
                        end_coord = end_time
                    else:
                        # Convert time to frames
                        start_coord = start_time * self.sampling_rate()
                        end_coord = end_time * self.sampling_rate()

                    # Plot colored rectangle over novelty curve
                    ax.axvspan(start_coord, end_coord, 
                                alpha=0.2, color=f'C{i % 10}', 
                                label=label)
                    
                    # Add text label at the center
                    ax.text((start_coord + end_coord) / 2, novelty_max * 0.9, 
                            label, ha='center', va='center', 
                            fontsize=8, rotation=0)
        elif time_annotations is not None and len(time_annotations) > 0 and type(time_annotations[0]) == float: # only the transitions
            novelty_max = np.max(self.data()) * 1.1
            for i, ann in enumerate(time_annotations):                    
                # Convert time annotations to appropriate x-coordinates
                if x_axis_type == 'time':
                    coord = ann
                else:
                    # Convert time to frames
                    coord = ann * self.sampling_rate()

                # Plot vertical line over novelty curve
                ax.axvline(coord, 
                            alpha=0.5, color=f'C{i % 10}', 
                            linestyle='--',
                            label=f'Transition {i+1}')
                
                # Add text label at the top
                ax.text(coord, novelty_max * 0.9, 
                        f'Transition {i+1}', ha='center', va='center', 
                        fontsize=8, rotation=90)
        
        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()