from abc import ABC, abstractmethod
import numpy as np
import src.libfmp.c3, src.libfmp.c4, src.libfmp.b
from scipy import signal
from typing import Optional, TYPE_CHECKING
from src.audio.signal import Signal
from typing import Self

if TYPE_CHECKING:
    from src.audio_features.features import NoveltyCurve


class Feature(ABC):
    def __init__(self, data: np.ndarray, sampling_rate: float):
        self._data = data
        self._sampling_rate = sampling_rate
        self._db_scaled = False  # To track if features are in dB scale

    def data(self) -> np.ndarray:
        return self._data

    def sampling_rate(self) -> float:
        return self._sampling_rate

    def is_db_scaled(self) -> bool:
        return self._db_scaled

    def copy(self) -> Self:
        obj_copy = object.__new__(self.__class__)
        obj_copy.__dict__.update(self.__dict__)
        return obj_copy

class BaseFeature(Feature):
    def __init__(self, data: np.ndarray, sampling_rate: float):
        super().__init__(data, sampling_rate)

    def normalize(self, norm: str = '2', threshold=0.001, v=None) -> Self:
        F_normalized = src.libfmp.c3.normalize_feature_sequence(self.data(), norm=norm, threshold=threshold, v=v)
        new_features = self.copy()
        new_features._data = F_normalized
        return new_features
    
    def downsample(self, factor: int = 10) -> Self:
        if factor < 1:
            raise ValueError("/!\\ Downsampling factor must be greater than 1. No downsampling applied.")
        
        new_features = self.copy()
        new_features._data = self.data()[:, ::factor]
        new_features._sampling_rate= self.sampling_rate() / factor
        return new_features
    
    def smooth(self, filter_length: int = 21, window_type='boxcar') -> Self:
        if filter_length < 1 or filter_length % 2 == 0:
            raise ValueError("Filter length must be a positive odd integer")
        
        filt_kernel = np.expand_dims(signal.get_window(window_type, filter_length), axis=0)
        
        new_features = self.copy()
        new_features._data = signal.convolve(self.data(), filt_kernel, mode='same') / float(filter_length)
        return new_features
    
    def log_compress(self, gamma: float = 1) -> Self:
        new_features = self.copy()
        new_features._data = np.log(1 + gamma * self.data())
        return new_features
    
    def _plot(self, 
              features: np.ndarray, 
              feature_sr: float, 
              features_name: str, 
              x_axis_type: str = 'time',
              time_annotations: Optional[list] = None,
              original_signal: Optional[Signal] = None):
        
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors

        if x_axis_type not in ['time', 'frame']:
            raise ValueError("x_axis_type must be 'time' or 'frame'")
        
        if features.ndim == 1:
            features = features[np.newaxis, :]

        # Add small epsilon to avoid log(0) for linear data
        epsilon = np.finfo(float).eps
        features_safe = np.maximum(features, epsilon)
        
        # Use grayscale colormap for both modes
        try:
            cmap = src.libfmp.b.compressed_gray_cmap(alpha=-10)
            # cmap = 'gray_r'
        except (ImportError, AttributeError):
            cmap = 'gray_r'  # reversed grayscale (white=high, black=low)

        # Compute coordinate systems for alignment
        n_frames = features.shape[1]
        feature_frames = np.arange(n_frames)
        feature_time = feature_frames / feature_sr
        
        # Create subplots based on whether we have original signal
        if original_signal is not None and original_signal.sample_rate is not None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1, 2])
            
            if x_axis_type == 'time':
                # Both plots use time axis - align by stretching signal plot to match feature time range
                audio_time = np.arange(len(original_signal.samples)) / original_signal.sample_rate
                max_time = feature_time[-1]
                
                # Plot original signal with its natural time coordinates
                ax1.plot(audio_time, original_signal.samples, color='gray', linewidth=0.8)
                ax1.set_xlabel('Time (s)')
                ax1.set_xlim(0, max_time)  # Align x-axis limits
                ax1.set_ylabel('Amplitude')
                ax1.set_title(f'Original Audio Signal (aligned with features)')
                ax1.grid(True, alpha=0.3)
                
                # Plot features with time-based heatmap
                
                im = ax2.imshow(features, aspect='auto', origin='lower', interpolation='nearest', cmap=cmap,
                                extent=(0, max_time, 0, features.shape[0]))
                ax2.set_xlabel('Time (s)')
                ax2.set_xlim(0, max_time)  # Match top plot x-axis limits
                
            else:  # x_axis_type == 'frame'
                # Both plots use frame axis - just change x-axis labels, don't stretch data
                
                # Plot original signal with natural sample indices but frame-based x-axis
                ax1.plot(original_signal.samples, color='gray', linewidth=0.8)
                ax1.set_xlabel('Frames')
                ax1.set_ylabel('Amplitude')
                ax1.set_title(f'Original Audio Signal (aligned with features)')
                ax1.grid(True, alpha=0.3)
                
                # Convert sample indices to frame indices for x-axis ticks
                sample_to_frame_ratio = len(original_signal.samples) / n_frames
                
                # Set x-axis limits to match frame count
                ax1.set_xlim(0, len(original_signal.samples) - 1)
                
                # Create custom x-axis ticks that show frame numbers
                signal_ticks = ax1.get_xticks()
                frame_tick_labels = [f'{int(tick / sample_to_frame_ratio)}' for tick in signal_ticks if tick >= 0 and tick < len(original_signal.samples)]
                ax1.set_xticklabels(frame_tick_labels)
                
                # Plot features with frame-based heatmap
                im = ax2.imshow(features, aspect='auto', origin='lower', cmap=cmap,
                                extent=(0, n_frames - 1, 0, features.shape[0]))
                ax2.set_xlabel('Frames')
                ax2.set_xlim(0, n_frames - 1)  # Match feature frame range
            
            ax2.set_ylabel('Feature Index')
            ax2.set_title(f'{features_name} Features')
            
            # Add horizontal colorbar below the feature plot
            cbar = plt.colorbar(im, ax=ax2, orientation='horizontal', pad=0.15, format='%+2.0f dB' if self.is_db_scaled() else None)
            
            # Plot time annotations as colored rectangles on both plots
            if time_annotations is not None and len(time_annotations) > 0:
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
                            start_coord = start_time * feature_sr
                            end_coord = end_time * feature_sr
                        
                        # Plot colored rectangle over original signal
                        signal_max = np.max(np.abs(original_signal.samples))
                        ax1.axvspan(start_coord, end_coord, 
                                   alpha=0.2, color=f'C{i % 10}', 
                                   label=label)
                        ax1.text((start_coord + end_coord) / 2, signal_max * 0.8, 
                               label, ha='center', va='center', 
                               fontsize=8, rotation=0)
                        
                        # Plot colored rectangle over features
                        feature_height = features.shape[0]
                        ax2.axvspan(start_coord, end_coord, 
                                   alpha=0.2, color=f'C{i % 10}')
                        ax2.text((start_coord + end_coord) / 2, feature_height * 0.9, 
                               label, ha='center', va='center', 
                               fontsize=8, rotation=0, color='white')
            
        else:
            # Single plot for features only
            fig, ax = plt.subplots(figsize=(12, 4))
            
            if x_axis_type == 'time':
                # Create time-based heatmap using grayscale
                im = ax.imshow(features, aspect='auto', origin='lower', cmap=cmap,
                            extent=(0, feature_time[-1], 0, features.shape[0]))
                ax.set_xlabel('Time (s)')
            else:
                # Create frame-based heatmap using grayscale
                im = ax.imshow(features, aspect='auto', origin='lower', cmap=cmap,
                            extent=(0, n_frames - 1, 0, features.shape[0]))
                ax.set_xlabel('Frames')

            ax.set_ylabel('Feature Index')
            plt.colorbar(im, format='%+2.0f dB' if self.is_db_scaled() else None)
            plt.title(f'{features_name} Features')
            
            # Plot time annotations as colored rectangles on features plot
            if time_annotations is not None and len(time_annotations) > 0:
                feature_height = features.shape[0]
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
                            start_coord = start_time * feature_sr
                            end_coord = end_time * feature_sr
                        
                        # Plot colored rectangle over features
                        ax.axvspan(start_coord, end_coord, 
                                  alpha=0.2, color=f'C{i % 10}', 
                                  label=label)
                        ax.text((start_coord + end_coord) / 2, feature_height * 0.9, 
                               label, ha='center', va='center', 
                               fontsize=8, rotation=0, color='white')

        plt.tight_layout()
        plt.show()

class SimilarityMatrix(Feature):
    def __init__(self, data: np.ndarray, sampling_rate: float):
        super().__init__(data, sampling_rate)

    @abstractmethod
    def compute_novelty_curve(self, *args, **kwargs) -> 'NoveltyCurve':
        pass

    def _plot(self, ssm_name: str, x_axis_type: str = 'time', time_annotations: Optional[list] = None,
                    original_base_feature: Optional[BaseFeature] = None):
            """
            Plot a Self-Similarity Matrix (SSM) optionally stacked below a features heatmap.
            Layout mimics src.libfmp.c4.plot_feature_ssm for perfect alignment.
            """
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            
            if x_axis_type not in ['time', 'frame']:
                raise ValueError("x_axis_type must be 'time' or 'frame'")

            n_frames = self.data().shape[0]
            ssm_time = np.arange(n_frames) / self.sampling_rate()

            # Use src.libfmp compressed gray colormap for SSM
            try:
                cmap_ssm = src.libfmp.b.compressed_gray_cmap(alpha=-10)
                # cmap_ssm = 'gray_r'
            except (ImportError, AttributeError):
                cmap_ssm = 'gray_r'

            # Case 1: with features - mimic src.libfmp layout exactly ----------------------
            if original_base_feature is not None:
                max_time = ssm_time[-1]
                
                fig_width = 8
                fig_height = fig_width * 1.25  # Adjust height for better aspect ratio
                
                # Adjust layout based on whether annotations are provided
                if time_annotations is not None and len(time_annotations) > 0:
                    # Full 3x3 layout with annotation areas
                    fig, ax = plt.subplots(3, 3, 
                                        gridspec_kw={'width_ratios': [0.1, 1, 0.05],
                                                    'wspace': 0.2,
                                                    'height_ratios': [0.3, 1, 0.1]},
                                        figsize=(fig_width, fig_height))

                else:
                    # Simplified 2x3 layout without annotation areas
                    fig, ax = plt.subplots(2, 3, 
                                        gridspec_kw={'width_ratios': [0.1, 1, 0.05],
                                                    'wspace': 0.2,
                                                    'height_ratios': [0.3, 1]},
                                        figsize=(fig_width, fig_height * 0.9))
                
                # Features plot - top center position [0, 1] with colorbar [0, 2]
                if x_axis_type == 'time':
                    extent_feat = (0, max_time, 0, original_base_feature.data().shape[0])
                    extent_ssm = (0, max_time, 0, max_time)
                    xlabel, ylabel = 'Time (s)', 'Time (s)'
                else:
                    extent_feat = (0, n_frames - 1, 0, original_base_feature.data().shape[0])
                    extent_ssm = (0, n_frames - 1, 0, n_frames - 1)
                    xlabel, ylabel = 'Frames', 'Frames'
                
                # Plot features using src.libfmp.b.plot_matrix style
                im_feat = ax[0, 1].imshow(original_base_feature.data(), aspect='auto', origin='lower', interpolation='nearest',
                                        cmap=cmap_ssm, extent=extent_feat)
                ax[0, 1].set_ylabel('Feature Index')
                ax[0, 1].set_title(f'Features used for {ssm_name}')
                ax[0, 1].set_xlabel('')  # No xlabel for top plot
                
                # Features colorbar
                cbar_feat = plt.colorbar(im_feat, cax=ax[0, 2])
                
                # Turn off corner axes like src.libfmp
                ax[0, 0].axis('off')
                
                # SSM plot - center position [1, 1] with colorbar [1, 2]
                im_ssm = ax[1, 1].imshow(self.data(), aspect='auto', origin='lower', 
                                        cmap=cmap_ssm, extent=extent_ssm, 
                                        interpolation='nearest')
                
                # Handle SSM axis labels based on annotation presence
                if time_annotations is not None and len(time_annotations) > 0:
                    # With annotations: labels go on annotation areas, main plot has no labels
                    ax[1, 1].set_xlabel('')  # No labels on main SSM plot
                    ax[1, 1].set_ylabel('')
                    ax[1, 1].set_xticks([])  # Remove ticks like src.libfmp
                    ax[1, 1].set_yticks([])
                else:
                    # Without annotations: labels go directly on main SSM plot
                    ax[1, 1].set_xlabel(xlabel)
                    ax[1, 1].set_ylabel(ylabel)
                    ax[1, 1].set_title(f'{ssm_name}')
                
                # SSM colorbar
                cbar_ssm = plt.colorbar(im_ssm, cax=ax[1, 2])
                
                # Turn off left axis for SSM row
                ax[1, 0].axis('off')
                
                # Handle annotation areas only if annotations are provided
                if time_annotations is not None and len(time_annotations) > 0:
                    # Bottom annotation area [2, 1] - populate with actual annotations
                    ax[2, 1].set_xlim(extent_ssm[0], extent_ssm[1])
                    ax[2, 1].set_ylim(-0.5, 0.5)  # Small height for annotation area
                    ax[2, 1].set_xlabel(xlabel)
                    ax[2, 1].set_ylabel('')
                    ax[2, 1].tick_params(left=False, labelleft=False)
                    
                    # Left annotation area [1, 0] - populate with actual annotations
                    ax[1, 0].set_ylim(extent_ssm[2], extent_ssm[3])
                    ax[1, 0].set_xlim(-0.5, 0.5)  # Small width for annotation area
                    ax[1, 0].set_ylabel(ylabel)
                    ax[1, 0].set_xlabel('')
                    ax[1, 0].tick_params(bottom=False, labelbottom=False)
                    ax[1, 0].axis('on')  # Turn back on for annotations
                    
                    # Turn off remaining corner axes
                    ax[2, 2].axis('off')
                    ax[2, 0].axis('off')
                    
                    # Convert time annotations to frame indices if x_axis_type is 'frame'
                    if x_axis_type == 'frame':
                        time_annotations = [(start * self.sampling_rate(), end * self.sampling_rate(), _) for start, end, _ in time_annotations]

                    # Manual annotation plotting with preserved axis settings
                    for i, ann in enumerate(time_annotations):
                        if len(ann) >= 2:  # Expecting [start, end, label] format
                            start, end = ann[0], ann[1]
                            
                            # Use coordinates directly as they are (frames or time based on x_axis_type)
                            start_coord = start
                            end_coord = end
                            
                            # Bottom horizontal segments
                            ax[2, 1].barh(0, end_coord - start_coord, left=start_coord, 
                                        height=1, alpha=0.2, 
                                        color=f'C{i % 10}')  # Cycle through colors
                            ax[2, 1].text((start_coord + end_coord) / 2, 0, "", 
                                        ha='center', va='center', fontsize=8, 
                                        rotation=0 if (end_coord - start_coord) > (extent_ssm[1] - extent_ssm[0]) * 0.1 else 90)
                            
                            # Left vertical segments
                            ax[1, 0].bar(0, end_coord - start_coord, bottom=start_coord, 
                                       width=1, alpha=0.2, 
                                       color=f'C{i % 10}')  # Same color as horizontal
                            ax[1, 0].text(0, (start_coord + end_coord) / 2, "", 
                                        ha='center', va='center', fontsize=8, 
                                        rotation=90)
                    
                    # Ensure axis labels and ticks are properly set and visible
                    # Bottom annotation area - preserve xlabel and show ticks
                    ax[2, 1].set_xlabel(xlabel)
                    ax[2, 1].set_ylabel('')
                    ax[2, 1].tick_params(bottom=True, labelbottom=True, left=False, labelleft=False)
                    
                    # Left annotation area - preserve ylabel and show ticks
                    ax[1, 0].set_ylabel(ylabel)
                    ax[1, 0].set_xlabel('')
                    ax[1, 0].tick_params(left=True, labelleft=True, bottom=False, labelbottom=False)

            # Case 2: SSM only - simple square layout ----------------------------------
            else:
                fig, ax_ssm = plt.subplots(figsize=(8, 8))
                
                if x_axis_type == 'time':
                    extent = (0, ssm_time[-1], 0, ssm_time[-1])
                    xlabel = ylabel = 'Time (s)'
                else:
                    extent = (0, n_frames - 1, 0, n_frames - 1)
                    xlabel = ylabel = 'Frames'
                    
                im_ssm = ax_ssm.imshow(self.data(), aspect='equal', origin='lower', 
                                    cmap=cmap_ssm, extent=extent, 
                                    interpolation='nearest')
                ax_ssm.set_xlabel(xlabel)
                ax_ssm.set_ylabel(ylabel)
                ax_ssm.set_title(f'{ssm_name}')
                
                # Add colorbar
                plt.colorbar(im_ssm)

            plt.tight_layout()
            plt.show()