import numpy as np
import libfmp.c4
import libfmp.b
from typing import Optional
from scipy.ndimage import filters





class FeatureProcessing:
    class SimilarityMatrix:
        @staticmethod
        def compute_ti_ssm(
            features: np.ndarray, 
            smoothing_filter_length: int = 1, 
            smoothing_filter_direction: int = 2,
            shift_set: np.ndarray = np.array([0]),
            tempo_relative_set: np.ndarray = np.array([1]),
            threshold: float = 0.15,
            threshhold_strategy: str = 'relative',
            threshold_scale: bool = True,
            threshold_penalty: float = 0.0,
            threshold_binarize: bool = False
            ):
            """
            Compute a transposition-invariant Self-Similarity Matrix (SSM) from audio features.
            
            This function creates an SSM that is robust to key transpositions and tempo variations
            by computing similarity matrices with different shift and tempo scaling parameters,
            then applying thresholding to enhance relevant structures.
            
            Args:
                features (np.ndarray): Input feature matrix of shape (F, N) where F is the 
                    number of feature dimensions and N is the number of time frames.
                smoothing_filter_length (int, optional): Length of the smoothing filter applied 
                    during SSM computation. Longer filters provide more temporal smoothing. 
                    Defaults to 1.
                smoothing_filter_direction (int, optional): Direction of smoothing filter application.
                    - 0: forward direction only
                    - 1: backward direction only  
                    - 2: both directions (bidirectional)
                    Defaults to 2.
                shift_set (np.ndarray, optional): Array of shift indices for transposition invariance.
                    Each value represents a cyclic shift applied to features (e.g., for chroma features,
                    shifts correspond to semitone transpositions). Defaults to np.array([0]).
                tempo_relative_set (np.ndarray, optional): Array of relative tempo scaling factors.
                    Values > 1 correspond to faster tempos, values < 1 to slower tempos.
                    Defaults to np.array([1]).
                threshold (float, optional): Threshold value for matrix thresholding. The meaning
                    depends on the thresholding strategy. Defaults to 0.15.
                threshhold_strategy (str, optional): Strategy for thresholding the similarity matrix.
                    Common options include 'relative', 'absolute'. Defaults to 'relative'.
                threshold_scale (bool, optional): If True, scales positive values to range [0,1]
                    after thresholding. Defaults to True.
                threshold_penalty (float, optional): Value assigned to elements below threshold.
                    Defaults to 0.0.
                threshold_binarize (bool, optional): If True, converts the final matrix to binary
                    (positive values become 1, others become 0). Defaults to False.
                    
            Returns:
                np.ndarray: Thresholded transposition-invariant SSM of shape (N, N) where N
                    is the number of time frames in the input feature sequence.
                    
            Note:
                This function uses libfmp.c4.compute_sm_ti for SSM computation and 
                libfmp.c4.threshold_matrix for post-processing. The resulting SSM can be used
                for structural analysis, music segmentation, and repetition detection tasks.
            """

            S, _ = libfmp.c4.compute_sm_ti(features, 
                                        features, 
                                        L=smoothing_filter_length, 
                                        tempo_rel_set=tempo_relative_set, 
                                        shift_set=shift_set, 
                                        direction=smoothing_filter_direction
                                        )
            
            S = libfmp.c4.threshold_matrix(S, 
                                        thresh=threshold, 
                                        strategy=threshhold_strategy,
                                        scale=threshold_scale, 
                                        penalty=threshold_penalty, 
                                        binarize=threshold_binarize
                                        )
            
            return S    

        @staticmethod
        def compute_tlm(ssm: np.ndarray, circular: bool = True) -> np.ndarray:
            """Computation of (circular) time-lag representation

            Notebook: C4/C4S4_StructureFeature.ipynb

            Args:
                ssm (np.ndarray): Self-similarity matrix
                circular (bool): Computes circular version (Default value = True)

            Returns:
                L (np.ndarray): (Circular) time-lag representation of S
            """
            N = ssm.shape[0]
            if circular:
                L = np.zeros((N, N))
                for n in range(N):
                    L[:, n] = np.roll(ssm[:, n], -n)
            else:
                L = np.zeros((2*N-1, N))
                for n in range(N):
                    L[((N-1)-n):((2*N)-1-n), n] = ssm[:, n]
            return L
        
        @staticmethod
        def apply_filter(M: np.ndarray, filter_type: str = "median", size: int = 3) -> np.ndarray:
            if filter_type == "median":
                from scipy.ndimage import median_filter
                M_filtered = median_filter(M, size=size)
            elif filter_type == "gaussian":
                from scipy.ndimage import gaussian_filter
                M_filtered = gaussian_filter(M, sigma=size)
            else:
                raise ValueError(f"Unsupported filter_type: {filter_type!r}. Supported types are 'median' and 'gaussian'.")
                
            return M_filtered            
            

        @staticmethod
        def print_sm_info(S: np.ndarray, S_name: str, features: Optional[np.ndarray] = None, feature_sr: Optional[float] = None):
            """
            Print a beautifully formatted analysis of a similarity Matrix (SM).
            
            Args:
                S (np.ndarray): Input self-similarity matrix of shape (N, N) where N is the number of time frames.
                features (Optional[np.ndarray]): Original feature matrix used to compute the SSM, if available.
                feature_sr (Optional[float]): Feature sampling rate, if available.
            """
            # Header with decorative border - simplified for notebook compatibility
            analysis_name = "SSM" if np.allclose(S, S.T) else "SM"
            title = f"{S_name.upper()} {analysis_name} ANALYSIS"
            # Use fixed width border that works well in most notebook environments
            print(f"***  {title}  ***")
            
            # SSM Matrix Properties
            n_frames = S.shape[0]
            
            print(f"\n🔲 Matrix Properties:")
            print(f"   ├─ Dimensions: {S.shape[0]} × {S.shape[1]} frames")
            print(f"   ├─ Symmetry: {'Yes' if np.allclose(S, S.T) else 'No'}")
            print(f"   └─ Memory Usage: {S.nbytes / 1024:.1f} KB")
            
            # Features → SSM Transformation (if features available)
            if features is not None and feature_sr is not None:
                n_features, n_feature_frames = features.shape
                duration_features = n_feature_frames / feature_sr
                complexity_reduction = (n_features * n_feature_frames) / (n_frames * n_frames)
                
                print(f"\n🔄 Features → SSM Transformation:")
                print(f"   ├─ Original Features: {n_features} features × {n_feature_frames} frames")
                print(f"   ├─ SSM Matrix: {n_frames} × {n_frames} frames")
                print(f"   ├─ Dimensionality Change: {n_features}D → 2D similarity space")
                print(f"   ├─ Complexity Ratio: {complexity_reduction:.3f}:1")
                print(f"   ├─ Time Resolution: {1000/feature_sr:.1f} ms per frame")
                
                # Compare data volumes
                feature_size_kb = features.nbytes / 1024
                ssm_size_kb = S.nbytes / 1024
                compression_ratio = feature_size_kb / ssm_size_kb if ssm_size_kb > 0 else 0
                print(f"   └─ Storage Ratio: {compression_ratio:.2f}:1 (features vs SSM)")
            
            print()

        @staticmethod
        def plot_ssm(ssm: np.ndarray, ssm_sr: float, ssm_name: str, x_axis_type: str = 'time', time_annotations: Optional[list] = None,
                    features: Optional[np.ndarray] = None, feature_sr: Optional[float] = None):
            """
            Plot a Self-Similarity Matrix (SSM) optionally stacked below a features heatmap.
            Layout mimics libfmp.c4.plot_feature_ssm for perfect alignment.
            """
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            
            if x_axis_type not in ['time', 'frame']:
                raise ValueError("x_axis_type must be 'time' or 'frame'")

            n_frames = ssm.shape[0]
            ssm_time = np.arange(n_frames) / ssm_sr

            # Use libfmp compressed gray colormap for SSM
            try:
                cmap_ssm = libfmp.b.compressed_gray_cmap(alpha=-10)
            except (ImportError, AttributeError):
                cmap_ssm = 'gray_r'

            # Case 1: with features - mimic libfmp layout exactly ----------------------
            if features is not None and feature_sr is not None:
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
                    extent_feat = (0, max_time, 0, features.shape[0])
                    extent_ssm = (0, max_time, 0, max_time)
                    xlabel, ylabel = 'Time (s)', 'Time (s)'
                else:
                    extent_feat = (0, n_frames - 1, 0, features.shape[0])
                    extent_ssm = (0, n_frames - 1, 0, n_frames - 1)
                    xlabel, ylabel = 'Frames', 'Frames'
                
                # Plot features using libfmp.b.plot_matrix style
                im_feat = ax[0, 1].imshow(features, aspect='auto', origin='lower', interpolation='nearest',
                                        cmap=cmap_ssm, extent=extent_feat)
                ax[0, 1].set_ylabel('Feature Index')
                ax[0, 1].set_title(f'Features used for {ssm_name}')
                ax[0, 1].set_xlabel('')  # No xlabel for top plot
                
                # Features colorbar
                cbar_feat = plt.colorbar(im_feat, cax=ax[0, 2])
                
                # Turn off corner axes like libfmp
                ax[0, 0].axis('off')
                
                # SSM plot - center position [1, 1] with colorbar [1, 2]
                im_ssm = ax[1, 1].imshow(ssm, aspect='auto', origin='lower', 
                                        cmap=cmap_ssm, extent=extent_ssm, 
                                        interpolation='nearest')
                
                # Handle SSM axis labels based on annotation presence
                if time_annotations is not None and len(time_annotations) > 0:
                    # With annotations: labels go on annotation areas, main plot has no labels
                    ax[1, 1].set_xlabel('')  # No labels on main SSM plot
                    ax[1, 1].set_ylabel('')
                    ax[1, 1].set_xticks([])  # Remove ticks like libfmp
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
                        time_annotations = [(start * ssm_sr, end * ssm_sr, _) for start, end, _ in time_annotations]

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
                    
                im_ssm = ax_ssm.imshow(ssm, aspect='equal', origin='lower', 
                                    cmap=cmap_ssm, extent=extent, 
                                    interpolation='nearest')
                ax_ssm.set_xlabel(xlabel)
                ax_ssm.set_ylabel(ylabel)
                ax_ssm.set_title(f'{ssm_name}')
                
                # Add colorbar
                plt.colorbar(im_ssm)

            plt.tight_layout()
            plt.show()
    
    class NoveltyCurve:
        @staticmethod
        def compute_curve_kernel(SSM: np.ndarray, kernel_size: int = 16, variance: float = 0.5, exclude_borders: bool = True) -> np.ndarray:
            """
            Compute a novelty curve from a self-similarity matrix (SSM) using a Gaussian checkerboard kernel.
            Args:
                SSM (np.ndarray): Input self-similarity matrix of shape (N, N) where N is the number of time frames.
                kernel_size (int, optional): Size of the Gaussian kernel. Must be an even integer. Defaults to 16.
                variance (float, optional): Variance of the Gaussian kernel. Controls the width of the Gaussian. Defaults to 0.5.
                exclude_borders (bool, optional): If True, sets the novelty values at the borders to zero to avoid edge effects. Defaults to True.
            Returns:
                np.ndarray: Novelty curve of shape (N,) where N is the number of time frames.
            """
            novelty_curve = libfmp.c4.compute_novelty_ssm(SSM, L=kernel_size, var=variance, exclude=exclude_borders)
            return novelty_curve
        
        @staticmethod
        def compute_curve_structure(tlm: np.ndarray, padding=True) -> np.ndarray:
            """Computation of the novelty function from a circular time-lag representation

            Notebook: C4/C4S4_StructureFeature.ipynb

            Args:
                L (np.ndarray): Circular time-lag representation
                padding (bool): Padding the result with the value zero (Default value = True)

            Returns:
                nov (np.ndarray): Novelty function
            """
            N = tlm.shape[0]
            if padding:
                nov = np.zeros(N)
            else:
                nov = np.zeros(N-1)
            for n in range(N-1):
                nov[n] = np.linalg.norm(tlm[:, n+1] - tlm[:, n])
            return nov
        
        @staticmethod
        def smooth_curve(novelty_curve: np.ndarray, sigma: float = 1.0) -> np.ndarray:
            """
            Smooth a novelty curve using a Gaussian filter.
            Args:
                novelty_curve (np.ndarray): Input novelty curve of shape (N,) where N is the number of time frames.
                sigma (float, optional): Standard deviation for Gaussian kernel. Controls the amount of smoothing. Defaults to 1.0.
            Returns:
                np.ndarray: Smoothed novelty curve of shape (N,).
            """
            smoothed_curve = filters.gaussian_filter1d(novelty_curve, sigma=sigma)
            return smoothed_curve


        @staticmethod
        def plot_curve(novelty_curve: np.ndarray, novelty_sr: float, novelty_name: str, x_axis_type: str = 'time',
                 time_annotations: Optional[list] = None,
                 SSM: Optional[np.ndarray] = None, SSM_sr: Optional[float] = None):
            """
            Plot a novelty curve optionally with its corresponding self-similarity matrix (SSM) above it.
            Layout follows the same pattern as plot_ssm for consistency.
            """
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec

            if x_axis_type not in ['time', 'frame']:
                raise ValueError("x_axis_type must be 'time' or 'frame'")

            n_frames = novelty_curve.shape[0]
            novelty_time = np.arange(n_frames) / novelty_sr
            fig_width = 8


            # Use libfmp compressed gray colormap for SSM
            try:
                cmap_ssm = libfmp.b.compressed_gray_cmap(alpha=-10)
            except (ImportError, AttributeError):
                cmap_ssm = 'gray_r'

            # Case 1: with SSM - libfmp-style layout -------------------------------
            if SSM is not None and SSM_sr is not None:
                max_time = novelty_time[-1]
                
                fig_height = fig_width * 1.25  # Adjust height for better aspect ratio
                
                # Use simplified 2x1 layout without colorbar (SSM on top, novelty below)
                fig = plt.figure(figsize=(fig_width, fig_height))
                gs = gridspec.GridSpec(2, 1, figure=fig, 
                                     height_ratios=[1, 0.3], 
                                     hspace=0.3)
                
                ax_ssm = fig.add_subplot(gs[0, 0])
                ax_novelty = fig.add_subplot(gs[1, 0])
                
                # Coordinate system setup
                if x_axis_type == 'time':
                    extent_ssm = (0, max_time, 0, max_time)
                    xlabel, ylabel = 'Time (s)', 'Time (s)'
                    novelty_x = novelty_time
                else:
                    extent_ssm = (0, n_frames - 1, 0, n_frames - 1)
                    xlabel, ylabel = 'Frames', 'Frames'
                    novelty_x = np.arange(n_frames)
                
                # SSM plot - no colorbar for cleaner appearance
                im_ssm = ax_ssm.imshow(SSM, aspect='auto', origin='lower', 
                                      cmap=cmap_ssm, extent=extent_ssm, 
                                      interpolation='nearest')
                ax_ssm.set_ylabel(ylabel)
                ax_ssm.set_title("SSM")
                ax_ssm.set_xlabel('')  # No xlabel for top plot
                
                # Novelty curve plot - bottom subplot
                ax_novelty.plot(novelty_x, novelty_curve, color='black', linewidth=1.5)
                ax_novelty.set_xlabel(xlabel)
                ax_novelty.set_ylabel('Novelty')
                ax_novelty.set_title(f'Novelty Curve: {novelty_name}')
                ax_novelty.grid(True, alpha=0.3)
                ax_novelty.set_ylim(0, np.max(novelty_curve) * 1.1)
                
                # Align x-axis limits with SSM
                if x_axis_type == 'time':
                    ax_novelty.set_xlim(0, max_time)
                else:
                    ax_novelty.set_xlim(0, n_frames - 1)
                
                # Plot time annotations as colored rectangles on top of novelty curve
                if time_annotations is not None and len(time_annotations) > 0:
                    novelty_max = np.max(novelty_curve) * 1.1
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
                                start_coord = start_time * novelty_sr
                                end_coord = end_time * novelty_sr
                            
                            # Plot colored rectangle over novelty curve
                            ax_novelty.axvspan(start_coord, end_coord, 
                                             alpha=0.2, color=f'C{i % 10}', 
                                             label=label)
                            
                            # Add text label at the center
                            ax_novelty.text((start_coord + end_coord) / 2, novelty_max * 0.9, 
                                          label, ha='center', va='center', 
                                          fontsize=8, rotation=0)

            # Case 2: Novelty only - simple layout ---------------------------------
            else:
                fig_height = fig_width * 0.25  # Shorter height for single plot
                fig, ax_novelty = plt.subplots(figsize=(fig_width, fig_height))
                
                if x_axis_type == 'time':
                    novelty_x = novelty_time
                    xlabel = 'Time (s)'
                else:
                    novelty_x = np.arange(n_frames)
                    xlabel = 'Frames'
                
                ax_novelty.plot(novelty_x, novelty_curve, color='black', linewidth=1.5)
                ax_novelty.set_title(f'Novelty Curve: {novelty_name}')
                ax_novelty.set_xlabel(xlabel)
                ax_novelty.set_ylabel('Novelty')
                ax_novelty.grid(True, alpha=0.3)
                ax_novelty.set_ylim(0, np.max(novelty_curve) * 1.1)
                
                if x_axis_type == 'time':
                    ax_novelty.set_xlim(0, novelty_time[-1])
                else:
                    ax_novelty.set_xlim(0, n_frames - 1)
                
                # Plot time annotations as colored rectangles on top of novelty curve
                if time_annotations is not None and len(time_annotations) > 0:
                    novelty_max = np.max(novelty_curve) * 1.1
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
                                start_coord = start_time * novelty_sr
                                end_coord = end_time * novelty_sr
                            
                            # Plot colored rectangle over novelty curve
                            ax_novelty.axvspan(start_coord, end_coord, 
                                             alpha=0.2, color=f'C{i % 10}', 
                                             label=label)
                            
                            # Add text label at the center
                            ax_novelty.text((start_coord + end_coord) / 2, novelty_max * 0.9, 
                                          label, ha='center', va='center', 
                                          fontsize=8, rotation=0)

            # plt.tight_layout()
            plt.show()

        @staticmethod
        def find_peaks(novelty_curve: np.ndarray, novelty_sr: float, threshold: float = 0.1, distance: int = 10) -> np.ndarray:
            """
            Identify peaks in a novelty curve using local maxima detection.
            
            Args:
                novelty_curve (np.ndarray): Input novelty curve of shape (N,) where N is the number of time frames.
                novelty_sr (float): Sampling rate of the novelty curve (frames per second).
                threshold (float, optional): Minimum height of peaks as a fraction of the maximum novelty value. Defaults to 0.1.
                distance (int, optional): Minimum number of frames between adjacent peaks. Defaults to 10.
            Returns:
                np.ndarray: Array of peak frame indices.
            """
            from scipy.signal import find_peaks
            
            # Calculate absolute height threshold based on relative threshold
            height_threshold = np.max(novelty_curve) * threshold
            
            # Use scipy's find_peaks to identify peaks
            peaks, _ = find_peaks(novelty_curve, height=height_threshold, distance=distance)
            
            return peaks
        
        @staticmethod
        def plot_peaks(novelty_curve: np.ndarray, novelty_sr: float, peaks: np.ndarray, novelty_name: str, x_axis_type: str = 'time',
                      time_annotations: Optional[list] = None):
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

            if x_axis_type not in ['time', 'frame']:
                raise ValueError("x_axis_type must be 'time' or 'frame'")

            n_frames = novelty_curve.shape[0]
            novelty_time = np.arange(n_frames) / novelty_sr

            fig, ax = plt.subplots(figsize=(10, 4))

            # Plot novelty curve depending on x axis type
            if x_axis_type == 'time':
                x_values = peaks / novelty_sr
                ax.plot(novelty_time, novelty_curve, label='Novelty Curve', color='black')
                ax.set_xlabel('Time (s)')
            else:
                x_values = peaks
                ax.plot(np.arange(n_frames), novelty_curve, label='Novelty Curve', color='black')
                ax.set_xlabel('Frames')

            # Draw vertical dashed red lines spanning the full axis height
            for i, xv in enumerate(x_values):
                # label only the first line so legend remains clean
                ax.axvline(xv, color='red', linestyle='--', linewidth=0.5, label='Predictions' if i == 0 else None)

                ax.set_title(f'Novelty Curve with Peaks: {novelty_name}')
                ax.set_ylabel('Novelty')
                max_val = np.max(novelty_curve)
                if max_val <= 0:
                    max_val = 1.0
                ax.set_ylim(0, max_val * 1.1)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Plot time annotations as colored rectangles on top of novelty curve
            if time_annotations is not None and len(time_annotations) > 0:
                novelty_max = np.max(novelty_curve) * 1.1
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
                            start_coord = start_time * novelty_sr
                            end_coord = end_time * novelty_sr
                        
                        # Plot colored rectangle over novelty curve
                        ax.axvspan(start_coord, end_coord, 
                                  alpha=0.2, color=f'C{i % 10}', 
                                  label=label)
                        
                        # Add text label at the center
                        ax.text((start_coord + end_coord) / 2, novelty_max * 0.9, 
                               label, ha='center', va='center', 
                               fontsize=8, rotation=0)
            
            plt.tight_layout()
            plt.show()

    class NonInstrumentalSegmentation:
        @staticmethod
        def detect_silences_from_waveform(audio: np.ndarray, sr: int | float, frame_length: int = 2048, hop_length: int = 512,
                            top_db: int = 60, min_silence_len_seconds: float = 0.3, 
                            silence_thresh: Optional[float] = None) -> np.ndarray:
            """
            Detect silent intervals in an audio signal using librosa's effects.split function.
            
            Args:
                audio (np.ndarray): Input audio signal (1D numpy array).
                sr (int): Sampling rate of the audio signal.
                frame_length (int, optional): Frame length for analysis. Defaults to 2048.
                hop_length (int, optional): Hop length for analysis. Defaults to 512.
                top_db (int, optional): Threshold in decibels below reference to consider as silence. Defaults to 60.
                min_silence_len (float, optional): Minimum length of silence in seconds to be considered a valid silent interval. Defaults to 0.3 seconds.
                silence_thresh (Optional[float], optional): Silence threshold in dB. If None, it is set to the mean dB level of the audio minus top_db. Defaults to None.
            
            Returns:
                np.ndarray: Array of shape (M, 2) where M is the number of detected silent intervals.
                    Each row contains the start and end sample indices of a silent interval.
            """
            import librosa
            
            non_silent_parts = librosa.effects.split(y=audio, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
            # Combine adjacent silent intervals that are closer than min_silence_len_seconds
            # Compute silent parts as the complement of non_silent_parts
            silent_parts = None
            if non_silent_parts is None or len(non_silent_parts) == 0:
                silent_parts = np.array([[0, len(audio)]], dtype=int)
            else:
                non_silent_parts = np.asarray(non_silent_parts, dtype=int)
                silent_list = []

                # Leading silence
                if non_silent_parts[0, 0] > 0:
                    silent_list.append([0, non_silent_parts[0, 0]])

                # Silences between non-silent regions
                for i in range(len(non_silent_parts) - 1):
                    start = non_silent_parts[i, 1]
                    end = non_silent_parts[i + 1, 0]
                    if end > start:
                        silent_list.append([start, end])

                # Trailing silence
                if non_silent_parts[-1, 1] < len(audio):
                    silent_list.append([non_silent_parts[-1, 1], len(audio)])

                silent_parts = np.array(silent_list, dtype=int) if silent_list else np.empty((0, 2), dtype=int)
            
            min_silence_len_samples = int(min_silence_len_seconds * sr)
        
            big_steps = [1]
            filtered_parts = []
            for i in range(len(silent_parts) - 1):
                if (silent_parts[i+1][0] - silent_parts[i][1] >= min_silence_len_samples):
                    big_steps.append(i+1)
                    filtered_parts.append([
                        silent_parts[big_steps[len(big_steps)-2]][0], 
                        silent_parts[big_steps[len(big_steps)-1]][1]
                    ])
            filtered_parts.append([
                silent_parts[big_steps[len(big_steps)-1]][0],
                silent_parts[len(silent_parts)-1][1]
            ])

            print(filtered_parts)

            return np.array(filtered_parts)
        
        @staticmethod
        def plot_signal_with_silences(audio: np.ndarray, sr: int | float, silence_intervals: np.ndarray, 
                                    title: str = "Audio Signal with Silence Detection",
                                    time_annotations: Optional[list] = None):
            """
            Plot audio signal with detected silence intervals marked as vertical dashed lines
            and optional ground truth annotations as colored stripes.
            
            Args:
                audio (np.ndarray): Input audio signal (1D numpy array).
                sr (int): Sampling rate of the audio signal.
                silence_intervals (np.ndarray): Array of shape (M, 2) where M is the number of 
                    detected silent intervals. Each row contains [start_sample, end_sample].
                title (str, optional): Title for the plot. Defaults to "Audio Signal with Silence Detection".
                time_annotations (Optional[list]): List of time-based ground truth annotations 
                    in format [start_time, end_time, label]. Always in time units (seconds).
            """
            import matplotlib.pyplot as plt
            
            # Create time axis
            time_axis = np.arange(len(audio)) / sr
            duration = len(audio) / sr
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot the audio waveform
            ax.plot(time_axis, audio, color='black', linewidth=0.8, alpha=0.3, label='Audio Signal')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, duration)
            
            # Get amplitude range for proper positioning
            audio_max = np.max(np.abs(audio))
            if audio_max == 0:
                audio_max = 1.0
            ax.set_ylim(-audio_max * 1.1, audio_max * 1.1)
            
            # Plot ground truth annotations as colored stripes (background)
            if time_annotations is not None and len(time_annotations) > 0:
                for i, ann in enumerate(time_annotations):
                    if len(ann) >= 2:  # Expecting [start_time, end_time, label] format
                        start_time, end_time = ann[0], ann[1]
                        label = ann[2] if len(ann) > 2 else f"Seg{i+1}"
                        
                        # Plot colored rectangle spanning full height
                        ax.axvspan(start_time, end_time, 
                                alpha=0.2, color=f'C{i % 10}', 
                                label=f'GT: {label}' if i == 0 else f'{label}')
                        
                        # Add text label at the top
                        ax.text((start_time + end_time) / 2, audio_max * 0.9, 
                            label, ha='center', va='center', 
                            fontsize=8, rotation=0, 
                            bbox=dict(boxstyle="round,pad=0.3", facecolor=f'C{i % 10}', alpha=0.3))
            
            # Plot detected silence intervals as vertical dashed lines
            if len(silence_intervals) > 0:
                # Convert sample indices to time
                silence_times = silence_intervals / sr
                
                for i, (start_time, end_time) in enumerate(silence_times):
                    # Plot start and end of silence as dashed red lines
                    ax.axvline(start_time, color='red', linestyle='--', linewidth=1.0, 
                            alpha=0.8, label='Silence Start' if i == 0 else None)
                    ax.axvline(end_time, color='red', linestyle='--', linewidth=1.0, 
                            alpha=0.8, label='Silence End' if i == 0 else None)
                    
                    # Optional: Add silence region shading
                    ax.axvspan(start_time, end_time, alpha=0.1, color='red', 
                            label='Silence Regions' if i == 0 else None)
            
            # Add legend if there are annotations or silence detections
            if (time_annotations is not None and len(time_annotations) > 0) or len(silence_intervals) > 0:
                ax.legend(loc='upper right', fontsize=8)
            
            plt.tight_layout()
            plt.show()

    class HarmonicResidualPercussiveSeparation:
        def __init__(self):
            pass

        @staticmethod
        def compute_separation():
            pass
        
    