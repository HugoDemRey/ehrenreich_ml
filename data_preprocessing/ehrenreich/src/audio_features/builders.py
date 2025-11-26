from src.audio_features.features import SilenceCurve
from src.audio.signal import Signal
from src.constants import FFT
import numpy as np
import librosa
import src.libfmp, src.libfmp.c2, src.libfmp.c3, src.libfmp.c4, src.libfmp.b
from typing import Optional
from abc import abstractmethod
from src.audio_features.features import Spectrogram, Chromagram, Tempogram, HRPS, MFCC, SelfSimilarityMatrix, TimeLagMatrix, NoveltyCurve
import scipy
from src.interfaces.builder import Builder
from src.interfaces.feature import BaseFeature, Feature, SimilarityMatrix

# FIXME: Should use the classes defined in src/interfaces/builder.py and src/interfaces/feature.py
# Features -> BaseFeature (Feature + Preprocessing functions), Feature = Simple data container (data and sample rate)
class BuilderFromSignal(Builder):
    @abstractmethod
    def build(self, signal: Signal) -> BaseFeature:
        pass

class BuilderFromBaseFeature(Builder):
    @abstractmethod
    def build(self, base_feature: BaseFeature) -> Feature:
        pass

class BuilderFromSimilarityMatrix(Builder):
    @abstractmethod
    def build(self, sm: SimilarityMatrix) -> Feature:
        pass

class SpectrogramBuilder(BuilderFromSignal):
    def __init__(self, frame_length: int = 4410, hop_length: int = 2205):
        self.frame_length = frame_length
        self.hop_length = hop_length

    def build(self, signal: Signal) -> Spectrogram:
        # Compute STFT
        stft = librosa.stft(y=signal.samples, hop_length=self.hop_length, n_fft=self.frame_length)
            
        spec = np.abs(stft) ** 2  # Store magnitude/power spectrogram

        spec_sr = signal.sample_rate / self.hop_length

        return Spectrogram(spec, spec_sr)

class ChromagramBuilder(BuilderFromSignal):
    def __init__(self, frame_length: int = 4410, hop_length: int = 2205):
        self.frame_length = frame_length
        self.hop_length = hop_length

    def build(self, signal: Signal) -> Chromagram:
        chroma = librosa.feature.chroma_stft(y=signal.samples, sr=signal.sample_rate, n_fft=self.frame_length, hop_length=self.hop_length)
        chroma_sr = signal.sample_rate / self.hop_length
        return Chromagram(chroma, chroma_sr)

class MFCCBuilder(BuilderFromSignal):
    def __init__(self, n_mfcc: int = 20, frame_length: int = 4410, hop_length: int = 2205):
        self.n_mfcc = n_mfcc
        self.frame_length = frame_length
        self.hop_length = hop_length

    def build(self, signal: Signal) -> MFCC:
        mfcc = librosa.feature.mfcc(y=signal.samples, sr=signal.sample_rate, n_mfcc=self.n_mfcc, hop_length=self.hop_length, n_fft=self.frame_length)
        mfcc_sr = signal.sample_rate / self.hop_length
        return MFCC(mfcc, mfcc_sr)

class TempogramBuilder(BuilderFromSignal):
    def __init__(self, frame_length: int = 4410, hop_length: int = 2205):
        self.frame_length = frame_length
        self.hop_length = hop_length

    def build(self, signal: Signal) -> Tempogram:
        onset_env = librosa.onset.onset_strength(y=signal.samples, sr=signal.sample_rate, hop_length=self.hop_length)
        tempogram = librosa.util.normalize(librosa.feature.tempogram(onset_envelope=onset_env, sr=signal.sample_rate, hop_length=self.hop_length))
        tempogram_sr = signal.sample_rate / self.hop_length
        return Tempogram(tempogram, tempogram_sr)

    pass

# Source
class PLPBuilder(BuilderFromSignal):
    """
    The goal of the Predominant Local Pulse (PLP) features is to enhance the novelty curve of the classical tempogram.
    Maybe, this class shouldn't be there since it can compute PLP from Tempogram, to be defined...
    
    Source: https://diglib.eg.org/server/api/core/bitstreams/fa7f1cad-a80e-423c-b866-0426908d4dba/content 
    """
    def __init__(self):
        super().__init__()
    
    def build(self, signal: Signal) -> BaseFeature:
        raise NotImplementedError("PLP feature extraction not implemented yet.")

class SilenceCurveBuilder(BuilderFromSignal):
    def __init__(self, silence_type="amplitude", frame_length=4410, hop_length=2205):
        """
        Initialize the SilenceCurveBuilder with parameters for different silence detection methods.
        Args:
            silence_type (str): Type of silence detection method ('amplitude', 'zcr', 'comb'= [amplitude + zcr], 'f0').
            frame_length (int): Frame length for feature computation.
            hop_length (int): Hop length for feature computation.
        """

        super().__init__()
        self.silence_type = silence_type
        self.frame_length = frame_length
        self.hop_length = hop_length

    def compute_amplitude_silence(self, signal):
        # Compute RMS energy per frame
        rms = librosa.feature.rms(y=signal.samples, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        # Silence curve: 1 if below amplitude threshold, else 0
        return -1 * rms

    def compute_zcr_silence(self, signal):
        # Compute zero crossing rate per frame
        zcr = librosa.feature.zero_crossing_rate(y=signal.samples, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        # Silence curve: 1 if ZCR below threshold (low energy), else 0
        return -1 * zcr

    def compute_f0_silence(self, signal):
        # Estimate fundamental frequency using librosa.pyin
        f0, _, _ = librosa.pyin(signal.samples, fmin=50, fmax=2000, frame_length=self.frame_length, hop_length=self.hop_length)
        # Silence curve: 1 where no f0 was detected (NaN), else 0
        silence_curve = np.where(np.isnan(f0), 1.0, 0.0)
        return silence_curve

    def build(self, signal: Signal) -> SilenceCurve:
        if self.silence_type == "amplitude":
            sc = SilenceCurve(self.compute_amplitude_silence(signal), signal.sample_rate / self.hop_length)
            return sc
        elif self.silence_type == "zcr":
            sc = SilenceCurve(self.compute_zcr_silence(signal), signal.sample_rate / self.hop_length)
            return sc
        elif self.silence_type == "comb":
            amp_sc = self.compute_amplitude_silence(signal)
            zcr_sc = self.compute_zcr_silence(signal)
            combined_sc = amp_sc * zcr_sc
            sc = SilenceCurve(combined_sc, signal.sample_rate / self.hop_length)
            return sc
        elif self.silence_type == "f0":
            sc = SilenceCurve(self.compute_f0_silence(signal), signal.sample_rate / self.hop_length)
            return sc
        else:
            raise ValueError(f"Unsupported silence_type {self.silence_type}")

class HRPSBuilder(BuilderFromSignal):
    def __init__(self, L_h_frames, L_p_bins, beta: float = 1, frame_length: int = 4410, hop_length: int = 2205):
        """
        Harmonic/Percussive source separation (HPSS) using median filtering.
        Args:
            L_h (int): Length of the median filter for harmonic components (in frames).
            L_p (int): Length of the median filter for percussive components (in bins).
            beta (float): Exponent for soft masking. Default is 1 (binary masking).
        """
        self.L_h_frames = L_h_frames + ((L_h_frames + 1) % 2)  # Ensure odd length
        self.L_p_bins = L_p_bins + ((L_p_bins + 1) % 2)        # Ensure odd length
        self.beta = beta
        self.frame_length = frame_length
        self.hop_length = hop_length

    def compute_signals(self, signal: Signal) -> tuple[Signal, Signal, Signal]:
        print("Computing STFT...", end="\r")
        X = librosa.stft(y=signal.samples, 
                         hop_length=self.hop_length, 
                         n_fft=self.frame_length,
                         win_length=self.frame_length, 
                         window='hann', 
                         center=True, 
                         pad_mode='constant')
        
        downsampling_factor = 5
        self.hop_length *= downsampling_factor
        filter_length = 101
        kernel = np.ones(filter_length) / filter_length
        X_smooth = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=1, arr=X)
        # Downsample (take every Nth frame)
        X = X_smooth[:, ::downsampling_factor]

        Y = np.abs(X) ** 2  # Power spectrogram

        print("Computed STFT                    \n")
        
        # Data of a Spectrogram is always complex-valued in linear scale

        print("Applying Median for Harmonic Component (1/2)")
        import time 
        start = time.time()
        print("Y.shape:", Y.shape)
        Y_h = scipy.signal.medfilt2d(Y, [1, self.L_h_frames])
        middle = time.time()
        print(f"Median filtering took {middle - start:.2f} seconds")
        print("Applying Median for Percussive Component (2/2)")
        Y_p = scipy.signal.medfilt2d(Y, [self.L_p_bins, 1])
        end = time.time()
        print(f"Median filtering took {end - middle:.2f} seconds")
        print("\n")

        # Masking
        print("Computing Masks...", end="\r")
        M_h = np.int8(Y_h >= self.beta * Y_p)
        M_p = np.int8(Y_p > self.beta * Y_h)
        M_r = 1 - (M_h + M_p)
        X_h = X * M_h
        X_p = X * M_p
        X_r = X * M_r
        print("Computed Masks                    \n")

        # istft
        print("Computing Inverse STFT for x_h (1/3)")
        x_h = librosa.istft(X_h, hop_length=self.hop_length, win_length=self.frame_length, window='hann', center=True, length=len(signal.samples))
        print("Computing Inverse STFT for x_r (2/3)")
        x_r = librosa.istft(X_r, hop_length=self.hop_length, win_length=self.frame_length, window='hann', center=True, length=len(signal.samples))
        print("Computing Inverse STFT for x_p (3/3)")
        x_p = librosa.istft(X_p, hop_length=self.hop_length, win_length=self.frame_length, window='hann', center=True, length=len(signal.samples))
        print("\n")    
        
        return Signal(x_h, signal.sample_rate, ""), Signal(x_r, signal.sample_rate, ""), Signal(x_p, signal.sample_rate, "")

    def compute_local_energy(self, signal: Signal) -> np.ndarray:
        """
        Compute the local energy for each frame of the signal using a sliding window technique.
        
        Args:
            signal (Signal): Input audio signal
            
        Returns:
            np.ndarray: Array of local energy values for each frame
        """
        # Get signal samples
        x = signal.samples
        
        # Calculate number of frames
        n_frames = 1 + (len(x) - self.frame_length) // self.hop_length
        
        # Initialize energy array
        local_energy = np.zeros(n_frames)
        
        # Compute local energy for each frame using sliding window
        for i in range(n_frames):
            # Calculate frame boundaries
            start = i * self.hop_length
            end = start + self.frame_length
            
            # Extract frame
            frame = x[start:end]
            
            # Compute energy as sum of squared samples
            local_energy[i] = np.sum(frame ** 2)
        
        return local_energy
    
    def build(self, signal: Signal) -> HRPS:
        
        sh, sr, sp = self.compute_signals(signal)

        print("Computing Local Energy for Harmonic Signal (1/3)")
        le_h = self.compute_local_energy(sh)

        print("Computing Local Energy for Residual Signal (2/3)")
        le_r = self.compute_local_energy(sr)

        print("Computing Local Energy for Percussive Signal (3/3)")
        le_p = self.compute_local_energy(sp)
        print("\n")

        stacked_energy = np.vstack([le_h, le_r, le_p])

        hrps_sr = signal.sample_rate / self.hop_length
        return HRPS(stacked_energy, hrps_sr)

class SSMBuilder(BuilderFromBaseFeature):
    def __init__(
            self,
            smoothing_filter_length: int = 1, 
            smoothing_filter_direction: int = 2,
            shift_set: np.ndarray = np.array([0]),
            tempo_relative_set: np.ndarray = np.array([1])):
        
        self.smoothing_filter_length = smoothing_filter_length
        self.smoothing_filter_direction = smoothing_filter_direction
        self.shift_set = shift_set
        self.tempo_relative_set = tempo_relative_set

    def build(self, base_feature: BaseFeature) -> SelfSimilarityMatrix:
        S, _ = src.libfmp.c4.compute_sm_ti(base_feature.data(), 
                                        base_feature.data(), 
                                        L=self.smoothing_filter_length, 
                                        tempo_rel_set=self.tempo_relative_set, 
                                        shift_set=self.shift_set, 
                                        direction=self.smoothing_filter_direction
                                        )
            
        return SelfSimilarityMatrix(S, base_feature.sampling_rate())

class TLMBuilder(BuilderFromSimilarityMatrix):
    def __init__(self):
        pass

    def build(self, sm: SimilarityMatrix) -> TimeLagMatrix:

        if not isinstance(sm, SelfSimilarityMatrix):
            raise ValueError("For computing Time-Lag Matrix, the input SimilarityMatrix must be a SelfSimilarityMatrix.")

        S = src.libfmp.c4.compute_time_lag_representation(sm.data(), circular=True)
        
        return TimeLagMatrix(S, sm.sampling_rate())















class FeatureExtractorOld:
    @staticmethod
    def compute_stft(signal: Signal) -> tuple[np.ndarray, np.ndarray]:
        H = FFT.DEFAULT_HOP_LENGTH
        N = FFT.DEFAULT_WINDOW_SIZE
        w = np.hanning(N)
        
        X = src.libfmp.c2.stft(signal.samples, w, H)
        Y = np.abs(X) ** 2
        eps = np.finfo(float).eps
        Y_db = 10 * np.log10(Y + eps)
        return X, Y_db
    
    @staticmethod
    def compute_spectrogram_linear(signal: Signal, window_size: int = FFT.DEFAULT_WINDOW_SIZE, hop_length: int = FFT.DEFAULT_HOP_LENGTH) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X, T_coef, F_coef = src.libfmp.c2.stft_convention_fmp(signal.samples, signal.sample_rate, window_size, hop_length)
        return X, T_coef, F_coef
    
    @staticmethod
    def compute_spectrogram_log_freq(signal: Signal, window_size: int = FFT.DEFAULT_WINDOW_SIZE, hop_length: int = FFT.DEFAULT_HOP_LENGTH) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the log-frequency spectrogram of a signal.
        Args:
            signal (Signal): The input signal.
            window_size (int): The window size for the STFT.
        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the log-frequency spectrogram and the corresponding pitch values, time coefficients, and frequency coefficients.
        """
        from src.cache import Cache

        print("x[0:10]:", signal.samples[0:10])
        print("Fs:", signal.sample_rate)
        print("N:", window_size)
        print("H:", hop_length)


        X, T_coef, F_coef = FeatureExtractorOld.compute_spectrogram_linear(signal, window_size, hop_length)
        print("X.shape:", X.shape, "X[0]", X[0])

        Y = np.abs(X) ** 2

        print("Y.shape:", Y.shape, "Y[0]", Y[0])

        Spec, Pitches = src.libfmp.c3.compute_spec_log_freq(Y, signal.sample_rate, window_size)

        return Spec, Pitches, T_coef, F_coef

    class Chromagram:
        @staticmethod
        def compute_features(x: np.ndarray, sr: float, hop_length: int = 2205, n_fft: int = 4410,
                                filter_length: int = 21, downsampling_factor: int = 5, 
                                postprocess: bool = True) -> tuple[np.ndarray, float]:
            """
            Compute chroma features from an audio signal with optional post-processing.
            
            Args:
                x (np.ndarray): Audio signal.
                sr (float): Sample rate of the audio signal.
                hop_length (int, optional): Hop length for STFT computation. Defaults to 2205.
                n_fft (int, optional): FFT window size. Defaults to 4410.
                filter_length (int, optional): Length of smoothing filter. Defaults to 21.
                downsampling_factor (int, optional): Downsampling factor. Defaults to 5.
                postprocess (bool, optional): Whether to apply post-processing (smoothing, downsampling, 
                    normalization). If False, returns raw chroma features. Defaults to True.
                
            Returns:
                tuple[np.ndarray, float]: Tuple containing:
                    - F (np.ndarray): Chroma feature sequence (post-processed if postprocess=True)
                    - new_sr (float): Feature sampling rate
            """
            # Compute chroma features
            chroma_raw = librosa.feature.chroma_stft(y=x, sr=sr, tuning=0, norm=2, 
                                                    hop_length=hop_length, n_fft=n_fft)
            Fs_raw = sr / hop_length
            
            # Conditionally apply post-processing
            if postprocess:
                F, new_sr = FeatureExtractorOld.postprocess_features(chroma_raw, Fs_raw, filter_length, downsampling_factor)
            else:
                F, new_sr = chroma_raw, Fs_raw
            
            return F, new_sr
    
    class MFCC:
        @staticmethod
        def compute_features(x: np.ndarray, sr: float, n_mfcc: int = 20, hop_length: int = 2205, 
                                n_fft: int = 4410, filter_length: int = 21, 
                                downsampling_factor: int = 5, postprocess: bool = True) -> tuple[np.ndarray, float]:
            """
            Compute MFCC features from an audio signal with optional post-processing.
            
            Args:
                x (np.ndarray): Audio signal.
                sr (float): Sample rate of the audio signal.
                n_mfcc (int, optional): Number of MFCC coefficients to compute. Defaults to 20.
                hop_length (int, optional): Hop length for STFT computation. Defaults to 2205.
                n_fft (int, optional): FFT window size. Defaults to 4410.
                filter_length (int, optional): Length of smoothing filter. Defaults to 21.
                downsampling_factor (int, optional): Downsampling factor. Defaults to 5.
                postprocess (bool, optional): Whether to apply post-processing (smoothing, downsampling, 
                    normalization). If False, returns raw MFCC features. Defaults to True.
                
            Returns:
                tuple[np.ndarray, float]: Tuple containing:
                    - F (np.ndarray): MFCC feature sequence (post-processed if postprocess=True)
                    - new_sr (float): Feature sampling rate
            """
            # Compute MFCC features
            mfcc_raw = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc, 
                                        hop_length=hop_length, n_fft=n_fft)
            Fs_raw = sr / hop_length
            
            # Conditionally apply post-processing
            if postprocess:
                F, new_sr = FeatureExtractorOld.postprocess_features(mfcc_raw, Fs_raw, filter_length, downsampling_factor)
            else:
                F, new_sr = mfcc_raw, Fs_raw
            
            return F, new_sr
    
    class Tempogram:
        @staticmethod
        def compute_features(x: np.ndarray, sr: float, hop_length: int = 2205, 
                                    filter_length: int = 21, downsampling_factor: int = 5, 
                                    postprocess: bool = True) -> tuple[np.ndarray, float]:
            """
            Compute tempogram features from an audio signal with optional post-processing.
            
            Args:
                x (np.ndarray): Audio signal.
                sr (float): Sample rate of the audio signal.
                hop_length (int, optional): Hop length for onset detection and tempogram computation. 
                    Defaults to 2205.
                filter_length (int, optional): Length of smoothing filter. Defaults to 21.
                downsampling_factor (int, optional): Downsampling factor. Defaults to 5.
                postprocess (bool, optional): Whether to apply post-processing (smoothing, downsampling, 
                    normalization). If False, returns raw tempogram features. Defaults to True.
                
            Returns:
                tuple[np.ndarray, float]: Tuple containing:
                    - F (np.ndarray): Tempogram feature sequence (post-processed if postprocess=True)
                    - new_sr (float): Feature sampling rate
            """
            # Compute tempogram features
            onset_env = librosa.onset.onset_strength(y=x, sr=sr, hop_length=hop_length)
            tempo_raw = librosa.util.normalize(librosa.feature.tempogram(onset_envelope=onset_env, 
                                                                    sr=sr, hop_length=hop_length))
            Fs_raw = sr / hop_length
            
            # Conditionally apply post-processing
            if postprocess:
                F, new_sr = FeatureExtractorOld.postprocess_features(tempo_raw, Fs_raw, filter_length, downsampling_factor)
            else:
                F, new_sr = tempo_raw, Fs_raw
            
            return F, new_sr

    class Spectrogram:
        @staticmethod
        def compute_features(x: np.ndarray, sr: float, hop_length: int = 2205, n_fft: int = 4410,
                                filter_length: int = 21, downsampling_factor: int = 5, 
                                postprocess: bool = True, power: float = 2.0,
                                to_db: bool = True) -> tuple[np.ndarray, float]:
            """
            Compute spectrogram features from an audio signal with optional post-processing.
            
            Args:
                x (np.ndarray): Audio signal.
                sr (float): Sample rate of the audio signal.
                hop_length (int, optional): Hop length for STFT computation. Defaults to 2205.
                n_fft (int, optional): FFT window size. Defaults to 4410.
                filter_length (int, optional): Length of smoothing filter. Defaults to 21.
                downsampling_factor (int, optional): Downsampling factor. Defaults to 5.
                postprocess (bool, optional): Whether to apply post-processing (smoothing, downsampling, 
                    normalization). If False, returns raw spectrogram features. Defaults to True.
                power (float, optional): Exponent for the magnitude spectrogram. 
                    1.0 for magnitude, 2.0 for power spectrogram. Defaults to 2.0.
                to_db (bool, optional): Whether to convert spectrogram to dB scale. Defaults to True.
                
            Returns:
                tuple[np.ndarray, float]: Tuple containing:
                    - F (np.ndarray): Spectrogram feature sequence (post-processed if postprocess=True)
                    - new_sr (float): Feature sampling rate
            """
            # Compute STFT
            stft_matrix = librosa.stft(y=x, hop_length=hop_length, n_fft=n_fft)
            
            # Compute magnitude/power spectrogram
            spectrogram_raw = np.abs(stft_matrix) ** power
            
            # Convert to dB scale if requested
            if to_db:
                spectrogram_raw = librosa.amplitude_to_db(spectrogram_raw, ref=np.max)
            
            Fs_raw = sr / hop_length
            
            # Conditionally apply post-processing
            if postprocess:
                F, new_sr = FeatureExtractorOld.postprocess_features(spectrogram_raw, Fs_raw, filter_length, downsampling_factor)
            else:
                F, new_sr = spectrogram_raw, Fs_raw
            
            return F, new_sr
        

    @staticmethod
    def postprocess_features(F: np.ndarray, sr_F: float, filter_length: int = 21, 
                              downsampling_factor: int = 5) -> tuple[np.ndarray, float]:
        """
        Post-process feature sequences with smoothing, downsampling and normalization.
        
        This is a common post-processing step applied to audio features before SSM computation.
        It applies temporal smoothing, downsamples the feature rate, and normalizes the features.
        
        Args:
            F (np.ndarray): Input feature matrix of shape (F, N) where F is the number of 
                feature dimensions and N is the number of time frames.
            Fs_F (float): Original feature rate (frames per second).
            filter_length (int, optional): Length of smoothing filter. Defaults to 21.
            downsampling_factor (int, optional): Factor by which to downsample the feature rate.
                Defaults to 5.
                
        Returns:
            tuple[np.ndarray, float]: Tuple containing:
                - X (np.ndarray): Post-processed feature sequence of shape (F, N/downsampling_factor)
                - Fs_feature (float): New feature rate after downsampling
        """
        X, Fs_feature = src.libfmp.c3.smooth_downsample_feature_sequence(F, sr_F, 
                                                                     filt_len=filter_length, 
                                                                     down_sampling=downsampling_factor)
        X = src.libfmp.c3.normalize_feature_sequence(X, norm='2', threshold=0.001)
        return X, Fs_feature
    
    @staticmethod
    def print_features_info(features: np.ndarray, feature_sr: float, features_name: str, 
                           original_signal: Optional[np.ndarray] = None, original_sr: Optional[float] = None):
        """
        Print a concise, beautifully formatted analysis of extracted audio features.
        
        Args:
            features (np.ndarray): The feature matrix of shape (n_features, n_frames).
            feature_sr (float): The sampling rate of the features (frames per second).
            features_name (str): Name of the feature type (e.g., "Chroma", "MFCC", "Tempogram").
            original_signal (Optional[np.ndarray]): The original audio signal, if available.
            original_sr (Optional[float]): The sampling rate of the original audio signal, if available.
        """
        # Header with decorative border
        title = f"{features_name.upper()} FEATURE ANALYSIS"
        print(f"***  {title}  ***")
        
        # Feature Matrix Information
        n_features, n_frames = features.shape
        duration_features = n_frames / feature_sr
        
        print(f"\n📊 Feature Matrix Properties:")
        print(f"   ├─ Dimensions: {n_features} features × {n_frames} frames")
        print(f"   ├─ Feature Rate: {feature_sr:.2f} Hz")
        print(f"   ├─ Duration: {duration_features:.3f} seconds")
        print(f"   └─ Memory Usage: {features.nbytes / 1024:.1f} KB")
        
        # Original Signal Comparison (if available)
        if original_signal is not None and original_sr is not None:
            duration_audio = len(original_signal) / original_sr
            compression_ratio = len(original_signal) / n_frames
            time_resolution = 1 / feature_sr
            
            print(f"\n🔄 Audio → Features Transformation:")
            print(f"   ├─ Original Signal: {len(original_signal):,} samples @ {original_sr} Hz ({duration_audio:.3f}s)")
            print(f"   ├─ Feature Frames: {n_frames} @ {feature_sr:.2f} Hz ({duration_features:.3f}s)")
            print(f"   ├─ Compression Ratio: {compression_ratio:.1f}:1")
            print(f"   ├─ Time Resolution: {time_resolution*1000:.1f} ms per frame")
            
            # Nyquist and aliasing analysis
            nyquist_audio = original_sr / 2
            effective_bandwidth = feature_sr / 2
            print(f"   ├─ Original Nyquist: {nyquist_audio} Hz")
            print(f"   └─ Feature Bandwidth: {effective_bandwidth:.1f} Hz")
        
        print()

    @staticmethod
    def plot_features(features: np.ndarray, feature_sr: float, features_name: str, x_axis_type: str = 'time',
                    time_annotations: Optional[list] = None,
                    original_signal: Optional[np.ndarray] = None, original_sr: Optional[float] = None):
        """
        Plot audio features using a heatmap representation, optionally with original signal on top.
        Features are assumed to be in frame units, and time axis is computed from feature_sr.
        
        Args:
            features (np.ndarray): The feature matrix of shape (F, N) where F is number of features and N is number of frames.
            feature_sr (float): The sampling rate of the features (frames per second).
            features_name (str): Name of the feature type (e.g., "Chroma", "MFCC", "Tempogram").
            x_axis_type (str, optional): Type of x-axis for the plot ('time' or 'frame'). Defaults to 'time'.
            time_annotations (Optional[list]): List of time-based annotations in format [start_time, end_time, label].
                Always in time units (seconds) regardless of x_axis_type.
            original_signal (Optional[np.ndarray]): The original audio signal of shape (T,) where T is number of time steps.
            original_sr (Optional[float]): The sampling rate of the original audio signal, if available.
        """
        import matplotlib.pyplot as plt
        import librosa.display

        if x_axis_type not in ['time', 'frame']:
            raise ValueError("x_axis_type must be 'time' or 'frame'")

        # Use src.libfmp-style compressed grayscale colormap
        try:
            # Try to use src.libfmp compressed gray colormap if available
            cmap = src.libfmp.b.compressed_gray_cmap(alpha=-10)
        except (ImportError, AttributeError):
            # Fallback to matplotlib grayscale if src.libfmp not available
            cmap = 'gray_r'  # reversed grayscale (white=high, black=low)

        # Compute coordinate systems for alignment
        n_frames = features.shape[1]
        feature_frames = np.arange(n_frames)
        feature_time = feature_frames / feature_sr
        
        # Create subplots based on whether we have original signal
        if original_signal is not None and original_sr is not None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1, 2])
            
            if x_axis_type == 'time':
                # Both plots use time axis - align by stretching signal plot to match feature time range
                audio_time = np.arange(len(original_signal)) / original_sr
                max_time = feature_time[-1]
                
                # Plot original signal with its natural time coordinates
                ax1.plot(audio_time, original_signal, color='gray', linewidth=0.8)
                ax1.set_xlabel('Time (s)')
                ax1.set_xlim(0, max_time)  # Align x-axis limits
                ax1.set_ylabel('Amplitude')
                ax1.set_title(f'Original Audio Signal (aligned with features)')
                ax1.grid(True, alpha=0.3)
                
                # Plot features with time-based heatmap using grayscale
                im = ax2.imshow(features, aspect='auto', origin='lower', interpolation='nearest', cmap=cmap, 
                            extent=(0, max_time, 0, features.shape[0]))
                ax2.set_xlabel('Time (s)')
                ax2.set_xlim(0, max_time)  # Match top plot x-axis limits
                
            else:  # x_axis_type == 'frame'
                # Both plots use frame axis - just change x-axis labels, don't stretch data
                
                # Plot original signal with natural sample indices but frame-based x-axis
                ax1.plot(original_signal, color='gray', linewidth=0.8)
                ax1.set_xlabel('Frames')
                ax1.set_ylabel('Amplitude')
                ax1.set_title(f'Original Audio Signal (aligned with features)')
                ax1.grid(True, alpha=0.3)
                
                # Convert sample indices to frame indices for x-axis ticks
                sample_to_frame_ratio = len(original_signal) / n_frames
                
                # Set x-axis limits to match frame count
                ax1.set_xlim(0, len(original_signal) - 1)
                
                # Create custom x-axis ticks that show frame numbers
                signal_ticks = ax1.get_xticks()
                frame_tick_labels = [f'{int(tick / sample_to_frame_ratio)}' for tick in signal_ticks if tick >= 0 and tick < len(original_signal)]
                ax1.set_xticklabels(frame_tick_labels)
                
                # Plot features with frame-based heatmap using grayscale
                im = ax2.imshow(features, aspect='auto', origin='lower', cmap=cmap,
                            extent=(0, n_frames - 1, 0, features.shape[0]))
                ax2.set_xlabel('Frames')
                ax2.set_xlim(0, n_frames - 1)  # Match feature frame range
            
            ax2.set_ylabel('Feature Index')
            ax2.set_title(f'{features_name} Features')
            
            # Add horizontal colorbar below the feature plot
            cbar = plt.colorbar(im, ax=ax2, orientation='horizontal', pad=0.15)
            
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
                        signal_max = np.max(np.abs(original_signal))
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
            plt.colorbar(im, format='%+2.0f dB')
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