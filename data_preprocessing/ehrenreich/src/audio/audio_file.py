from src.audio.signal import Signal
from src.utils.Time import Time
import librosa
import os
import soundfile

class AudioFile:
    
    def __init__(self, path: str) -> None:
        """
        Initialize the AudioFile with the given path.
        Args:
            path (str): Path to the audio file.
        """
        self.path: str = path
        self.size = None
        self.loaded = False
        self.signal = None

    def load(self, sr: int | None = None) -> Signal:
        """Load audio file from the specified path.
        Args:
            sr (int, optional): Desired sample rate. If None, uses the file's original sample rate.
        Returns:
            (samples, sr) = Tuple[List[float], int]: A tuple containing the audio signal and the sample rate.
        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"File {self.path} does not exist.")
        
        samples, sample_rate = librosa.load(self.path, sr=sr)
        self.signal = Signal(samples, int(sample_rate), self.path)
        self.size = os.path.getsize(self.path)
        self.loaded = True

        size_mb = self.size / (1024 * 1024)
        print(f'* Loaded Audio Path \'{self.path}\' \n* Samples Number: {samples.shape[0]} \n* Sample Rate: {sample_rate} Hz \n* Duration: {Time.seconds_to_hms(self.signal._duration)} \n* File Size: {size_mb:.2f} MB')

        return self.signal

    @staticmethod
    def save(path: str, signal: Signal) -> None:
        """Save the audio signal to a file.
        Args:
            path (str): The path where the audio file will be saved.
            signal (Signal): The audio signal to save.
        """
        soundfile.write(path, signal.samples, signal.sample_rate)
        print(f'Audio saved to \"{path}\" with Sample Rate: {signal.sample_rate}')
