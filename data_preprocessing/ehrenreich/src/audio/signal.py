from typing import List
import numpy as np
from abc import ABC

class Signal(ABC):
    def __init__(self, samples: np.ndarray, sample_rate: float, origine_filename: str) -> None:

        if sample_rate <= 0:
            raise ValueError("Sample rate must be a positive integer.")
        if len(samples) == 0:
            raise ValueError("Samples list cannot be empty.")

        self.samples = samples
        self.sample_rate = sample_rate
        self._duration = len(samples) / sample_rate
        self.origin_filename = origine_filename
        id = origine_filename.rsplit('.', 1)[0].replace("data/", "")
        self.id: str = id

    def set_id(self, id: str) -> None:
        """Set the ID of the signal.
        Args:
            id (str): The ID to set.
        """
        self.id = id

    def get_id(self) -> str:
        """Get the ID of the signal.
        Returns:
            str: The ID of the signal.
        """
        return self.id
    
    def duration_seconds(self) -> float:
        return self._duration

    def subsignal(self, from_time: float, to_time: float) -> "SubSignal":
        """Extract a subsignal from the loaded audio signal.
        Args:
            from_time (float): Start time in seconds.
            to_time (float): End time in seconds.
        Returns:
            (samples, sample_rate) = Tuple[List[float], int]: A tuple containing the subsignal samples and the sample rate.
        Raises:
            ValueError: If the audio signal is not loaded.
        """
        if self.samples is None or self.sample_rate is None:
            raise ValueError("Signal not loaded. Please load the audio first.")
        
        start_sample = int(from_time * self.sample_rate)
        end_sample = int(to_time * self.sample_rate)

        print(f"Extracting subsignal from {from_time}s to {to_time}s")
        subsignal = SubSignal(self, from_time, to_time)
        print("Subsignal extracted successfully.")
        print(subsignal)
        
        # return Signal(self.samples[start_sample:end_sample], self.sample_rate, self.origin_filename, from_time=from_time, to_time=to_time)
        return subsignal
    
    def norm(self) -> float:
        """Compute the norm of the signal samples.
        Returns:
            float: The norm of the signal samples.
        Raises:
            ValueError: If the audio signal is not loaded.
        """
        return np.max(np.abs(self.samples)) + 1e-12  # Avoid division by zero
    
    def offset_time(self) -> float:
        """Get the offset time of the signal.
        Returns:
            float: The offset time in seconds.
        """
        return 0.0
    
    def save_wav(self, filename: str) -> None:
        """Save the signal as a WAV file.
        Args:
            filename (str): The filename to save the WAV file as.
        """
        from scipy.io import wavfile
        wavfile.write(filename, self.sample_rate, self.samples)
        print(f"Signal saved as WAV file: {filename}")
    
    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  id: '{self.id}',\n"
            f"  origin_filename: '{self.origin_filename}',\n"
            f"  sample_rate: {self.sample_rate} Hz,\n"
            f"  duration: {self._duration:.3f} s,\n"
            f"  samples number: {self.samples.shape[0]}\n"
            f")"
        )

class SubSignal(Signal):
    def __init__(self, parent_signal: Signal, from_time: float, to_time: float) -> None:
        if from_time < 0 or to_time > parent_signal._duration:
            raise ValueError(f"Invalid from_time and to_time for subsignal. from_time and to_time must fall in the interval [0 - {parent_signal._duration}]. ---> from_time: {from_time}, to_time: {to_time}")
        
        if from_time >= to_time:
            raise ValueError(f"from_time must be less than to_time for subsignal. from_time: {from_time}, to_time: {to_time}")

        start_sample = int(from_time * parent_signal.sample_rate)
        end_sample = int(to_time * parent_signal.sample_rate)
        
        super().__init__(parent_signal.samples[start_sample:end_sample], 
                         parent_signal.sample_rate, 
                         parent_signal.origin_filename)
        
        self.id += f"_{int(from_time + parent_signal.offset_time())}-{int(to_time + parent_signal.offset_time())}"
        self.duration = to_time - from_time
        self.from_time = from_time + parent_signal.offset_time()

    def offset_time(self) -> float:
        """Get the offset time of the subsignal.
        Returns:
            float: The offset time in seconds.
        """
        return self.from_time