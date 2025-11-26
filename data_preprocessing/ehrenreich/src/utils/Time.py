import time
class Time:
    def __init__(self) -> None:
        self.start_time = None
        self.end_time = None

    @staticmethod
    def seconds_to_hms(seconds: float) -> str:
        """Convert seconds to a string in the format HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02}:{minutes:02}:{secs:02}"
    
    def start_timer(self) -> None:
        """Start a timer and return the start time."""
        self.start_time = time.time()
    
    def stop_timer(self) -> None:
        """Stop the timer and return the elapsed time in seconds."""
        if self.start_time is None:
            raise ValueError("Timer was not started. Please call start_timer() before stop_timer().")
        self.end_time = time.time()
        elapsed_time = self.end_time - self.start_time
        self.start_time = None  # Reset start time for future use
        print(f"Duration: {elapsed_time:.2f} seconds")