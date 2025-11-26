from src.audio.audio_file import AudioFile
from src.audio.signal import Signal
from src.io.ts_annotation import TSAnnotations


def load_signal(file_path: str, cut_start_seconds: float | None = None, cut_end_seconds: float | None = None) -> Signal:
    audio_file = AudioFile(file_path)
    signal = audio_file.load()

    cut_start, cut_end = 0, signal.duration_seconds()
    if cut_start_seconds is not None:
        cut_start = cut_start_seconds
    if cut_end_seconds is not None:
        cut_end = cut_end_seconds

    return signal.subsignal(cut_start, cut_end)


def load_transitions_labels(file_path: str, cut_start_seconds: float | None = None, cut_end_seconds: float | None = None):
    transitions_ts = TSAnnotations.load_transitions_txt(file_path)
    if cut_end_seconds is not None:
        transitions_ts = filter(
            lambda t: t <= cut_end_seconds,
            transitions_ts
        )

    if cut_start_seconds is not None:
        transitions_ts = filter(
            lambda t: t >= cut_start_seconds,
            transitions_ts
        )

    return list(transitions_ts)