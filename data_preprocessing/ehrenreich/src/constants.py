import enum
import os
import numpy as np

import musical_scales
from musical_scales import Note

_current_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.join(_current_dir, "..")

class FFT:
    DEFAULT_HOP_LENGTH = 512
    DEFAULT_WINDOW_SIZE = 4096

class SILENCE:
    TOP_DB = 35
    FRAME_LENGTH = 2048
    HOP_LENGTH = 512
    REF = 1.0

class CacheToken(enum.Enum):
    STFT = "stft.npz"
    SPECTROGRAM_LOG_FREQ = "spec_log_freq.npz"
    SILENCES = "silence_splits.npy"
    GROUND_TRUTH = "ground_truth.txt"

class Paths:
    DATA = os.path.join(_root_dir, "data")
    PLOTS = os.path.join(DATA, "plots")
    

class TLM:
    MEDIAN_FILTER_SIZE = 10
    GAUSSIAN_FILTER_SIZE = 25

class Scales:

    MAJOR_MINOR_RELATIVE_KEYS = {
            (("C", "ionian"), ("A", "aeolian")),
            (("G", "ionian"), ("E", "aeolian")),
            (("D", "ionian"), ("B", "aeolian")),
            (("A", "ionian"), ("F#", "aeolian")),
            (("E", "ionian"), ("C#", "aeolian")),
            (("B", "ionian"), ("G#", "aeolian")),
            (("F#", "ionian"), ("D#", "aeolian")),
            (("C#", "ionian"), ("A#", "aeolian")),
            (("G#", "ionian"), ("F", "aeolian")),
            (("D#", "ionian"), ("C", "aeolian")),
            (("A#", "ionian"), ("G", "aeolian")),
            (("F", "ionian"), ("D", "aeolian")),
    }

    SCALES_TO_INDEX: dict[str, int] = {
        "C-Am": 0,
        "C": 0,
        "Am": 0,
        "G-Em": 1,
        "G": 1,
        "Em": 1,
        "D-Bm": 2,
        "D": 2,
        "Bm": 2,
        "A-F#m": 3,
        "A": 3,
        "F#m": 3,
        "E-C#m": 4,
        "E": 4,
        "C#m": 4,
        "B-G#m": 5,
        "B": 5,
        "G#m": 5,
        "F#-D#m": 6,
        "F#": 6,
        "D#m": 6,
        "C#-A#m": 7,
        "C#": 7,
        "A#m": 7,
        "G#-Fm": 8,
        "G#": 8,
        "Fm": 8,
        "D#-Cm": 9,
        "D#": 9,
        "Cm": 9,
        "A#-Gm": 10,
        "A#": 10,
        "Gm": 10,
        "F-Dm": 11,
        "F": 11,
        "Dm": 11,
    }

    @staticmethod
    def get_scales_vectors() -> dict[str, np.ndarray]:
        scale_to_vec: dict[str, tuple[str]] = {}
        vec_to_scale: dict[tuple[str], str] = {}

        for major_key, minor_key in Scales.MAJOR_MINOR_RELATIVE_KEYS:
            key_name = f"{major_key[0]}-{minor_key[0]}m"
            
            major_scale: list[Note] = musical_scales.scale(major_key[0], major_key[1])
            major_scale_str: list[str] = [note.name for note in major_scale]

            indices = [musical_scales.interval_from_names[name] for name in major_scale_str]
            vector = np.zeros(12, dtype=int)
            for index in indices:
                vector[index] = 1

            scale_to_vec[key_name] = tuple(vector.tolist())
            vec_to_scale[tuple(vector.tolist())] = key_name

        return scale_to_vec, vec_to_scale
    
    NOTE_TO_INDEX: dict[str, int] = musical_scales.interval_from_names

# Move SCALES_VECTORS assignment outside the class definition
Scales.SCALE_TO_VEC, Scales.VEC_TO_SCALE = Scales.get_scales_vectors()

         