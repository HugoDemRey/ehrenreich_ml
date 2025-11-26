from abc import ABC, abstractmethod
import numpy as np


class ProcessedFeatures(ABC):
    @abstractmethod
    def plot(self) -> None:
        pass

class SelfSimilarityMatrix(ProcessedFeatures):
    def __init__(self, SSM: np.ndarray):
        self.SSM = SSM


    def plot(self) -> None:
        pass