from abc import ABC, abstractmethod

class Plottable(ABC):
    @abstractmethod
    def plot(self, *args, **kwargs):
        pass