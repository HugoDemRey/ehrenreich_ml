from abc import ABC, abstractmethod
from typing import Any

class Builder(ABC):
    @abstractmethod
    def build(self, *args, **kwargs) -> Any:
        pass