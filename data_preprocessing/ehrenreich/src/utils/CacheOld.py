import os
from src.audio.signal import Signal
from src.constants import CacheToken
import numpy as np

class CacheOld:
    @staticmethod
    def resource_exists(id: str, token: CacheToken):
        """Check if a resource exists in the cache.
        Args:
            id (str): The ID of the resource.
            token (CacheToken): The type of resource to check.
        Returns:
            bool: True if the resource exists, False otherwise.
        """
        dir_path = CacheOld._id_to_cache_dir(id)
        exists = os.path.exists(os.path.join(dir_path, token.value))

        if exists:
            print(f"🟢 \033[92mCache hit for {id}/{token.value}\033[0m")
        else:
            print(f"🟠 \033[93mCache miss for {id}/{token.value}\033[0m")
        return exists
    
    @staticmethod
    def resource_path(id: str, token: CacheToken) -> str:
        """Get the file path for a cached resource.
        Args:
            id (str): The ID of the resource.
            token (CacheToken): The type of resource.
        Returns:
            str: The file path of the cached resource.
        """
        CacheOld._make_cache_dir(id)
        dir_path = CacheOld._id_to_cache_dir(id)
        return os.path.join(dir_path, token.value)

    @staticmethod
    def _id_to_cache_dir(id: str) -> str:
        """Convert an ID to a file path.
        Args:
            id (str): The ID to convert.
        Returns:
            str: The corresponding file path.
        """
        return f"data/cache/{id}"
    
    
    @staticmethod
    def _make_cache_dir(id: str) -> None:
        """Create a directory for the given ID if it doesn't exist.
        Args:
            id (str): The ID for which to create the directory.
        """
        dir_path = CacheOld._id_to_cache_dir(id)
        os.makedirs(dir_path, exist_ok=True)