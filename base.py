"""
cache/base.py — Abstract cache interface.
All cache backends implement this so the pipeline never knows
which backend is running underneath.
"""
from abc import ABC, abstractmethod
from typing import Any


class BaseCache(ABC):

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """Return cached value or None if not found / expired."""
        ...

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store value with optional TTL in seconds."""
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove a single key."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Wipe the entire cache."""
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key is present and not expired."""
        ...
