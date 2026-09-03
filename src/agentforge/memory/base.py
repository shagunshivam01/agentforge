from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Memory(ABC):
    """
    Public contract for agent memory.
    """

    @abstractmethod
    async def get_messages(self) -> list[dict[str, Any]]:
        """
        Return the stored conversation messages.
        """
        raise NotImplementedError

    @abstractmethod
    async def add_message(
        self, 
        role: str, 
        content: str, 
        **kwargs: Any
    ) -> None:
        """
        Add a message to memory.
        """
        raise NotImplementedError

    @abstractmethod
    async def clear(self) -> None:
        """
        Clear all stored messages.
        """
        raise NotImplementedError

