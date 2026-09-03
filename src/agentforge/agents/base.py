from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Agent(ABC):
    """
    Public contract for an agent.
    """

    name: str
     
    @abstractmethod
    async def run(self, message: str, **kwargs: Any) -> str:
        """
        Process user input and return the agent response.
        """
        raise NotImplementedError

