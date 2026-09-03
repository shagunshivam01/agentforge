from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agentforge.models.schemas import ModelResponse


class Model(ABC):
    """
    Public contract for an LLM.
    """

    name: str

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, Any]], 
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Generate a complete response.
        """
        raise NotImplementedError

