from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """
    Public contract for a capability/tool.

    A tool exposes:
    - a stable name
    - a description
    - a schema
    - an invocation method
    """

    name: str
    description: str

    @property
    @abstractmethod
    def schema(self) -> dict[str, Any]:
        """
        Return JSON-schema-like description of the tool's schema.
        """
        raise NotImplementedError

    @abstractmethod
    async def invoke(self, **kwargs: Any) -> Any:
        """
        Execute the capability with keyword arguments.
        """
        raise NotImplementedError

    def definition(self) -> dict[str, Any]:
        """
        Return the tool definition exposed to an agent/model.
        """
        return {
            "name": self.name,
            "description": self.description,
            "schema": self.schema,
        }

