from __future__ import annotations

from typing import Any

from .base import Tool


class ToolRegistry:
    """
    Registry through which an agent discovers and invokes tools.

    The registry is responsible for:
    - registering tools
    - discovering tools
    - exposing tool definitions
    - invoking a tool by name
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """
        Register a tool by its stable name.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")

        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """
        Remove a tool from registry.
        """
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """
        Discover a tool by name.
        """
        return self._tools.get(name)

    def list(self) -> list[Tool]:
        """
        Return all registered tools.
        """
        return list(self._tools.values())

    def definitions(self) -> list[dict[str, Any]]:
        """
        Return the capabilities exposed to an agent/model.
        """
        return [tool.definition() for tool in self._tools.values()]

    async def invoke(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        """
        Discover and invoke a tool by name.
        """
        tool = self.get(name)
        if tool is None:
            raise KeyError(f"Unknown tool: {name}")

        return await tool.invoke(**(arguments or {}))

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

