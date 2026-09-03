from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    """
    A request from a model to invoke a tool.
    """

    id: str
    name: str
    arguments: dict[str, Any]

@dataclass(frozen=True)
class ModelResponse:
    """
    Model-independent response returned by a Model.
    """

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

