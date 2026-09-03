from __future__ import annotations

from typing import Any

import pytest

from agentforge.tools.base import Tool
from agentforge.tools.registry import ToolRegistry


class FakeTool(Tool):
    name = "fake_tool"
    description = "A fake tool used for testing."

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                },
            },
            "required": ["value"],
        }

    async def invoke(self, **kwargs: Any) -> Any:
        return f"received: {kwargs['value']}"


def test_registry_starts_empty() -> None:
    registry = ToolRegistry()

    assert len(registry) == 0
    assert registry.list() == []


def test_register_tool() -> None:
    registry = ToolRegistry()
    tool = FakeTool()

    registry.register(tool)

    assert len(registry) == 1
    assert registry.get("fake_tool") is tool


def test_get_unknown_tool_returns_none() -> None:
    registry = ToolRegistry()

    assert registry.get("unknown") is None


def test_register_duplicate_tool_raises_error() -> None:
    registry = ToolRegistry()
    tool = FakeTool()

    registry.register(tool)

    with pytest.raises(ValueError, match="Tool already registered"):
        registry.register(tool)


def test_list_tools() -> None:
    registry = ToolRegistry()
    tool = FakeTool()

    registry.register(tool)

    assert registry.list() == [tool]


def test_tool_definitions() -> None:
    registry = ToolRegistry()
    tool = FakeTool()

    registry.register(tool)

    assert registry.definitions() == [
        {
            "name": "fake_tool",
            "description": "A fake tool used for testing.",
            "schema": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                    },
                },
                "required": ["value"],
            },
        }
    ]


@pytest.mark.asyncio
async def test_invoke_tool() -> None:
    registry = ToolRegistry()
    tool = FakeTool()

    registry.register(tool)

    result = await registry.invoke(
        "fake_tool",
        {"value": "hello"},
    )

    assert result == "received: hello"


@pytest.mark.asyncio
async def test_invoke_unknown_tool_raises_error() -> None:
    registry = ToolRegistry()

    with pytest.raises(KeyError, match="Unknown tool"):
        await registry.invoke(
            "unknown",
            {"value": "hello"},
        )


def test_unregister_tool() -> None:
    registry = ToolRegistry()
    tool = FakeTool()

    registry.register(tool)
    registry.unregister("fake_tool")

    assert len(registry) == 0
    assert registry.get("fake_tool") is None


def test_unregister_unknown_tool_does_nothing() -> None:
    registry = ToolRegistry()

    registry.unregister("unknown")

    assert len(registry) == 0


def test_tool_name_is_supported_by_contains() -> None:
    registry = ToolRegistry()
    tool = FakeTool()

    registry.register(tool)

    assert "fake_tool" in registry
    assert "unknown" not in registry
