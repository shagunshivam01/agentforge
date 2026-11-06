from __future__ import annotations

from typing import Any

import pytest

from agentforge.agents.architectures.react.loop import ReActLoop
from agentforge.models.base import Model
from agentforge.models.schemas import ModelResponse, ToolCall
from agentforge.tools.base import Tool
from agentforge.tools.registry import ToolRegistry


class FakeModel(Model):
    """
    Deterministic model that returns predefined responses.
    """

    name = "fake-model"

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ModelResponse:
        self.calls.append(
            {
                "messages": messages,
                **kwargs,
            }
        )

        if not self._responses:
            raise AssertionError(
                "FakeModel was called more times than expected"
            )

        return self._responses.pop(0)


class FakeCalculator(Tool):
    """
    Deterministic calculator returning 4.
    """

    name = "calculator"
    description = "Performs arithmetic"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                },
            },
            "required": ["expression"],
        }

    async def invoke(self, **arguments: Any) -> Any:
        self.calls.append(arguments)

        assert arguments == {
            "expression": "2 + 2",
        }

        return 4


@pytest.mark.asyncio
async def test_react_loop_executes_tool_and_returns_final_answer() -> None:
    # ---------------------------------------------------------
    # Fake model #1: asks to call calculator
    # ---------------------------------------------------------

    tool_call = ToolCall(
        id="call_1",
        name="calculator",
        arguments={
            "expression": "2 + 2",
        },
    )

    first_response = ModelResponse(
        content=None,
        tool_calls=[tool_call],
    )

    # ---------------------------------------------------------
    # Fake model #2: returns the final answer
    # ---------------------------------------------------------

    final_response = ModelResponse(
        content="The answer is 4.",
    )

    model = FakeModel(
        responses=[
            first_response,
            final_response,
        ]
    )

    # ---------------------------------------------------------
    # Fake calculator
    # ---------------------------------------------------------

    calculator = FakeCalculator()

    # ---------------------------------------------------------
    # Tool registry
    # ---------------------------------------------------------

    registry = ToolRegistry()
    registry.register(calculator)

    # ---------------------------------------------------------
    # ReAct loop
    # ---------------------------------------------------------

    loop = ReActLoop(
        model=model,
        tool_registry=registry,
        max_iterations=5,
    )

    messages = [
        {
            "role": "user",
            "content": "What is 2 + 2?",
        }
    ]

    result = await loop.run(messages)

    # ---------------------------------------------------------
    # Final answer
    # ---------------------------------------------------------

    assert result is final_response
    assert result.content == "The answer is 4."

    # ---------------------------------------------------------
    # Model was called exactly twice.
    #
    # Call #1 -> tool request
    # Call #2 -> final answer
    # ---------------------------------------------------------

    assert len(model.calls) == 2

    # ---------------------------------------------------------
    # Calculator was invoked exactly once.
    # ---------------------------------------------------------

    assert len(calculator.calls) == 1

    assert calculator.calls[0] == {
        "expression": "2 + 2",
    }


@pytest.mark.asyncio
async def test_react_loop_passes_tool_observation_to_second_model_call() -> None:
    tool_call = ToolCall(
        id="call_1",
        name="calculator",
        arguments={
            "expression": "2 + 2",
        },
    )

    model = FakeModel(
        responses=[
            ModelResponse(
                content=None,
                tool_calls=[tool_call],
            ),
            ModelResponse(
                content="The answer is 4.",
            ),
        ]
    )

    calculator = FakeCalculator()

    registry = ToolRegistry()
    registry.register(calculator)

    loop = ReActLoop(
        model=model,
        tool_registry=registry,
        max_iterations=5,
    )

    await loop.run(
        [
            {
                "role": "user",
                "content": "What is 2 + 2?",
            }
        ]
    )

    # The second model invocation must contain:
    #
    #   assistant -> calculator request
    #   tool      -> 4
    second_messages = model.calls[1]["messages"]

    assert {
        "role": "assistant",
        "content": None,
        "tool_call": {
            "name": "calculator",
            "arguments": {
                "expression": "2 + 2",
            },
        },
    } in second_messages

    assert {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "4",
    } in second_messages

@pytest.mark.asyncio
async def test_react_loop_builds_expected_message_history() -> None:
    tool_call = ToolCall(
        id="call_1",
        name="calculator",
        arguments={
            "expression": "2 + 2",
        },
    )

    model = FakeModel(
        responses=[
            ModelResponse(
                content=None,
                tool_calls=[tool_call],
            ),
            ModelResponse(
                content="The answer is 4.",
            ),
        ]
    )

    calculator = FakeCalculator()

    registry = ToolRegistry()
    registry.register(calculator)

    loop = ReActLoop(
        model=model,
        tool_registry=registry,
        max_iterations=5,
    )

    await loop.run(
        [
            {
                "role": "user",
                "content": "What is 2 + 2?",
            }
        ]
    )

    assert model.calls[1]["messages"] == [
        {
            "role": "user",
            "content": "What is 2 + 2?",
        },
        {
            "role": "assistant",
            "content": None,
            "tool_call": {
                "name": "calculator",
                "arguments": {
                    "expression": "2 + 2",
                },
            },
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "4",
        },
    ]

@pytest.mark.asyncio
async def test_react_loop_passes_tool_definitions_to_model() -> None:
    model = FakeModel(
        [
            ModelResponse(content="The answer is 4."),
        ]
    )

    registry = ToolRegistry()
    registry.register(FakeCalculator())

    loop = ReActLoop(
        model=model,
        tool_registry=registry,
        max_iterations=3,
    )

    result = await loop.run(
        [
            {
                "role": "user",
                "content": "What is 2 + 2?",
            }
        ]
    )

    assert result.content == "The answer is 4."

    assert model.calls[0]["tools"] == [
        {
            "name": "calculator",
            "description": "Performs arithmetic",
            "schema": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                },
                "required": ["expression"],
            },
        }
    ]
