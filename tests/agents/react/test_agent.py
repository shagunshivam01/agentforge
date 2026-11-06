from __future__ import annotations

import pytest

from agentforge.agents.architectures.react.agent import ReActAgent
from agentforge.memory.conversation import ConversationMemory
from agentforge.models.schemas import ModelResponse
from agentforge.tools.registry import ToolRegistry
from tests.fakes.model import FakeModel


@pytest.mark.asyncio
async def test_react_agent_runs_prompt() -> None:
    model = FakeModel(
        responses=[
            ModelResponse(
                content="The answer is 4.",
            )
        ]
    )

    registry = ToolRegistry()

    agent = ReActAgent(
        model=model,
        tool_registry=registry,
    )

    result = await agent.run("What is 2 + 2?")

    assert result == "The answer is 4."

    assert model.calls == [
        [
            {
                "role": "user",
                "content": "What is 2 + 2?",
            }
        ]
    ]


@pytest.mark.asyncio
async def test_react_agent_passes_max_iterations_to_loop() -> None:
    model = FakeModel(
        responses=[
            ModelResponse(
                content="Done.",
            )
        ]
    )

    registry = ToolRegistry()

    agent = ReActAgent(
        model=model,
        tool_registry=registry,
        max_iterations=1,
    )

    result = await agent.run("Hello")

    assert result == "Done."

@pytest.mark.asyncio
async def test_react_agent_preserves_conversation_memory() -> None:
    model = FakeModel(
        responses=[
            ModelResponse(content="Hello!"),
            ModelResponse(content="I'm doing well."),
        ]
    )

    memory = ConversationMemory()

    registry = ToolRegistry()

    agent = ReActAgent(
        model=model,
        tool_registry=registry,
        memory=memory,
    )

    first_result = await agent.run("Hello")
    second_result = await agent.run("How are you?")

    assert first_result == "Hello!"
    assert second_result == "I'm doing well."

    assert model.calls[0] == [
        {
            "role": "user",
            "content": "Hello",
        }
    ]

    assert model.calls[1] == [
        {
            "role": "user",
            "content": "Hello",
        },
        {
            "role": "assistant",
            "content": "Hello!",
        },
        {
            "role": "user",
            "content": "How are you?",
        },
    ]
