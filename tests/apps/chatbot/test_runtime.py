from __future__ import annotations

import pytest

from agentforge.models.schemas import ModelResponse, ToolCall
from apps.chatbot.runtime import ChatbotRuntime
from agentforge.tools.registry import ToolRegistry

from tests.fakes.model import FakeModel
from tests.fakes.tools import FakeCalculator


@pytest.mark.asyncio
async def test_runtime_returns_model_response() -> None:
    model = FakeModel(responses="Hello from fake model")
    runtime = ChatbotRuntime(model=model)

    result = await runtime.execute(
        session_id="user-1",
        message="Hello",
    )

    assert result == "Hello from fake model"


@pytest.mark.asyncio
async def test_runtime_preserves_conversation_for_same_user() -> None:
    model = FakeModel()
    runtime = ChatbotRuntime(model=model)

    await runtime.execute(
        session_id="user-1",
        message="Hello",
    )

    await runtime.execute(
        session_id="user-1",
        message="How are you?",
    )

    assert len(model.calls) == 2

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
            "content": "Fake response",
        },
        {
            "role": "user",
            "content": "How are you?",
        },
    ]


@pytest.mark.asyncio
async def test_runtime_isolates_different_users() -> None:
    model = FakeModel()
    runtime = ChatbotRuntime(model=model)

    await runtime.execute(
        session_id="user-1",
        message="Hello from user 1",
    )

    await runtime.execute(
        session_id="user-2",
        message="Hello from user 2",
    )

    assert model.calls[0] == [
        {
            "role": "user",
            "content": "Hello from user 1",
        }
    ]

    assert model.calls[1] == [
        {
            "role": "user",
            "content": "Hello from user 2",
        }
    ]


@pytest.mark.asyncio
async def test_runtime_forwards_model_kwargs() -> None:
    model = FakeModel()
    runtime = ChatbotRuntime(model=model)

    await runtime.execute(
        session_id="user-1",
        message="Hello",
        temperature=0.5,
        max_tokens=100,
    )

    assert model.kwargs == [
        {
            "temperature": 0.5,
            "max_tokens": 100,
        }
    ]


@pytest.mark.asyncio
async def test_runtime_reuses_agent_for_same_user() -> None:
    model = FakeModel()
    runtime = ChatbotRuntime(model=model)

    first_agent = runtime._get_agent("user-1", "direct")
    second_agent = runtime._get_agent("user-1", "direct")

    assert first_agent is second_agent


@pytest.mark.asyncio
async def test_runtime_creates_different_agents_for_different_users() -> None:
    model = FakeModel()
    runtime = ChatbotRuntime(model=model)

    user_one_agent = runtime._get_agent("user-1", "direct")
    user_two_agent = runtime._get_agent("user-2", "direct")

    assert user_one_agent is not user_two_agent

@pytest.mark.asyncio
async def test_runtime_executes_react_agent_with_tool() -> None:
    tool_call = ToolCall(
        id="call_1",
        name="calculator",
        arguments={
            "expression": "2 + 2",
        },
    )

    model = FakeModel(
        responses=[
            # Model asks ReAct to invoke the calculator.
            ModelResponse(
                content=None,
                tool_calls=[tool_call],
            ),
            # Model receives the observation and gives the final answer.
            ModelResponse(
                content="The answer is 4.",
            ),
        ]
    )

    calculator = FakeCalculator()

    registry = ToolRegistry()
    registry.register(calculator)

    runtime = ChatbotRuntime(
        model=model,
        architecture="react",
        tool_registry=registry,
    )

    result = await runtime.execute(
        session_id="session-1",
        message="What is 2 + 2?",
    )

    assert result == "The answer is 4."

    assert len(model.calls) == 2

    assert len(calculator.calls) == 1
    assert calculator.calls[0] == {
        "expression": "2 + 2",
    }

@pytest.mark.asyncio
async def test_runtime_uses_requested_architecture() -> None:
    model = FakeModel(
        responses=[
            ModelResponse(content="Direct response"),
            ModelResponse(content="ReAct response"),
        ]
    )

    runtime = ChatbotRuntime(model=model)

    direct_result = await runtime.execute(
        session_id="user-1",
        message="Hello",
        architecture="direct",
    )

    react_result = await runtime.execute(
        session_id="user-1",
        message="Hello",
        architecture="react",
    )

    assert direct_result == "Direct response"
    assert react_result == "ReAct response"
