from __future__ import annotations

from typing import Any

import pytest

from agentforge.models.base import Model
from agentforge.models.schemas import ModelResponse
from apps.chatbot.runtime import ChatbotRuntime


class FakeModel(Model):
    name = "fake"

    def __init__(self, response: str = "Fake response") -> None:
        self.response = response
        self.calls: list[list[dict[str, Any]]] = []
        self.kwargs: list[dict[str, Any]] = []

    async def generate(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ModelResponse:
        self.calls.append(messages)
        self.kwargs.append(kwargs)

        return ModelResponse(content=self.response)


@pytest.mark.asyncio
async def test_runtime_returns_model_response() -> None:
    model = FakeModel(response="Hello from fake model")
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

    first_agent = runtime._get_agent("user-1")
    second_agent = runtime._get_agent("user-1")

    assert first_agent is second_agent


@pytest.mark.asyncio
async def test_runtime_creates_different_agents_for_different_users() -> None:
    model = FakeModel()
    runtime = ChatbotRuntime(model=model)

    user_one_agent = runtime._get_agent("user-1")
    user_two_agent = runtime._get_agent("user-2")

    assert user_one_agent is not user_two_agent
