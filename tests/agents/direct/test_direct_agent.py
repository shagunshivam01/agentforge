from __future__ import annotations

from typing import Any

import pytest

from agentforge.agents.architectures.direct.agent import DirectAgent
from agentforge.memory.conversation import ConversationMemory
from agentforge.models.base import Model
from agentforge.models.schemas import ModelResponse


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
async def test_direct_agent_returns_model_response() -> None:
    model = FakeModel(response="Hello from fake model")
    agent = DirectAgent(model=model)

    response = await agent.run("Hello")

    assert response == "Hello from fake model"


@pytest.mark.asyncio
async def test_direct_agent_sends_user_message_to_model() -> None:
    model = FakeModel()
    agent = DirectAgent(model=model)

    await agent.run("Hello")

    assert model.calls == [
        [
            {
                "role": "user",
                "content": "Hello",
            }
        ]
    ]


@pytest.mark.asyncio
async def test_direct_agent_uses_existing_memory() -> None:
    model = FakeModel()
    memory = ConversationMemory()

    await memory.add_message(
        role="user",
        content="Previous question",
    )

    await memory.add_message(
        role="assistant",
        content="Previous answer",
    )

    agent = DirectAgent(
        model=model,
        memory=memory,
    )

    await agent.run("New question")

    assert model.calls == [
        [
            {
                "role": "user",
                "content": "Previous question",
            },
            {
                "role": "assistant",
                "content": "Previous answer",
            },
            {
                "role": "user",
                "content": "New question",
            },
        ]
    ]


@pytest.mark.asyncio
async def test_direct_agent_stores_conversation_in_memory() -> None:
    model = FakeModel(response="Hello back")
    memory = ConversationMemory()

    agent = DirectAgent(
        model=model,
        memory=memory,
    )

    response = await agent.run("Hello")

    assert response == "Hello back"

    messages = await memory.get_messages()

    assert messages == [
        {
            "role": "user",
            "content": "Hello",
        },
        {
            "role": "assistant",
            "content": "Hello back",
        },
    ]


@pytest.mark.asyncio
async def test_direct_agent_preserves_conversation_across_runs() -> None:
    model = FakeModel()
    memory = ConversationMemory()

    agent = DirectAgent(
        model=model,
        memory=memory,
    )

    await agent.run("First message")
    await agent.run("Second message")

    assert model.calls == [
        [
            {
                "role": "user",
                "content": "First message",
            }
        ],
        [
            {
                "role": "user",
                "content": "First message",
            },
            {
                "role": "assistant",
                "content": "Fake response",
            },
            {
                "role": "user",
                "content": "Second message",
            },
        ],
    ]


@pytest.mark.asyncio
async def test_direct_agent_works_without_memory() -> None:
    model = FakeModel()

    agent = DirectAgent(model=model)

    response = await agent.run("Hello")

    assert response == "Fake response"

    assert model.calls == [
        [
            {
                "role": "user",
                "content": "Hello",
            }
        ]
    ]

@pytest.mark.asyncio
async def test_direct_agent_forwards_model_kwargs() -> None:
    model = FakeModel()
    agent = DirectAgent(model=model)

    await agent.run(
        "Hello",
        temperature=0.5,
        max_tokens=100,
    )

    assert model.kwargs == [
        {
            "temperature": 0.5,
            "max_tokens": 100,
        }
    ]

