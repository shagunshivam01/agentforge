from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agentforge.models.groq import GroqModel
from agentforge.models.schemas import ModelResponse, ToolCall


class FakeCompletions:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response


class FakeGroqClient:
    def __init__(self, response: Any) -> None:
        self.chat = SimpleNamespace(
            completions=FakeCompletions(response)
        )


def make_groq_response(
    *,
    content: str | None = None,
    tool_calls: list[Any] | None = None,
) -> Any:
    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
    )

    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=message)
        ]
    )


def make_groq_tool_call(
    *,
    call_id: str = "call_1",
    name: str = "calculator",
    arguments: str = '{"expression": "2 + 2"}',
) -> Any:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name=name,
            arguments=arguments,
        ),
    )


@pytest.mark.asyncio
async def test_groq_model_returns_final_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_response = make_groq_response(
        content="The answer is 4.",
    )

    fake_client = FakeGroqClient(fake_response)

    monkeypatch.setattr(
        "agentforge.models.groq.AsyncGroq",
        lambda api_key: fake_client,
    )

    model = GroqModel(
        model="test-model",
        api_key="test-key",
    )

    result = await model.generate(
        messages=[
            {
                "role": "user",
                "content": "What is 2 + 2?",
            }
        ],
    )

    assert result == ModelResponse(
        content="The answer is 4.",
        tool_calls=[],
    )


@pytest.mark.asyncio
async def test_groq_model_converts_tool_call_to_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_response = make_groq_response(
        content=None,
        tool_calls=[
            make_groq_tool_call(
                call_id="call_1",
                name="calculator",
                arguments='{"expression": "2 + 2"}',
            )
        ],
    )

    fake_client = FakeGroqClient(fake_response)

    monkeypatch.setattr(
        "agentforge.models.groq.AsyncGroq",
        lambda api_key: fake_client,
    )

    model = GroqModel(
        model="test-model",
        api_key="test-key",
    )

    result = await model.generate(
        messages=[
            {
                "role": "user",
                "content": "What is 2 + 2?",
            }
        ],
    )

    assert result == ModelResponse(
        content=None,
        tool_calls=[
            ToolCall(
                id="call_1",
                name="calculator",
                arguments={
                    "expression": "2 + 2",
                },
            )
        ],
    )


@pytest.mark.asyncio
async def test_groq_model_translates_tool_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_response = make_groq_response(
        content="The answer is 4.",
    )

    fake_client = FakeGroqClient(fake_response)

    monkeypatch.setattr(
        "agentforge.models.groq.AsyncGroq",
        lambda api_key: fake_client,
    )

    model = GroqModel(
        model="test-model",
        api_key="test-key",
    )

    messages = [
        {
            "role": "user",
            "content": "What is 2 + 2?",
        },
        {
            "role": "assistant",
            "content": None,
            "tool_call": {
                "id": "call_1",
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

    await model.generate(messages=messages)

    request = fake_client.chat.completions.calls[0]

    assert request["messages"] == [
        {
            "role": "user",
            "content": "What is 2 + 2?",
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": '{"expression": "2 + 2"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "4",
        },
    ]

@pytest.mark.asyncio
async def test_groq_model_translates_tool_definitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_response = make_groq_response(
        content="The answer is 4.",
    )

    fake_client = FakeGroqClient(fake_response)

    monkeypatch.setattr(
        "agentforge.models.groq.AsyncGroq",
        lambda api_key: fake_client,
    )

    model = GroqModel(
        model="test-model",
        api_key="test-key",
    )

    tools = [
        {
            "name": "calculator",
            "description": "Performs arithmetic",
            "schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                    },
                },
                "required": ["expression"],
            },
        }
    ]

    await model.generate(
        messages=[
            {
                "role": "user",
                "content": "What is 2 + 2?",
            }
        ],
        tools=tools,
    )

    request = fake_client.chat.completions.calls[0]

    assert request["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Performs arithmetic",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                        },
                    },
                    "required": ["expression"],
                },
            },
        }
    ]
