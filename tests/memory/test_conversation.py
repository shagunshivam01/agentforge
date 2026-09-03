from __future__ import annotations

import pytest

from agentforge.memory.conversation import ConversationMemory


@pytest.mark.asyncio
async def test_memory_starts_empty() -> None:
    memory = ConversationMemory()

    messages = await memory.get_messages()

    assert messages == []


@pytest.mark.asyncio
async def test_add_message() -> None:
    memory = ConversationMemory()

    await memory.add_message(
        role="user",
        content="Hello",
    )

    messages = await memory.get_messages()

    assert messages == [
        {
            "role": "user",
            "content": "Hello",
        }
    ]


@pytest.mark.asyncio
async def test_messages_preserve_order() -> None:
    memory = ConversationMemory()

    await memory.add_message(
        role="user",
        content="Hello",
    )

    await memory.add_message(
        role="assistant",
        content="Hi!",
    )

    await memory.add_message(
        role="user",
        content="How are you?",
    )

    messages = await memory.get_messages()

    assert messages == [
        {
            "role": "user",
            "content": "Hello",
        },
        {
            "role": "assistant",
            "content": "Hi!",
        },
        {
            "role": "user",
            "content": "How are you?",
        },
    ]


@pytest.mark.asyncio
async def test_add_message_supports_extra_fields() -> None:
    memory = ConversationMemory()

    await memory.add_message(
        role="user",
        content="Hello",
        name="Alice",
    )

    messages = await memory.get_messages()

    assert messages == [
        {
            "role": "user",
            "content": "Hello",
            "name": "Alice",
        }
    ]


@pytest.mark.asyncio
async def test_clear_removes_all_messages() -> None:
    memory = ConversationMemory()

    await memory.add_message(
        role="user",
        content="Hello",
    )

    await memory.add_message(
        role="assistant",
        content="Hi!",
    )

    await memory.clear()

    messages = await memory.get_messages()

    assert messages == []

@pytest.mark.asyncio
async def test_get_messages_returns_copy() -> None:
    memory = ConversationMemory()

    await memory.add_message(
        role="user",
        content="Hello",
    )

    messages = await memory.get_messages()
    messages.clear()

    assert await memory.get_messages() == [
        {
            "role": "user",
            "content": "Hello",
        }
    ]

