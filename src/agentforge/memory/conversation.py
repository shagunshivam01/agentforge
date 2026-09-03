from __future__ import annotations

from typing import Any

from agentforge.memory.base import Memory


class ConversationMemory(Memory):
    """
    In-memory conversation history.
    """

    name = "conversation"

    def __init__(
        self,
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        self._messages: list[dict[str, Any]] = list(messages or [])

    async def get_messages(self) -> list[dict[str, Any]]:
        """
        Return the stored conversation messages.
        """
        return list(self._messages)

    async def add_message(
        self,
        role: str,
        content: str,
        **kwargs: Any,
    ) -> None:
        """
        Add a message to the conversation.
        """
        message: dict[str, Any] = {
            "role": role,
            "content": content,
        }

        message.update(kwargs)

        self._messages.append(message)

    async def clear(self) -> None:
        """
        Clear the conversation history.
        """
        self._messages.clear()

