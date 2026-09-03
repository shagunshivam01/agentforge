from __future__ import annotations

from typing import Any

from agentforge.agents.base import Agent
from agentforge.memory.base import Memory
from agentforge.models.base import Model


class DirectAgent(Agent):
    """
    Direct agent implementation.

    Sends the conversation directly to the model 
    and returns the generated response.
    """

    name = "direct"

    def __init__(
        self,
        model: Model,
        memory: Memory | None = None,
    ) -> None:
        self._model = model
        self._memory = memory

    async def run(
        self,
        message: str,
        **kwargs: Any,
    ) -> str:
        """
        Process user input and return the model response.
        """
        messages: list[dict[str, Any]] = []

        if self._memory is not None:
            messages = await self._memory.get_messages()

        messages.append(
            {
                "role": "user",
                "content": message,
            }
        )

        model_response = await self._model.generate(
            messages,
            **kwargs,
        )

        response = model_response.content or ""

        if self._memory is not None:
            await self._memory.add_message(
                role="user",
                content=message,
            )

            await self._memory.add_message(
                role="assistant",
                content=response,
            )

        return response

