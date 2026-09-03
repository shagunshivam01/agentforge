from __future__ import annotations

from typing import Any, cast

from groq import AsyncGroq
from groq.types.chat import ChatCompletionMessageParam

from agentforge.models.base import Model
from agentforge.models.schemas import ModelResponse


class GroqModel(Model):
    """
    Groq-backed LLM implementation.
    """

    name = "groq"

    def __init__(
        self,
        model: str,
        api_key: str,
        **kwargs: Any,
    ) -> None:
        self._model = model
        self._client = AsyncGroq(api_key=api_key)
        self._default_kwargs = kwargs

    async def generate(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Generate a response using Groq.
        """
        request_kwargs = {
            **self._default_kwargs,
            **kwargs,
        }

        groq_messages = cast(
            list[ChatCompletionMessageParam],
            messages,
        )
        
        response = await self._client.chat.completions.create(
            model=self._model,
            messages = groq_messages,
            **request_kwargs,
        )

        message = response.choices[0].message
        
        return ModelResponse(content=message.content)
