from __future__ import annotations

import json
from typing import Any

from groq import AsyncGroq
from groq.types.chat import ChatCompletionMessageParam

from agentforge.models.base import Model
from agentforge.models.schemas import ModelResponse, ToolCall


class GroqModel(Model):
    """
    Groq-backed LLM implementation.

    Translates AgentForge's provider-independent message format 
    into Groq's chat-completion format and converts Groq's responses
    back into ModelResponse objects.
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

        groq_messages = self._to_groq_messages(messages)

        if "tools" in request_kwargs:
            request_kwargs["tools"] = self._to_groq_tools(
                request_kwargs["tools"]
            )
        
        response = await self._client.chat.completions.create(
            model=self._model,
            messages = groq_messages,
            **request_kwargs,
        )

        message = response.choices[0].message

        tool_calls: list[ToolCall] = []

        for tool_call in message.tool_calls or []:
            tool_calls.append(
                ToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=json.loads(tool_call.function.arguments),
                )
            )
        
        return ModelResponse(
            content=message.content,
            tool_calls=tool_calls,
        )

    @staticmethod
    def _to_groq_messages(
        messages: list[dict[str, Any]]
    ) -> list[ChatCompletionMessageParam]:
        """
        Translate AgentForge messages into Groq messages.
        """

        groq_messages: list[ChatCompletionMessageParam] = []

        for message in messages:
            role = message["role"]

            if role == "user":
                groq_messages.append(
                    {
                        "role": "user",
                        "content": message["content"],
                    }
                )

            elif role == "assistant":
                tool_call = message.get("tool_call")

                if tool_call is None:
                    groq_messages.append(
                        {
                            "role": "assistant",
                            "content": message.get("content"),
                        }
                    )

                    continue
                    
                groq_messages.append(
                    {
                        "role": "assistant",
                        "content": message.get("content"),
                        "tool_calls": [
                            {
                                "id": tool_call["id"],
                                "type": "function",
                                "function": {
                                    "name": tool_call["name"],
                                    "arguments": json.dumps(
                                        tool_call["arguments"]
                                    ),
                                },
                            },
                        ],
                    }
                )

            elif role == "tool":
                groq_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": message["tool_call_id"],
                        "content": message["content"],
                    }
                )

            else:
                raise ValueError(
                    f"Unsupported message role: {role}"
                )

        return groq_messages


    @staticmethod
    def _to_groq_tools(
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Translate AgentForge tool definitions into Groq tool definitions.
        """

        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["schema"],
                },
            }
            for tool in tools
        ]
