from __future__ import annotations

from typing import Any

from agentforge.agents.architectures.react.schemas import (
    ReActDecision,
    ToolCall as ReActToolCall,
)
from agentforge.agents.architectures.react.state import ReActState
from agentforge.models.base import Model
from agentforge.models.schemas import ModelResponse
from agentforge.tools.registry import ToolRegistry


class ReActLoop:
    """
    Execute an ReAct loop using a model and tool registry.

    The loop is responsible for:

    - requesting a model response
    - converting the response into a ReAct ReAct decision
    - executing the requested tools
    - recording observations
    - continuing until a final answer is produced
    - enforcing the maximum iteration limit

    The loop is independent of concrete model providers and 
    tool implementations.
    """

    def __init__(
        self,
        model: Model,
        tool_registry: ToolRegistry,
        max_iterations: int
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max iterations must be greater than 0")

        self._model = model
        self._tool_registry = tool_registry
        self._max_iterations = max_iterations

    async def run(
        self,
        messages: list[dict[str, Any]],
    ) -> ModelResponse:
        """
        Execute the ReAct loop and return the final model response.
        """
        
        state = ReActState(
            messages=list(messages),
        )

        while state.iteration < self._max_iterations:
            response = await self._model.generate(
                messages=state.messages,
                tools=self._tool_registry.definitions(),
            )

            decision = self._decision_from_response(response)

            state.action = decision
            state.observation = None

            if decision.type == "final":
                state.finished = True
                return response

            tool_call = decision.tool_call
            assert tool_call is not None

            observation = await self._tool_registry.invoke(
                name=tool_call.tool_name,
                arguments=tool_call.arguments,
            )

            state.observation = observation

            state.messages.append(
                self._assistant_tool_call_message(
                    response=response,
                    tool_call=tool_call,
                )
            )

            state.messages.append(
                self._tool_observation_message(
                    tool_call_id=self._model_tool_call_id(response),
                    observation=observation,
                )
            )

            state.iteration += 1

        raise RuntimeError(
            f"ReAct loop exceeded maximum iterations: "
            f"{self._max_iterations}"
        )

    @staticmethod
    def _decision_from_response(
        response: ModelResponse,
    ) -> ReActDecision:
        """
        Converts a model response into a ReAct decision.
        """
        
        if len(response.tool_calls) > 1:
            raise ValueError(
                "ReAct currently supports one tool call per iteration"
            )

        if response.tool_calls:
            tool_call = response.tool_calls[0]

            return ReActDecision(
                type="tool_call",
                tool_call=ReActToolCall(
                    tool_name=tool_call.name,
                    arguments=tool_call.arguments,
                ),
            )

        return ReActDecision(
            type="final",
            final_answer=response.content,
        )

    @staticmethod
    def _assistant_tool_call_message(
        response: ModelResponse,
        tool_call: ReActToolCall,
    ) -> dict[str, Any]:
        """
        Build the internal assistant message representing a tool call.
        """

        return {
            "role": "assistant",
            "content": response.content,
            "tool_call": {
                "name": tool_call.tool_name,
                "arguments": tool_call.arguments,
            },
        }

    @staticmethod
    def _tool_observation_message(
        tool_call_id: str,
        observation: Any
    ) -> dict[str, Any]:
        """
        Build the internal tool observation message.
        """

        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": str(observation),
        }

    @staticmethod
    def _model_tool_call_id(
        response: ModelResponse,
    ) -> str:
        """
        Return the provider-level tool call ID.

        The ID belongs to the model protocol, not the ReAct semantic
        decision.
        """

        if not response.tool_calls:
            raise ValueError(
                "Cannot get tool call ID from response without a tool call"
            )

        return response.tool_calls[0].id

