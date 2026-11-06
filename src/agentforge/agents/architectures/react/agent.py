from __future__ import annotations

from typing import Any

from agentforge.agents.architectures.react.loop import ReActLoop
from agentforge.agents.base import Agent
from agentforge.memory.base import Memory
from agentforge.memory.conversation import ConversationMemory
from agentforge.models.base import Model
from agentforge.tools.registry import ToolRegistry


class ReActAgent(Agent):
    """
    Public ReAct Agent.

    The agent owns the high-level interface for executing a ReAct task.
    The actual iterative reasoning/tool-execution logic is delegated to ReActLoop.
    """

    name = "react"

    def __init__(
        self,
        model: Model,
        tool_registry: ToolRegistry,
        memory: Memory | None = None,
        max_iterations: int = 5,
    ) -> None:
        self._memory = ConversationMemory()
        self._loop = ReActLoop(
            model=model,
            tool_registry=tool_registry,
            max_iterations=max_iterations,
        )

    async def run(
        self, 
        message: str,
        **kwargs: Any,
    ) -> str:
        """
        Execute a ReAct task from a user message.
        """
        messages = await self._memory.get_messages()

        messages.append(
            {
                "role": "user",
                "content": message
            }
        )
        
        response = await self._loop.run(
            messages=messages,
            **kwargs,
        )

        answer = response.content or ""

        await self._memory.add_message(
            role="user",
            content=message,
        )

        await self._memory.add_message(
            role="assistant",
            content=answer,
        )

        return answer
        
