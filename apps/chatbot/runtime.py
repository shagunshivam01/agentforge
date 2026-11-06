from __future__ import annotations

from typing import Any

from agentforge.agents.architectures.direct.agent import DirectAgent
from agentforge.agents.architectures.react.agent import ReActAgent
from agentforge.agents.base import Agent
from agentforge.memory.conversation import ConversationMemory
from agentforge.models.base import Model
from agentforge.tools.registry import ToolRegistry


class ChatbotRuntime:
    """
    Runtime responsible for executing the chatbot for a user.
    """

    def __init__(
        self, 
        model: Model,
        tool_registry: ToolRegistry | None = None,
        architecture: str = "direct",
    ) -> None:
        self._model = model
        self._tool_registry = (
            tool_registry if tool_registry is not None else ToolRegistry()
        )
        self._default_architecture = architecture
        self._agents: dict[tuple[str, str], Agent] = {}

    def _create_agent(self, architecture: str) -> Agent:
        """
        Create an agent with its runtime dependencies.
        """
        memory = ConversationMemory()
        
        if self._default_architecture == "direct":
            return DirectAgent(
                model=self._model,
                memory=memory,
            )

        if self._default_architecture == "react":
            return ReActAgent(
                model=self._model,
                tool_registry=self._tool_registry,
                memory=memory,
                max_iterations=5,
            )

        raise ValueError(
            f"Unsupported architecture: {self._default_architecture}"
        )

    def _get_agent(
        self, 
        session_id: str,
        architecture: str,
    ) -> Agent:
        """
        Get or create an agent for a session and architecture.
        """
        key = (session_id, architecture)

        agent = self._agents.get(key)

        if agent is None:
            agent = self._create_agent(architecture)

            self._agents[key] = agent

        return agent

    async def execute(
        self,
        session_id: str,
        message: str,
        architecture: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Execute the chatbot for a session using the requested architecture.
        """
        selected_architecture = (
            architecture
            if architecture is not None
            else self._default_architecture
        )

        agent = self._get_agent(
            session_id,
            architecture=selected_architecture,
        )

        return await agent.run(
            message,
            **kwargs,
        )

