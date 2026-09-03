from __future__ import annotations

from typing import Any

from agentforge.agents.architectures.direct.agent import DirectAgent
from agentforge.agents.base import Agent
from agentforge.memory.conversation import ConversationMemory
from agentforge.models.base import Model


class ChatbotRuntime:
    """
    Runtime responsible for executing the chatbot for a user.
    """

    def __init__(
        self, 
        model: Model,
        architecture: str = "direct",
    ) -> None:
        self._model = model
        self._architecture = architecture
        self._agents: dict[str, Agent] = {}

    def _create_agent(self) -> Agent:
        """
        Create an agent with its runtime dependencies.
        """
        memory = ConversationMemory()

        if self._architecture == "direct":
            return DirectAgent(
                model=self._model,
                memory=memory,
            )

        raise ValueError(
            f"Unsupported architecture: {self._architecture}"
        )

    def _get_agent(self, session_id: str) -> Agent:
        """
        Get or create an agent for a user.
        """
        agent = self._agents.get(session_id)

        if agent is None:
            agent = self._create_agent()

            self._agents[session_id] = agent

        return agent

    async def execute(
        self,
        session_id: str,
        message: str,
        **kwargs: Any,
    ) -> str:
        """
        Execute the chatbot for a session.
        """
        agent = self._get_agent(session_id)

        return await agent.run(
            message,
            **kwargs,
        )

