from __future__ import annotations

from typing import Any

from apps.chatbot.api.schemas import Architecture
from apps.chatbot.runtime import ChatbotRuntime


class ChatbotApp:
    """
    Application-level interface for the chatbot.
    """

    def __init__(self, runtime: ChatbotRuntime) -> None:
        self._runtime = runtime

    async def run(
        self, 
        session_id: str, 
        message: str,
        architecture: Architecture,
        **kwargs: Any,
    ) -> str:
        return await self._runtime.execute(
            session_id=session_id,
            message=message,
            architecture=architecture,
            **kwargs,
        )

