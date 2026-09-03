from __future__ import annotations

from typing import Any

from apps.chatbot.runtime import ChatbotRuntime


class ChatbotApp:
    def __init__(self, runtime: ChatbotRuntime) -> None:
        self._runtime = runtime

    async def run(
        self, 
        session_id: str, 
        message: str,
        **kwargs: Any,
    ) -> str:
        return await self._runtime.execute(
            session_id=session_id,
            message=message,
            **kwargs,
        )

