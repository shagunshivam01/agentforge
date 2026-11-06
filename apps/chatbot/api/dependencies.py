from __future__ import annotations

from functools import lru_cache

from agentforge.infrastructure.config import settings
from agentforge.models.groq import GroqModel
from apps.chatbot.app import ChatbotApp
from apps.chatbot.runtime import ChatbotRuntime


@lru_cache
def get_app() -> ChatbotApp:
    model = GroqModel(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        max_completion_tokens=500,
    )

    runtime = ChatbotRuntime(
        model=model,
        architecture=settings.agent_architecture,
    )

    return ChatbotApp(runtime=runtime)

