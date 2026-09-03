from __future__ import annotations

import asyncio

from agentforge.infrastructure.config import settings
from agentforge.models.groq import GroqModel
from apps.chatbot.app import ChatbotApp
from apps.chatbot.runtime import ChatbotRuntime


def create_app() -> ChatbotApp:
    model = GroqModel(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
    )

    runtime = ChatbotRuntime(model=model)

    return ChatbotApp(runtime=runtime)

async def main() -> None:
    app = create_app()

    session_id = "local-user"

    print("Agentforge Chatbot")
    print("Type 'exit' to quit.\n")

    while True:
        message = input("You: ")

        if message.lower() in {"exit"}:
            print("Goodbye!")
            break

        response = await app.run(
            session_id=session_id,
            message=message,
        )

        print(f"Assistant: {response}\n")

if __name__ == "__main__":
    asyncio.run(main())

