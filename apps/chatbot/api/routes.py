from __future__ import annotations

from fastapi import APIRouter, Depends

from apps.chatbot.api.dependencies import get_app
from apps.chatbot.api.schemas import ChatRequest, ChatResponse
from apps.chatbot.app import ChatbotApp

router = APIRouter(
    prefix="/api",
    tags=["chat"],
)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    app: ChatbotApp = Depends(get_app),
) -> ChatResponse:
    response = await app.run(
        session_id=request.session_id,
        message=request.message,
    )

    return ChatResponse(
        session_id=request.session_id,
        response=response,
    )

