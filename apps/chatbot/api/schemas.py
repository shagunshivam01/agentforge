from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Architecture = Literal["direct", "react"]


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    architecture: Architecture = "direct"


class ChatResponse(BaseModel):
    session_id: str
    response: str
    architecture: Architecture

