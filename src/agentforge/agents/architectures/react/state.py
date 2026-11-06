from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentforge.agents.architectures.react.schemas import ReActDecision


class ReActState(BaseModel):
    """
    State carried through a ReAct execution.

    The state represents the current execution context of the agent.
    """

    messages: list[dict[str, Any]] = Field(default_factory=list)

    iteration: int = 0

    action: ReActDecision | None = None

    observation: Any | None = None

    finished: bool = False

