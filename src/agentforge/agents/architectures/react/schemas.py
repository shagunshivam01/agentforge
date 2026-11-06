from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ToolCall(BaseModel):
    """
    A request to invoke a tool.

    This is the ReAct-level representation of a tool action.
    """

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ReActDecision(BaseModel):
    """
    Decision produced from a model response at one ReAct step.

    A decision is exactly one of:
    - a tool call
    - a final answer
    """

    type: Literal['tool_call', "final"]

    tool_call: ToolCall | None = None
    final_answer: str | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> ReActDecision:
        if self.type == "tool_call":
            if self.tool_call is None:
                raise ValueError(
                    "tool_call is required when type is 'tool_call'"
                )

            if self.final_answer is not None:
                raise ValueError(
                    "final_answer must be None when type is 'tool_call'"
                )

        elif self.type == "final":
            if self.final_answer is None:
                raise ValueError(
                    "final_answer is required when type is 'final'"
                )

            if self.tool_call is not None:
                raise ValueError(
                    "tool_call must be None when type is 'final'"
                )

        return self

