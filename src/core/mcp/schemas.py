from pydantic import BaseModel, Field
from typing import Dict, Any, List


class ToolCall(BaseModel):
    """
    Represents a single tool invocation.
    """

    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class PlanOutput(BaseModel):
    """
    Planner output consumed by the runtime.
    """

    requires_tools: bool = False
    confidence: float = 1.0
    reason: str = ""

    tool_calls: List[ToolCall] = Field(default_factory=list)

