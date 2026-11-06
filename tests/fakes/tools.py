from __future__ import annotations

from typing import Any

from agentforge.tools.base import Tool


class FakeCalculator(Tool):
    """
    Deterministic calculator for tests.

    Accepts 2 + 2 and always returns 4.
    """

    name = "calculator"
    description = "Performs arithmetic"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                },
            },
            "required": ["expression"],
        }

    async def invoke(self, **arguments: Any) -> Any:
        self.calls.append(arguments)

        if arguments != {
            "expression": "2 + 2",
        }:
            raise AssertionError(
                f"Unexpected calculator arguments: {arguments}"
            )

        return 4
