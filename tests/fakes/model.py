from __future__ import annotations

from typing import Any

from agentforge.models.base import Model
from agentforge.models.schemas import ModelResponse


class FakeModel(Model):
    """
    Deterministic model for tests.
    """

    name = "fake-model"

    def __init__(
        self,
        responses: list[ModelResponse] | str | None = None,
    ) -> None:
        self._responses: list[ModelResponse] = []

        if isinstance(responses, str):
            self._response = responses
        else:
            self._response = None
            self._responses = list(responses or [])

        self.calls: list[list[dict[str, Any]]] = []
        self.kwargs: list[dict[str, Any]] = []

    async def generate(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ModelResponse:
        self.calls.append(messages)
        self.kwargs.append(kwargs)

        if self._responses:
            return self._responses.pop(0)

        if self._response is not None:
            return ModelResponse(
                content=self._response,
            )

        return ModelResponse(
            content="Fake response",
        )
