from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from apps.chatbot.api.dependencies import get_app
from apps.chatbot.api.main import create_api


class FakeChatbotApp:
    def __init__(self, response: str = "Fake response") -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def run(
        self,
        session_id: str,
        message: str,
        **kwargs: Any,
    ) -> str:
        self.calls.append(
            {
                "session_id": session_id,
                "message": message,
                "kwargs": kwargs,
            }
        )

        return self.response


@pytest.fixture
def fake_app() -> FakeChatbotApp:
    return FakeChatbotApp()


@pytest.fixture
def client(
    fake_app: FakeChatbotApp
) -> Generator[TestClient, None, None]:
    app = create_api()

    app.dependency_overrides[get_app] = lambda: fake_app

    client = TestClient(app)

    yield client

    app.dependency_overrides.clear()


def test_chat_returns_response(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/chat",
        json={
            "session_id": "session-123",
            "message": "Hello",
        },
    )

    assert response.status_code == 200

    assert response.json() == {
        "session_id": "session-123",
        "response": "Fake response",
    }


def test_chat_passes_session_id_to_app(
    client: TestClient,
    fake_app: FakeChatbotApp,
) -> None:
    client.post(
        "/api/chat",
        json={
            "session_id": "session-123",
            "message": "Hello",
        },
    )

    assert fake_app.calls == [
        {
            "session_id": "session-123",
            "message": "Hello",
            "kwargs": {},
        }
    ]


def test_chat_passes_message_to_app(
    client: TestClient,
    fake_app: FakeChatbotApp,
) -> None:
    client.post(
        "/api/chat",
        json={
            "session_id": "session-123",
            "message": "What is AgentForge?",
        },
    )

    assert fake_app.calls[0]["message"] == "What is AgentForge?"


def test_chat_rejects_empty_session_id(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/chat",
        json={
            "session_id": "",
            "message": "Hello",
        },
    )

    assert response.status_code == 422


def test_chat_rejects_empty_message(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/chat",
        json={
            "session_id": "session-123",
            "message": "",
        },
    )

    assert response.status_code == 422


def test_chat_rejects_missing_session_id(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/chat",
        json={
            "message": "Hello",
        },
    )

    assert response.status_code == 422


def test_chat_rejects_missing_message(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/chat",
        json={
            "session_id": "session-123",
        },
    )

    assert response.status_code == 422
