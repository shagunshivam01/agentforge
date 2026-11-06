from __future__ import annotations

import pytest

from agentforge.infrastructure.config import Settings

def test_settings_default_to_direct_architecture(
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        monkeypatch.delenv("AGENT_ARCHITECTURE", raising=False)

        config = Settings.from_env()

        assert config.agent_architecture == "direct"


def test_settings_load_react_architecture(
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        monkeypatch.setenv("AGENT_ARCHITECTURE", "react")

        config = Settings.from_env()

        assert config.agent_architecture == "react"


def test_settings_load_direct_architecture(
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        monkeypatch.setenv("AGENT_ARCHITECTURE", "direct")

        config = Settings.from_env()

        assert config.agent_architecture == "direct"
