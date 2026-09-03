from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """
    Application configuration.
    """
    
    groq_api_key: str
    groq_model: str

    @classmethod
    def from_env(cls) -> Settings:
        """
        Load configuration from environment variables.
        """
        groq_api_key = os.getenv("GROQ_API_KEY")

        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")

        return cls(
            groq_api_key=groq_api_key,
            groq_model="qwen/qwen3.8-27b"
        )

settings = Settings.from_env()
