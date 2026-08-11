"""
Abstract LLM interface. Every provider (OpenAI, Gemini, Ollama, or the
offline extractive fallback) implements `generate(prompt) -> str`, so
`rag/pipeline.py` never imports a specific SDK — switching providers
is a one-line config change (`ASSISTANT_LLM_PROVIDER=...`).
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Interface every LLM provider implements."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a completion for `prompt`."""
