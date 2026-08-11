"""
Concrete LLM providers.

`OpenAIProvider`, `GeminiProvider`, and `OllamaProvider` are the
production paths — each needs credentials or a local server this
sandbox doesn't have. `ExtractiveLLM` needs neither: it composes a
grounded answer directly from the retrieved context already embedded
in the prompt, using simple keyword-overlap sentence ranking. It's
intentionally "dumber" than a real LLM, but it can never hallucinate
beyond the retrieved text, and it lets the full RAG pipeline run and
be verified end-to-end without any external API.
"""
from __future__ import annotations

import logging
import re

from support_assistant.config import settings
from support_assistant.llm.base_llm import BaseLLM

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLM):
    """Production provider using the OpenAI Chat Completions API."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        from openai import OpenAI  # lazy import: optional dependency

        self._model = model or settings.llm_model
        self._client = OpenAI(api_key=api_key or settings.openai_api_key)

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""


class GeminiProvider(BaseLLM):
    """Production provider using Google's Gemini API."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        import google.generativeai as genai  # lazy import: optional dependency

        genai.configure(api_key=api_key or settings.gemini_api_key)
        self._model = genai.GenerativeModel(model or "gemini-1.5-flash")

    def generate(self, prompt: str) -> str:
        response = self._model.generate_content(prompt)
        return response.text or ""


class OllamaProvider(BaseLLM):
    """Production provider for a locally-running Ollama server."""

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        import requests  # local import to keep the module import-light

        self._requests = requests
        self._model = model or settings.llm_model
        self._base_url = base_url or settings.ollama_base_url

    def generate(self, prompt: str) -> str:
        response = self._requests.post(
            f"{self._base_url}/api/generate",
            json={"model": self._model, "prompt": prompt, "stream": False},
            timeout=60,
        )
        response.raise_for_status()
        return response.json().get("response", "")


class ExtractiveLLM(BaseLLM):
    """
    Dependency-free offline fallback. Ranks sentences from the
    prompt's context section by keyword overlap with the question and
    stitches the top few into an answer — grounded by construction
    since it only ever emits text taken from the retrieved context.
    """

    _CONTEXT_RE = re.compile(r"Context:\s*(.*?)\s*Question:", re.DOTALL)
    _QUESTION_RE = re.compile(r"Question:\s*(.*?)\s*Answer:", re.DOTALL)
    _STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "to", "of", "and",
        "in", "on", "for", "with", "how", "what", "do", "i", "does",
        "can", "you", "my", "it", "this", "that",
    }

    def generate(self, prompt: str) -> str:
        context_match = self._CONTEXT_RE.search(prompt)
        question_match = self._QUESTION_RE.search(prompt)

        if not context_match or not question_match:
            return "I don't have enough context to answer that."

        context = context_match.group(1)
        question = question_match.group(1)

        question_keywords = self._keywords(question)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if s.strip()]

        scored = sorted(
            sentences,
            key=lambda s: len(self._keywords(s) & question_keywords),
            reverse=True,
        )
        top_sentences = [s for s in scored[:3] if self._keywords(s) & question_keywords]

        if not top_sentences:
            return "I don't have enough information in the provided documents to answer that."

        return " ".join(top_sentences)

    @classmethod
    def _keywords(cls, text: str) -> set[str]:
        words = re.findall(r"[a-zA-Z]+", text.lower())
        return {w for w in words if w not in cls._STOPWORDS and len(w) > 2}


def build_llm() -> BaseLLM:
    """Factory: instantiate the LLM provider configured in `settings`."""
    provider = settings.llm_provider
    if provider == "openai":
        return OpenAIProvider()
    if provider == "gemini":
        return GeminiProvider()
    if provider == "ollama":
        return OllamaProvider()
    if provider == "extractive":
        return ExtractiveLLM()
    raise ValueError(f"Unknown llm_provider: {provider!r}")
