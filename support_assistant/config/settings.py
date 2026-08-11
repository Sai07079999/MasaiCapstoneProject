"""
Centralized configuration for the RAG assistant. Every tunable —
chunk size, embedding backend, LLM provider, vector store location —
is a field here, overridable via env vars or `.env`.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class AssistantSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ASSISTANT_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Ingestion ---
    documents_dir: Path = Field(
        default=PROJECT_ROOT / "support_assistant" / "data"
    )
    chunk_size: int = Field(default=800, description="Target characters per chunk.")
    chunk_overlap: int = Field(default=120, description="Overlap between consecutive chunks.")

    # --- Embeddings ---
    # "sentence-transformers" (production; needs network for first download)
    # or "tfidf" (dependency-free fallback for offline/sandboxed environments).
    embedding_backend: str = Field(default="sentence-transformers")
    sentence_transformer_model: str = Field(default="all-MiniLM-L6-v2")

    # --- Vector store ---
    vector_store_dir: Path = Field(
        default=PROJECT_ROOT / "support_assistant" / "data" / "chroma_db"
    )
    collection_name: str = Field(default="support_docs")
    top_k: int = Field(default=4, description="Number of chunks retrieved per query.")

    # --- LLM ---
    # "openai" | "gemini" | "ollama" | "extractive" (dependency-free offline fallback)
    llm_provider: str = Field(default="extractive")
    llm_model: str = Field(default="gpt-4o-mini")
    openai_api_key: str = Field(default="")
    gemini_api_key: str = Field(default="")
    ollama_base_url: str = Field(default="http://localhost:11434")

    # --- Conversation ---
    max_history_turns: int = Field(default=5)

    log_level: str = Field(default="INFO")


settings = AssistantSettings()
