"""
Top-level RAG orchestration: ingestion (documents -> chunks ->
embeddings -> vector store) and querying (question -> retrieval ->
prompt -> LLM -> grounded, cited answer). This is the module both the
CLI and the FastAPI app call into — neither talks to embeddings,
the vector store, or the LLM directly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from support_assistant.config import settings
from support_assistant.embeddings.embedder import BaseEmbedder, build_embedder
from support_assistant.ingestion.chunker import chunk_documents
from support_assistant.ingestion.loader import load_pdf_directory
from support_assistant.llm.base_llm import BaseLLM
from support_assistant.llm.providers import build_llm
from support_assistant.rag.prompt_templates import build_rag_prompt
from support_assistant.rag.retriever import Retriever
from support_assistant.vectorstore.store import RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RagAnswer:
    """A generated answer plus the sources it was grounded in."""

    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)


class RagPipeline:
    """Owns conversation state and coordinates ingest/query end-to-end."""

    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        store: VectorStore | None = None,
        llm: BaseLLM | None = None,
    ) -> None:
        self._embedder = embedder or build_embedder()
        self._store = store or VectorStore()
        self._llm = llm or build_llm()
        self._retriever = Retriever(self._embedder, self._store)
        self._history: list[tuple[str, str]] = []

    def ingest_directory(self, directory: Path | None = None) -> int:
        """Load, chunk, embed, and store every PDF in `directory`. Returns chunk count."""
        directory = directory or settings.documents_dir
        pages = load_pdf_directory(directory)
        chunks = chunk_documents(pages)

        if not chunks:
            logger.warning("No chunks produced from %s; nothing ingested.", directory)
            return 0

        texts = [c.text for c in chunks]
        if hasattr(self._embedder, "fit"):
            self._embedder.fit(texts)  # TfidfEmbedder must see the full corpus first

        embeddings = self._embedder.embed(texts)
        self._store.add(chunks, embeddings)
        return len(chunks)

    def ask(self, question: str, top_k: int | None = None) -> RagAnswer:
        """Answer one question, grounded in retrieved context, updating history."""
        chunks = self._retriever.retrieve(question, top_k=top_k)
        prompt = build_rag_prompt(question, chunks, history=self._history)
        answer_text = self._llm.generate(prompt)

        self._history.append((question, answer_text))
        self._history = self._history[-settings.max_history_turns:]

        return RagAnswer(answer=answer_text, sources=chunks)

    def reset_history(self) -> None:
        self._history = []

    @property
    def history(self) -> list[tuple[str, str]]:
        return list(self._history)
