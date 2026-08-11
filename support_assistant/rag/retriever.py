"""
Retriever: the single call site for turning a question into ranked
context chunks. Keeps `pipeline.py` from having to know how a query
gets embedded or searched.
"""
from __future__ import annotations

from support_assistant.embeddings.embedder import BaseEmbedder
from support_assistant.vectorstore.store import RetrievedChunk, VectorStore


class Retriever:
    def __init__(self, embedder: BaseEmbedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        query_embedding = self._embedder.embed([question])[0]
        return self._store.query(query_embedding, top_k=top_k)
