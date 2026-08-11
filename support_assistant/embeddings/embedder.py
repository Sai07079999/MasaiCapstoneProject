"""
Embedding backend abstraction, so the vector store and retriever never
know or care which model produced the vectors.

Two implementations:
  - `SentenceTransformerEmbedder` — the production backend (dense,
    semantic embeddings). Requires downloading model weights from
    HuggingFace on first use.
  - `TfidfEmbedder` — a dependency-free fallback with no network
    requirement, used automatically when `settings.embedding_backend`
    is "tfidf" or when the sentence-transformers model can't be
    downloaded (e.g. a network-restricted environment). It's weaker
    semantically (bag-of-words, not contextual) but keeps the whole
    RAG pipeline runnable end-to-end offline.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from support_assistant.config import settings

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Interface every embedding backend implements."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n_texts, dim) float32 array of embeddings."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimensionality."""


class SentenceTransformerEmbedder(BaseEmbedder):
    """Production embedder using a local Sentence-Transformers model."""

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model_name = model_name or settings.sentence_transformer_model
        logger.info("Loading Sentence-Transformers model: %s", self._model_name)
        self._model = SentenceTransformer(self._model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()


class TfidfEmbedder(BaseEmbedder):
    """
    Dependency-free fallback embedder (TF-IDF, dimensionality-reduced
    with TruncatedSVD to a fixed-size dense vector so it's a drop-in
    replacement for the vector store's fixed-dimension expectation).
    """

    def __init__(self, dim: int = 128) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._dim = dim
        self._vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self._svd = TruncatedSVD(n_components=dim, random_state=42)
        self._fitted = False

    def fit(self, corpus: list[str]) -> None:
        """Fit the TF-IDF vocabulary + SVD projection on the full corpus once."""
        tfidf_matrix = self._vectorizer.fit_transform(corpus)
        n_components = min(self._dim, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
        if n_components < self._dim:
            logger.warning(
                "Corpus too small for dim=%d; reducing TruncatedSVD to %d components.",
                self._dim, n_components,
            )
            self._svd.n_components = max(n_components, 2)
        self._svd.fit(tfidf_matrix)
        self._fitted = True

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder must be fit() on a corpus before embed().")
        tfidf_matrix = self._vectorizer.transform(texts)
        vectors = self._svd.transform(tfidf_matrix)
        return vectors.astype(np.float32)

    @property
    def dimension(self) -> int:
        return self._svd.n_components


def build_embedder() -> BaseEmbedder:
    """Factory: instantiate the embedder configured in `settings`."""
    backend = settings.embedding_backend
    if backend == "sentence-transformers":
        try:
            return SentenceTransformerEmbedder()
        except Exception:
            logger.exception(
                "Failed to load Sentence-Transformers model (likely no network "
                "access to download weights); falling back to TfidfEmbedder."
            )
            return TfidfEmbedder()
    if backend == "tfidf":
        return TfidfEmbedder()
    raise ValueError(f"Unknown embedding_backend: {backend!r}")
