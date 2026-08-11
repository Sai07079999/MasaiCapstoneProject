"""
Splits loaded pages into overlapping chunks sized for embedding.

Chunking happens on paragraph boundaries first, falling back to a
character-window split for oversized paragraphs — this keeps chunks
semantically coherent instead of cutting mid-sentence whenever
possible, which matters for retrieval quality.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from support_assistant.config import settings
from support_assistant.ingestion.loader import PageRecord

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A retrievable unit of text with full citation metadata."""

    chunk_id: str
    text: str
    source_file: str
    page_number: int


def _split_long_paragraph(paragraph: str, chunk_size: int, overlap: int) -> list[str]:
    """Character-window split for a paragraph longer than `chunk_size`."""
    if len(paragraph) <= chunk_size:
        return [paragraph]

    windows: list[str] = []
    start = 0
    while start < len(paragraph):
        end = start + chunk_size
        windows.append(paragraph[start:end])
        start = end - overlap
    return windows


def chunk_page(page: PageRecord, chunk_size: int, overlap: int) -> list[str]:
    """Group a page's paragraphs into chunks close to `chunk_size` characters."""
    paragraphs = [p.strip() for p in page.text.split("\n") if p.strip()]

    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        for piece in _split_long_paragraph(paragraph, chunk_size, overlap):
            if len(current) + len(piece) + 1 <= chunk_size:
                current = f"{current}\n{piece}".strip()
            else:
                if current:
                    chunks.append(current)
                current = piece

    if current:
        chunks.append(current)

    return chunks


def chunk_documents(
    pages: list[PageRecord],
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """Chunk every page into `Chunk` records ready for embedding."""
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap

    all_chunks: list[Chunk] = []
    for page in pages:
        for i, text in enumerate(chunk_page(page, chunk_size, overlap)):
            chunk_id = f"{page.source_file}::p{page.page_number}::c{i}"
            all_chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    source_file=page.source_file,
                    page_number=page.page_number,
                )
            )

    logger.info("Produced %d chunks from %d pages.", len(all_chunks), len(pages))
    return all_chunks
