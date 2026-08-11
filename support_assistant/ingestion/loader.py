"""
Loads PDF documents into page-level text records, preserving the
source filename and page number so downstream chunks can be cited
back to an exact page — this is what makes source citations possible
later in the pipeline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass
class PageRecord:
    """One page of extracted text with enough metadata to cite it."""

    text: str
    source_file: str
    page_number: int  # 1-indexed


def load_pdf(path: Path) -> list[PageRecord]:
    """Extract text from every page of a single PDF."""
    reader = PdfReader(str(path))
    records: list[PageRecord] = []

    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            records.append(PageRecord(text=text, source_file=path.name, page_number=i))
        else:
            logger.debug("Page %d of %s produced no extractable text.", i, path.name)

    logger.info("Loaded %d pages with text from %s", len(records), path.name)
    return records


def load_pdf_directory(directory: Path) -> list[PageRecord]:
    """Load and concatenate page records from every PDF in `directory`."""
    directory = Path(directory)
    pdf_paths = sorted(directory.glob("*.pdf"))
    if not pdf_paths:
        logger.warning("No PDF files found in %s", directory)

    records: list[PageRecord] = []
    for path in pdf_paths:
        records.extend(load_pdf(path))
    return records
