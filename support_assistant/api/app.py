"""
FastAPI endpoint for the support assistant.

Run with:  uvicorn support_assistant.api.app:app --reload
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from pydantic import BaseModel

from support_assistant.rag.pipeline import RagPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

app = FastAPI(
    title="Support Assistant API",
    description="RAG-powered support assistant with source citations.",
    version="1.0.0",
)

_pipeline: RagPipeline | None = None


def get_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline()
    return _pipeline


class IngestResponse(BaseModel):
    chunks_ingested: int


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None


class SourceModel(BaseModel):
    source_file: str
    page_number: int
    distance: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceModel]


@app.post("/ingest", response_model=IngestResponse)
def ingest() -> IngestResponse:
    """(Re)ingest every PDF in the configured documents directory."""
    pipeline = get_pipeline()
    count = pipeline.ingest_directory()
    return IngestResponse(chunks_ingested=count)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Ask a question, grounded in the ingested documents."""
    pipeline = get_pipeline()
    result = pipeline.ask(request.question, top_k=request.top_k)
    return AskResponse(
        answer=result.answer,
        sources=[
            SourceModel(
                source_file=s.source_file, page_number=s.page_number, distance=s.distance
            )
            for s in result.sources
        ],
    )


@app.post("/reset")
def reset() -> dict:
    """Clear conversation history."""
    get_pipeline().reset_history()
    return {"status": "history cleared"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
