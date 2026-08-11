"""
Prompt construction. Kept separate from `pipeline.py` so prompt
wording can be iterated on independently of retrieval/generation
plumbing — a common need once an assistant is in front of real users.
"""
from __future__ import annotations

from support_assistant.vectorstore.store import RetrievedChunk

SYSTEM_INSTRUCTIONS = (
    "You are a support assistant that answers ONLY using the provided context. "
    "If the context does not contain the answer, say you don't have enough "
    "information rather than guessing. Cite the source file and page number "
    "for any claim you make, using the format [source_file, p.N]. "
    "Keep answers concise and directly address the question."
)


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks into a labeled context block for the prompt."""
    blocks = []
    for chunk in chunks:
        blocks.append(
            f"[{chunk.source_file}, p.{chunk.page_number}]\n{chunk.text}"
        )
    return "\n\n".join(blocks)


def format_history(history: list[tuple[str, str]]) -> str:
    """Render prior (question, answer) turns for conversational continuity."""
    if not history:
        return ""
    lines = []
    for question, answer in history:
        lines.append(f"Previous Q: {question}\nPrevious A: {answer}")
    return "\n\n".join(lines) + "\n\n"


def build_rag_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    history: list[tuple[str, str]] | None = None,
) -> str:
    """Assemble the full grounded prompt sent to the LLM."""
    context = format_context(chunks)
    history_block = format_history(history or [])

    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"{history_block}"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
