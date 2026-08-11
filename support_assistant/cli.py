"""
CLI for the support assistant.

Run with:  python -m support_assistant.cli ingest
           python -m support_assistant.cli chat
"""
from __future__ import annotations

import argparse
import logging
import sys

from support_assistant.rag.pipeline import RagPipeline


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
    )


def run_ingest(pipeline: RagPipeline) -> None:
    count = pipeline.ingest_directory()
    print(f"Ingested {count} chunks.")


def run_chat(pipeline: RagPipeline) -> None:
    print("Support Assistant — type 'exit' to quit, 'reset' to clear history.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() == "exit":
            break
        if question.lower() == "reset":
            pipeline.reset_history()
            print("(history cleared)\n")
            continue

        result = pipeline.ask(question)
        print(f"\nAssistant: {result.answer}\n")
        if result.sources:
            print("Sources:")
            for s in result.sources:
                print(f"  - {s.source_file}, p.{s.page_number} (distance={s.distance:.3f})")
        print()


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Support Assistant CLI")
    parser.add_argument("command", choices=["ingest", "chat"])
    args = parser.parse_args()

    pipeline = RagPipeline()

    if args.command == "ingest":
        run_ingest(pipeline)
    elif args.command == "chat":
        run_chat(pipeline)


if __name__ == "__main__":
    main()
