"""
CLI entry point for the DocOps agent.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python run_agent.py --query "What does RAGAS measure?"
    python run_agent.py --query "Summarize the RAG explainer" --doc rag_explainer.pdf
"""
from __future__ import annotations

import argparse
import os
import sys

from orchestration.graph import DEFAULT_FOLDER, run


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question or request a summary over a folder of documents.")
    parser.add_argument("--query", required=True, help="Your question, or a summarization request.")
    parser.add_argument("--folder", default=DEFAULT_FOLDER, help="Folder of documents to search.")
    parser.add_argument("--doc", default=None, help="Specific document to summarize (optional).")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ANTHROPIC_API_KEY is not set. Export it before running the agent, e.g.\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...\n",
            file=sys.stderr,
        )
        sys.exit(1)

    result = run(args.query, folder=args.folder, target_doc=args.doc)
    print(f"\nTask: {result.get('task', 'summarize' if args.doc else 'qa')}")
    print(f"Answer:\n{result['answer']}")


if __name__ == "__main__":
    main()
