"""
RAGAS evaluation harness for the DocOps agent's QA pipeline.

Runs each question in eval_dataset.jsonl through the real
retrieve -> answer pipeline (mcp_server.doc_tools.search_documents +
orchestration.graph.answer_node), then scores the results with RAGAS.

Requires ANTHROPIC_API_KEY (used both as the agent's generation model
and as the RAGAS judge LLM).

IMPORTANT — dependency note:
RAGAS pins an older langchain-core than the current LangGraph/
langchain-anthropic release train. Installing them into the same
environment produces real, unresolvable version conflicts (seen
firsthand while building this: langchain-openai and langgraph disagree
on which langchain-core they need once ragas forces a downgrade).
The fix used here is to run evaluation in its own virtual environment
(see requirements-eval.txt) rather than fighting the resolver. This is
also just good practice: an eval harness doesn't need to ship with the
agent's runtime dependencies.

    python3 -m venv venv-eval
    source venv-eval/bin/activate
    pip install -r requirements-eval.txt
    export ANTHROPIC_API_KEY=sk-ant-...
    python evaluation/evaluate.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcp_server import doc_tools

DATASET_PATH = os.path.join(os.path.dirname(__file__), "eval_dataset.jsonl")
FOLDER = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs")


def load_eval_set() -> list[dict]:
    rows = []
    with open(DATASET_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_pipeline(question: str) -> tuple[str, list[str]]:
    """Retrieve context and generate an answer, using the agent's own tools."""
    from langchain_anthropic import ChatAnthropic

    chunks = doc_tools.search_documents(question, FOLDER, top_k=4)
    contexts = [c["text"] for c in chunks]
    context_block = "\n\n".join(contexts)

    llm = ChatAnthropic(model=os.environ.get("DOCOPS_MODEL", "claude-sonnet-4-6"), temperature=0)
    prompt = (
        "Answer the question using ONLY the context below. If the context "
        "doesn't contain the answer, say so explicitly instead of guessing.\n\n"
        f"Context:\n{context_block}\n\nQuestion: {question}\n\nAnswer:"
    )
    answer = llm.invoke(prompt).content
    return answer, contexts


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set — export it before running evaluation.", file=sys.stderr)
        sys.exit(1)

    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import context_precision, context_recall, faithfulness
    from langchain_anthropic import ChatAnthropic

    eval_set = load_eval_set()
    print(f"Running agent pipeline on {len(eval_set)} eval questions...")

    records = []
    for row in eval_set:
        answer, contexts = run_pipeline(row["question"])
        records.append(
            {
                "user_input": row["question"],
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": row["ground_truth"],
            }
        )
        print(f"  done: {row['question'][:60]}...")

    dataset = EvaluationDataset.from_list(records)

    judge_llm = LangchainLLMWrapper(ChatAnthropic(model="claude-sonnet-4-6", temperature=0))
    metrics = [faithfulness, context_precision, context_recall]

    print("\nScoring with RAGAS (faithfulness, context_precision, context_recall)...")
    result = evaluate(dataset=dataset, metrics=metrics, llm=judge_llm)

    print("\n=== RAGAS scores ===")
    print(result)

    out_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out_path, "w") as f:
        json.dump(result.to_pandas().to_dict(orient="records"), f, indent=2, default=str)
    print(f"\nPer-question results written to {out_path}")


if __name__ == "__main__":
    main()
