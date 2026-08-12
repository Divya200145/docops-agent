# DocOps Agent

A small agent that answers questions and generates summaries over a folder of
documents (.txt, .md, .pdf), built to get hands-on with three specific
pieces of the current agentic AI stack: an MCP server, LangGraph
orchestration, and a RAGAS evaluation harness.

## What it actually does

- **Ask a question** over a folder of documents — the agent searches the
  folder, pulls the most relevant passages, and answers using only that
  retrieved context (it's told explicitly to say so if the context doesn't
  contain the answer, rather than guess).
- **Summarize a document** — point it at a specific file (including PDFs)
  and it returns a short summary.
- **Evaluate itself** — a separate RAGAS harness runs a fixed set of test
  questions through the real pipeline and scores faithfulness and
  retrieval quality, instead of relying on manual spot-checks.

## Architecture

```
User query
    │
    ▼
LangGraph orchestrator  ── classifies the request (question vs. summarize)
    │                       and routes to the right path
    ▼
MCP server               ── exposes list_documents / read_document /
                              search_documents as callable tools
    │
    ▼
RAGAS evaluation harness ── scores retrieval + generation quality
                              (run separately, not on every request)
    │
    ▼
Response
```

## Project structure

```
docops-agent/
├── mcp_server/
│   ├── doc_tools.py     # list/read/chunk/search — no LLM dependency, unit tested
│   └── server.py        # wraps doc_tools as MCP tools over stdio transport
├── orchestration/
│   └── graph.py          # LangGraph: classify -> (retrieve -> answer) | summarize
├── evaluation/
│   ├── eval_dataset.jsonl
│   └── evaluate.py       # RAGAS harness (separate venv — see below)
├── data/sample_docs/     # sample .txt/.md/.pdf files used by tests
├── tests/test_tools.py   # 9 tests, no API key required
├── run_agent.py          # CLI entry point
├── requirements.txt
└── requirements-eval.txt
```

## How the pieces fit together

- **MCP server** (`mcp_server/`): exposes `list_documents`, `read_document`,
  and `search_documents` as MCP tools. Retrieval is TF-IDF over chunked
  documents (`scikit-learn`) — deliberately not an embedding-based vector
  search, so the tool layer works fully offline with no external API calls
  and is cheap to unit test. Swapping in an embedding model is the natural
  next step (see Roadmap).
- **LangGraph orchestration** (`orchestration/graph.py`): a small graph —
  `classify` routes to either `retrieve -> answer` (question answering) or
  `summarize`, both of which call into the same document tools. State is a
  typed dict threaded through each node.
- **RAGAS evaluation harness** (`evaluation/`): runs the real
  retrieve-and-answer pipeline against a fixed set of 5 test questions
  (`eval_dataset.jsonl`) and scores the results on faithfulness, context
  precision, and context recall.

## Setup

```bash
git clone https://github.com/Divya200145/docops-agent.git
cd docops-agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

```bash
# Ask a question over the sample docs
python run_agent.py --query "What does RAGAS's faithfulness metric measure?"

# Summarize a specific document (including PDFs)
python run_agent.py --query "Summarize this" --doc rag_explainer.pdf

# Run the MCP server standalone (stdio transport)
python -m mcp_server.server
```

## Running the tests

```bash
pip install pytest
pytest tests/ -v
```

9 tests covering document listing, PDF text extraction, chunking, TF-IDF
search ranking, and graph compilation — all pass without an API key or
network access.

## Running the evaluation harness

RAGAS pins an older `langchain-core` than the current LangGraph /
`langchain-anthropic` release train. Installing both dependency sets into
one environment produces real, unresolvable version conflicts — I hit this
directly while building this project (`langchain-openai` and `langgraph`
disagreeing on which `langchain-core` they need once `ragas` forces a
downgrade). The fix is to run evaluation in its own virtual environment
rather than fight the resolver:

```bash
python3 -m venv venv-eval
source venv-eval/bin/activate
pip install -r requirements-eval.txt
export ANTHROPIC_API_KEY=sk-ant-...
python evaluation/evaluate.py
```

This writes per-question results to `evaluation/results.json` and prints
aggregate faithfulness / context precision / context recall scores.

**Evaluation results:** not yet published here — pending a run with a
live API key. Once available, real scores go here rather than a
placeholder number.

## Roadmap

- Swap TF-IDF retrieval for an embedding-based vector search
- Add `answer_relevancy` to the RAGAS metric set (requires an embedding
  model — left out for now to keep the eval environment lightweight)
- Resolve the RAGAS / LangGraph dependency conflict into a single
  environment once the ecosystem versions catch up to each other
- Package the MCP server for remote (SSE / streamable-HTTP) deployment,
  not just local stdio

## Status

Personal project, actively developed. Not affiliated with or deployed at
any employer.

## Author

Bala Divya · [LinkedIn](https://linkedin.com/in/baladivya-aiml/) · [GitHub](https://github.com/Divya200145)
