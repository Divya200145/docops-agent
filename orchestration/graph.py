"""
LangGraph orchestration for the DocOps agent.

The graph has one entry point (classify) that routes to one of two
task-specific paths (answer_question or summarize), each of which
calls into the MCP document tools before generating a response with
an LLM.

    classify --route--> retrieve --> answer_question --> END
             \\--route--> summarize --> END

Running `run()` end-to-end requires ANTHROPIC_API_KEY to be set, since
both the router and the generation nodes call the model. The retrieval
and tool-calling logic itself (search_documents, read_document) has no
LLM dependency and is covered directly by tests/test_tools.py.
"""
from __future__ import annotations

import os
from typing import Literal, TypedDict

from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, StateGraph

from mcp_server import doc_tools

DEFAULT_FOLDER = "data/sample_docs"
MODEL_NAME = os.environ.get("DOCOPS_MODEL", "claude-sonnet-4-6")


class AgentState(TypedDict, total=False):
    query: str
    folder: str
    task: Literal["qa", "summarize"]
    target_doc: str | None
    context_chunks: list[dict]
    answer: str


def _get_llm() -> ChatAnthropic:
    return ChatAnthropic(model=MODEL_NAME, temperature=0)


def classify_node(state: AgentState) -> AgentState:
    """Decide whether this is a question-answering or summarization request."""
    query = state["query"]
    llm = _get_llm()
    prompt = (
        "Classify the user's request as exactly one word: 'qa' or 'summarize'.\n"
        "Use 'summarize' only if the user is explicitly asking for a summary of a "
        "specific document. Otherwise use 'qa'.\n\n"
        f"Request: {query}\n\nAnswer with one word:"
    )
    result = llm.invoke(prompt).content.strip().lower()
    task = "summarize" if "summarize" in result else "qa"
    return {"task": task}


def retrieve_node(state: AgentState) -> AgentState:
    """Search the document folder for chunks relevant to the query."""
    chunks = doc_tools.search_documents(
        state["query"], state.get("folder", DEFAULT_FOLDER), top_k=4
    )
    return {"context_chunks": chunks}


def answer_node(state: AgentState) -> AgentState:
    """Generate an answer grounded in the retrieved chunks."""
    context = "\n\n".join(
        f"[{c['doc_path']} #{c['chunk_id']}]\n{c['text']}" for c in state.get("context_chunks", [])
    )
    llm = _get_llm()
    prompt = (
        "Answer the question using ONLY the context below. If the context "
        "doesn't contain the answer, say so explicitly instead of guessing.\n\n"
        f"Context:\n{context}\n\nQuestion: {state['query']}\n\nAnswer:"
    )
    answer = llm.invoke(prompt).content
    return {"answer": answer}


def summarize_node(state: AgentState) -> AgentState:
    """Summarize a specific document."""
    folder = state.get("folder", DEFAULT_FOLDER)
    target = state.get("target_doc")
    if not target:
        # fall back to the top search hit as the doc to summarize
        hits = doc_tools.search_documents(state["query"], folder, top_k=1)
        if not hits:
            return {"answer": "No matching document found to summarize."}
        target = hits[0]["doc_path"]

    full_path = os.path.join(folder, target)
    text = doc_tools.read_document(full_path)
    llm = _get_llm()
    prompt = f"Summarize the following document in 3-5 sentences:\n\n{text}"
    summary = llm.invoke(prompt).content
    return {"answer": summary, "target_doc": target}


def _route(state: AgentState) -> str:
    return "summarize" if state.get("task") == "summarize" else "retrieve"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("classify", classify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_node("summarize", summarize_node)

    graph.set_entry_point("classify")
    graph.add_conditional_edges("classify", _route, {"retrieve": "retrieve", "summarize": "summarize"})
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    graph.add_edge("summarize", END)
    return graph.compile()


def run(query: str, folder: str = DEFAULT_FOLDER, target_doc: str | None = None) -> AgentState:
    """Run the agent end-to-end. Requires ANTHROPIC_API_KEY to be set."""
    app = build_graph()
    return app.invoke({"query": query, "folder": folder, "target_doc": target_doc})
