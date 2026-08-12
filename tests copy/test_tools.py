"""
Unit tests for the document tools and graph structure.

These deliberately avoid calling any LLM, so they run without
ANTHROPIC_API_KEY and without network access — fast to run in CI.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcp_server import doc_tools
from orchestration.graph import build_graph

FOLDER = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs")


def test_list_documents_finds_all_sample_files():
    docs = doc_tools.list_documents(FOLDER)
    assert "mcp_overview.txt" in docs
    assert "langgraph_overview.txt" in docs
    assert "ragas_overview.txt" in docs
    assert "rag_explainer.pdf" in docs
    assert len(docs) == 4


def test_read_document_txt():
    text = doc_tools.read_document(os.path.join(FOLDER, "mcp_overview.txt"))
    assert "Model Context Protocol" in text


def test_read_document_pdf_extracts_real_text():
    text = doc_tools.read_document(os.path.join(FOLDER, "rag_explainer.pdf"))
    assert "Retrieval-Augmented Generation" in text
    assert "RAGAS" in text


def test_read_document_missing_file_raises():
    try:
        doc_tools.read_document(os.path.join(FOLDER, "does_not_exist.txt"))
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_chunk_text_respects_overlap():
    text = "a" * 2000
    chunks = doc_tools.chunk_text(text, chunk_size=800, overlap=100)
    assert len(chunks) >= 2
    # consecutive chunks should overlap
    assert chunks[0][-100:] == chunks[1][:100]


def test_search_documents_ranks_relevant_doc_first():
    results = doc_tools.search_documents(
        "what metrics does RAGAS compute for evaluating a RAG pipeline", FOLDER, top_k=3
    )
    assert len(results) > 0
    assert results[0]["doc_path"] == "ragas_overview.txt"


def test_search_documents_finds_pdf_content():
    results = doc_tools.search_documents(
        "why does RAG reduce hallucination", FOLDER, top_k=3
    )
    doc_paths = [r["doc_path"] for r in results]
    assert "rag_explainer.pdf" in doc_paths


def test_search_documents_empty_folder_returns_empty(tmp_path):
    results = doc_tools.search_documents("anything", str(tmp_path), top_k=3)
    assert results == []


def test_graph_compiles_and_has_expected_nodes():
    app = build_graph()
    nodes = set(app.get_graph().nodes.keys())
    assert {"classify", "retrieve", "answer", "summarize"}.issubset(nodes)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
