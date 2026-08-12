"""
Core document operations used by the MCP server tools.

Kept separate from server.py so this logic can be unit tested directly,
without spinning up an MCP transport (stdio/SSE) or needing an LLM API key.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}
CHUNK_SIZE = 800  # characters per chunk
CHUNK_OVERLAP = 100


@dataclass
class Chunk:
    doc_path: str
    chunk_id: int
    text: str


def list_documents(folder: str) -> list[str]:
    """Return relative paths of all supported documents in a folder."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    results = []
    for root, _dirs, files in os.walk(folder):
        for name in sorted(files):
            ext = os.path.splitext(name)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                results.append(os.path.relpath(os.path.join(root, name), folder))
    return results


def _extract_pdf_text(path: str) -> str:
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts)


def read_document(path: str) -> str:
    """Read a document's full text content. Supports .txt, .md, and .pdf."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _extract_pdf_text(path)
    if ext in {".txt", ".md"}:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping fixed-size chunks."""
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def build_corpus(folder: str) -> list[Chunk]:
    """Read every document in a folder and split it into chunks."""
    corpus: list[Chunk] = []
    for rel_path in list_documents(folder):
        full_path = os.path.join(folder, rel_path)
        try:
            text = read_document(full_path)
        except Exception:
            continue
        for i, chunk in enumerate(chunk_text(text)):
            corpus.append(Chunk(doc_path=rel_path, chunk_id=i, text=chunk))
    return corpus


def search_documents(query: str, folder: str, top_k: int = 3) -> list[dict]:
    """
    TF-IDF search over all documents in a folder.

    Deliberately not an embedding-based vector search — TF-IDF needs no
    external API call, so this tool works fully offline and is cheap to
    unit test. Swapping in an embedding model is a natural next step
    (see README "Roadmap").
    """
    corpus = build_corpus(folder)
    if not corpus:
        return []

    texts = [c.text for c in corpus]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts + [query])
    doc_vectors, query_vector = matrix[:-1], matrix[-1]

    scores = cosine_similarity(query_vector, doc_vectors)[0]
    ranked = sorted(zip(corpus, scores), key=lambda pair: pair[1], reverse=True)

    results = []
    for chunk, score in ranked[:top_k]:
        if score <= 0:
            continue
        results.append(
            {
                "doc_path": chunk.doc_path,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "score": round(float(score), 4),
            }
        )
    return results
