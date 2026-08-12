"""
MCP server exposing document operations as tools.

Run standalone (stdio transport) with:
    python -m mcp_server.server

Or import `mcp_app` and mount it another way (SSE / streamable-http) if
you want to connect it to a hosted client instead of a local one.
"""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from mcp_server import doc_tools

DEFAULT_FOLDER = "data/sample_docs"

mcp_app = MCPServer(
    name="docops-agent",
    description="Tools for listing, reading, searching, and summarizing a folder of documents.",
)


@mcp_app.tool()
def list_documents(folder: str = DEFAULT_FOLDER) -> list[str]:
    """List all supported documents (.txt, .md, .pdf) in a folder."""
    return doc_tools.list_documents(folder)


@mcp_app.tool()
def read_document(path: str) -> str:
    """Read the full text content of a single document."""
    return doc_tools.read_document(path)


@mcp_app.tool()
def search_documents(query: str, folder: str = DEFAULT_FOLDER, top_k: int = 3) -> list[dict]:
    """
    Search a folder of documents for chunks relevant to a query.

    Returns up to top_k chunks with their source document, chunk id,
    text, and a TF-IDF relevance score.
    """
    return doc_tools.search_documents(query, folder, top_k)


if __name__ == "__main__":
    mcp_app.run(transport="stdio")
