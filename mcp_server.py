"""MCP server exposing the knowledge base as tools for Claude Desktop."""

from mcp.server.fastmcp import FastMCP

from ingestion import KnowledgeStore, Document

store = KnowledgeStore()
mcp = FastMCP("BrainBot")


@mcp.tool()
def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """Search the knowledge base by semantic similarity.

    Use this to find saved articles, notes, or URLs related to a topic.
    Returns the most relevant chunks with their sources.
    """
    results = store.search(query, top_k=top_k)

    if not results:
        return "No results found for that query."

    lines = []
    for i, r in enumerate(results, 1):
        score_pct = f"{r['score']:.0%}"
        source = r["source"] if r["source"] != "manual" else "manual note"
        lines.append(f"[{i}] {r['title']} ({score_pct} match) — {source}")
        lines.append(r["text"])
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
def save_to_knowledge_base(text: str, title: str, source: str = "manual") -> str:
    """Save a text snippet to the knowledge base.

    Args:
        text: The content to save.
        title: A short descriptive title.
        source: Optional source URL or identifier.
    """
    doc = Document(text=text, title=title, source=source, added_by="mcp")
    num_chunks, is_duplicate = store.ingest(doc)

    if is_duplicate:
        return f"Already in knowledge base — skipped. ({num_chunks} chunks exist)"

    return f"Saved '{title}' — {num_chunks} chunks created."


@mcp.tool()
def list_documents(limit: int = 10) -> str:
    """List recently saved documents in the knowledge base."""
    docs = store.list_documents(limit=limit)

    if not docs:
        return "Knowledge base is empty."

    lines = [f"{len(docs)} document(s) found:\n"]
    for doc in docs:
        source = doc["source"] if doc["source"] != "manual" else "manual note"
        lines.append(f"• {doc['title']}")
        lines.append(f"  Source: {source} | Chunks: {doc['total_chunks']} | ID: {doc['doc_id']}")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
