"""MCP server exposing the knowledge base as tools for Claude Desktop."""

from mcp.server.fastmcp import FastMCP

from knowledge_bot.ingestion import KnowledgeStore, Document
from knowledge_bot.retrieval_graph import app

store = KnowledgeStore()
mcp = FastMCP("BrainBot")


@mcp.tool()
def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """Search the knowledge base using agentic retrieval.

    Runs hybrid search (BM25 + vector), fuses results with RRF, re-ranks with a
    cross-encoder, then grades relevance with an LLM. If results are not relevant,
    the query is automatically rewritten and the search is retried once.

    Use this to find saved articles, notes, or URLs related to a topic.
    Returns the most relevant chunks with their sources.
    """
    final_state = app.invoke({
        "query": query,
        "original_query": query,
        "results": [],
        "attempts": 0,
        "should_rewrite": False,
    })
    results = final_state["results"]

    if not results:
        return "No results found for that query."

    lines = []
    for i, r in enumerate(results, 1):
        source = r["source"] if r["source"] != "manual" else "manual note"
        lines.append(f"[{i}] {r['title']} — {source}")
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
    doc = Document(text=text, title=title, source=source)
    num_chunks, is_duplicate = store.ingest(doc)

    if is_duplicate:
        return f"Already in knowledge base — skipped. ({num_chunks} chunks exist)"

    return f"Saved '{title}' — {num_chunks} chunks created."


@mcp.tool()
def save_url_to_knowledge_base(url: str) -> str:
    """Scrape a URL and save its content to the knowledge base.

    Use this to save an article or web page by URL instead of pasting raw text.
    """
    from knowledge_bot.web_scraper import scrape_url
    scraped = scrape_url(url)
    if not scraped.is_valid:
        return "Couldn't extract meaningful content from that URL."
    doc = Document(text=scraped.text, title=scraped.title, source=url)
    num_chunks, is_duplicate = store.ingest(doc)
    if is_duplicate:
        return f"Already in knowledge base — skipped. ({num_chunks} chunks exist)"
    return f"Saved '{scraped.title}' — {num_chunks} chunks created."

@mcp.tool()
def save_pdf_to_knowledge_base(file_path: str) -> str:
    """Parse a local PDF file and save its content to the knowledge base.

    Args:
        file_path: Absolute path to the PDF file on disk.
    """
    from knowledge_bot.pdf_parser import parse_pdf
    pdf = parse_pdf(file_path)

    if not pdf.is_valid:
        return "Couldn't extract meaningful text from that PDF."
    doc = Document(text=pdf.text, title=pdf.title, source="pdf")
    num_chunks, is_duplicate = store.ingest(doc)

    if is_duplicate:
        return f"Already in knowledge base — skipped. ({num_chunks} chunks exist)"
    return f"Saved '{pdf.title}' — {num_chunks} chunks created."

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
