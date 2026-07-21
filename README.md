# knowledge-bot

A personal RAG knowledge base exposed as MCP tools for Claude Code. Save articles, notes, and URLs — then ask Claude Code questions grounded in your saved content.

## How It Works

**Saving:** content is scraped (if a URL) or taken as-is, split into overlapping chunks, embedded with a local sentence-transformers model, and stored in ChromaDB.

**Querying:** Claude Code calls `search_knowledge_base`, retrieves the most semantically similar chunks, and uses them as context to answer your question. Claude does the reasoning — the bot handles the retrieval.

## Setup

```bash
git clone <repo>
cd knowledge-bot
uv sync
```

Copy the example env file (defaults work out of the box):

```bash
cp .env.example .env
```

Register the MCP server with Claude Code:

```bash
claude mcp add brainbot /path/to/knowledge-bot/.venv/bin/python /path/to/knowledge-bot/interfaces/mcp_server.py
```

Restart Claude Code — the tools will be available in every session.

## MCP Tools

| Tool | Description |
|---|---|
| `search_knowledge_base(query, top_k)` | Semantic search over saved content |
| `save_to_knowledge_base(text, title, source)` | Save a text snippet |
| `save_url_to_knowledge_base(url)` | Scrape and save a URL |
| `list_documents(limit)` | List recently saved documents |

## Stack

- **ChromaDB** — vector store (persisted locally)
- **sentence-transformers** — multilingual embeddings, runs on CPU
- **trafilatura** — web scraping
- **FastMCP** — MCP server

## Tests

```bash
uv run pytest tests/ -v
```
