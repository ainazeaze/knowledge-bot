# knowledge-bot

A personal RAG knowledge base with two interfaces: MCP tools for Claude Code, and a REST API with a web frontend. Save articles, notes, PDFs, and URLs — then search them semantically with LLM-generated answers.

## How It Works

**Saving:** content is scraped (URL), parsed (PDF), or taken as-is, split into overlapping chunks, embedded with a local sentence-transformers model, and stored in ChromaDB.

**Searching:** hybrid search combines vector similarity and BM25 keyword matching, fused with Reciprocal Rank Fusion. Results are re-ranked by a cross-encoder. An LLM then grades relevance — if results are poor, the query is automatically rewritten and the search retried. Finally, Groq generates a concise answer from the top chunks.

## Interfaces

### MCP (Claude Code)

Register the MCP server once:

```bash
claude mcp add brainbot /path/to/knowledge-bot/.venv/bin/python /path/to/knowledge-bot/interfaces/mcp_server.py
```

Restart Claude Code — the tools are available in every session. Ask naturally: "What do I know about RAG?" or "Save this URL to my knowledge base."

| Tool | Description |
|---|---|
| `search_knowledge_base(query, top_k)` | Agentic hybrid search with query rewriting |
| `save_to_knowledge_base(text, title, source)` | Save a text snippet |
| `save_url_to_knowledge_base(url)` | Scrape and save a URL |
| `save_pdf_to_knowledge_base(file_path)` | Parse and save a local PDF |
| `list_documents(limit)` | List recently saved documents |

### REST API

```bash
uv run uvicorn api.main:app --reload
```

Docs available at `http://localhost:8000/docs`.

| Endpoint | Description |
|---|---|
| `POST /ingest/text` | Ingest raw text (background task, returns job ID) |
| `POST /ingest/url` | Scrape and ingest a URL (background task) |
| `POST /ingest/pdf` | Upload and ingest a PDF (background task) |
| `GET /search?q=...` | Hybrid search with LLM-generated answer |
| `GET /documents` | List saved documents |
| `DELETE /documents/{doc_id}` | Delete a document |
| `GET /jobs/{job_id}` | Poll ingestion job status |

## Setup

```bash
git clone <repo>
cd knowledge-bot
uv sync
cp .env.example .env
```

Add your Groq API key to `.env` — get one free at [console.groq.com](https://console.groq.com).

> **Why Groq?** It offers a generous free tier (14,400 requests/day) which is more than enough for a personal knowledge base. The LLM integration uses a standard OpenAI-compatible interface, so swapping to another provider (OpenAI, Anthropic, Ollama) only requires changing `GROQ_API_KEY` and `GROQ_MODEL` in `.env`.

## Search Pipeline

```
Query
  ├─ Vector search (ChromaDB bi-encoder)  ─┐
  └─ BM25 keyword search                  ─┤ RRF fusion → cross-encoder re-rank
                                            │
                                     LLM grades relevance
                                            │
                              (if poor) rewrite query → retry once
                                            │
                                     LLM generates answer
```

## Stack

- **ChromaDB** — local vector store
- **sentence-transformers** — multilingual bi-encoder embeddings, runs on CPU
- **cross-encoder/ms-marco-MiniLM-L-6-v2** — re-ranking model, runs on CPU
- **rank-bm25** — keyword search index
- **LangGraph + langchain-groq** — agentic retrieval graph
- **FastAPI + uvicorn** — REST API
- **trafilatura** — web scraping
- **pymupdf** — PDF parsing
- **FastMCP** — MCP server

## Tests

```bash
uv run pytest tests/ -v
```
