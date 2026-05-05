# BrainBot — Claude Code Handoff

## What this is

A Discord knowledge base bot using RAG (Retrieval-Augmented Generation). You save URLs, articles, and notes via Discord slash commands, and query them in natural language. It's a personal second brain that lives in your server.

## Current state

The scaffold is complete and functional. All files are in the `knowledge-bot/` directory:

- `bot.py` — Main entry point, Discord slash commands (`/save`, `/ask`, `/search`, `/list`, `/delete`, `/stats`)
- `ingestion.py` — `KnowledgeStore` class: chunking, embedding with sentence-transformers, storage in ChromaDB
- `retriever.py` — `Retriever` class: vector search + Claude API for RAG answer generation (degrades gracefully without API key)
- `web_scraper.py` — URL content extraction using trafilatura
- `config.py` — Env var loading via python-dotenv
- `.env.example` — Template for `DISCORD_TOKEN`, `ANTHROPIC_API_KEY`, and tuning params
- `requirements.txt` — discord.py, chromadb, sentence-transformers, anthropic, trafilatura, python-dotenv

Stack: Python, discord.py (slash commands via `app_commands`), ChromaDB (local persistent vector DB), sentence-transformers (`all-MiniLM-L6-v2`, CPU), Anthropic Claude API (optional).

## What needs to be built / improved

These are the real engineering challenges — roughly ordered by interest:

### 1. Semantic chunking (replace the naive splitter)
`ingestion.py` → `chunk_text()` currently splits on character count with paragraph-boundary heuristics. Replace with semantic chunking: embed sliding windows and split where cosine similarity between consecutive windows drops (= topic shift). This produces much better retrieval quality.

### 2. Hybrid search (BM25 + vector)
`retriever.py` currently does pure vector similarity. Add BM25 keyword search alongside it and combine scores (reciprocal rank fusion works well). This catches cases where exact keyword matches matter but the embedding misses them.

### 3. Re-ranking
After retrieving top-20 candidates, re-rank with a cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`). Cross-encoders are much more accurate than bi-encoders for scoring query-document pairs but too slow for first-pass retrieval. Classic two-stage pattern.

### 4. Deduplication
Save the same URL twice → double chunks, polluted results. Add near-duplicate detection using either MinHash/simhash on chunk text or a simple content hash check before upserting. `ingestion.py` → `ingest()` is the place.

### 5. Conversational memory for /ask
Currently each `/ask` is stateless. Add a short conversation buffer (last 3-5 Q&A pairs per user) so follow-up questions work: "what else did that paper say?" should know which paper. Store in a simple dict keyed by user ID, or use ChromaDB's metadata filtering.

### 6. PDF ingestion
`web_scraper.py` only handles URLs. Add a `/save_file` command that accepts Discord attachments (PDFs). Use PyMuPDF or pdfplumber to extract text, then feed into the same `KnowledgeStore.ingest()` pipeline.

### 7. Auto-ingest from RSS feeds
Add a background task (`discord.ext.tasks.loop`) that periodically checks a list of RSS feeds (stored in a JSON config or a ChromaDB collection) and auto-ingests new posts. Good for staying on top of arXiv, blogs, newsletters.

### 8. Per-user namespaces
Currently one shared knowledge base. Use Discord user IDs as ChromaDB collection namespaces so each person gets their own. Straightforward change in `KnowledgeStore.__init__()`.

### 9. Tests
No tests yet. Add pytest tests for:
- Chunking (edge cases: empty text, very short text, text shorter than chunk size)
- Ingestion round-trip (ingest → search → find it)
- Deduplication
- Retriever with/without LLM key

## Architecture reference

```
User in Discord
  │
  ├── /save <url>  ──→  scrape URL  ──→  chunk text  ──→  embed  ──→  ChromaDB
  ├── /save <text> ──→                    chunk text  ──→  embed  ──→  ChromaDB
  │
  ├── /ask <question> ──→  embed query  ──→  vector search  ──→  top-k chunks
  │                                                                 │
  │                                          ┌──────────────────────┘
  │                                          ▼
  │                                    Claude API (context + question)
  │                                          │
  │                                          ▼
  │                                    Answer with sources ──→ Discord embed
  │
  ├── /search <keywords> ──→  embed query  ──→  vector search  ──→  results
  ├── /list  ──→  ChromaDB metadata query  ──→  recent docs
  └── /delete <id>  ──→  ChromaDB delete by doc_id
```

## How to run

```bash
cd knowledge-bot
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DISCORD_TOKEN (required) + ANTHROPIC_API_KEY (optional)
python bot.py
```

## Notes

- The embedding model downloads automatically on first run (~80MB)
- ChromaDB persists to `./chroma_data/` — survives restarts
- Without an Anthropic API key, `/ask` still works but returns raw search results instead of generated answers
- Slash commands may take a few minutes to sync with Discord on first run
