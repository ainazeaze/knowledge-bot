# BrainBot — Architecture & Code Walkthrough

## The big picture

BrainBot is a personal RAG knowledge base exposed as MCP tools to Claude Code. Claude Code handles the reasoning, and the bot handles the retrieval.

```
Claude Code
     │  (MCP stdio)
  interfaces/mcp_server.py   ← tool definitions
     │
  knowledge_bot/ingestion.py ← full RAG pipeline
     │
  knowledge_bot/web_scraper.py ← URL → clean text (for save_url tool)
     │
  knowledge_bot/config.py    ← env vars used by all of the above
     │
  ChromaDB (./chroma_data/)  ← persisted vector store
```

---

## File by file

### `config.py` — env vars, globally imported

Loads `.env` and exposes everything as module-level constants. Every other module imports this.

| Variable | Default | Purpose |
|---|---|---|
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Bi-encoder for chunk/query embeddings |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder for re-ranking |
| `CHROMA_DB_PATH` | `./chroma_data` | Where ChromaDB persists to disk |
| `TOP_K` | `5` | Final number of results returned per search |
| `CHUNK_SIZE` | `2000` | Max chunk size in characters |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |

---

### `web_scraper.py` — URL → clean text

Only used when saving a URL. Calls `trafilatura` to download and strip the page down to article body text. Returns a `ScrapedContent` dataclass with `.text`, `.title`, and `.is_valid` (requires at least 50 chars).

---

### `ingestion.py` — the full RAG pipeline

The core of the project. `KnowledgeStore` owns ingestion and a three-stage search pipeline.

#### Ingestion

1. **Chunking** (`chunk_text`): splits text into overlapping character-based chunks, preferring paragraph boundaries. Overlap means sentences near a boundary appear in both adjacent chunks so retrieval doesn't miss them.

2. **Deduplication**: MD5 hash of the full text becomes the `doc_id`. If a matching `doc_id` already exists in ChromaDB, ingestion is skipped.

3. **Embedding + storage**: all chunks are embedded in one batch with the bi-encoder (`SentenceTransformer`), then upserted into ChromaDB with metadata (source, title, timestamp, doc\_id, chunk index).

#### Search — three-stage pipeline

```
Query
  ├─ 1. Vector search (ChromaDB)  ─────────────────┐
  │       bi-encoder embeds query                   │
  │       cosine similarity against stored vectors  │  RRF merge
  │       → ranked list of chunk IDs               ├──────────────→ candidate pool
  │                                                 │
  └─ 2. BM25 keyword search ───────────────────────┘
          tokenized exact-term scoring
          → ranked list of chunk IDs

  3. Cross-encoder re-rank
          (query, chunk_text) pairs fed to cross-encoder
          raw relevance score per pair
          sort descending → top K returned
```

**Stage 1 — Vector search**: the bi-encoder embeds the query and ChromaDB does cosine similarity against all stored chunk embeddings. Returns `top_k × 4` candidates. Great for semantic matches ("car" finds "automobile").

**Stage 2 — BM25 keyword search**: a classic term-frequency index (`rank-bm25`) built lazily from all chunks in ChromaDB. Scores the same candidate pool by exact keyword overlap. Great for precise terms that embeddings can compress away (version numbers, names, acronyms).

**Stage 3 — RRF fusion**: Reciprocal Rank Fusion merges the two ranked lists without needing to tune weights. Each chunk gets a score of `Σ 1 / (60 + rank)` across both lists — chunks that rank well in both float to the top.

**Stage 4 — Cross-encoder re-rank**: the cross-encoder sees `(query, chunk)` as a single input (unlike the bi-encoder which processes them separately), giving it a richer signal for relevance. Runs on the fused candidate pool and produces the final ranking.

The BM25 index is held in memory and invalidated whenever chunks are added or deleted, so it's always rebuilt fresh from ChromaDB on the next search.

---

### `interfaces/mcp_server.py` — MCP tool definitions

Exposes four tools to Claude Code via `FastMCP` over stdio:

| Tool | What it does |
|---|---|
| `search_knowledge_base(query, top_k)` | Runs the full hybrid search pipeline, returns formatted results |
| `save_to_knowledge_base(text, title, source)` | Ingests raw text |
| `save_url_to_knowledge_base(url)` | Scrapes URL then ingests |
| `list_documents(limit)` | Lists recently saved documents |

Claude Code calls these automatically when the user's intent matches (e.g. "what do I know about X?" triggers `search_knowledge_base`).

---

## Data flows

### Saving a URL

```
save_url_to_knowledge_base(url)
  → web_scraper.scrape_url(url)
      → trafilatura: HTTP fetch + content extraction → ScrapedContent
  → Document{text, title, source=url}
  → store.ingest(doc)
      → MD5 dedup check
      → chunk_text() → ["chunk0", "chunk1", ...]
      → embedder.encode(chunks) → vectors
      → collection.upsert() → written to ./chroma_data/
  → "Saved 'Title' — N chunks created."
```

### Searching

```
search_knowledge_base("what is RAG?")
  → store.search("what is RAG?", top_k=5)
      → vector search: embed query → ChromaDB top 20
      → BM25 search: tokenize query → score all chunks → top 20
      → RRF merge → unified candidate pool
      → cross-encoder.predict([(query, chunk), ...]) → scores
      → sort → top 5
  → formatted result string → Claude Code uses as context
```

---

## What persists between restarts

Only ChromaDB (`./chroma_data/`). Everything else — the embedding model, BM25 index, cross-encoder — is loaded or rebuilt fresh each time the MCP server starts.

The BM25 index and both ML models are loaded lazily: the bi-encoder loads at server startup, the cross-encoder and BM25 index on the first search call.
