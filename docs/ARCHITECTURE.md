# BrainBot — Architecture & Code Walkthrough

## The big picture

When you type `/save <url>` or `/ask <question>` in Discord, a chain of components kicks in. Each file owns one layer of that chain:

```
Discord user
     │
  bot.py          ← entry point, Discord interface
     │
  web_scraper.py  ← fetches & cleans URL content (only for /save <url>)
     │
  ingestion.py    ← chunks text, embeds it, stores in ChromaDB
     │
  retriever.py    ← searches ChromaDB, calls Claude API for answers
     │
  config.py       ← env vars used by all of the above
```

---

## File by file

### `config.py` — env vars, globally imported

The simplest file. Loads `.env` and exposes everything as module-level constants (`DISCORD_TOKEN`, `ANTHROPIC_API_KEY`, `CHUNK_SIZE`, etc.). Every other file imports this — it's the single source of truth for configuration.

---

### `web_scraper.py` — URL → clean text

Only used when you `/save` a URL (not raw text). Calls `trafilatura.fetch_url()` to download the page, then `trafilatura.extract()` to strip out nav, ads, footers, and return just the article body. Returns a `ScrapedContent` dataclass with `.text`, `.title`, and `.is_valid` (checks that at least 50 chars came back).

No ChromaDB, no embeddings — just HTTP in, clean text out.

---

### `ingestion.py` — text → vectors → ChromaDB

The storage engine. Owns two things:

**`Document` dataclass** — a plain container for content before storage: `text`, `source` (URL or `"manual"`), `title`, `added_by` (Discord user ID).

**`KnowledgeStore` class** — does three jobs:

1. **Chunking** (`chunk_text`): Splits long text into overlapping pieces so a 10,000-word article becomes ~15 chunks of 2000 chars with 200-char overlap. The overlap means a sentence straddling a boundary appears in both chunks, so retrieval doesn't miss it.

2. **Embedding** (`ingest`): Runs each chunk through `SentenceTransformer("all-MiniLM-L6-v2")` — a small local model that converts text to a 384-dimensional vector. No API call, runs on CPU. These vectors are what make semantic search possible (similar meaning = similar vector).

3. **Storage** (`ingest`, `search`, `delete_document`, `list_documents`): Talks to ChromaDB, a local vector database that persists to `./chroma_data/`. ChromaDB stores the chunk text, embedding vector, and metadata (source URL, title, timestamp, doc ID) together. The `doc_id` is an MD5 hash of the first 500 chars of the document — it ties all chunks from the same document together so you can delete a whole document at once.

---

### `retriever.py` — question → answer

**`Retriever` class** — the RAG layer. Wraps `KnowledgeStore` and adds Claude on top.

**`search(query)`**: Calls `store.search()` and returns ranked results. Pure vector similarity, no LLM.

**`ask(question)`** — the full RAG pipeline:
1. Embeds the question using the same model as ingestion
2. Asks ChromaDB for the top-K most similar chunks (default 5)
3. If no API key → formats those chunks as readable text and returns
4. If API key present → builds a context string from the chunks, then calls Claude with a system prompt constrained to "answer only from the provided context"
5. Returns the answer + source list + a `used_llm` flag

The key insight of RAG is that Claude never "remembers" your documents — it reads them fresh every time as context in the prompt. This keeps answers grounded and avoids hallucination.

---

### `bot.py` — Discord interface, entry point

Sets up a `discord.Client` and an `app_commands.CommandTree` (Discord's slash command system). All six commands follow the same pattern:

1. `await interaction.response.defer(thinking=True)` — immediately tells Discord "I'm working on it", preventing the 3-second timeout
2. Do the work (call `store` or `retriever`)
3. Build a `discord.Embed` (the formatted card Discord displays)
4. `await interaction.followup.send(embed=embed)`

`store` and `retriever` are initialized in `on_ready()`, not at module level, because they need the event loop to be running.

| Command | Calls |
|---|---|
| `/save <url>` | `scrape_url()` → `store.ingest()` |
| `/save <text>` | `store.ingest()` directly |
| `/ask <question>` | `retriever.ask()` |
| `/search <keywords>` | `retriever.search()` |
| `/list` | `store.list_documents()` |
| `/delete <id>` | `store.delete_document()` |
| `/stats` | `store.list_documents()` + `store.total_chunks` |

---

## Data flows

### `/save https://example.com/article`

```
bot.py: save_command()
  → web_scraper.scrape_url("https://...")
      → trafilatura downloads + extracts → ScrapedContent{text, title}
  → Document{text, source="https://...", title, added_by="user_id"}
  → store.ingest(doc)
      → chunk_text(doc.text) → ["chunk0", "chunk1", ...]
      → embedder.encode(chunks) → [[0.12, -0.43, ...], ...]  (384-dim each)
      → collection.upsert(ids, documents, embeddings, metadatas)
          → written to ./chroma_data/ on disk
  → returns num_chunks
bot.py: sends Discord embed "Saved — 8 chunks"
```

### `/ask "what did that article say about X?"`

```
bot.py: ask_command()
  → retriever.ask("what did that article say about X?")
      → store.search(question)
          → embedder.encode(["what did..."]) → [0.08, 0.71, ...]
          → ChromaDB cosine similarity search → top 5 chunks
      → _build_context(results) → formatted string of 5 chunks
      → claude_client.messages.create(
            system="answer only from context",
            user="Context: <5 chunks>\n\nQuestion: what did..."
        )
      → returns answer text
  → returns {answer, sources, used_llm=True}
bot.py: sends Discord embed with answer + source links
```

---

## What persists between restarts

Only ChromaDB. The `./chroma_data/` directory is a local SQLite + index file. Everything else (Python objects, embedding model loaded into RAM) is reconstructed fresh each time the bot starts. There's no database server — ChromaDB is an embedded library, like SQLite.
