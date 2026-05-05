# BrainBot — Discord Knowledge Base Bot

A personal knowledge base that lives in your Discord server. Save articles, notes, and URLs, then query them in natural language using RAG (Retrieval-Augmented Generation).

## Features

- `/save <url_or_text>` — Save a URL or text snippet to your knowledge base
- `/ask <question>` — Ask a natural language question, get an answer grounded in your saved content
- `/search <keywords>` — Browse saved items by keyword similarity
- `/list` — Show recently saved items
- `/delete <id>` — Remove an item from the knowledge base

## Tech Stack

- **discord.py** — Bot framework
- **ChromaDB** — Local vector database (zero config)
- **sentence-transformers** — Local embeddings (`all-MiniLM-L6-v2`, runs on CPU)
- **Anthropic Claude API** — Answer generation (optional, can swap for any LLM)
- **trafilatura** — Web page content extraction

## Setup

### 1. Create a Discord Bot

1. Go to https://discord.com/developers/applications
2. Click "New Application" → name it → go to "Bot" tab
3. Click "Reset Token" → copy the token
4. Enable "Message Content Intent" under Privileged Gateway Intents
5. Go to OAuth2 → URL Generator → select `bot` + `applications.commands`
6. Select permissions: Send Messages, Embed Links, Read Message History
7. Copy the URL and open it to invite the bot to your server

### 2. Get an Anthropic API Key (optional)

If you want AI-generated answers (not just search results):
1. Go to https://console.anthropic.com/
2. Create an API key

### 3. Install Dependencies

```bash
cd knowledge-bot
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Configure

```bash
cp .env.example .env
# Edit .env with your tokens
```

### 5. Run

```bash
python bot.py
```

## Project Structure

```
knowledge-bot/
├── bot.py              # Main bot entry point + slash commands
├── ingestion.py        # Document chunking and embedding pipeline
├── retriever.py        # Vector search and RAG answer generation
├── web_scraper.py      # URL content extraction
├── config.py           # Settings and env vars
├── requirements.txt
├── .env.example
└── README.md
```

## How It Works

**Saving:** When you `/save` a URL, the bot scrapes the page content, splits it into
overlapping chunks (~500 tokens each), embeds each chunk using a local model, and
stores the vectors + metadata in ChromaDB.

**Asking:** When you `/ask` a question, the bot embeds your query, retrieves the top-5
most similar chunks from ChromaDB, and sends them as context to Claude to generate a
grounded answer with source references.

**Searching:** `/search` does a pure vector similarity search without LLM generation —
useful for browsing what you've saved.

## Customization Ideas

- Add PDF ingestion (use `PyMuPDF` or `pdfplumber`)
- Add scheduled ingestion from RSS feeds
- Add per-user knowledge bases (use Discord user ID as collection namespace)
- Add a web UI with Gradio alongside the Discord bot

## LLM Usage
This project was made in collaboration with Claude Code, testing how fast we can build actually useful things now with AI.
## Done
- Swap Claude for a local LLM (Ollama + Mistral/Llama)
