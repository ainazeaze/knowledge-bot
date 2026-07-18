# BrainBot — Discord Knowledge Base Bot

A personal knowledge base that lives in your Discord server. Save articles, notes, and URLs, then query them in natural language using RAG (Retrieval-Augmented Generation).

## Features

- `/save <url_or_text>` — Save a URL or text snippet to your knowledge base
- `/ask <question>` — Ask a natural language question, get an answer grounded in your saved content
- `/search <keywords>` — Browse saved items by keyword similarity
- `/list` — Show recently saved items
- `/delete <id>` — Remove an item from the knowledge base

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
uv venv
source .venv/bin/activate  
uv sync
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

## How It Works

**Saving:** When you `/save` a URL, the bot scrapes the page content, splits it into
overlapping chunks (~500 tokens each), embeds each chunk using a local model, and
stores the vectors + metadata in ChromaDB.

**Asking:** When you `/ask` a question, the bot embeds your query, retrieves the top-5
most similar chunks from ChromaDB, and sends them as context to Claude to generate a
grounded answer with source references.

**Searching:** `/search` does a pure vector similarity search without LLM generation —
useful for browsing what you've saved.

## LLM Providers

The bot supports two LLM backends, switchable via `.env`:

```
LLM_PROVIDER=ollama        # local, recommended
LLM_PROVIDER=claude        # Anthropic API
```
## MCP Server

The knowledge base is also exposed as an MCP server, letting you query it directly from Claude Code without going through Discord.

**Tools available:**
- `search_knowledge_base(query, top_k)` — semantic search over saved content
- `save_to_knowledge_base(text, title, source)` — add a text snippet
- `list_documents(limit)` — list recently saved documents

**Register with Claude Code:**

```bash
claude mcp add brainbot /path/to/.venv/bin/python /path/to/knowledge-bot/mcp_server.py
```
The MCP server shares the same ChromaDB as the Discord bot — anything saved via `/save` is instantly queryable through MCP and vice versa.
