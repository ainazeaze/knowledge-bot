"""BrainBot — Discord Knowledge Base Bot.

A personal second brain that lives in your Discord server.
Save articles, notes, and URLs, then query them in natural language.
"""

import asyncio
import logging

import discord
from discord import app_commands

import config
from ingestion import Document, KnowledgeStore
from retriever import Retriever
from web_scraper import scrape_url

# ── Logging ──────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
log = logging.getLogger("brainbot")

# ── Bot setup ────────────────────────────────────────────────────────

intents = discord.Intents.default()
intents.message_content = True

bot = discord.Client(intents=intents)
tree = app_commands.CommandTree(bot)

# These get initialized in on_ready (after the event loop is running)
store: KnowledgeStore | None = None
retriever: Retriever | None = None


@bot.event
async def on_ready():
    global store, retriever
    store = KnowledgeStore()
    retriever = Retriever(store)

    # Sync slash commands with Discord
    await tree.sync()
    log.info(f"Logged in as {bot.user} | {store.total_chunks} chunks in knowledge base")


# ── /save ────────────────────────────────────────────────────────────


@tree.command(
    name="save", description="Save a URL or text snippet to your knowledge base"
)
@app_commands.describe(content="A URL to scrape, or raw text to save directly")
async def save_command(interaction: discord.Interaction, content: str):
    await interaction.response.defer(thinking=True)

    try:
        is_url = content.startswith("http://") or content.startswith("https://")

        if is_url:
            # Scrape the URL
            scraped = scrape_url(content)
            if not scraped.is_valid:
                await interaction.followup.send(
                    "Couldn't extract meaningful content from that URL. "
                    "Try saving the text directly instead."
                )
                return

            doc = Document(
                text=scraped.text,
                source=content,
                title=scraped.title,
                added_by=str(interaction.user.id),
            )
        else:
            # Save raw text
            doc = Document(
                text=content,
                source="manual",
                title=content[:80] + ("..." if len(content) > 80 else ""),
                added_by=str(interaction.user.id),
            )

        num_chunks, is_duplicate = await asyncio.to_thread(store.ingest, doc)

        if is_duplicate:
            await interaction.followup.send(
                f"Already in your knowledge base — skipped. ({num_chunks} chunks exist)"
            )
            return

        embed = discord.Embed(
            title="Saved to knowledge base",
            color=0x2ECC71,
        )
        embed.add_field(name="Title", value=doc.title[:256], inline=False)
        embed.add_field(name="Source", value=doc.source[:256], inline=True)
        embed.add_field(name="Chunks", value=str(num_chunks), inline=True)
        embed.set_footer(text=f"Total: {store.total_chunks} chunks in knowledge base")

        await interaction.followup.send(embed=embed)
        log.info(f"Saved: {doc.title[:60]} ({num_chunks} chunks) by {interaction.user}")

    except Exception as e:
        log.error(f"Save failed: {e}", exc_info=True)
        await interaction.followup.send(f"Failed to save: {e}")


# ── /ask ─────────────────────────────────────────────────────────────


@tree.command(
    name="ask", description="Ask a question — get an answer from your knowledge base"
)
@app_commands.describe(question="Your question in natural language")
async def ask_command(interaction: discord.Interaction, question: str):
    await interaction.response.defer(thinking=True)

    try:
        result = await asyncio.to_thread(retriever.ask, question)

        embed = discord.Embed(
            title=question[:256],
            description=result["answer"][:4000],
            color=0x3498DB,
        )

        # Add source references
        if result["sources"]:
            sources_text = []
            seen_titles = set()
            for s in result["sources"]:
                if s["title"] not in seen_titles:
                    seen_titles.add(s["title"])
                    label = s["title"][:60]
                    if s["source"] != "manual":
                        label = f"[{label}]({s['source']})"
                    sources_text.append(f"• {label}")

            if sources_text:
                embed.add_field(
                    name="Sources",
                    value="\n".join(sources_text[:5]),
                    inline=False,
                )

        method = "RAG + LLM" if result["used_llm"] else "Vector search"
        embed.set_footer(text=f"Method: {method}")

        await interaction.followup.send(embed=embed)
        log.info(f"Ask: '{question[:60]}' by {interaction.user}")

    except Exception as e:
        log.error(f"Ask failed: {e}", exc_info=True)
        await interaction.followup.send(f"Something went wrong: {e}")


# ── /search ──────────────────────────────────────────────────────────


@tree.command(name="search", description="Search your knowledge base by keywords")
@app_commands.describe(query="Keywords or phrase to search for")
async def search_command(interaction: discord.Interaction, query: str):
    await interaction.response.defer(thinking=True)

    try:
        results = await asyncio.to_thread(retriever.search, query, 5)

        if not results:
            await interaction.followup.send("No results found. Try different keywords.")
            return

        embed = discord.Embed(
            title=f"Search: {query[:200]}",
            color=0x9B59B6,
        )

        for i, r in enumerate(results, 1):
            score_pct = f"{r['score']:.0%}"
            snippet = r["text"][:200].replace("\n", " ").strip()
            if len(r["text"]) > 200:
                snippet += "..."

            source_info = ""
            if r["source"] != "manual":
                source_info = f"\n[Link]({r['source']})"

            embed.add_field(
                name=f"{i}. {r['title'][:80]} ({score_pct})",
                value=f"{snippet}{source_info}",
                inline=False,
            )

        embed.set_footer(text=f"{store.total_chunks} chunks in knowledge base")
        await interaction.followup.send(embed=embed)

    except Exception as e:
        log.error(f"Search failed: {e}", exc_info=True)
        await interaction.followup.send(f"Something went wrong: {e}")


# ── /list ────────────────────────────────────────────────────────────


@tree.command(name="list", description="Show recently saved documents")
async def list_command(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True)

    try:
        docs = store.list_documents(limit=10)

        if not docs:
            await interaction.followup.send(
                "Knowledge base is empty. Save something with `/save`!"
            )
            return

        embed = discord.Embed(
            title="Knowledge base",
            description=f"{store.total_chunks} chunks across {len(docs)}+ documents",
            color=0xF39C12,
        )

        for doc in docs:
            source = doc["source"] if doc["source"] != "manual" else "Manual note"
            embed.add_field(
                name=f"{doc['title'][:80]}",
                value=f"Source: {source[:100]}\nChunks: {doc['total_chunks']} | ID: `{doc['doc_id']}`",
                inline=False,
            )

        await interaction.followup.send(embed=embed)

    except Exception as e:
        log.error(f"List failed: {e}", exc_info=True)
        await interaction.followup.send(f"Something went wrong: {e}")


# ── /delete ──────────────────────────────────────────────────────────


@tree.command(name="delete", description="Delete a document from the knowledge base")
@app_commands.describe(doc_id="Document ID (from /list)")
async def delete_command(interaction: discord.Interaction, doc_id: str):
    await interaction.response.defer(thinking=True)

    try:
        count = store.delete_document(doc_id)

        if count == 0:
            await interaction.followup.send(
                f"No document found with ID `{doc_id}`. Use `/list` to see IDs."
            )
        else:
            await interaction.followup.send(
                f"Deleted document `{doc_id}` ({count} chunks removed). "
                f"{store.total_chunks} chunks remaining."
            )

    except Exception as e:
        log.error(f"Delete failed: {e}", exc_info=True)
        await interaction.followup.send(f"Something went wrong: {e}")


# ── /stats ───────────────────────────────────────────────────────────


@tree.command(name="stats", description="Show knowledge base statistics")
async def stats_command(interaction: discord.Interaction):
    docs = store.list_documents(limit=999)

    embed = discord.Embed(title="Knowledge base stats", color=0x1ABC9C)
    embed.add_field(name="Total chunks", value=str(store.total_chunks), inline=True)
    embed.add_field(name="Documents", value=str(len(docs)), inline=True)
    embed.add_field(
        name="LLM",
        value=retriever.provider_name
        if retriever.llm_available
        else "Disabled (search only)",
        inline=True,
    )
    embed.add_field(
        name="Embedding model",
        value=config.EMBEDDING_MODEL,
        inline=False,
    )

    await interaction.response.send_message(embed=embed)


# ── Run ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not config.DISCORD_TOKEN:
        print("Error: DISCORD_TOKEN not set. Copy .env.example to .env and fill it in.")
        raise SystemExit(1)

    if not config.ANTHROPIC_API_KEY:
        log.warning(
            "ANTHROPIC_API_KEY not set — /ask will return raw search results "
            "instead of LLM-generated answers. Set it in .env for full RAG."
        )

    bot.run(config.DISCORD_TOKEN)
