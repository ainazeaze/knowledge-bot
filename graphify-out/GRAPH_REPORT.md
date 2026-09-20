# Graph Report - knowledge-bot  (2026-09-18)

## Corpus Check
- Corpus is ~4,542 words - fits in a single context window. You may not need a graph.

## Summary
- 193 nodes · 337 edges · 14 communities (10 shown, 4 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 29 edges (avg confidence: 0.94)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Ingestion API Layer
- Search API & Schemas
- Core RAG Pipeline
- Architecture Concepts
- App Bootstrap & Middleware
- KnowledgeStore & Retrieval
- MCP Server Tools
- Web Scraper
- PDF Parser
- Ingestion Tests
- Package Root

## God Nodes (most connected - your core abstractions)
1. `KnowledgeStore` - 19 edges
2. `Document` - 14 edges
3. `parse_pdf()` - 12 edges
4. `ingest_url()` - 10 edges
5. `ingest_pdf()` - 10 edges
6. `ingest_text()` - 9 edges
7. `JobResponse` - 9 edges
8. `run_ingest()` - 8 edges
9. `JobStatus` - 7 edges
10. `scrape_url()` - 7 edges

## Surprising Connections (you probably didn't know these)
- `run_ingest()` --uses--> `KnowledgeStore`  [INFERRED]
  api/routers/ingest.py → knowledge_bot/ingestion.py
- `save_to_knowledge_base()` --uses--> `Document`  [INFERRED]
  interfaces/mcp_server.py → knowledge_bot/ingestion.py
- `save_url_to_knowledge_base()` --uses--> `Document`  [INFERRED]
  interfaces/mcp_server.py → knowledge_bot/ingestion.py
- `save_pdf_to_knowledge_base()` --uses--> `Document`  [INFERRED]
  interfaces/mcp_server.py → knowledge_bot/ingestion.py
- `store()` --uses--> `KnowledgeStore`  [INFERRED]
  tests/conftest.py → knowledge_bot/ingestion.py

## Import Cycles
- None detected.

## Communities (14 total, 4 thin omitted)

### Community 0 - "Ingestion API Layer"
Cohesion: 0.19
Nodes (22): get_jobs(), ingest_pdf(), ingest_text(), ingest_url(), run_ingest(), get_job(), get, IngestResponse (+14 more)

### Community 1 - "Search API & Schemas"
Cohesion: 0.11
Nodes (23): get, search(), BaseModel, SearchResponse, SearchResult, _get_logger(), grade_node(), RetrievalState (+15 more)

### Community 2 - "Core RAG Pipeline"
Cohesion: 0.09
Nodes (10): asyncio, dotenv, fixture, hashlib, Ingestion pipeline: chunk text, embed, and store in ChromaDB., os, pytest, rank_bm25 (+2 more)

### Community 3 - "Architecture Concepts"
Cohesion: 0.11
Nodes (20): Agentic Retrieval, BM25 Index, Bi-Encoder (sentence-transformers), BrainBot, ChromaDB, Claude Code, Cross-Encoder (ms-marco), FastMCP (+12 more)

### Community 4 - "App Bootstrap & Middleware"
Cohesion: 0.13
Nodes (18): catch_exceptions(), lifespan(), _validate_groq_model(), api_routers, delete_document(), get_documents(), get, DocumentItem (+10 more)

### Community 5 - "KnowledgeStore & Retrieval"
Cohesion: 0.12
Nodes (8): CrossEncoder, KnowledgeStore, Merge ranked ID lists with Reciprocal Rank Fusion., Hybrid search: BM25 + vector retrieval, fused with RRF, re-ranked by cross-…, List recently added documents (unique by doc_id)., Delete all chunks belonging to a document. Returns count deleted., Handles chunking, embedding, and storage of documents., Chunk, embed, and store a document. Returns (num_chunks, is_duplicate). If…

### Community 6 - "MCP Server Tools"
Cohesion: 0.17
Nodes (14): list_documents(), MCP server exposing the knowledge base as tools for Claude Desktop., List recently saved documents in the knowledge base., Search the knowledge base using agentic retrieval. Runs hybrid search (BM25 +…, Save a text snippet to the knowledge base. Args: text: The content to save.…, Scrape a URL and save its content to the knowledge base. Use this to save an…, Parse a local PDF file and save its content to the knowledge base. Args:…, save_pdf_to_knowledge_base() (+6 more)

### Community 7 - "Web Scraper"
Cohesion: 0.22
Nodes (11): dataclasses, datetime, Web content extraction using trafilatura., Download and extract main content from a URL., scrape_url(), ScrapedContent, _content(), test_empty_content_is_invalid() (+3 more)

### Community 8 - "PDF Parser"
Cohesion: 0.35
Nodes (9): fitz, parse_pdf(), ParsedPDF, _make_pdf(), test_is_valid(), test_page_count(), test_text_content(), test_title_falls_back_to_filename() (+1 more)

### Community 9 - "Ingestion Tests"
Cohesion: 0.36
Nodes (6): _doc(), test_delete_removes_all_chunks(), test_duplicate_detection(), test_empty_text_not_ingested(), test_ingest_returns_chunk_count(), test_search_finds_ingested_content()

## Knowledge Gaps
- **9 isolated node(s):** `knowledge-bot`, `React Web UI`, `Cross-Encoder (ms-marco)`, `Reciprocal Rank Fusion`, `LangGraph` (+4 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 74 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `KnowledgeStore` connect `KnowledgeStore & Retrieval` to `Ingestion API Layer`, `Core RAG Pipeline`, `MCP Server Tools`?**
  _High betweenness centrality (0.173) - this node is a cross-community bridge._
- **Why does `parse_pdf()` connect `PDF Parser` to `Ingestion API Layer`, `MCP Server Tools`?**
  _High betweenness centrality (0.098) - this node is a cross-community bridge._
- **Why does `scrape_url()` connect `Web Scraper` to `Ingestion API Layer`, `MCP Server Tools`?**
  _High betweenness centrality (0.090) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `KnowledgeStore` (e.g. with `run_ingest()` and `store()`) actually correct?**
  _`KnowledgeStore` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `Document` (e.g. with `ingest_pdf()` and `ingest_text()`) actually correct?**
  _`Document` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `ingest_url()` (e.g. with `run_ingest()` and `IngestResponse`) actually correct?**
  _`ingest_url()` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `ingest_pdf()` (e.g. with `run_ingest()` and `IngestResponse`) actually correct?**
  _`ingest_pdf()` has 5 INFERRED edges - model-reasoned connections that need verification._