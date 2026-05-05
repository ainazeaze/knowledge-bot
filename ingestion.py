"""Ingestion pipeline: chunk text, embed, and store in ChromaDB."""

import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import chromadb
from sentence_transformers import SentenceTransformer

import config


@dataclass
class Document:
    """A document to ingest into the knowledge base."""

    text: str
    source: str  # URL or "manual"
    title: str
    added_by: str  # Discord user ID


class KnowledgeStore:
    """Handles chunking, embedding, and storage of documents."""

    def __init__(self):
        self.client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)

    def chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks.

        Uses a simple character-based splitter that tries to break on
        paragraph boundaries. Good enough for v1 — swap in a token-aware
        splitter (like langchain's) if you need more precision.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + config.CHUNK_SIZE

            # Try to break on a paragraph boundary
            if end < len(text):
                # Look for a paragraph break near the end
                newline_pos = text.rfind("\n\n", start + config.CHUNK_SIZE // 2, end)
                if newline_pos != -1:
                    end = newline_pos

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - config.CHUNK_OVERLAP

        return chunks

    def ingest(self, doc: Document) -> int:
        """Chunk, embed, and store a document. Returns number of chunks created."""
        chunks = self.chunk_text(doc.text)
        if not chunks:
            return 0

        # Generate a stable document ID from content hash
        doc_id = hashlib.md5(doc.text[:500].encode()).hexdigest()[:12]
        now = datetime.now(timezone.utc).isoformat()

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append(
                {
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "source": doc.source,
                    "title": doc.title,
                    "added_by": doc.added_by,
                    "added_at": now,
                }
            )

        # Embed all chunks in one batch
        embeddings = self.embedder.encode(documents).tolist()

        # Upsert into ChromaDB
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(chunks)

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Search the knowledge base by semantic similarity.

        Returns a list of dicts with keys: text, source, title, score, doc_id
        """
        k = top_k or config.TOP_K
        query_embedding = self.embedder.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        items = []
        for i in range(len(results["ids"][0])):
            items.append(
                {
                    "text": results["documents"][0][i],
                    "source": results["metadatas"][0][i]["source"],
                    "title": results["metadatas"][0][i]["title"],
                    "doc_id": results["metadatas"][0][i]["doc_id"],
                    "added_at": results["metadatas"][0][i]["added_at"],
                    "score": 1 - results["distances"][0][i],  # cosine similarity
                }
            )

        return items

    def list_documents(self, limit: int = 10) -> list[dict]:
        """List recently added documents (unique by doc_id)."""
        all_items = self.collection.get(
            include=["metadatas"],
            limit=500,  # get a batch to deduplicate
        )

        seen = {}
        for meta in all_items["metadatas"]:
            doc_id = meta["doc_id"]
            if doc_id not in seen:
                seen[doc_id] = {
                    "doc_id": doc_id,
                    "title": meta["title"],
                    "source": meta["source"],
                    "added_by": meta["added_by"],
                    "added_at": meta["added_at"],
                    "total_chunks": meta["total_chunks"],
                }

        # Sort by added_at descending
        docs = sorted(seen.values(), key=lambda x: x["added_at"], reverse=True)
        return docs[:limit]

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks belonging to a document. Returns count deleted."""
        all_items = self.collection.get(
            include=["metadatas"],
            where={"doc_id": doc_id},
        )

        if not all_items["ids"]:
            return 0

        self.collection.delete(ids=all_items["ids"])
        return len(all_items["ids"])

    @property
    def total_chunks(self) -> int:
        return self.collection.count()
