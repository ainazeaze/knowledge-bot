"""Ingestion pipeline: chunk text, embed, and store in ChromaDB."""

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from . import config

RERANK_FACTOR = 4  # fetch top_k * this, then re-rank down to top_k
RRF_K = 60  # reciprocal rank fusion constant (standard value)


@dataclass
class Document:
    text: str
    source: str
    title: str


class KnowledgeStore:
    """Handles chunking, embedding, and storage of documents."""

    def __init__(self):
        self.client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        self._reranker: CrossEncoder | None = None
        self._bm25: BM25Okapi | None = None
        self._bm25_chunks: list[dict] = []

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

    def ingest(self, doc: Document) -> tuple[int, bool]:
        """Chunk, embed, and store a document.

        Returns (num_chunks, is_duplicate). If duplicate, no work is done.
        """
        doc_id = hashlib.md5(doc.text.encode()).hexdigest()[:16]

        existing = self.collection.get(where={"doc_id": doc_id}, limit=1, include=[])
        if existing["ids"]:
            existing_count = self.collection.get(where={"doc_id": doc_id}, include=[])
            return len(existing_count["ids"]), True

        chunks = self.chunk_text(doc.text)
        if not chunks:
            return 0, False

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

        self._bm25 = None
        return len(chunks), False

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = CrossEncoder(config.RERANKER_MODEL)
        return self._reranker

    def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return
        all_items = self.collection.get(include=["documents", "metadatas"])
        assert all_items["documents"] is not None
        assert all_items["metadatas"] is not None
        self._bm25_chunks = [
            {
                "id": all_items["ids"][i],
                "text": all_items["documents"][i],
                "source": all_items["metadatas"][i]["source"],
                "title": all_items["metadatas"][i]["title"],
                "doc_id": all_items["metadatas"][i]["doc_id"],
                "added_at": all_items["metadatas"][i]["added_at"],
            }
            for i in range(len(all_items["ids"]))
        ]
        tokenized = [c["text"].lower().split() for c in self._bm25_chunks]
        self._bm25 = BM25Okapi(tokenized)

    def _rrf_merge(self, *rankings: list[str]) -> list[str]:
        """Merge ranked ID lists with Reciprocal Rank Fusion."""
        scores: dict[str, float] = {}
        for ranking in rankings:
            for rank, chunk_id in enumerate(ranking):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        return sorted(scores, key=lambda x: scores[x], reverse=True)

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Hybrid search: BM25 + vector retrieval, fused with RRF, re-ranked by cross-encoder."""
        k = top_k or config.TOP_K
        total = self.collection.count()
        if total == 0:
            return []

        n_candidates = min(k * RERANK_FACTOR, total)

        # Vector search
        query_embedding = self.embedder.encode([query]).tolist()
        vec_results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_candidates,
            include=["documents", "metadatas"],
        )
        assert vec_results["documents"] is not None
        assert vec_results["metadatas"] is not None
        vec_ids = vec_results["ids"][0]
        vec_chunks = {
            vec_ids[i]: {
                "text": vec_results["documents"][0][i],
                "source": vec_results["metadatas"][0][i]["source"],
                "title": vec_results["metadatas"][0][i]["title"],
                "doc_id": vec_results["metadatas"][0][i]["doc_id"],
                "added_at": vec_results["metadatas"][0][i]["added_at"],
            }
            for i in range(len(vec_ids))
        }

        # BM25 search
        self._ensure_bm25()
        assert self._bm25 is not None
        bm25_scores = self._bm25.get_scores(query.lower().split())
        bm25_ranked = [
            self._bm25_chunks[i]["id"]
            for i in sorted(range(len(bm25_scores)), key=lambda x: bm25_scores[x], reverse=True)
        ][:n_candidates]

        # RRF fusion
        merged_ids = self._rrf_merge(vec_ids, bm25_ranked)[:n_candidates]

        # Build candidate list (prefer vec_chunks, fall back to bm25_chunks index)
        bm25_by_id = {c["id"]: c for c in self._bm25_chunks}
        candidates = []
        for chunk_id in merged_ids:
            chunk = vec_chunks.get(chunk_id) or bm25_by_id.get(chunk_id)
            if chunk:
                candidates.append(chunk)

        # Cross-encoder re-rank
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs).tolist()
        for candidate, score in zip(candidates, scores):
            candidate["score"] = score

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:k]

    def list_documents(self, limit: int = 10) -> list[dict]:
        """List recently added documents (unique by doc_id)."""
        all_items = self.collection.get(
            include=["metadatas"],
            limit=500,  # get a batch to deduplicate
        )
        assert all_items["metadatas"] is not None

        seen = {}
        for meta in all_items["metadatas"]:
            doc_id = meta["doc_id"]
            if doc_id not in seen:
                seen[doc_id] = {
                    "doc_id": doc_id,
                    "title": meta["title"],
                    "source": meta["source"],
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
        self._bm25 = None
        return len(all_items["ids"])

    @property
    def total_chunks(self) -> int:
        return self.collection.count()
