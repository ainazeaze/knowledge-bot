from knowledge_bot.ingestion import Document


def _doc(text, title="test"):
    return Document(text=text, title=title, source="manual")


def test_ingest_returns_chunk_count(store):
    n, is_dup = store.ingest(_doc("a" * 100))
    assert n >= 1
    assert is_dup is False


def test_duplicate_detection(store):
    doc = _doc("unique content for duplicate test " * 10)
    n1, is_dup1 = store.ingest(doc)
    n2, is_dup2 = store.ingest(doc)
    assert is_dup1 is False
    assert is_dup2 is True
    assert n2 == n1


def test_empty_text_not_ingested(store):
    n, is_dup = store.ingest(_doc(""))
    assert n == 0
    assert is_dup is False


def test_search_finds_ingested_content(store):
    store.ingest(_doc("Quantum entanglement is a phenomenon in quantum physics.", title="Quantum"))
    results = store.search("quantum entanglement", top_k=3)
    assert any(r["title"] == "Quantum" for r in results)


def test_search_score_is_float(store):
    # Cross-encoder scores are raw logits — no fixed range, just floats
    results = store.search("test", top_k=1)
    if results:
        assert isinstance(results[0]["score"], float)


def test_delete_removes_all_chunks(store):
    doc = _doc("Content to be deleted " * 20, title="DeleteMe")
    n, _ = store.ingest(doc)
    assert n >= 1

    results_before = store.search("Content to be deleted", top_k=5)
    doc_id = next(r["doc_id"] for r in results_before if r["title"] == "DeleteMe")

    deleted = store.delete_document(doc_id)
    assert deleted == n

    results_after = store.search("Content to be deleted", top_k=5)
    assert all(r["title"] != "DeleteMe" for r in results_after)


def test_delete_nonexistent_returns_zero(store):
    assert store.delete_document("doesnotexist") == 0
