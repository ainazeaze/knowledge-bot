import knowledge_bot.config as config


def test_short_text_is_single_chunk(store):
    chunks = store.chunk_text("Hello world")
    assert len(chunks) == 1
    assert chunks[0] == "Hello world"


def test_empty_text_returns_no_chunks(store):
    assert store.chunk_text("") == []


def test_whitespace_only_returns_no_chunks(store):
    assert store.chunk_text("   \n\n   ") == []


def test_long_text_is_split(store):
    chunks = store.chunk_text("word " * 1000)
    assert len(chunks) > 1


def test_chunks_do_not_exceed_chunk_size(store):
    for chunk in store.chunk_text("a" * 10000):
        assert len(chunk) <= config.CHUNK_SIZE


def test_overlap_between_consecutive_chunks(store):
    chunks = store.chunk_text("abcdefghij " * 500)
    assert len(chunks) > 1
    assert chunks[1][:50] in chunks[0]


def test_paragraph_boundary_preferred(store):
    text = "A" * 1200 + "\n\n" + "B" * 1200
    chunks = store.chunk_text(text)
    assert len(chunks) > 1
    assert all(c == "A" for c in chunks[0])


def test_all_content_covered(store):
    words = [f"word{i}" for i in range(200)]
    chunks = store.chunk_text(" ".join(words))
    combined = " ".join(chunks)
    for word in words:
        assert word in combined
