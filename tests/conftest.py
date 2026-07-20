import pytest
import knowledge_bot.config as config
from knowledge_bot.ingestion import KnowledgeStore


@pytest.fixture(scope="session")
def store(tmp_path_factory):
    # Session-scoped so the embedding model loads only once (~3s)
    config.CHROMA_DB_PATH = str(tmp_path_factory.mktemp("chroma"))
    return KnowledgeStore()
