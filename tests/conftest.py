import pytest
import knowledge_bot.config as config
import knowledge_bot.store as store_module
from knowledge_bot.ingestion import KnowledgeStore


@pytest.fixture(scope="session")
def store(tmp_path_factory):
    # Session-scoped so the embedding model loads only once (~3s)
    config.CHROMA_DB_PATH = str(tmp_path_factory.mktemp("chroma"))
    instance = KnowledgeStore()
    store_module._store = instance  # override singleton so get_store() returns the test instance
    return instance
