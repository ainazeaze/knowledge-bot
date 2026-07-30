from api.schemas.jobs import JobResponse
from knowledge_bot.store import get_store

_jobs: dict[str, JobResponse] = {}


def get_jobs() -> dict[str, JobResponse]:
    return _jobs
