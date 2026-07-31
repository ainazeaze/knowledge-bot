from enum import Enum

from pydantic import BaseModel


class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    detail: str | None = None
