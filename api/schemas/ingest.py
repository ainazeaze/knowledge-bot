from pydantic import BaseModel


class IngestTextRequest(BaseModel):
    text: str
    title: str
    source: str = "manual"


class IngestUrlRequest(BaseModel):
    url: str


class IngestResponse(BaseModel):
    job_id: str
    status: str
