import uuid

from fastapi import APIRouter, Depends
from fastapi.background import BackgroundTasks

from api.dependencies import get_jobs, get_store
from api.schemas.ingest import IngestResponse, IngestTextRequest
from api.schemas.jobs import JobResponse, JobStatus
from knowledge_bot.ingestion import Document, KnowledgeStore

router = APIRouter(prefix="/ingest", tags=["ingest"])


async def run_ingest(job_id: str, store: KnowledgeStore, doc: Document):
    get_jobs()[job_id].status = JobStatus.PROCESSING
    try:
        num_chunks, is_duplicate = await store.async_ingest(doc)
        if is_duplicate:
            get_jobs()[job_id].status = JobStatus.DONE
            get_jobs()[job_id].detail = "Already in knowledge base"
            return
        get_jobs()[job_id].status = JobStatus.DONE
        get_jobs()[job_id].detail = f"Saved {num_chunks} chunks"

    except Exception as e:
        get_jobs()[job_id].status = JobStatus.FAILED
        get_jobs()[job_id].detail = str(e)



@router.post("", response_model=IngestResponse, status_code=202)
async def ingest_text(
    request: IngestTextRequest,
    background_tasks: BackgroundTasks,
    store=Depends(get_store), # noqa :  B008
    jobs=Depends(get_jobs), # noqa :  B008
):
    job_id = str(uuid.uuid4())
    doc = Document(text=request.text, title=request.title, source=request.source)
    jobs[job_id] = JobResponse(job_id=job_id, status=JobStatus.PENDING)
    background_tasks.add_task(run_ingest, job_id, store, doc)
    return IngestResponse(job_id=job_id, status="pending")
