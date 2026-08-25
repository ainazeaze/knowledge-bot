import os
import tempfile
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.background import BackgroundTasks

from api.dependencies import get_jobs, get_store
from api.schemas.ingest import IngestResponse, IngestTextRequest, IngestUrlRequest
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



@router.post("/text", response_model=IngestResponse, status_code=202)
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


@router.post("/url", response_model=IngestResponse, status_code=202)
async def ingest_url(
    request: IngestUrlRequest,
    background_tasks: BackgroundTasks,
    store=Depends(get_store),  # noqa: B008
    jobs=Depends(get_jobs),  # noqa: B008
):
    from knowledge_bot.web_scraper import scrape_url
    scraped = scrape_url(request.url)
    if not scraped.is_valid:
        raise HTTPException(status_code=422, detail="Couldn't extract meaningful content from that URL.")
    job_id = str(uuid.uuid4())
    doc = Document(text=scraped.text, title=scraped.title, source=request.url)
    jobs[job_id] = JobResponse(job_id=job_id, status=JobStatus.PENDING)
    background_tasks.add_task(run_ingest, job_id, store, doc)
    return IngestResponse(job_id=job_id, status="pending")


@router.post("/pdf", response_model=IngestResponse, status_code=202)
async def ingest_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    store=Depends(get_store),  # noqa: B008
    jobs=Depends(get_jobs),  # noqa: B008
):
    from knowledge_bot.pdf_parser import parse_pdf
    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    pdf = parse_pdf(tmp_path)
    os.unlink(tmp_path)
    if not pdf.is_valid:
        raise HTTPException(status_code=422, detail="Couldn't extract meaningful text from that PDF.")
    job_id = str(uuid.uuid4())
    doc = Document(text=pdf.text, title=pdf.title, source=file.filename or "upload.pdf")
    jobs[job_id] = JobResponse(job_id=job_id, status=JobStatus.PENDING)
    background_tasks.add_task(run_ingest, job_id, store, doc)
    return IngestResponse(job_id=job_id, status="pending")
