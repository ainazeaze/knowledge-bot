from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_jobs
from api.schemas.jobs import JobResponse

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str, jobs: dict = Depends(get_jobs)): # noqa: B008
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]
