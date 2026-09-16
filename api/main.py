from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routers import documents, ingest, jobs, search
from knowledge_bot.logger import logger

app = FastAPI(
    title="BrainBot API",
    description="RAG knowledge base — ingest documents and search them semantically.",
    version="0.1.0",
)


@app.middleware("http")
async def catch_exceptions(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        logger.exception("unhandled error: %s", exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router)
app.include_router(search.router)
app.include_router(documents.router)
app.include_router(jobs.router)
