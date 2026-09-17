from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from groq import Groq

from api.routers import documents, ingest, jobs, search
from knowledge_bot import config
from knowledge_bot.logger import logger


def _validate_groq_model() -> None:
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set")
    client = Groq(api_key=config.GROQ_API_KEY)
    available = {m.id for m in client.models.list().data}
    if config.GROQ_MODEL not in available:
        logger.error(
            "GROQ_MODEL=%r is not available. Available models: %s",
            config.GROQ_MODEL,
            sorted(available),
        )
        raise RuntimeError(
            f"GROQ_MODEL={config.GROQ_MODEL!r} not found. "
            f"Check logs for available models."
        )
    logger.info("startup | GROQ_MODEL=%r OK", config.GROQ_MODEL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _validate_groq_model()
    yield


app = FastAPI(
    title="BrainBot API",
    description="RAG knowledge base — ingest documents and search them semantically.",
    version="0.1.0",
    lifespan=lifespan,
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
