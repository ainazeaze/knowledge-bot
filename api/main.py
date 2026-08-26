from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import documents, ingest, jobs, search

app = FastAPI(
    title="BrainBot API",
    description="RAG knowledge base — ingest documents and search them semantically.",
    version="0.1.0",
)

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
