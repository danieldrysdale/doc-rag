"""
FastAPI REST interface for doc-rag.

Endpoints:
    POST /query    — Query the document knowledge base
    POST /ingest   — Ingest a document by file path
    GET  /list     — List all ingested documents
    GET  /health   — Health check
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from doc_rag import store, rag, chunker
from doc_rag.config import TOP_K

app = FastAPI(
    title="doc-rag",
    description="RAG pipeline API — query your documents using natural language.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language question to ask")
    n_results: int = Field(default=TOP_K, ge=1, le=20, description="Number of chunks to retrieve")


class SourceResponse(BaseModel):
    text: str
    source: str
    page: Optional[int]
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    found_answer: bool
    sources: list[SourceResponse]


class IngestRequest(BaseModel):
    path: str = Field(..., description="Absolute path to the file or directory to ingest")


class IngestResponse(BaseModel):
    path: str
    chunks_added: int
    message: str


class DocumentInfo(BaseModel):
    source: str
    chunks: int


class ListResponse(BaseModel):
    documents: list[DocumentInfo]
    total_chunks: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """
    Query the document knowledge base with a natural language question.
    Returns the answer and the source chunks used to generate it.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        result = rag.ask(request.query, top_k=request.n_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {e}")

    return QueryResponse(
        question=result.question,
        answer=result.answer,
        found_answer=result.found_answer,
        sources=[
            SourceResponse(
                text=s.text,
                source=s.source,
                page=s.page,
                score=s.score,
            )
            for s in result.sources
        ],
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    """
    Ingest a file or directory into the document knowledge base.
    Supports PDF, Markdown, and plain text files.
    """
    path = Path(request.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {request.path}")

    try:
        files = [path] if path.is_file() else list(path.rglob("*"))
        files = [f for f in files if f.is_file() and f.suffix in {".pdf", ".md", ".txt"}]

        if not files:
            raise HTTPException(
                status_code=400,
                detail="No supported files found (pdf, md, txt)"
            )

        total_chunks = 0
        for file in files:
            chunks = chunker.load_and_chunk(file)
            store.add_chunks(chunks)
            total_chunks += len(chunks)

        return IngestResponse(
            path=str(path),
            chunks_added=total_chunks,
            message=f"Ingested {len(files)} file(s), {total_chunks} chunk(s) added",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest error: {e}")


@app.get("/list", response_model=ListResponse)
def list_documents() -> ListResponse:
    """List all documents currently in the knowledge base."""
    try:
        collection = store._get_collection()
        results = collection.get(include=["metadatas"])
        metadatas = results.get("metadatas") or []

        # Count chunks per source
        from collections import Counter
        counts = Counter(m.get("source", "unknown") for m in metadatas)

        return ListResponse(
            documents=[
                DocumentInfo(source=src, chunks=count)
                for src, count in sorted(counts.items())
            ],
            total_chunks=len(metadatas),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"List error: {e}")


@app.get("/health")
def health():
    """Health check — confirms the service and ChromaDB are available."""
    try:
        collection = store._get_collection()
        count = collection.count()
        return {
            "status": "ok",
            "chunks_in_store": count,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Store unavailable: {e}")
