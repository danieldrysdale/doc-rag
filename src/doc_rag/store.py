"""ChromaDB vector store — embedding, storage, and retrieval."""

from __future__ import annotations

from dataclasses import dataclass

import chromadb
from chromadb.utils import embedding_functions

from doc_rag.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K,
)
from doc_rag.chunker import Chunk


@dataclass
class RetrievedChunk:
    """A chunk retrieved from the vector store with its similarity score."""
    text: str
    source: str
    page: int | None
    score: float       # cosine distance (lower = more similar)


def _get_collection() -> chromadb.Collection:
    """Return (or create) the ChromaDB collection."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def ingest(chunks: list[Chunk], source_name: str) -> int:
    """Embed and store chunks in ChromaDB.

    Returns the number of chunks stored.
    Skips chunks that are already stored (idempotent).
    """
    if not chunks:
        return 0

    collection = _get_collection()

    ids = [f"{source_name}::{c.chunk_index}" for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [
        {
            "source": c.source,
            "page": str(c.page) if c.page is not None else "",
            "chunk_index": str(c.chunk_index),
        }
        for c in chunks
    ]

    # Upsert — safe to re-ingest the same document
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(chunks)


def query(question: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
    """Find the most relevant chunks for a question."""
    collection = _get_collection()

    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[question],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        page = int(meta["page"]) if meta.get("page") else None
        chunks.append(RetrievedChunk(
            text=doc,
            source=meta["source"],
            page=page,
            score=dist,
        ))

    return chunks


def list_sources() -> list[dict]:
    """Return a list of ingested documents with chunk counts."""
    collection = _get_collection()

    if collection.count() == 0:
        return []

    # Get all metadata and aggregate by source
    results = collection.get(include=["metadatas"])
    source_counts: dict[str, int] = {}
    for meta in results["metadatas"]:
        source = meta.get("source", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1

    return [
        {"source": source, "chunks": count}
        for source, count in sorted(source_counts.items())
    ]


def delete_source(source_name: str) -> int:
    """Delete all chunks for a given source document.

    Returns the number of chunks deleted.
    """
    collection = _get_collection()
    results = collection.get(
        where={"source": source_name},
        include=["metadatas"],
    )
    if not results["ids"]:
        return 0
    collection.delete(ids=results["ids"])
    return len(results["ids"])


def clear_all() -> int:
    """Delete all documents from the store.

    Returns the number of chunks deleted.
    """
    collection = _get_collection()
    count = collection.count()
    if count > 0:
        all_ids = collection.get(include=[])["ids"]
        collection.delete(ids=all_ids)
    return count
