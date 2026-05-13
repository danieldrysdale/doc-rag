# doc-rag — Project Context

## What this is
A RAG (Retrieval Augmented Generation) pipeline that lets you query your own documents using natural language. Uses ChromaDB for vector storage, sentence-transformers for embeddings, and Ollama for local LLM generation.

## Stack
- Python 3.12
- ChromaDB — vector store
- sentence-transformers — embeddings
- Ollama — local LLM (llama3.2 by default)
- FastAPI + uvicorn — REST API (api.py)
- src layout with pyproject.toml

## Project structure
```
doc-rag/
├── api.py                  — FastAPI REST interface (added separately)
├── pyproject.toml          — Package config, dependencies
├── src/doc_rag/
│   ├── cli.py              — CLI entry point
│   ├── rag.py              — RAG query pipeline (ask() function)
│   ├── store.py            — ChromaDB operations
│   ├── chunker.py          — Document chunking
│   └── config.py           — Configuration
├── sample_docs/            — Sample documents for testing
└── tests/
    ├── test_chunker.py
    └── test_store.py
```

## Key function names (important — get these right)
- `chunker.load_and_chunk(path)` — NOT chunk_file
- `store.ingest(chunks, source_name)` — NOT add_chunks
- `store.query(question, top_k)` — returns list[RetrievedChunk]
- `store.list_sources()` — returns list[dict]
- `rag.ask(question, top_k)` — returns RAGResponse

## API endpoints (api.py)
- `POST /query` — query the knowledge base
- `POST /ingest` — ingest a file or directory path
- `GET /list` — list ingested documents
- `GET /health` — health check with chunk count

## Running locally
```bash
source venv/bin/activate
ollama serve  # required — Ollama must be running
uvicorn api:app --reload --port 8001
pip install -e ".[dev]"
pytest tests/ -v
```

## Docker notes
- Ollama runs on the HOST, not in the container
- Use `OLLAMA_URL=http://host.docker.internal:11434` when running in Docker
- Image is large (~3GB) due to sentence-transformers model weights
- ChromaDB data persists via Docker volume `doc-rag-data:/data`

## CI/CD
- Uses shared reusable workflow from `danieldrysdale/.github`
- Multi-platform Docker image (amd64 + arm64) to GHCR
- Image: `ghcr.io/danieldrysdale/doc-rag:latest`
- arm64 build takes ~17 minutes due to QEMU emulation of heavy ML dependencies

## Environment variables
- `OLLAMA_URL` — default: `http://localhost:11434`
- `OLLAMA_MODEL` — default: `llama3.2`

## Conventions
- Conventional Commits
- api.py lives at project root, not inside src/
- sample_docs/ mounted as volume in Docker Compose for testing
