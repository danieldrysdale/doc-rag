"""Configuration for doc-rag."""

from __future__ import annotations

import os
from pathlib import Path

# Embedding model -- runs locally, no API key needed
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ChromaDB persistence directory
CHROMA_DIR = Path(os.environ.get("DOC_RAG_CHROMA_DIR", Path.home() / ".doc-rag" / "chroma"))

# Collection name inside ChromaDB
COLLECTION_NAME = "documents"

# Chunking parameters
CHUNK_SIZE = 500        # characters per chunk
CHUNK_OVERLAP = 100     # overlap between consecutive chunks

# RAG query parameters
TOP_K = 5               # number of chunks to retrieve per query

# Ollama settings
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

# Supported file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
