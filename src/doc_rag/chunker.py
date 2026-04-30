"""Document loading and text chunking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pypdf

from doc_rag.config import CHUNK_SIZE, CHUNK_OVERLAP, SUPPORTED_EXTENSIONS


@dataclass
class Chunk:
    """A chunk of text extracted from a document."""
    text: str
    source: str        # filename
    page: int | None   # page number for PDFs, None for plain text
    chunk_index: int   # position within the document


def load_text(path: Path) -> list[tuple[str, int | None]]:
    """Load a document and return a list of (text, page_number) tuples.

    For PDFs each page is a separate entry. For text/markdown,
    the whole file is a single entry with page=None.
    """
    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: {SUPPORTED_EXTENSIONS}")

    if suffix == ".pdf":
        reader = pypdf.PdfReader(str(path))
        return [
            (page.extract_text() or "", i + 1)
            for i, page in enumerate(reader.pages)
        ]
    else:
        # .txt or .md
        text = path.read_text(encoding="utf-8", errors="replace")
        return [(text, None)]


def chunk_text(text: str, source: str, page: int | None, start_index: int = 0) -> list[Chunk]:
    """Split text into overlapping chunks.

    Uses character-based chunking with overlap to ensure context
    isn't lost at chunk boundaries.
    """
    chunks = []
    start = 0
    chunk_index = start_index

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                source=source,
                page=page,
                chunk_index=chunk_index,
            ))
            chunk_index += 1

        if end >= len(text):
            break

        # Move forward by chunk_size minus overlap
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def load_and_chunk(path: Path) -> list[Chunk]:
    """Load a document and return all chunks."""
    pages = load_text(path)
    all_chunks = []
    chunk_index = 0

    for text, page in pages:
        if not text.strip():
            continue
        chunks = chunk_text(text, path.name, page, start_index=chunk_index)
        all_chunks.extend(chunks)
        chunk_index += len(chunks)

    return all_chunks
