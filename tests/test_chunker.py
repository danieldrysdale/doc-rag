"""Tests for document loading and chunking."""

import pytest
from pathlib import Path
from doc_rag.chunker import chunk_text, load_and_chunk, Chunk
from doc_rag.config import CHUNK_SIZE, CHUNK_OVERLAP


def test_chunk_text_basic():
    text = "A" * 1000
    chunks = chunk_text(text, "test.txt", None)
    assert len(chunks) > 1
    assert all(isinstance(c, Chunk) for c in chunks)


def test_chunk_text_overlap():
    text = "A" * CHUNK_SIZE + "B" * CHUNK_SIZE
    chunks = chunk_text(text, "test.txt", None)
    # With overlap, second chunk should start before end of first chunk's content
    assert len(chunks) >= 2


def test_chunk_text_small_document():
    text = "Short document."
    chunks = chunk_text(text, "test.txt", None)
    assert len(chunks) == 1
    assert chunks[0].text == "Short document."


def test_chunk_text_empty():
    chunks = chunk_text("", "test.txt", None)
    assert chunks == []


def test_chunk_preserves_source_and_page():
    text = "Some content here."
    chunks = chunk_text(text, "myfile.pdf", 3)
    assert chunks[0].source == "myfile.pdf"
    assert chunks[0].page == 3


def test_chunk_indices_are_sequential():
    text = "X" * 2000
    chunks = chunk_text(text, "test.txt", None, start_index=5)
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == 5 + i


def test_load_and_chunk_txt(tmp_path):
    doc = tmp_path / "test.txt"
    doc.write_text("Hello world. " * 100)
    chunks = load_and_chunk(doc)
    assert len(chunks) > 0
    assert all(c.source == "test.txt" for c in chunks)
    assert all(c.page is None for c in chunks)


def test_load_and_chunk_md(tmp_path):
    doc = tmp_path / "readme.md"
    doc.write_text("# Title\n\n" + "Some content. " * 50)
    chunks = load_and_chunk(doc)
    assert len(chunks) > 0


def test_load_and_chunk_unsupported(tmp_path):
    doc = tmp_path / "file.csv"
    doc.write_text("a,b,c")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_and_chunk(doc)
