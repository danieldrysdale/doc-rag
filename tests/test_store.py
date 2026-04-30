"""Tests for the ChromaDB vector store."""

import pytest
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from doc_rag.chunker import Chunk
from doc_rag import store


class _FakeEF(EmbeddingFunction[Documents]):
    """Deterministic fake embeddings -- no network needed."""

    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        return [
            [float(hash(text + str(i)) % 1000) / 1000.0 for i in range(384)]
            for text in input
        ]

    @classmethod
    def name(cls) -> str:
        return "fake-ef"

    @classmethod
    def build_from_config(cls, config):
        return cls()

    def get_config(self):
        return {}


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Redirect ChromaDB to a temp directory and inject fake embeddings."""
    import doc_rag.store as store_module

    test_chroma_dir = tmp_path / "chroma"

    def _get_test_collection():
        test_chroma_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(test_chroma_dir))
        return client.get_or_create_collection(
            name="documents",
            embedding_function=_FakeEF(),
            metadata={"hnsw:space": "cosine"},
        )

    monkeypatch.setattr(store_module, "_get_collection", _get_test_collection)
    yield


def make_chunks(texts, source="test.txt"):
    return [
        Chunk(text=t, source=source, page=i + 1, chunk_index=i)
        for i, t in enumerate(texts)
    ]


def test_ingest_returns_chunk_count():
    chunks = make_chunks(["Hello world", "Second chunk", "Third chunk"])
    count = store.ingest(chunks, "test.txt")
    assert count == 3


def test_ingest_empty_returns_zero():
    assert store.ingest([], "test.txt") == 0


def test_ingest_is_idempotent():
    chunks = make_chunks(["Some content here"])
    store.ingest(chunks, "test.txt")
    count = store.ingest(chunks, "test.txt")
    assert count == 1


def test_query_returns_results():
    chunks = make_chunks([
        "The refund policy allows returns within 30 days.",
        "Employees get 20 days of annual leave.",
        "Contact support at support@example.com.",
    ])
    store.ingest(chunks, "test.txt")
    results = store.query("What is the refund policy?")
    assert len(results) > 0


def test_query_empty_store_returns_empty():
    results = store.query("anything")
    assert results == []


def test_list_sources_empty():
    assert store.list_sources() == []


def test_list_sources_after_ingest():
    chunks = make_chunks(["Content A", "Content B"], source="doc_a.txt")
    store.ingest(chunks, "doc_a.txt")
    sources = store.list_sources()
    assert len(sources) == 1
    assert sources[0]["source"] == "doc_a.txt"
    assert sources[0]["chunks"] == 2


def test_delete_source():
    chunks = make_chunks(["Delete me"], source="to_delete.txt")
    store.ingest(chunks, "to_delete.txt")
    count = store.delete_source("to_delete.txt")
    assert count == 1
    assert store.list_sources() == []


def test_delete_nonexistent_source():
    count = store.delete_source("doesnt_exist.txt")
    assert count == 0


def test_clear_all():
    store.ingest(make_chunks(["A"], "a.txt"), "a.txt")
    store.ingest(make_chunks(["B"], "b.txt"), "b.txt")
    count = store.clear_all()
    assert count == 2
    assert store.list_sources() == []
