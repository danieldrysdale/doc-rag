"""RAG query pipeline — retrieves relevant chunks and generates an answer with Ollama."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass

from doc_rag.config import OLLAMA_MODEL, OLLAMA_URL, TOP_K
from doc_rag.store import query, RetrievedChunk


@dataclass
class RAGResponse:
    """The result of a RAG query."""
    question: str
    answer: str
    sources: list[RetrievedChunk]
    found_answer: bool    # False if the model couldn't find relevant info


SYSTEM_PROMPT = """You are a helpful document assistant. Answer questions using the context provided below.

Rules:
- Read ALL the context carefully before answering.
- If the context contains relevant information, use it to answer the question directly and specifically.
- Quote or closely paraphrase the relevant text from the context.
- Only say "I could not find an answer to that question in the provided documents." if the context contains absolutely nothing relevant.
- Never use knowledge outside the provided context.
- Keep answers concise but complete."""


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a context string."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        location = f"page {chunk.page}" if chunk.page else "document"
        parts.append(
            f"[Source {i}: {chunk.source}, {location}]\n{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


def ask(question: str, top_k: int = TOP_K) -> RAGResponse:
    """Run the full RAG pipeline: retrieve -> augment -> generate.

    1. Embed the question and retrieve the most relevant chunks
    2. Build a context string from the chunks
    3. Send context + question to Ollama
    4. Return a structured response with sources
    """
    # Step 1: Retrieve
    chunks = query(question, top_k=top_k)

    if not chunks:
        return RAGResponse(
            question=question,
            answer="No documents have been ingested yet. Run `doc-rag ingest <path>` first.",
            sources=[],
            found_answer=False,
        )

    # Step 2: Build context
    context = build_context(chunks)

    # Step 3: Generate with Ollama
    user_message = f"""Context from documents:

{context}

---

Question: {question}"""

    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        answer = data["message"]["content"]
    except Exception as exc:
        return RAGResponse(
            question=question,
            answer=f"Error contacting Ollama: {exc}\nIs `ollama serve` running?",
            sources=[],
            found_answer=False,
        )

    found_answer = "I could not find an answer" not in answer

    return RAGResponse(
        question=question,
        answer=answer,
        sources=chunks,
        found_answer=found_answer,
    )
