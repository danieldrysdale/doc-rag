# doc-rag

A RAG (Retrieval-Augmented Generation) pipeline that lets you ask natural language questions of your own documents, powered by Claude and ChromaDB.

Drop in PDFs, text files, or markdown — ask questions in plain English — get answers grounded in your documents with source citations.

## How it works

1. **Ingest** — documents are split into overlapping chunks and embedded locally using `all-MiniLM-L6-v2`
2. **Store** — embeddings are persisted in a local ChromaDB vector database
3. **Query** — your question is embedded, the most relevant chunks are retrieved, and Claude answers based only on those chunks
4. **Answer** — response includes source citations; Claude says when it can't find an answer rather than hallucinating

## Features

- PDF, plain text, and markdown support
- Overlapping chunk strategy preserves context at boundaries
- Source citations in every answer
- Confidence-aware — Claude explicitly says when the answer isn't in the documents
- Persistent vector store — ingest once, query many times
- 19 pytest tests, no network required (fake embeddings in tests)

## Project structure

```
doc-rag/
├── src/doc_rag/
│   ├── __init__.py
│   ├── cli.py          # argparse CLI
│   ├── config.py       # configuration and constants
│   ├── chunker.py      # document loading and text chunking
│   ├── store.py        # ChromaDB vector store
│   └── rag.py          # RAG pipeline (retrieve + generate with Claude)
├── tests/
│   ├── test_chunker.py
│   └── test_store.py
├── sample_docs/
│   ├── refund_policy.md
│   └── employee_handbook.md
└── pyproject.toml
```

## Installation

```bash
git clone https://github.com/danieldrysdale/doc-rag
cd doc-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Setup

You need an Anthropic API key:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Add to `~/.zshrc` or `~/.bashrc` to persist across sessions.

## Usage

### Ingest documents

```bash
# Ingest a single file
doc-rag ingest sample_docs/refund_policy.md

# Ingest an entire folder
doc-rag ingest ./my_documents/
```

### Ask questions

```bash
doc-rag query "What is the refund policy for digital downloads?"
doc-rag query "How many days of annual leave do employees get?"

# Show source documents
doc-rag query --show-sources "What happens if I receive a damaged item?"

# Retrieve more context chunks
doc-rag query --top-k 8 "Explain the parental leave policy"
```

### Manage documents

```bash
# List all ingested documents
doc-rag list

# Delete a specific document
doc-rag delete refund_policy.md

# Clear everything
doc-rag clear
```

## Example session

```
$ doc-rag ingest sample_docs/
Ingesting: refund_policy.md... 4 chunks stored.
Ingesting: employee_handbook.md... 7 chunks stored.

Done. 11 total chunks ingested from 2 file(s).

$ doc-rag query --show-sources "Can I return software I've already activated?"

Question: Can I return software I've already activated?
Searching...

Answer: No. According to the refund policy, software licenses and digital
downloads are non-refundable once activated.

Sources:
  - refund_policy.md (document)  [similarity: 0.91]
```

## Running tests

Tests use fake embeddings — no API keys or network access required:

```bash
pytest -v
```

## Configuration

Key settings in `src/doc_rag/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model |
| `CHUNK_SIZE` | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | 100 | Overlap between chunks |
| `TOP_K` | 5 | Chunks retrieved per query |
| `CLAUDE_MODEL` | `claude-haiku-4-5-20251001` | Claude model for answers |

Override the storage directory:

```bash
export DOC_RAG_CHROMA_DIR=/path/to/custom/store
```
