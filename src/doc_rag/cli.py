"""Command-line interface for doc-rag."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from doc_rag import store, chunker
from doc_rag.rag import ask
from doc_rag.config import SUPPORTED_EXTENSIONS


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------

def cmd_ingest(args: argparse.Namespace) -> int:
    """Ingest a file or directory of documents."""
    path = Path(args.path).resolve()

    if not path.exists():
        print(f"ERROR: path not found: {path}", file=sys.stderr)
        return 1

    # Collect files to ingest
    if path.is_dir():
        files = [
            f for f in path.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not files:
            print(f"No supported files found in {path}")
            print(f"Supported types: {', '.join(SUPPORTED_EXTENSIONS)}")
            return 0
    else:
        files = [path]

    total_chunks = 0
    for file in files:
        print(f"Ingesting: {file.name}...", end=" ", flush=True)
        try:
            chunks = chunker.load_and_chunk(file)
            count = store.ingest(chunks, file.name)
            print(f"{count} chunks stored.")
            total_chunks += count
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)

    print(f"\nDone. {total_chunks} total chunks ingested from {len(files)} file(s).")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    """Ask a question of the ingested documents."""
    question = " ".join(args.question)

    print(f"\nQuestion: {question}")
    print("Searching...\n")

    try:
        response = ask(question, top_k=args.top_k)
    except EnvironmentError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"Answer: {response.answer}\n")

    if args.show_sources and response.sources:
        print("Sources:")
        seen = set()
        for chunk in response.sources:
            location = f"page {chunk.page}" if chunk.page else "document"
            key = f"{chunk.source}:{chunk.page}"
            if key not in seen:
                print(f"  - {chunk.source} ({location})  [similarity: {1 - chunk.score:.2f}]")
                seen.add(key)

    return 0 if response.found_answer else 1


def cmd_list(args: argparse.Namespace) -> int:
    """List all ingested documents."""
    sources = store.list_sources()

    if not sources:
        print("No documents ingested yet. Run `doc-rag ingest <path>` to get started.")
        return 0

    print(f"{'Document':<40}  {'Chunks':>6}")
    print("-" * 50)
    for s in sources:
        print(f"{s['source']:<40}  {s['chunks']:>6}")
    print(f"\n{len(sources)} document(s) total.")
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    """Delete a specific document from the store."""
    count = store.delete_source(args.source)
    if count == 0:
        print(f"Document not found: {args.source}")
        return 1
    print(f"Deleted {count} chunks for: {args.source}")
    return 0


def cmd_clear(args: argparse.Namespace) -> int:
    """Clear all documents from the store."""
    if not args.yes:
        confirm = input("This will delete ALL ingested documents. Type 'yes' to confirm: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            return 0

    count = store.clear_all()
    print(f"Cleared {count} chunks from the store.")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="doc-rag",
        description="RAG pipeline — ask questions of your own documents.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ingest
    ing = sub.add_parser("ingest", help="Ingest a file or folder of documents")
    ing.add_argument("path", help="File or directory to ingest (.pdf, .txt, .md)")

    # query
    q = sub.add_parser("query", help="Ask a question of the ingested documents")
    q.add_argument("question", nargs="+", help="Your question")
    q.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve (default: 5)")
    q.add_argument("--show-sources", action="store_true", help="Show source documents")

    # list
    sub.add_parser("list", help="List all ingested documents")

    # delete
    d = sub.add_parser("delete", help="Delete a specific document from the store")
    d.add_argument("source", help="Document filename to delete")

    # clear
    c = sub.add_parser("clear", help="Clear all documents from the store")
    c.add_argument("--yes", action="store_true", help="Skip confirmation prompt")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    handlers = {
        "ingest": cmd_ingest,
        "query": cmd_query,
        "list": cmd_list,
        "delete": cmd_delete,
        "clear": cmd_clear,
    }

    sys.exit(handlers[args.command](args))


if __name__ == "__main__":
    main()
