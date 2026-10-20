"""Build the reference index the coach chat cites from.

    python scripts/build_index.py            # rebuild from data/reference/
    python scripts/build_index.py --dry-run  # chunk it, without embedding
    python scripts/build_index.py --check    # what is indexed right now

Source documents go in ``data/reference/`` as Markdown or plain text. The index
is always rebuilt from scratch, and records the embedding model that built it:
change ``CAE_EMBEDDING_MODEL`` and the old index is ignored until this runs again.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cae.config import PROVIDER_KEYS, REFERENCE_DIR, get_settings  # noqa: E402
from cae.rag import ingest, store  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="chunk the corpus and count the tokens without calling the API",
    )
    parser.add_argument(
        "--check", action="store_true", help="report what is already indexed"
    )
    args = parser.parse_args()

    if args.check:
        return report_existing()

    chunks = ingest.load_corpus()
    if not chunks:
        print(
            f"No documents found in {REFERENCE_DIR}.\n"
            "Add the reference material there as .md or .txt files -- see the "
            "README in that directory -- and run this again.\n"
            "The chat still works without it; it just cannot cite sources."
        )
        return 1

    documents = len({c.source for c in chunks})
    tokens = ingest.estimated_tokens(chunks)
    print(f"{documents} document(s) -> {len(chunks)} chunks, ~{tokens:,} tokens")
    print(f"embedding model: {store.embedding_label()} (free tier)")

    if args.dry_run:
        print("\n--dry-run: nothing was embedded.")
        _preview(chunks)
        return 0

    settings = get_settings()
    if not settings.has_embedding_key:
        variable, url = PROVIDER_KEYS[settings.embedding_route().provider]
        print(
            f"\n{variable} is not set, so the corpus cannot be embedded.\n"
            f"Get a free key from {url}, add it to .env and run this again."
        )
        return 1

    print("\nembedding and persisting...")
    report = ingest.build_index()
    print(f"indexed {report['chunks']} chunks from {report['documents']} document(s)")
    print(f"collection '{store.COLLECTION}' now holds {store.count()} chunks")
    return 0


def report_existing() -> int:
    try:
        chunks = store.count()
        built_with = store.indexed_model() or "(not recorded)"
    except Exception:  # noqa: BLE001
        chunks, built_with = 0, ""
    if not chunks:
        print(
            "Nothing is indexed yet. Run `python scripts/build_index.py` once the "
            f"documents are in {REFERENCE_DIR}."
        )
        return 1
    print(f"collection '{store.COLLECTION}' holds {chunks} chunks")
    print(f"built with  {built_with}")
    if not store.index_matches():
        print(
            f"configured  {store.embedding_label()}\n"
            "The index was built with another embedding model, so the coach will "
            "ignore it. Run `python scripts/build_index.py` to rebuild it."
        )
        return 1
    return 0


def _preview(chunks, limit: int = 3) -> None:
    """Show the first few chunks, so a bad split is visible before embedding."""
    print("\nfirst chunks:")
    for chunk in chunks[:limit]:
        head = chunk.text[:120].replace("\n", " ")
        print(f"  [{chunk.citation}] {head}...")


if __name__ == "__main__":
    raise SystemExit(main())
