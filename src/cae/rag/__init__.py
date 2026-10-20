"""Vector retrieval over the reference corpus the coach chat cites from.

* :mod:`cae.rag.ingest` builds the index from ``data/reference/``.
* :mod:`cae.rag.store` opens it and answers :func:`~cae.rag.store.search`.
"""

from __future__ import annotations

from cae.rag.ingest import Chunk, load_corpus, split_document
from cae.rag.store import COLLECTION, index_exists, search

__all__ = [
    "COLLECTION",
    "Chunk",
    "index_exists",
    "load_corpus",
    "search",
    "split_document",
]
