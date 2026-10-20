"""The persisted Chroma collection.

A missing index is a supported state: :func:`search` returns ``[]`` instead of
raising. So is an index built with a different embedding model than the one now
configured: its vectors are not comparable with the query's, so it is treated as
missing until ``scripts/build_index.py`` rebuilds it.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from cae.config import INDEX_DIR, PROVIDER_KEYS, get_settings

COLLECTION = "cae-reference"
MODEL_KEY = "cae_embedding_model"

DEFAULT_K = 4


def get_embeddings():
    """The embedding client, shared by indexing and search so both use one model.

    There is deliberately no fallback chain here: a second embedding model would
    search the index in a different vector space and return plausible nonsense.
    """
    from cae.llm import MissingAPIKeyError

    settings = get_settings()
    route = settings.embedding_route()
    key = settings.key_for(route.provider)
    if not key:
        variable, url = PROVIDER_KEYS[route.provider]
        raise MissingAPIKeyError(
            f"{variable} is not set, so the reference corpus cannot be indexed or "
            f"searched. Get a free key from {url}"
        )
    if route.provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=route.model, google_api_key=key)
    if route.provider == "mistral":
        from langchain_mistralai import MistralAIEmbeddings

        return MistralAIEmbeddings(model=route.model, api_key=key)
    raise MissingAPIKeyError(
        f"CAE_EMBEDDING_MODEL names '{route.label}', but {route.provider} serves no "
        "embedding models. Use google:gemini-embedding-001 or mistral:mistral-embed."
    )


def embedding_label(embeddings=None) -> str:
    """The name an index built with ``embeddings`` is recorded under."""
    if embeddings is not None:
        return f"custom:{type(embeddings).__name__}"
    return get_settings().embedding_route().label


@lru_cache(maxsize=1)
def _cached_store():
    return _open_store(None)


def _open_store(embeddings, *, metadata: dict[str, str] | None = None):
    from langchain_chroma import Chroma

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=str(INDEX_DIR),
        collection_metadata=metadata,
    )


def get_store(*, embeddings=None):
    """Open the persisted collection. Tests pass a fake ``embeddings``."""
    if embeddings is None:
        return _cached_store()
    return _open_store(embeddings)


def reset_collection(*, embeddings=None):
    """Drop and reopen the collection, recording the model that will fill it."""
    store = get_store(embeddings=embeddings)
    try:
        store.delete_collection()
    except Exception:
        pass
    _cached_store.cache_clear()
    return _open_store(embeddings, metadata={MODEL_KEY: embedding_label(embeddings)})


def indexed_model(*, embeddings=None) -> str:
    """The embedding model the index was built with; ``""`` if not recorded."""
    store = get_store(embeddings=embeddings)
    return str((store._collection.metadata or {}).get(MODEL_KEY, ""))


def index_matches(*, embeddings=None) -> bool:
    """Whether the index was built with the embedding model now configured."""
    return indexed_model(embeddings=embeddings) == embedding_label(embeddings)


def index_exists(*, embeddings=None) -> bool:
    """Whether a usable index is there: non-empty and the same model. Never raises."""
    try:
        return count(embeddings=embeddings) > 0 and index_matches(embeddings=embeddings)
    except Exception:
        return False


def count(*, embeddings=None) -> int:
    """How many chunks the collection holds."""
    store = get_store(embeddings=embeddings)
    return int(store._collection.count())


def search(query: str, k: int = DEFAULT_K, *, embeddings=None) -> list[dict[str, Any]]:
    """The ``k`` most relevant passages with their citations.

    Returns ``[]`` when there is no index, no key, no match, or an index built with
    another embedding model.
    """
    if not query.strip():
        return []
    try:
        if not index_matches(embeddings=embeddings):
            return []
        store = get_store(embeddings=embeddings)
        hits = store.similarity_search(query, k=k)
    except Exception:
        return []
    return [
        {
            "text": hit.page_content,
            "source": hit.metadata.get("source", ""),
            "section": hit.metadata.get("section", ""),
            "citation": hit.metadata.get("citation", "") or hit.metadata.get("source", ""),
        }
        for hit in hits
    ]


__all__ = [
    "COLLECTION",
    "DEFAULT_K",
    "MODEL_KEY",
    "count",
    "embedding_label",
    "get_embeddings",
    "get_store",
    "index_exists",
    "index_matches",
    "indexed_model",
    "reset_collection",
    "search",
]
