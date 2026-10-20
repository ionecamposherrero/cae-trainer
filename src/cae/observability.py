"""LangSmith tracing setup and trace annotations.

LangChain reads the tracing variables from ``os.environ``, not from ``Settings``,
so :func:`configure_tracing` must run once per process.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from cae.config import get_settings

logger = logging.getLogger("cae")

_LEGACY_TRACING_VAR = "LANGCHAIN_TRACING_V2"


def configure_tracing() -> bool:
    """Mirror the tracing settings into ``os.environ``; return whether it is on."""
    settings = get_settings()

    if not settings.has_tracing:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ[_LEGACY_TRACING_VAR] = "false"
        if settings.langsmith_tracing:
            logger.warning(
                "LANGSMITH_TRACING is on but LANGSMITH_API_KEY is empty, so tracing "
                "stays off. Add the key to .env to enable it."
            )
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ[_LEGACY_TRACING_VAR] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    if settings.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    logger.info("LangSmith tracing on, project %r", settings.langsmith_project)
    return True


def tracing_enabled() -> bool:
    """Whether :func:`configure_tracing` left tracing switched on."""
    return os.environ.get("LANGSMITH_TRACING", "").lower() == "true"


def note_degraded(node: str, exc: BaseException, **details: Any) -> None:
    """Log that ``node`` fell back to a degraded answer and tag the current run.

    Never raises.
    """
    logger.warning("%s degraded, falling back: %s: %s", node, type(exc).__name__, exc)
    try:
        from langsmith.run_helpers import get_current_run_tree

        run = get_current_run_tree()
        if run is None:
            return
        run.add_metadata(
            {
                "degraded": True,
                "degraded_node": node,
                "degraded_reason": f"{type(exc).__name__}: {exc}",
                **details,
            }
        )
    except Exception:  # pragma: no cover
        logger.debug("could not annotate the trace for %s", node, exc_info=True)


def call_config(
    *,
    label: str,
    tier: str,
    model: str,
    temperature: float,
    schema: str = "-",
    fallback_from: str = "",
) -> dict[str, Any]:
    """LangChain config that names one model call and tags it in the trace.

    ``model`` is the route that actually answered; ``fallback_from`` is the tier's
    first choice when a later route in the chain had to step in.
    """
    tags = [f"tier:{tier}", f"model:{model}"]
    metadata: dict[str, Any] = {
        "cae_label": label,
        "cae_tier": tier,
        "cae_model": model,
        "cae_schema": schema,
        "cae_temperature": temperature,
    }
    if fallback_from:
        tags.append("fallback")
        metadata["cae_fallback_from"] = fallback_from
    return {"run_name": label, "tags": tags, "metadata": metadata}


def graph_config(
    metadata: dict[str, Any] | None = None, **extra: Any
) -> dict[str, Any]:
    """Config for a graph invocation; LangGraph passes its metadata to child runs."""
    config: dict[str, Any] = dict(extra)
    merged = {k: v for k, v in (metadata or {}).items() if v is not None}
    if merged:
        config["metadata"] = merged
    return config


__all__ = [
    "call_config",
    "configure_tracing",
    "graph_config",
    "note_degraded",
    "tracing_enabled",
]
