"""Check that every model in every tier's fallback chain is actually usable.

Each route is called through the same client ``cae.llm`` builds for the app,
with the same kind of request the app makes: structured output for ``fast``,
``smart``, ``rue_task``, ``rue_evaluator``, ``writing_task``,
``writing_evaluator`` and ``coach``, a bound tool for ``chat``. A bare call is
not a useful check -- a model can answer one and still refuse the tools the
coach depends on.

    python scripts/check_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pydantic import BaseModel  # noqa: E402

from cae.config import PROVIDER_KEYS, RouteConfigError, get_settings  # noqa: E402

PURPOSE = {
    "fast": "no graph: get_llm() and model sweeps",
    "smart": "no graph: get_llm() and model sweeps",
    "chat": "coach chat: tool calls and its own SQL",
    "rue_task": "Reading exercise generation, moderation and repair",
    "rue_evaluator": "Reading marking: near-miss adjudication, error tagging",
    "writing_task": "Writing task generation and repair",
    "writing_evaluator": "Writing assessment: subscales, annotation, rewrite",
    "coach": "coach report: the narrative over the learner's figures",
}


class Ping(BaseModel):
    word: str


def main() -> int:
    settings = get_settings()
    print("=== Keys ===\n")
    for provider, (variable, url) in PROVIDER_KEYS.items():
        state = "set" if settings.has_key(provider) else f"not set  ({url})"
        print(f"  {variable:<16} {state}")

    ok = True
    for tier in (
        "fast",
        "smart",
        "rue_task",
        "rue_evaluator",
        "writing_task",
        "writing_evaluator",
        "coach",
        "chat",
    ):
        ok = check_tier(tier) and ok
    ok = check_embeddings() and ok

    print(
        "\nA tier works as long as one model in its chain is OK; the others are "
        "fallbacks.\nA spent allowance does not fail the check: the next model "
        "takes over until it resets."
    )
    return 0 if ok else 1


def check_tier(tier: str) -> bool:
    settings = get_settings()
    setting = settings.chain_setting(tier)  # type: ignore[arg-type]
    print(f"\n=== {tier} ({PURPOSE[tier]}) -- {setting} ===\n")
    try:
        routes = settings.routes(tier)  # type: ignore[arg-type]
    except RouteConfigError as exc:
        print(f"  {exc}")
        return False

    usable = 0
    for position, route in enumerate(routes, start=1):
        label = f"  {position}. {route.label:<44}"
        if not settings.has_key(route.provider):
            print(f"{label} SKIPPED (no {PROVIDER_KEYS[route.provider][0]})")
            continue
        working, verdict = _try_route(tier, route)
        print(f"{label} {verdict}")
        usable += working
    if not usable:
        print(f"\n  No model in {setting} is usable right now.")
    return usable > 0


def _try_route(tier: str, route) -> tuple[bool, str]:
    """Call one route the way the app does. Returns ``(usable, verdict)``."""
    from langchain_core.tools import tool

    from cae.llm import _with_schema, build_llm

    try:
        llm = build_llm(route, 0.3)
        if tier == "chat":

            @tool
            def part_score(part: int) -> str:
                """Return the learner's score for one part of the exam."""
                return "62%"

            reply = llm.bind_tools([part_score]).invoke(
                [
                    ("system", "You are a tutor. Use the tool when asked about a score."),
                    ("human", "What did I score on part 5?"),
                ]
            )
            if not reply.tool_calls:
                return True, "OK, but it did not call the tool -- the coach may not either"
            return True, "OK (tool calling works)"

        _with_schema(llm, route, Ping).invoke(
            [("human", "Reply with the single word ok in the field 'word'.")]
        )
    except Exception as exc:  # noqa: BLE001
        return _verdict(exc)
    return True, "OK (structured output works)"


def _verdict(exc: Exception) -> tuple[bool, str]:
    from cae.llm import _has_no_allowance, _is_daily_quota, _is_rate_limit, _is_transient

    message = str(exc)
    if "limit: 0" in message or _has_no_allowance(exc):
        return False, "UNUSABLE (no free-tier quota for this model on this account)"
    if "tier_not_allowed" in message or "not available in your subscription" in message:
        return False, "UNUSABLE (not included in this account's free plan)"
    if _is_daily_quota(message):
        return True, "works, but its free allowance is spent right now"
    if _is_rate_limit(message):
        return True, "works, but rate-limited right now"
    if _is_transient(message):
        return False, "BUSY (provider overloaded or unreachable right now; try again later)"
    if "reasoning_effort" in message:
        return False, (
            "REJECTED reasoning_effort="
            f"{get_settings().cae_groq_reasoning_effort!r} -- see "
            "CAE_GROQ_REASONING_EFFORT in .env"
        )
    if any(m in message for m in ("NOT_FOUND", "404", "does not exist", "model_not_found", "Invalid model")):
        return False, "UNUSABLE (no such model for this key)"
    if any(m in message for m in ("401", "invalid_api_key", "Unauthorized", "API_KEY_INVALID")):
        return False, "BAD KEY"
    if "tool" in message.lower() and "support" in message.lower():
        return False, "UNUSABLE (no tool calling)"
    return False, f"FAILED ({type(exc).__name__}: {message[:70]})"


def check_embeddings() -> bool:
    settings = get_settings()
    print("\n=== embeddings (reference corpus) -- CAE_EMBEDDING_MODEL ===\n")
    try:
        route = settings.embedding_route()
    except RouteConfigError as exc:
        print(f"  {exc}")
        return False
    label = f"  {route.label:<47}"
    if not settings.has_embedding_key:
        print(f"{label} SKIPPED (no {PROVIDER_KEYS[route.provider][0]})")
        print("  The coach still runs; it just cannot cite the reference material.")
        return True

    from cae.rag import store

    try:
        store.get_embeddings().embed_query("ok")
    except Exception as exc:  # noqa: BLE001
        print(f"{label} {_verdict(exc)[1]}")
        return False
    print(f"{label} OK")
    if store.index_exists():
        print("  index: built with this model")
    else:
        print("  index: missing or built with another model -- run scripts/build_index.py")
    return True


if __name__ == "__main__":
    raise SystemExit(main())
