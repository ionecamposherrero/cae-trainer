"""Chat-model factory and retrying call helpers shared by every graph.

Every tier is a fallback chain of free models (``CAE_FAST_MODEL`` and friends,
``provider:model`` separated by commas). A call goes to the first model in the
chain; when that one's allowance is spent, the provider is down or the reply
cannot be read, the next one answers instead. No graph knows which it got.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import Counter
from functools import lru_cache
from typing import Any, Callable, Sequence, TypeVar

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ValidationError
from tenacity import RetryCallState, Retrying, stop_after_attempt

from cae.config import (
    PROVIDER_KEYS,
    TIER_SETTINGS,
    Route,
    RouteConfigError,
    Tier,
    get_settings,
)
from cae.observability import call_config

logger = logging.getLogger("cae")

TModel = TypeVar("TModel", bound=BaseModel)
T = TypeVar("T")


class LLMUnavailableError(RuntimeError):
    """The model cannot be reached for a reason the user has to act on."""


class MissingAPIKeyError(LLMUnavailableError):
    """No model in the requested tier has a key, or the chain is misconfigured."""


class QuotaExhaustedError(LLMUnavailableError):
    """Raised when the free allowance is gone. Retrying will not help."""

    def __init__(self, model: str, tier: str = ""):
        self.model = model
        setting = (
            get_settings().chain_setting(tier)  # type: ignore[arg-type]
            if tier in TIER_SETTINGS
            else " / ".join(TIER_SETTINGS.values())
        )
        super().__init__(
            f"The free allowance for '{model}' is used up, and so is every other "
            f"model configured for this job. Free allowances reset daily. To carry "
            f"on now, add another provider's free key (GOOGLE_API_KEY, "
            f"MISTRAL_API_KEY, GROQ_API_KEY) or add more models to {setting}."
        )


class ModelUnreachableError(LLMUnavailableError):
    """The provider failed to answer, and the retries did not get through.

    Overload (503), a server fault (500), a gateway error, a per-minute limit that
    did not clear or a dropped connection. There is nothing for the learner to
    configure, so the message says only that it is temporary: the provider's own
    wording leaks model ids and status codes into the page, which the retry loop
    has already acted on.
    """

    def __init__(self, cause: str = ""):
        self.cause = cause
        super().__init__(
            "The model is busy or temporarily unavailable, so this could not be "
            "done just now. It normally clears within a minute or two — please "
            "try again."
        )


class _UnreadableOutput(Exception):
    """The model answered, but not in the shape asked for.

    Internal: it moves the call to the next model in the chain. If every model
    fails this way the original exception propagates, because at that point a
    schema bug on our side is likelier than every provider misbehaving at once.
    """

    def __init__(self, original: BaseException):
        self.original = original
        super().__init__(str(original))


EXHAUSTED_FOR = 3600.0
UNREACHABLE_FOR = 300.0
_exhausted: dict[Route, float] = {}
_unreachable: dict[Route, float] = {}
_usage: Counter[tuple[str, str]] = Counter()
_state_lock = threading.Lock()


def _mark(marks: dict[Route, float], route: Route, seconds: float) -> None:
    with _state_lock:
        marks[route] = time.monotonic() + seconds


def _is_marked(marks: dict[Route, float], route: Route) -> bool:
    with _state_lock:
        until = marks.get(route)
        if until is None:
            return False
        if time.monotonic() >= until:
            del marks[route]
            return False
        return True


def _mark_exhausted(route: Route) -> None:
    _mark(_exhausted, route, EXHAUSTED_FOR)


def _is_exhausted(route: Route) -> bool:
    return _is_marked(_exhausted, route)


def _mark_unreachable(route: Route) -> None:
    _mark(_unreachable, route, UNREACHABLE_FOR)


def _is_unreachable(route: Route) -> bool:
    return _is_marked(_unreachable, route)


def _record_usage(tier: str, route: Route) -> None:
    with _state_lock:
        _usage[(tier, route.label)] += 1


def route_usage() -> dict[tuple[str, str], int]:
    """How many calls each ``(tier, route)`` answered since the last reset."""
    with _state_lock:
        return dict(_usage)


def reset_route_usage() -> None:
    with _state_lock:
        _usage.clear()


def _missing_key_message(tier: Tier, configured: Sequence[Route]) -> str:
    wanted = []
    for provider in dict.fromkeys(route.provider for route in configured):
        variable, url = PROVIDER_KEYS[provider]
        wanted.append(f"{variable} ({url})")
    job = "the coach chat" if tier == "chat" else "exercises cannot be generated or marked"
    return (
        f"No key is set for any model in {get_settings().chain_setting(tier)}, so {job}. Copy "
        f".env.example to .env and add at least one of: {'; '.join(wanted)}. All "
        "of them are free."
    )


def configured_routes(tier: Tier) -> list[Route]:
    """The tier's chain as configured, whether or not the keys are there."""
    try:
        return get_settings().routes(tier)
    except RouteConfigError as exc:
        raise MissingAPIKeyError(str(exc)) from exc


def routes_for(tier: Tier) -> list[Route]:
    """The routes a call will try, in order: keyed, not spent, not known to be down.

    A spent route is a hard skip. A down one is soft: if every route left is
    marked down, they are all tried again rather than giving up without a call.
    """
    settings = get_settings()
    configured = configured_routes(tier)
    keyed = [route for route in configured if settings.has_key(route.provider)]
    if not keyed:
        raise MissingAPIKeyError(_missing_key_message(tier, configured))
    unspent = [route for route in keyed if not _is_exhausted(route)]
    if not unspent:
        raise QuotaExhaustedError(keyed[0].label, tier)
    healthy = [route for route in unspent if not _is_unreachable(route)]
    return healthy or unspent


def model_for(tier: Tier) -> str:
    """The route a call on this tier would try first, without building a client."""
    try:
        return routes_for(tier)[0].label
    except LLMUnavailableError:
        try:
            configured = get_settings().routes(tier)
        except RouteConfigError:
            return "-"
        return configured[0].label if configured else "-"


REQUEST_TIMEOUT = 120.0


@lru_cache(maxsize=1)
def _mistral_limiter():
    """One limiter for every Mistral route: the free plan's limit is per key."""
    from langchain_core.rate_limiters import InMemoryRateLimiter

    rps = get_settings().cae_mistral_requests_per_second
    return InMemoryRateLimiter(requests_per_second=rps, check_every_n_seconds=0.1)


@lru_cache(maxsize=32)
def build_llm(route: Route, temperature: float = 0.7) -> BaseChatModel:
    """The client for one route. Retries are left to this module, not the SDK."""
    settings = get_settings()
    key = settings.key_for(route.provider)
    if not key:
        variable, url = PROVIDER_KEYS[route.provider]
        raise MissingAPIKeyError(f"{variable} is not set. Get a free key from {url}")

    if route.provider == "mistral":
        from langchain_mistralai import ChatMistralAI

        return ChatMistralAI(
            model=route.model,
            api_key=key,
            temperature=temperature,
            rate_limiter=_mistral_limiter(),
            max_retries=1,
            timeout=int(REQUEST_TIMEOUT),
        )

    if route.provider == "groq":
        from langchain_groq import ChatGroq

        extra: dict[str, Any] = {}
        effort = settings.cae_groq_reasoning_effort.strip()
        if effort:
            extra["reasoning_effort"] = effort
        return ChatGroq(
            model=route.model,
            api_key=key,
            temperature=temperature,
            max_retries=0,
            timeout=REQUEST_TIMEOUT,
            **extra,
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=route.model,
        google_api_key=key,
        temperature=temperature,
        max_retries=1,
        timeout=REQUEST_TIMEOUT,
    )


def get_llm(tier: Tier = "fast", temperature: float = 0.7) -> BaseChatModel:
    """The client for the first live route of a tier."""
    return build_llm(routes_for(tier)[0], temperature)


def clear_caches() -> None:
    """Forget built clients and spent routes, e.g. after changing the settings."""
    build_llm.cache_clear()
    _mistral_limiter.cache_clear()
    with _state_lock:
        _exhausted.clear()
        _unreachable.clear()


_RETRY_DELAY_RE = re.compile(
    r"(?:[Rr]etry|try again) in (?:(\d+)m)?(\d+(?:\.\d+)?)s"
)
MAX_RATE_LIMIT_WAIT = 70.0
FALLBACK_WAIT = 8.0


def _is_rate_limit(message: str) -> bool:
    return "RESOURCE_EXHAUSTED" in message or "429" in message


def _asked_delay(message: str) -> float | None:
    match = _RETRY_DELAY_RE.search(message)
    if not match:
        return None
    minutes = float(match.group(1) or 0)
    return minutes * 60 + float(match.group(2))


def _is_daily_quota(message: str) -> bool:
    """A spent allowance, which retrying cannot fix, as opposed to a rate limit.

    Gemini reports ``PerDay`` (or ``limit: 0`` for a model with no free tier);
    Groq names the window, ``tokens per day (TPD)`` or ``requests per day
    (RPD)``; Mistral's free plan runs out by the month.
    """
    if "insufficient_quota" in message:
        return True
    lowered = message.lower()
    if not _is_rate_limit(message) and "limit" not in lowered:
        return False
    return (
        "PerDay" in message
        or "per_day" in lowered
        or "per day" in lowered
        or "(tpd)" in lowered
        or "(rpd)" in lowered
        or "monthly" in lowered
        or "limit: 0" in message
    )


_TRANSIENT_MARKERS: tuple[str, ...] = (
    "500",
    "502",
    "503",
    "504",
    "UNAVAILABLE",
    "INTERNAL",
    "overloaded",
    "DEADLINE_EXCEEDED",
    "Timeout",
    "timed out",
    "Connection",
    "ServerError",
    "ServiceUnavailable",
    "capacity exceeded",
)

_OUTPUT_MARKERS: tuple[str, ...] = (
    "tool_use_failed",
    "json_validate_failed",
    "output_parse_failed",
    "Failed to call a function",
)


def _is_transient(message: str) -> bool:
    """A provider or transport failure, as opposed to a bad request from us."""
    return any(marker in message for marker in _TRANSIENT_MARKERS)


def _has_no_allowance(exc: BaseException) -> bool:
    """A model this account may not use at all, whatever the wording says.

    Mistral answers a model with no free quota with the same "Rate limit
    exceeded" as a busy minute; only the header tells them apart (a limit of 0).
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    return str(headers.get("x-ratelimit-limit-req-minute", "")).strip() == "0"


def _is_timeout(message: str) -> bool:
    lowered = message.lower()
    return "timed out" in lowered or "timeout" in lowered or "deadline_exceeded" in lowered


def _is_unreadable_output(exc: BaseException) -> bool:
    if isinstance(exc, (OutputParserException, ValidationError)):
        return True
    message = str(exc)
    return any(marker in message for marker in _OUTPUT_MARKERS)


def _should_retry(state: RetryCallState) -> bool:
    exception = state.outcome.exception() if state.outcome else None
    if exception is None or isinstance(exception, _UnreadableOutput):
        return False
    if _is_unreadable_output(exception) or _has_no_allowance(exception):
        return False
    return not _is_daily_quota(str(exception))


def _should_retry_before_fallback(state: RetryCallState) -> bool:
    """With another route waiting, only a short hiccup is worth waiting out.

    A request that timed out has already cost ``REQUEST_TIMEOUT``; a second one
    would double that before the next model is even asked.
    """
    if not _should_retry(state):
        return False
    message = str(state.outcome.exception())
    if _is_timeout(message):
        return False
    if not _is_rate_limit(message):
        return True
    asked = _asked_delay(message)
    return asked is not None and asked <= FALLBACK_WAIT


def _wait_for_quota(state: RetryCallState) -> float:
    """Exponential backoff, extended to the delay a rate limit asks for."""
    exponential = min(20.0, 2.0 * 2 ** max(0, state.attempt_number - 1))
    exception = state.outcome.exception() if state.outcome else None
    if exception is None:
        return exponential

    message = str(exception)
    if not _is_rate_limit(message):
        return exponential
    asked = _asked_delay(message)
    wanted = asked + 2.0 if asked is not None else 30.0
    return min(MAX_RATE_LIMIT_WAIT, max(exponential, wanted))


def _wait_before_fallback(state: RetryCallState) -> float:
    return min(FALLBACK_WAIT, _wait_for_quota(state))


def _retrying(last: bool) -> Retrying:
    """Per-route retries: patient on the last route, brief when one is left."""
    if last:
        return Retrying(
            stop=stop_after_attempt(4),
            wait=_wait_for_quota,
            retry=_should_retry,
            reraise=True,
        )
    return Retrying(
        stop=stop_after_attempt(2),
        wait=_wait_before_fallback,
        retry=_should_retry_before_fallback,
        reraise=True,
    )


def _translating_failures(model: str, call: Callable[[], T]) -> T:
    """Run ``call``, turning provider failures into :class:`LLMUnavailableError`.

    A spent allowance becomes :class:`QuotaExhaustedError`; an overloaded or
    unreachable provider, or a rate limit that did not clear, becomes
    :class:`ModelUnreachableError`. Both are what the pages catch, so neither
    reaches the learner as a traceback. A reply in the wrong shape becomes
    :class:`_UnreadableOutput`, which moves the call along the chain. Anything
    else is a bug on our side — a malformed request, a schema the backend
    rejects — and must keep propagating rather than hide behind a "try again"
    the learner would follow forever.
    """
    try:
        return call()
    except (LLMUnavailableError, _UnreadableOutput):
        raise
    except Exception as exc:
        message = str(exc)
        if _is_daily_quota(message) or _has_no_allowance(exc):
            raise QuotaExhaustedError(model) from exc
        if _is_unreadable_output(exc):
            raise _UnreadableOutput(exc) from exc
        if _is_transient(message) or _is_rate_limit(message):
            raise ModelUnreachableError(message) from exc
        raise


def _along_the_chain(
    tier: Tier,
    attempt: Callable[[Route, dict], T],
    *,
    label: str,
    schema: str,
    temperature: float,
) -> T:
    """Try each live route of ``tier`` in turn until one answers.

    A spent allowance sets its route aside for an hour and moves on; an
    unreachable provider sets it aside for a few minutes; an unreadable reply
    moves on without marking. Anything else propagates at once.
    """
    routes = routes_for(tier)
    first = routes[0]
    errors: list[BaseException] = []
    for index, route in enumerate(routes):
        config = call_config(
            label=label,
            tier=tier,
            model=route.label,
            schema=schema,
            temperature=temperature,
            fallback_from=first.label if route != first else "",
        )
        retrying = _retrying(last=index == len(routes) - 1)
        try:
            result = _translating_failures(
                route.label, lambda: retrying(attempt, route, config)
            )
        except QuotaExhaustedError as exc:
            _mark_exhausted(route)
            errors.append(exc)
        except ModelUnreachableError as exc:
            _mark_unreachable(route)
            errors.append(exc)
        except (MissingAPIKeyError, _UnreadableOutput) as exc:
            errors.append(exc)
        else:
            _record_usage(tier, route)
            return result
        logger.warning(
            "%s: %s failed (%s)%s",
            label,
            route.label,
            type(errors[-1]).__name__,
            ", trying the next model" if index < len(routes) - 1 else "",
        )

    last = errors[-1]
    if isinstance(last, _UnreadableOutput):
        raise last.original
    if all(isinstance(error, QuotaExhaustedError) for error in errors):
        raise QuotaExhaustedError(first.label, tier) from last
    raise last


def _with_schema(llm: BaseChatModel, route: Route, schema: type[TModel]):
    """Bind ``schema`` the way each provider actually enforces it.

    Mistral and Groq honour a function-calling schema only loosely: a model can
    invent error categories outside the closed taxonomy or drop required
    fields. Their strict ``json_schema``
    mode is constrained decoding, so neither can happen. Gemini's default
    already enforces the schema.
    """
    if route.provider == "mistral":
        return llm.with_structured_output(schema, method="json_schema")
    if route.provider == "groq":
        return llm.with_structured_output(schema, method="json_schema", strict=True)
    return llm.with_structured_output(schema)


def _invoke(
    route: Route,
    schema: type[TModel],
    system: str,
    user: str,
    temperature: float,
    config: dict | None = None,
) -> TModel:
    llm = _with_schema(build_llm(route, temperature), route, schema)
    result = llm.invoke([("system", system), ("human", user)], config=config)
    if isinstance(result, schema):
        return result
    return schema.model_validate(result)


def _invoke_chat(
    route: Route,
    messages: Sequence[Any],
    tools: Sequence[BaseTool],
    temperature: float,
    config: dict | None = None,
) -> AIMessage:
    llm = build_llm(route, temperature)
    if tools:
        llm = llm.bind_tools(tools)
    return llm.invoke(list(messages), config=config)


def structured_call(
    schema: type[TModel],
    system: str,
    user: str,
    *,
    tier: Tier = "fast",
    temperature: float = 0.7,
    label: str = "",
) -> TModel:
    """Call the tier's model chain and parse the reply into ``schema``.

    Transient errors and rate limits are retried, then passed along the chain; a
    spent allowance on every route raises :class:`QuotaExhaustedError` and an
    unreachable last route :class:`ModelUnreachableError`, both of which the pages
    already catch. ``label`` names the call in the trace, by convention
    ``<graph>.<node>``.
    """
    return _along_the_chain(
        tier,
        lambda route, config: _invoke(route, schema, system, user, temperature, config),
        label=label or schema.__name__,
        schema=schema.__name__,
        temperature=temperature,
    )


def chat_call(
    messages: Sequence[Any],
    *,
    tools: Sequence[BaseTool] = (),
    tier: Tier = "chat",
    temperature: float = 0.3,
    label: str = "",
) -> AIMessage:
    """Call the chat chain with a message history and return the raw ``AIMessage``.

    Tool calls are left on the reply so the calling graph can run them and loop.
    Retries, fallback and quota handling are shared with :func:`structured_call`.
    """
    return _along_the_chain(
        tier,
        lambda route, config: _invoke_chat(route, messages, tools, temperature, config),
        label=label or "chat",
        schema="-",
        temperature=temperature,
    )
