"""Rate limits versus exhausted allowances, and provider failures.

Fallback along a chain is covered in ``test_llm_routing.py``.
"""

from __future__ import annotations

import pytest

from cae import llm

PER_MINUTE = (
    "429 RESOURCE_EXHAUSTED. Quota exceeded for metric: generate_content_free_tier_requests, "
    "'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier'. Please retry in 53.07s."
)
PER_DAY = (
    "429 RESOURCE_EXHAUSTED. Quota exceeded for metric: generate_content_free_tier_requests, "
    "limit: 20, 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'. "
    "Please retry in 53.07s."
)
NO_FREE_TIER = (
    "429 RESOURCE_EXHAUSTED. Quota exceeded for metric: "
    "generate_content_free_tier_input_token_count, limit: 0, model: gemini-3.1-pro"
)
NO_BALANCE = (
    "Error code: 429 - {'error': {'message': 'You exceeded your current quota, please "
    "check your plan and billing details.', 'type': 'insufficient_quota', 'code': "
    "'insufficient_quota'}}"
)
TRANSIENT = "503 UNAVAILABLE. This model is currently experiencing high demand."
REAL_503 = (
    "google.genai.errors.ServerError: 503 UNAVAILABLE. {'error': {'code': 503, "
    "'message': 'This model is currently experiencing high demand. Spikes in demand "
    "are usually temporary. Please try again later.', 'status': 'UNAVAILABLE'}}"
)


class _State:
    """The slice of tenacity's RetryCallState these helpers actually read."""

    def __init__(self, exception: Exception | None, attempt: int = 1):
        self.attempt_number = attempt
        self.outcome = _Outcome(exception)


class _Outcome:
    def __init__(self, exception: Exception | None):
        self._exception = exception

    def exception(self) -> Exception | None:
        return self._exception


@pytest.mark.parametrize("message", [PER_DAY, NO_FREE_TIER])
def test_daily_quotas_are_recognised(message: str) -> None:
    assert llm._is_daily_quota(message)


@pytest.mark.parametrize("message", [PER_MINUTE, TRANSIENT, "some other failure"])
def test_other_failures_are_not_daily_quotas(message: str) -> None:
    assert not llm._is_daily_quota(message)


def test_a_daily_quota_is_not_retried() -> None:
    assert not llm._should_retry(_State(RuntimeError(PER_DAY)))


@pytest.mark.parametrize("message", [PER_MINUTE, TRANSIENT])
def test_recoverable_failures_are_retried(message: str) -> None:
    assert llm._should_retry(_State(RuntimeError(message)))


def test_a_rate_limit_waits_as_long_as_it_was_asked_to() -> None:
    """Exponential backoff tops out at 20s, well short of a 53s window."""
    wait = llm._wait_for_quota(_State(RuntimeError(PER_MINUTE)))
    assert 53 < wait <= llm.MAX_RATE_LIMIT_WAIT


def test_a_rate_limit_with_no_stated_delay_still_waits_meaningfully() -> None:
    wait = llm._wait_for_quota(_State(RuntimeError("429 RESOURCE_EXHAUSTED")))
    assert wait >= 30


def test_an_ordinary_failure_backs_off_exponentially() -> None:
    first = llm._wait_for_quota(_State(RuntimeError(TRANSIENT), attempt=1))
    third = llm._wait_for_quota(_State(RuntimeError(TRANSIENT), attempt=3))
    assert first < third <= 20


def test_a_daily_quota_surfaces_as_an_actionable_error(
    free_providers, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError(PER_DAY)

    monkeypatch.setattr(llm, "_invoke", fail)

    from cae.schemas.coach import FocusArea

    with pytest.raises(llm.QuotaExhaustedError) as excinfo:
        llm.structured_call(FocusArea, "system", "user")

    message = str(excinfo.value)
    assert "gemini-3.5-flash" in message
    assert "CAE_FAST_MODEL" in message
    assert isinstance(excinfo.value, llm.LLMUnavailableError)


def test_a_spent_allowance_is_not_retried(
    free_providers, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A spent allowance must fail on the first attempt, not after four."""
    free_providers({"CAE_CHAT_MODEL": "groq:openai/gpt-oss-120b"})
    attempts: list[int] = []

    def fail(*args, **kwargs):
        attempts.append(1)
        raise RuntimeError(PER_DAY)

    monkeypatch.setattr(llm, "_invoke_chat", fail)

    with pytest.raises(llm.QuotaExhaustedError) as excinfo:
        llm.chat_call([("human", "hello")])

    assert len(attempts) == 1, "a spent allowance must fail on the first attempt"
    assert "CAE_CHAT_MODEL" in str(excinfo.value)
    assert isinstance(excinfo.value, llm.LLMUnavailableError)


def test_a_rate_limit_is_still_retried_after_the_billing_check() -> None:
    """The billing check must not swallow ordinary per-minute throttling."""
    assert llm._is_daily_quota(NO_BALANCE) is True
    assert llm._is_daily_quota(PER_MINUTE) is False
    assert llm._should_retry(_State(RuntimeError(PER_MINUTE))) is True
    assert llm._should_retry(_State(RuntimeError(NO_BALANCE))) is False


@pytest.mark.parametrize(
    "message",
    [
        REAL_503,
        TRANSIENT,
        "500 INTERNAL. Internal error encountered.",
        "502 Bad Gateway",
        "504 DEADLINE_EXCEEDED",
        "The model is overloaded. Please try again later.",
        "Connection aborted: remote end closed connection",
    ],
)
def test_provider_failures_are_recognised_as_transient(message: str) -> None:
    assert llm._is_transient(message)


def _raising(exc: Exception):
    def call():
        raise exc

    return call


def test_an_overloaded_model_reaches_the_page_as_an_llm_error() -> None:
    with pytest.raises(llm.ModelUnreachableError) as excinfo:
        llm._translating_failures("gemini-3.5-flash", _raising(RuntimeError(REAL_503)))

    assert isinstance(excinfo.value, llm.LLMUnavailableError)


def test_the_message_keeps_the_provider_wording_off_the_page() -> None:
    with pytest.raises(llm.ModelUnreachableError) as excinfo:
        llm._translating_failures("gemini-3.5-flash", _raising(RuntimeError(REAL_503)))

    message = str(excinfo.value)
    for leak in ("503", "UNAVAILABLE", "google", "genai", "gemini", "{"):
        assert leak not in message, f"{leak!r} leaked into the learner's message"
    assert "try again" in message.lower()
    assert "503" in excinfo.value.cause


def test_a_spent_allowance_still_wins_over_the_transient_branch() -> None:
    """PER_DAY carries a 429, not a 503, but the order must not be luck."""
    with pytest.raises(llm.QuotaExhaustedError):
        llm._translating_failures("gemini-3.5-flash", _raising(RuntimeError(PER_DAY)))


def test_our_own_bugs_are_not_disguised_as_a_busy_model() -> None:
    """A malformed request must keep propagating, not invite an endless retry."""
    with pytest.raises(ValueError):
        llm._translating_failures(
            "gemini-3.5-flash",
            _raising(ValueError("Invalid schema: Optional fields are not supported")),
        )


def test_an_overloaded_model_is_retried_before_giving_up() -> None:
    assert llm._should_retry(_State(RuntimeError(REAL_503))) is True


def test_other_failures_are_not_disguised_as_quota_errors(
    free_providers, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llm, "_invoke", lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")))

    from cae.schemas.coach import FocusArea

    with pytest.raises(ValueError, match="boom"):
        llm.structured_call(FocusArea, "system", "user")
