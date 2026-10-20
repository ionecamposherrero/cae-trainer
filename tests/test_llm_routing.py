"""Fallback chains: which model answers when the first choice cannot.

No API is called. ``_invoke`` / ``_invoke_chat`` are replaced by fakes that fail
or answer per route, which is the one seam every call goes through.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from cae import llm
from cae.config import Route, RouteConfigError, get_settings, parse_chain, parse_route
from cae.schemas.coach import FocusArea

GEMINI_PER_DAY = (
    "429 RESOURCE_EXHAUSTED. Quota exceeded for metric: generate_content_free_tier_requests, "
    "limit: 20, 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'. "
    "Please retry in 53.07s."
)
GROQ_TPD = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for model "
    "`openai/gpt-oss-120b` in organization `org_x` service tier `on_demand` on tokens "
    "per day (TPD): Limit 200000, Used 199500, Requested 1800. Please try again in "
    "12m3.5s.', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}"
)
GROQ_TPM = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for model "
    "`openai/gpt-oss-120b` in organization `org_x` service tier `on_demand` on tokens "
    "per minute (TPM): Limit 8000, Used 7000, Requested 2000. Please try again in "
    "7.5s.', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}"
)
GROQ_BAD_TOOL_CALL = (
    "Error code: 400 - {'error': {'message': \"Failed to call a function. Please "
    "adjust your prompt. See 'failed_generation' for more details.\", 'type': "
    "'invalid_request_error', 'code': 'tool_use_failed'}}"
)
MISTRAL_BUSY = (
    "Error response 429 while fetching https://api.mistral.ai/v1/chat/completions: "
    '{"object":"error","message":"Service tier capacity exceeded for this model.",'
    '"type":"service_tier_capacity_exceeded","code":"3505"}'
)
OVERLOADED = "503 UNAVAILABLE. This model is currently experiencing high demand."

FOCUS = FocusArea(title="Collocations", why="Most dropped marks.", action="Drill them.")


def _fake_invoke(outcomes: dict[str, Any], calls: list[str]):
    """An ``_invoke`` whose result depends on the route: raise it or return it."""

    def invoke(route: Route, *args, **kwargs):
        calls.append(route.label)
        outcome = outcomes.get(route.label, FOCUS)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    return invoke


def test_a_chain_is_read_in_order_with_its_providers() -> None:
    routes = parse_chain(
        "google:gemini-3.6-flash, mistral:ministral-14b-latest ,groq:openai/gpt-oss-120b"
    )
    assert routes == [
        Route("google", "gemini-3.6-flash"),
        Route("mistral", "ministral-14b-latest"),
        Route("groq", "openai/gpt-oss-120b"),
    ]


@pytest.mark.parametrize(
    ("entry", "provider"),
    [
        ("gemini-3.5-flash", "google"),
        ("ministral-14b-latest", "mistral"),
        ("codestral-latest", "mistral"),
        ("openai/gpt-oss-120b", "groq"),
        ("llama-3.3-70b-versatile", "groq"),
    ],
)
def test_a_bare_id_is_attributed_by_its_prefix(entry: str, provider: str) -> None:
    """So a sweep can say --models gemini-3.5-flash."""
    assert parse_route(entry).provider == provider


@pytest.mark.parametrize("entry", ["gpt-5.6-luna", "openrouter:some/model"])
def test_a_model_no_free_provider_serves_is_refused(entry: str) -> None:
    """A .env naming a paid model must fail loudly, not 404 later."""
    with pytest.raises(RouteConfigError):
        parse_route(entry, setting="CAE_CHAT_MODEL")


def test_a_duplicate_in_a_chain_is_dropped() -> None:
    assert len(parse_chain("gemini-3.5-flash,google:gemini-3.5-flash")) == 1


@pytest.mark.parametrize(
    ("tiers", "job_setting"),
    [
        (("rue_task", "writing_task"), "CAE_GENERATION_MODEL"),
        (("rue_evaluator", "writing_evaluator"), "CAE_MARKING_MODEL"),
        (("coach", "chat"), "CAE_COACH_MODEL"),
    ],
)
def test_a_tier_without_its_own_chain_reads_its_jobs(
    free_providers, tiers: tuple[str, str], job_setting: str
) -> None:
    """So .env needs three chains: generation, marking and coaching."""
    free_providers(
        {**{f"CAE_{tier.upper()}_MODEL": "" for tier in tiers if tier != "coach"},
         job_setting: "google:gemini-3.8-flash"}
    )
    settings = get_settings()
    for tier in tiers:
        assert settings.chain_setting(tier) == job_setting
        assert settings.routes(tier) == [Route("google", "gemini-3.8-flash")]


def test_a_tiers_own_chain_overrides_its_jobs(free_providers) -> None:
    """What run_evals.py --tier relies on to measure one graph alone."""
    free_providers({
        "CAE_GENERATION_MODEL": "google:gemini-3.1-flash-lite",
        "CAE_RUE_TASK_MODEL": "google:gemini-3.8-flash",
        "CAE_WRITING_TASK_MODEL": "",
    })
    settings = get_settings()
    assert settings.routes("rue_task") == [Route("google", "gemini-3.8-flash")]
    assert settings.routes("writing_task") == [Route("google", "gemini-3.1-flash-lite")]


def test_a_missing_key_names_the_setting_the_chain_came_from(free_providers) -> None:
    free_providers({
        "MISTRAL_API_KEY": "",
        "CAE_RUE_EVALUATOR_MODEL": "",
        "CAE_MARKING_MODEL": "mistral:ministral-14b-latest",
    })
    with pytest.raises(llm.MissingAPIKeyError, match="CAE_MARKING_MODEL"):
        llm.routes_for("rue_evaluator")


def test_a_provider_without_a_key_is_skipped(free_providers) -> None:
    free_providers({"MISTRAL_API_KEY": ""})
    assert [r.provider for r in llm.routes_for("smart")] == ["google", "groq", "google"]


def test_no_key_for_a_whole_tier_says_which_keys_would_do(free_providers) -> None:
    free_providers({"MISTRAL_API_KEY": "", "GROQ_API_KEY": "", "CAE_CHAT_MODEL": "mistral:ministral-14b-latest,groq:openai/gpt-oss-120b"})
    with pytest.raises(llm.MissingAPIKeyError) as excinfo:
        llm.routes_for("chat")
    message = str(excinfo.value)
    assert "CAE_CHAT_MODEL" in message
    assert "MISTRAL_API_KEY" in message and "GROQ_API_KEY" in message


def test_no_chat_key_does_not_stop_the_exercises(free_providers) -> None:
    """The tiers fail independently: a learner can still sit an exercise."""
    free_providers({"MISTRAL_API_KEY": "", "GROQ_API_KEY": "", "CAE_CHAT_MODEL": "mistral:ministral-14b-latest"})
    settings = get_settings()
    assert settings.tier_available("fast")
    assert settings.tier_available("smart")
    assert not settings.tier_available("chat")


def test_a_misconfigured_chain_reaches_the_page_as_an_llm_error(free_providers) -> None:
    free_providers({"CAE_CHAT_MODEL": "gpt-5.6-luna"})
    assert not get_settings().tier_available("chat")
    with pytest.raises(llm.LLMUnavailableError):
        llm.chat_call([("human", "hello")])


def test_a_spent_allowance_hands_the_call_to_the_next_model(
    free_providers, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    spent = RuntimeError(GEMINI_PER_DAY)
    monkeypatch.setattr(
        llm,
        "_invoke",
        _fake_invoke({"google:gemini-3.6-flash": spent}, calls),
    )

    assert llm.structured_call(FocusArea, "s", "u", tier="smart") == FOCUS
    assert calls == ["google:gemini-3.6-flash", "mistral:ministral-14b-latest"]
    assert llm.route_usage() == {("smart", "mistral:ministral-14b-latest"): 1}


def test_a_spent_model_is_not_asked_again_until_it_resets(
    free_providers, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        llm,
        "_invoke",
        _fake_invoke({"google:gemini-3.6-flash": RuntimeError(GEMINI_PER_DAY)}, calls),
    )
    llm.structured_call(FocusArea, "s", "u", tier="smart")
    calls.clear()

    llm.structured_call(FocusArea, "s", "u", tier="smart")
    assert calls == ["mistral:ministral-14b-latest"]

    monkeypatch.setattr(llm, "EXHAUSTED_FOR", 0.0)
    llm._mark_exhausted(Route("google", "gemini-3.6-flash"))
    calls.clear()
    llm.structured_call(FocusArea, "s", "u", tier="smart")
    assert calls[0] == "google:gemini-3.6-flash"


def test_an_unreachable_provider_moves_on_without_being_marked_spent(
    free_providers, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        llm,
        "_invoke",
        _fake_invoke({"google:gemini-3.6-flash": RuntimeError(OVERLOADED)}, calls),
    )

    assert llm.structured_call(FocusArea, "s", "u", tier="smart") == FOCUS
    assert calls == [
        "google:gemini-3.6-flash",
        "google:gemini-3.6-flash",
        "mistral:ministral-14b-latest",
    ]
    assert not llm._is_exhausted(Route("google", "gemini-3.6-flash"))


def test_a_model_that_was_down_is_not_waited_on_again_straight_away(
    free_providers, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-asking an overloaded model on every call would pay its timeout every time."""
    calls: list[str] = []
    monkeypatch.setattr(
        llm,
        "_invoke",
        _fake_invoke({"google:gemini-3.6-flash": RuntimeError(OVERLOADED)}, calls),
    )
    llm.structured_call(FocusArea, "s", "u", tier="smart")
    calls.clear()

    llm.structured_call(FocusArea, "s", "u", tier="smart")
    assert calls == ["mistral:ministral-14b-latest"]


def test_when_every_model_was_down_they_are_all_tried_again(
    free_providers, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Being down is a soft mark: it must not turn into a refusal without a call."""
    free_providers({"CAE_SMART_MODEL": "google:gemini-3.6-flash"})
    route = Route("google", "gemini-3.6-flash")
    llm._mark_unreachable(route)
    assert llm.routes_for("smart") == [route]


@pytest.mark.parametrize(
    "failure",
    [RuntimeError(GROQ_BAD_TOOL_CALL), FocusArea.model_validate_json],
    ids=["tool_use_failed", "validation"],
)
def test_an_unreadable_reply_moves_on(
    free_providers, monkeypatch: pytest.MonkeyPatch, failure
) -> None:
    """Smaller free models fail the schema more often than Gemini does."""
    if not isinstance(failure, BaseException):
        try:
            failure("{}")
        except Exception as exc:
            failure = exc
    calls: list[str] = []
    monkeypatch.setattr(
        llm, "_invoke", _fake_invoke({"google:gemini-3.6-flash": failure}, calls)
    )

    assert llm.structured_call(FocusArea, "s", "u", tier="smart") == FOCUS
    assert calls == ["google:gemini-3.6-flash", "mistral:ministral-14b-latest"]


def test_an_unreadable_reply_from_every_model_surfaces_as_itself(
    free_providers, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At that point our schema is the likelier culprit, so it must not be hidden."""
    monkeypatch.setattr(
        llm,
        "_invoke",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError(GROQ_BAD_TOOL_CALL)),
    )
    with pytest.raises(RuntimeError, match="tool_use_failed"):
        llm.structured_call(FocusArea, "s", "u", tier="smart")


def test_our_own_bug_is_not_passed_along_the_chain(
    free_providers, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        llm,
        "_invoke",
        _fake_invoke({"google:gemini-3.6-flash": ValueError("Invalid schema")}, calls),
    )
    with pytest.raises(ValueError, match="Invalid schema"):
        llm.structured_call(FocusArea, "s", "u", tier="smart")
    assert set(calls) == {"google:gemini-3.6-flash"}


def test_every_allowance_spent_is_one_actionable_error(
    free_providers, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        llm,
        "_invoke",
        lambda route, *a, **k: calls.append(route.label) or (_ for _ in ()).throw(
            RuntimeError(GEMINI_PER_DAY)
        ),
    )
    with pytest.raises(llm.QuotaExhaustedError) as excinfo:
        llm.structured_call(FocusArea, "s", "u", tier="smart")
    assert len(calls) == 4
    assert "CAE_SMART_MODEL" in str(excinfo.value)

    calls.clear()
    with pytest.raises(llm.QuotaExhaustedError):
        llm.structured_call(FocusArea, "s", "u", tier="smart")
    assert calls == []


def test_the_last_model_left_is_given_the_patient_retries(
    free_providers, monkeypatch: pytest.MonkeyPatch
) -> None:
    free_providers({"CAE_SMART_MODEL": "google:gemini-3.6-flash"})
    calls: list[str] = []
    monkeypatch.setattr(
        llm, "_invoke", _fake_invoke({"google:gemini-3.6-flash": RuntimeError(OVERLOADED)}, calls)
    )
    with pytest.raises(llm.ModelUnreachableError):
        llm.structured_call(FocusArea, "s", "u", tier="smart")
    assert len(calls) == 4


def test_the_trace_names_the_model_that_answered_and_the_one_it_replaced(
    free_providers, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs: dict[str, dict] = {}

    def invoke(route, schema, system, user, temperature, config):
        configs[route.label] = config
        if route.provider == "google":
            raise RuntimeError(GEMINI_PER_DAY)
        return FOCUS

    monkeypatch.setattr(llm, "_invoke", invoke)
    llm.structured_call(FocusArea, "s", "u", tier="smart", label="coach.focus")

    answered = configs["mistral:ministral-14b-latest"]
    assert answered["metadata"]["cae_model"] == "mistral:ministral-14b-latest"
    assert answered["metadata"]["cae_fallback_from"] == "google:gemini-3.6-flash"
    assert "fallback" in answered["tags"]
    assert "cae_fallback_from" not in configs["google:gemini-3.6-flash"]["metadata"]


def test_the_chat_falls_back_too(free_providers, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def invoke_chat(route, messages, tools, temperature, config):
        calls.append(route.label)
        if route.provider == "mistral":
            raise RuntimeError(MISTRAL_BUSY)
        return AIMessage(content="Hello.")

    monkeypatch.setattr(llm, "_invoke_chat", invoke_chat)
    reply = llm.chat_call([("human", "hello")])
    assert reply.content == "Hello."
    assert calls == ["mistral:ministral-14b-latest", "groq:openai/gpt-oss-120b"]


def test_groq_states_its_retry_delay_in_its_own_words() -> None:
    assert llm._asked_delay(GROQ_TPM) == 7.5
    assert llm._asked_delay(GROQ_TPD) == pytest.approx(723.5)
    assert llm._asked_delay("Please retry in 53.07s.") == pytest.approx(53.07)


def test_groq_daily_tokens_are_a_spent_allowance_and_its_minute_window_is_not() -> None:
    assert llm._is_daily_quota(GROQ_TPD)
    assert not llm._is_daily_quota(GROQ_TPM)


def test_a_mistral_model_with_no_quota_counts_as_spent_not_busy() -> None:
    """Mistral answers a model with no quota with "Rate limit exceeded" and a limit of 0."""
    import httpx

    request = httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions")
    response = httpx.Response(
        429,
        request=request,
        headers={"x-ratelimit-limit-req-minute": "0"},
        text='{"message":"Rate limit exceeded","type":"rate_limited","code":"1300"}',
    )
    error = httpx.HTTPStatusError(
        "Error response 429 while fetching ...: Rate limit exceeded",
        request=request,
        response=response,
    )

    assert not llm._should_retry(_State(error))
    with pytest.raises(llm.QuotaExhaustedError):
        llm._translating_failures("mistral:ministral-14b-latest", lambda: (_ for _ in ()).throw(error))


def test_a_groq_malformed_tool_call_is_an_unreadable_reply_not_our_bug() -> None:
    assert llm._is_unreadable_output(RuntimeError(GROQ_BAD_TOOL_CALL))
    assert not llm._is_unreadable_output(ValueError("Invalid schema"))


class _State:
    def __init__(self, exception: Exception):
        self.attempt_number = 1
        self.outcome = type("O", (), {"exception": lambda self: exception})()


def test_a_short_rate_limit_is_waited_out_before_falling_back() -> None:
    assert llm._should_retry_before_fallback(_State(RuntimeError(GROQ_TPM)))


def test_a_long_or_unstated_rate_limit_falls_back_at_once() -> None:
    assert not llm._should_retry_before_fallback(
        _State(RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 53.07s."))
    )
    assert not llm._should_retry_before_fallback(_State(RuntimeError(MISTRAL_BUSY)))


def test_a_timed_out_request_is_not_repeated_while_another_model_waits() -> None:
    """It has already cost the whole timeout once."""
    assert not llm._should_retry_before_fallback(
        _State(RuntimeError("httpx.ReadTimeout: The read operation timed out"))
    )
    assert llm._should_retry_before_fallback(_State(RuntimeError(OVERLOADED)))


def test_a_rate_limit_that_never_clears_reaches_the_page_as_busy() -> None:
    def call():
        raise RuntimeError(GROQ_TPM)

    with pytest.raises(llm.ModelUnreachableError):
        llm._translating_failures("groq:openai/gpt-oss-120b", call)


def test_each_provider_gets_its_own_client(free_providers) -> None:
    """A graph asks for a tier, never for a provider."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_groq import ChatGroq
    from langchain_mistralai import ChatMistralAI

    assert isinstance(llm.build_llm(Route("google", "gemini-3.6-flash")), ChatGoogleGenerativeAI)
    assert isinstance(llm.build_llm(Route("mistral", "ministral-14b-latest")), ChatMistralAI)
    assert isinstance(llm.build_llm(Route("groq", "openai/gpt-oss-120b")), ChatGroq)
    assert isinstance(llm.get_llm(tier="chat"), ChatMistralAI)


def test_no_sdk_retries_behind_the_routers_back(free_providers) -> None:
    """An SDK retrying a spent allowance for minutes would stall the fallback."""
    assert llm.build_llm(Route("google", "gemini-3.6-flash")).max_retries == 1
    assert llm.build_llm(Route("groq", "openai/gpt-oss-120b")).max_retries == 0


def test_groq_reasoning_effort_is_only_sent_when_configured(free_providers) -> None:
    """The parameter is model-specific, so an empty setting must omit it."""
    route = Route("groq", "openai/gpt-oss-120b")
    assert llm.build_llm(route).reasoning_effort == "low"

    free_providers({"CAE_GROQ_REASONING_EFFORT": ""})
    assert llm.build_llm(route).reasoning_effort is None


def test_every_mistral_route_shares_one_rate_limiter(free_providers) -> None:
    """The free plan's limit is per key, so two models must not each get a second."""
    large = llm.build_llm(Route("mistral", "ministral-14b-latest"))
    small = llm.build_llm(Route("mistral", "ministral-8b-latest"))
    assert small is not large, "two models, two clients"
    assert large.rate_limiter is not None
    assert small.rate_limiter is large.rate_limiter


def _turn_with_a_tool_round() -> list:
    """A history as the graph builds it after one ``query_record`` round."""
    return [
        SystemMessage(content="You are a tutor."),
        HumanMessage(content="Which Part 4 items did I get wrong?"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "query_record",
                    "args": {"sql": "SELECT item_number FROM item_results WHERE is_correct = 0"},
                    "id": "call_9f8e7d6c5b4a3f2e1d0c",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content='{"columns": ["item_number"], "rows": [[27], [30]]}',
            tool_call_id="call_9f8e7d6c5b4a3f2e1d0c",
            name="query_record",
        ),
    ]


def test_mistral_can_take_over_a_turn_another_provider_started() -> None:
    """Mistral only accepts 9-character tool call ids, and must still pair them."""
    from langchain_mistralai.chat_models import _convert_message_to_mistral_chat_message

    converted = [_convert_message_to_mistral_chat_message(m) for m in _turn_with_a_tool_round()]
    call_id = converted[2]["tool_calls"][0]["id"]
    assert len(call_id) == 9 and call_id.isalnum()
    assert converted[3]["tool_call_id"] == call_id


def test_groq_can_take_over_a_turn_another_provider_started() -> None:
    from langchain_groq.chat_models import _convert_message_to_dict

    converted = [_convert_message_to_dict(m) for m in _turn_with_a_tool_round()]
    assert converted[2]["tool_calls"][0]["function"]["name"] == "query_record"
    assert converted[3]["tool_call_id"] == converted[2]["tool_calls"][0]["id"]


def test_gemini_can_take_over_a_turn_another_provider_started() -> None:
    from langchain_google_genai.chat_models import _parse_chat_history

    system, contents = _parse_chat_history(_turn_with_a_tool_round())
    assert system is not None
    parts = [part for content in contents for part in content.parts]
    assert any(part.function_call and part.function_call.name == "query_record" for part in parts)
    assert any(
        part.function_response and part.function_response.name == "query_record"
        for part in parts
    )
