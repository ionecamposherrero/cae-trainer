"""Session fixtures: a throwaway SQLite database, with tracing forced off."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def offline() -> None:
    """Force tracing off, whatever ``.env`` says.

    Must run before ``temp_database`` clears the settings cache.
    """
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ.pop("LANGCHAIN_TRACING_V2", None)
    os.environ["OPENAI_API_KEY"] = ""


@pytest.fixture(scope="session", autouse=True)
def temp_database(offline: None, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Point the app at a scratch database for the whole session."""
    db_path = tmp_path_factory.mktemp("data") / "test.db"
    os.environ["CAE_DATABASE_URL"] = f"sqlite:///{db_path.as_posix()}"

    from cae.config import get_settings
    from cae.db.session import get_engine, get_sessionmaker

    get_settings.cache_clear()
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()
    get_engine()


FAKE_KEYS = {
    "GOOGLE_API_KEY": "test-google",
    "MISTRAL_API_KEY": "test-mistral",
    "GROQ_API_KEY": "test-groq",
}

TEST_CHAINS = {
    "CAE_GENERATION_MODEL": "google:gemini-3.1-flash-lite",
    "CAE_MARKING_MODEL": "google:gemini-3.6-flash,google:gemini-3.1-flash-lite",
    "CAE_FAST_MODEL": "google:gemini-3.5-flash,google:gemini-3.5-flash-lite,"
    "mistral:ministral-14b-latest,groq:openai/gpt-oss-120b",
    "CAE_SMART_MODEL": "google:gemini-3.6-flash,mistral:ministral-14b-latest,"
    "groq:openai/gpt-oss-120b,google:gemini-3.5-flash-lite",
    "CAE_CHAT_MODEL": "mistral:ministral-14b-latest,groq:openai/gpt-oss-120b,"
    "google:gemini-3.5-flash-lite",
    "CAE_RUE_TASK_MODEL": "google:gemini-3.6-flash,mistral:ministral-14b-latest,"
    "groq:openai/gpt-oss-120b,google:gemini-3.5-flash-lite",
    "CAE_RUE_EVALUATOR_MODEL": "google:gemini-3.5-flash,google:gemini-3.5-flash-lite,"
    "mistral:ministral-14b-latest,groq:openai/gpt-oss-120b",
    "CAE_WRITING_TASK_MODEL": "google:gemini-3.5-flash,google:gemini-3.5-flash-lite,"
    "mistral:ministral-14b-latest,groq:openai/gpt-oss-120b",
    "CAE_WRITING_EVALUATOR_MODEL": "google:gemini-3.6-flash,mistral:ministral-14b-latest,"
    "groq:openai/gpt-oss-120b,google:gemini-3.5-flash-lite",
    "CAE_COACH_MODEL": "google:gemini-3.6-flash,mistral:ministral-14b-latest,"
    "groq:openai/gpt-oss-120b,google:gemini-3.5-flash-lite",
    "CAE_EMBEDDING_MODEL": "google:gemini-embedding-001",
    "CAE_GROQ_REASONING_EFFORT": "low",
}


@pytest.fixture
def free_providers(monkeypatch: pytest.MonkeyPatch):
    """Every provider keyed, the default chains, no waiting, and clean caches.

    Call the yielded function with ``{"GROQ_API_KEY": ""}`` and the like to drop a
    key or override a chain.
    """
    from cae import llm
    from cae.config import get_settings

    def configure(overrides: dict[str, str] | None = None) -> None:
        for name, value in {**FAKE_KEYS, **TEST_CHAINS, **(overrides or {})}.items():
            monkeypatch.setenv(name, value)
        get_settings.cache_clear()
        llm.clear_caches()
        llm.reset_route_usage()

    monkeypatch.setattr(llm, "_wait_for_quota", lambda state: 0.0)
    monkeypatch.setattr(llm, "_wait_before_fallback", lambda state: 0.0)
    configure()
    yield configure
    monkeypatch.undo()
    get_settings.cache_clear()
    llm.clear_caches()
    llm.reset_route_usage()


@pytest.fixture
def user() -> dict:
    """A freshly registered account, unique per test."""
    import uuid

    from cae import auth

    return auth.register(f"tester{uuid.uuid4().hex[:8]}", "correct-horse-battery")
