"""Application settings, loaded from the environment / .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, NamedTuple

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
REFERENCE_DIR = DATA_DIR / "reference"
INDEX_DIR = DATA_DIR / "chroma"

Provider = Literal["google", "mistral", "groq"]
Tier = Literal[
    "fast",
    "smart",
    "chat",
    "rue_task",
    "rue_evaluator",
    "writing_task",
    "writing_evaluator",
    "coach",
]

PROVIDER_KEYS: dict[str, tuple[str, str]] = {
    "google": ("GOOGLE_API_KEY", "https://aistudio.google.com/apikey"),
    "mistral": ("MISTRAL_API_KEY", "https://console.mistral.ai/api-keys"),
    "groq": ("GROQ_API_KEY", "https://console.groq.com/keys"),
}

TIER_SETTINGS: dict[str, str] = {
    "fast": "CAE_FAST_MODEL",
    "smart": "CAE_SMART_MODEL",
    "chat": "CAE_CHAT_MODEL",
    "rue_task": "CAE_RUE_TASK_MODEL",
    "rue_evaluator": "CAE_RUE_EVALUATOR_MODEL",
    "writing_task": "CAE_WRITING_TASK_MODEL",
    "writing_evaluator": "CAE_WRITING_EVALUATOR_MODEL",
    "coach": "CAE_COACH_MODEL",
}

TIER_GROUPS: dict[str, str] = {
    "rue_task": "CAE_GENERATION_MODEL",
    "writing_task": "CAE_GENERATION_MODEL",
    "rue_evaluator": "CAE_MARKING_MODEL",
    "writing_evaluator": "CAE_MARKING_MODEL",
    "chat": "CAE_COACH_MODEL",
}

_BARE_PREFIXES: tuple[tuple[str, Provider], ...] = (
    ("gemini", "google"),
    ("gemma", "google"),
    ("mistral", "mistral"),
    ("magistral", "mistral"),
    ("ministral", "mistral"),
    ("codestral", "mistral"),
    ("devstral", "mistral"),
    ("open-mistral", "mistral"),
    ("llama", "groq"),
    ("openai/", "groq"),
    ("qwen/", "groq"),
    ("moonshotai/", "groq"),
    ("meta-llama/", "groq"),
)


class Route(NamedTuple):
    """One model on one provider: a link in a tier's fallback chain."""

    provider: Provider
    model: str

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


class RouteConfigError(ValueError):
    """A model chain in ``.env`` names something no free provider serves."""


def parse_route(entry: str, *, setting: str = "") -> Route:
    """``"mistral:ministral-14b-latest"`` → ``Route("mistral", ...)``.

    Groq ids contain a slash (``openai/gpt-oss-120b``) but never a colon, so the
    first colon is always the provider separator.
    """
    entry = entry.strip()
    provider, sep, model = entry.partition(":")
    if sep:
        provider = provider.strip().lower()
        if provider not in PROVIDER_KEYS:
            raise RouteConfigError(
                f"{setting or 'A model setting'} names provider '{provider}' in "
                f"'{entry}'. The free providers are: {', '.join(PROVIDER_KEYS)}."
            )
        return Route(provider, model.strip())  # type: ignore[arg-type]
    for prefix, inferred in _BARE_PREFIXES:
        if entry.lower().startswith(prefix):
            return Route(inferred, entry)
    raise RouteConfigError(
        f"{setting or 'A model setting'} names '{entry}', which is not served by any "
        "configured free provider. Write it as provider:model, e.g. "
        "'mistral:ministral-14b-latest' (providers: "
        f"{', '.join(PROVIDER_KEYS)})."
    )


def parse_chain(value: str, *, setting: str = "") -> list[Route]:
    """A comma-separated chain, first choice first, duplicates dropped."""
    routes: list[Route] = []
    for entry in value.split(","):
        if entry.strip():
            route = parse_route(entry, setting=setting)
            if route not in routes:
                routes.append(route)
    return routes


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    google_api_key: str = ""
    mistral_api_key: str = ""
    groq_api_key: str = ""

    cae_fast_model: str = (
        "google:gemini-3.5-flash,groq:openai/gpt-oss-120b,"
        "google:gemini-3.5-flash-lite,mistral:ministral-14b-latest"
    )
    cae_smart_model: str = (
        "google:gemini-3.6-flash,google:gemini-3.5-flash-lite,"
        "mistral:ministral-14b-latest,groq:openai/gpt-oss-120b"
    )
    cae_generation_model: str = "google:gemini-3.1-flash-lite"
    cae_marking_model: str = (
        "google:gemini-3.6-flash,google:gemini-3.7-flash,"
        "google:gemini-3.8-flash,google:gemini-3.1-flash-lite"
    )
    cae_coach_model: str = "google:gemini-3.5-flash-lite"
    cae_rue_task_model: str = ""
    cae_writing_task_model: str = ""
    cae_rue_evaluator_model: str = ""
    cae_writing_evaluator_model: str = ""
    cae_chat_model: str = ""
    cae_embedding_model: str = "google:gemini-embedding-001"

    cae_groq_reasoning_effort: str = "low"
    cae_mistral_requests_per_second: float = 0.5

    cae_database_url: str = ""
    cae_max_generation_attempts: int = 3

    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "cae-trainer"
    langsmith_endpoint: str = ""

    @property
    def database_url(self) -> str:
        if self.cae_database_url:
            return self.cae_database_url
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{DATA_DIR / 'cae.db'}"

    def key_for(self, provider: str) -> str:
        return getattr(self, f"{provider}_api_key", "").strip()

    def has_key(self, provider: str) -> bool:
        return bool(self.key_for(provider))

    def chain_setting(self, tier: Tier) -> str:
        """The variable the tier's chain comes from: its own, or its job's if empty."""
        own = TIER_SETTINGS[tier]
        if tier in TIER_GROUPS and not getattr(self, own.lower()).strip():
            return TIER_GROUPS[tier]
        return own

    def chain(self, tier: Tier) -> str:
        """The tier's chain as written in ``.env`` (or its default)."""
        return getattr(self, self.chain_setting(tier).lower())

    def routes(self, tier: Tier) -> list[Route]:
        """The tier's chain as configured, keys or not. Raises on a bad entry."""
        return parse_chain(self.chain(tier), setting=self.chain_setting(tier))

    def embedding_route(self) -> Route:
        return parse_route(self.cae_embedding_model, setting="CAE_EMBEDDING_MODEL")

    def tier_available(self, tier: Tier) -> bool:
        """Whether at least one model in the tier's chain has a key.

        Checked per tier on purpose: no key for the chat's providers must not stop
        a learner sitting an exercise.
        """
        try:
            return any(self.has_key(route.provider) for route in self.routes(tier))
        except RouteConfigError:
            return False

    @property
    def has_embedding_key(self) -> bool:
        """Whether the reference corpus can be indexed and searched."""
        try:
            return self.has_key(self.embedding_route().provider)
        except RouteConfigError:
            return False

    @property
    def has_tracing(self) -> bool:
        """Tracing needs both the switch and a key; either alone traces nothing."""
        return self.langsmith_tracing and bool(self.langsmith_api_key.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
