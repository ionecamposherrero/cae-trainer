"""Shared page furniture: the login gate, headers and cached resources."""

from __future__ import annotations

import time
from typing import Any

import streamlit as st

from cae.config import get_settings
from cae.db.session import init_db
from cae.llm import LLMUnavailableError, ModelUnreachableError, QuotaExhaustedError
from cae.observability import configure_tracing
from cae.ui import style

APP_TITLE = "C1 Advanced Trainer"


@st.cache_resource
def bootstrap() -> bool:
    """Create the database and switch tracing on, once per server process."""
    init_db()
    configure_tracing()
    return True


def page_setup(
    title: str, icon: str = ":material/school:", layout: str = "wide"
) -> None:
    st.set_page_config(page_title=f"{title} · {APP_TITLE}", page_icon=icon, layout=layout)
    bootstrap()
    style.inject()


def current_user() -> dict[str, Any] | None:
    return st.session_state.get("user")


def require_login() -> dict[str, Any]:
    """Return the signed-in user, or stop the page with a prompt to sign in."""
    user = current_user()
    if user is None:
        _, centre, _ = st.columns([1, 2, 1])
        with centre:
            style.eyebrow("Members only")
            st.title(APP_TITLE)
            st.warning(
                "Please sign in on the Home page to continue.", icon=":material/lock:"
            )
            st.page_link("app.py", label="Go to Home", icon=":material/home:")
        st.stop()
    return user


def sign_out() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]


DATA_NOTICE = (
    "Your answers, writing and chat are sent to Google's Gemini API to be marked. "
    "On the free tier Google may use what it is sent to improve its models."
)


def sidebar_account(user: dict[str, Any]) -> None:
    with st.sidebar:
        style.user_badge(user["username"])
        if st.button("Sign out", width="stretch", icon=":material/logout:"):
            sign_out()
            st.rerun()
        st.caption(DATA_NOTICE)


def api_key_banner() -> bool:
    """Warn once if no model for exercises has a key. Returns True when one does."""
    settings = get_settings()
    tiers = ("writing_task", "writing_evaluator", "coach", "rue_task", "rue_evaluator")
    if all(settings.tier_available(tier) for tier in tiers):
        return True
    st.error(
        "No free model is configured for generating and marking exercises. Copy "
        "`.env.example` to `.env`, add a free key from "
        "https://aistudio.google.com/apikey (`GOOGLE_API_KEY`), then restart "
        "the app.",
        icon=":material/key_off:",
    )
    return False


def chat_key_banner() -> bool:
    """Warn if no model for the coach chat has a key. Returns True when one does."""
    if get_settings().tier_available("chat"):
        return True
    st.error(
        "No free model is configured for the coach chat. Copy `.env.example` to "
        "`.env`, add a free key from https://console.mistral.ai/api-keys "
        "(`MISTRAL_API_KEY`) or https://console.groq.com/keys (`GROQ_API_KEY`), "
        "then restart the app. Exercises and marking are unaffected.",
        icon=":material/key_off:",
    )
    return False


def llm_error(exc: LLMUnavailableError) -> None:
    """Show why the model could not run, with an icon that matches the cause.

    A crossed-out key is right for a missing key and misleading for an overloaded
    provider, so the icon follows the exception rather than the call site. The
    message is the exception's own: none of them carry provider wording.
    """
    if isinstance(exc, ModelUnreachableError):
        icon = ":material/cloud_off:"
    elif isinstance(exc, QuotaExhaustedError):
        icon = ":material/hourglass_disabled:"
    else:
        icon = ":material/key_off:"
    st.error(str(exc), icon=icon)


def page_header(title: str, subtitle: str = "", eyebrow: str = "") -> None:
    if eyebrow:
        style.eyebrow(eyebrow)
    st.title(title)
    if subtitle:
        style.lead(subtitle)


def clear_widget_state(prefix: str) -> None:
    """Drop the answer widgets belonging to one exercise between attempts."""
    for key in [k for k in st.session_state if k.startswith(prefix)]:
        del st.session_state[key]


def format_duration(seconds: int) -> str:
    minutes, secs = divmod(max(0, seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


@st.fragment(run_every=5)
def countdown(deadline: float, total_minutes: int, label: str = "Time remaining") -> None:
    """A ticking exam clock, run as a fragment so the page does not rerun.

    At zero it reruns the whole page, which is what hands the paper in: the page
    must check the deadline itself *before* calling this, or it would rerun
    forever.
    """
    remaining = int(deadline - time.time())
    total = total_minutes * 60
    if remaining <= 0:
        st.rerun(scope="app")
    st.progress(
        max(0.0, min(1.0, remaining / total)),
        text=f"{label}: {format_duration(remaining)}",
    )
    if remaining < 300:
        st.warning(
            "Less than five minutes left. The paper is handed in automatically "
            "when the clock reaches zero.",
            icon=":material/alarm:",
        )
