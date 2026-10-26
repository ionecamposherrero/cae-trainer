"""Talk with your coach about results, marked questions or English itself."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st  # noqa: E402

from cae.llm import LLMUnavailableError  # noqa: E402
from cae.services import coach_reply, progress  # noqa: E402
from cae.ui.shell import (  # noqa: E402
    chat_key_banner,
    llm_error,
    page_header,
    page_setup,
    require_login,
    sidebar_account,
)

THREAD_KEY = "coach_chat_thread"

SUGGESTIONS = (
    "How am I doing overall?",
    "Which mistake is costing me the most marks?",
    "When do I use inversion after a negative adverbial?",
)

AVATARS = {"user": ":material/person:", "assistant": ":material/school:"}

page_setup("Talk with your coach", icon=":material/forum:")
user = require_login()
sidebar_account(user)

page_header(
    "Talk with your coach",
    "Ask about your results, about a question you got wrong, or about English "
    "itself. Your coach can see everything you have been marked on.",
    eyebrow="Your coach",
)

snapshot = progress(user["id"])
thread: list[dict] = st.session_state.setdefault(THREAD_KEY, [])


status = st.columns([4, 1], vertical_alignment="bottom")
with status[0]:
    if snapshot.activity_count == 0:
        st.info(
            "You have not been marked on anything yet, so your coach has no "
            "figures to discuss. It can still answer questions about English and "
            "about the exam.",
            icon=":material/hourglass:",
        )
    else:
        st.caption(
            f"Your coach can see {snapshot.reading_attempts} marked exercise(s) and "
            f"{snapshot.writing_submissions} writing task(s)."
        )
with status[1]:
    if st.button(
        "Clear chat",
        width="stretch",
        disabled=not thread,
        icon=":material/delete_sweep:",
    ):
        st.session_state[THREAD_KEY] = []
        st.rerun()

st.divider()


def render_turn(turn: dict) -> None:
    with st.chat_message(turn["role"], avatar=AVATARS[turn["role"]]):
        st.markdown(turn["text"])


for turn in thread:
    render_turn(turn)

if not thread:
    st.markdown("**Not sure where to start?**")
    picked = ""
    columns = st.columns(len(SUGGESTIONS))
    for column, suggestion in zip(columns, SUGGESTIONS):
        with column:
            if st.button(
                suggestion,
                width="stretch",
                key=f"suggest_{suggestion[:20]}",
                icon=":material/chat_bubble:",
            ):
                picked = suggestion
    if picked:
        st.session_state["coach_chat_pending"] = picked
        st.rerun()

typed = st.chat_input("Ask your coach anything")
question = typed or st.session_state.pop("coach_chat_pending", "")

if question:
    if chat_key_banner():
        thread.append({"role": "user", "text": question})
        render_turn(thread[-1])

        history = [(t["role"], t["text"]) for t in thread[:-1]]
        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            with st.spinner("Reading your record."):
                try:
                    result = coach_reply(
                        user["id"], user["username"], history, question
                    )
                except LLMUnavailableError as exc:
                    llm_error(exc)
                    result = None
                except Exception as exc:
                    st.error(
                        f"The coach could not answer that: {exc}",
                        icon=":material/error:",
                    )
                    result = None

            if result is not None:
                st.markdown(result["reply"])
                thread.append({"role": "assistant", "text": result["reply"]})
            else:
                thread.pop()
