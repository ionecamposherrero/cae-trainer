"""C1 Advanced Trainer — Streamlit entry point.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st  # noqa: E402

from cae import auth  # noqa: E402
from cae.domain.parts import (  # noqa: E402
    PART_SPECS,
    READING_MAX_MARKS,
    READING_TIME_MINUTES,
    WRITING_TIME_MINUTES,
    WRITING_WORD_RANGE,
)
from cae.services import progress  # noqa: E402
from cae.ui import style  # noqa: E402
from cae.ui.shell import (  # noqa: E402
    APP_TITLE,
    DATA_NOTICE,
    api_key_banner,
    current_user,
    page_setup,
    sidebar_account,
)

page_setup("Home")

QUESTIONS = sum(spec.item_count for spec in PART_SPECS.values())
MIN_WORDS, MAX_WORDS = WRITING_WORD_RANGE
DISCLAIMER = (
    "An independent practice tool. Not affiliated with or endorsed by Cambridge "
    "University Press & Assessment."
)


def render_sign_in() -> None:
    pitch, form = st.columns([7, 5], gap="large")
    with pitch:
        with st.container(key="cae_hero"):
            style.eyebrow("Cambridge C1 Advanced · exam practice")
            st.title(APP_TITLE)
            style.lead(
                "Practise the Reading and Use of English paper and both Writing "
                "tasks. Every exercise is written for you, marked item by item, and "
                "added to a record of what you keep getting wrong."
            )
            style.features(
                [
                    (
                        "All eight Reading and Use of English parts,",
                        "from multiple-choice cloze to multiple matching.",
                    ),
                    (
                        "Writing marked on the four official subscales,",
                        "with your script annotated and rewritten at band 5.",
                    ),
                    (
                        "Timed mock exams",
                        "with an estimated scale score against the 180 pass mark.",
                    ),
                    (
                        "A personal coach",
                        "that reads your record and tells you what to work on next.",
                    ),
                ]
            )
            style.chips(
                [
                    f"{QUESTIONS} questions · {READING_MAX_MARKS} marks",
                    f"{READING_TIME_MINUTES} + {WRITING_TIME_MINUTES} minutes",
                    f"{MIN_WORDS}-{MAX_WORDS} words a task",
                ]
            )

    with form:
        with st.container(key="cae_panel_sign_in"):
            st.markdown("### Start practising")
            sign_in_tab, register_tab = st.tabs(["Sign in", "Create an account"])

        with sign_in_tab:
            with st.form("sign_in", border=False):
                username = st.text_input("Username", key="login_username")
                password = st.text_input(
                    "Password", type="password", key="login_password"
                )
                submitted = st.form_submit_button(
                    "Sign in", type="primary", width="stretch"
                )
            if submitted:
                try:
                    st.session_state["user"] = auth.login(username, password)
                    st.rerun()
                except auth.AuthError as exc:
                    st.error(str(exc))

        with register_tab:
            with st.form("register", border=False):
                new_username = st.text_input("Username", key="reg_username")
                new_password = st.text_input(
                    "Password", type="password", key="reg_password"
                )
                st.caption("At least 8 characters.")
                st.caption(DATA_NOTICE)
                created = st.form_submit_button(
                    "Create account", type="primary", width="stretch"
                )
            if created:
                try:
                    st.session_state["user"] = auth.register(new_username, new_password)
                    st.rerun()
                except auth.AuthError as exc:
                    st.error(str(exc))

        style.disclaimer(DISCLAIMER)


def render_home(user: dict) -> None:
    sidebar_account(user)
    with st.container(key="cae_hero"):
        style.eyebrow("Your C1 Advanced study space")
        st.title(f"Welcome back, {user['username']}")
        style.lead(
            "Pick up where you left off: practise one part at a time, write a task, "
            "or sit a whole paper against the clock."
        )
    api_key_banner()

    snapshot = progress(user["id"])
    cols = st.columns(4)
    cols[0].metric("Exercises marked", snapshot.reading_attempts)
    cols[1].metric("Writing tasks marked", snapshot.writing_submissions)
    cols[2].metric("Overall accuracy", f"{snapshot.overall_accuracy:g}%")
    weakest = snapshot.weaknesses[0].name if snapshot.weaknesses else "—"
    cols[3].metric("Weakest part", weakest)

    if snapshot.activity_count == 0:
        st.info(
            "Nothing marked yet. Start with **Reading and Use of English** to work "
            "on one part at a time, or go straight to a **Mock exam**.",
            icon=":material/rocket_launch:",
        )

    st.markdown("### Practise")
    practise = st.columns(3, gap="medium")
    with practise[0]:
        style.nav_card(
            "reading",
            "pages/1_Reading_and_Use_of_English.py",
            kicker="Paper 1",
            title="Reading and Use of English",
            icon="book",
            text=f"Eight parts, {QUESTIONS} questions. One part at a time, aimed at "
            "the mistakes on your record.",
            cta="Practise a part",
        )
    with practise[1]:
        style.nav_card(
            "writing",
            "pages/2_Writing.py",
            kicker="Paper 2",
            title="Writing",
            icon="pen",
            text=f"An essay or a Part 2 task of {MIN_WORDS}-{MAX_WORDS} words, "
            "marked on the four official subscales.",
            cta="Write a task",
        )
    with practise[2]:
        style.nav_card(
            "mock",
            "pages/3_Mock_exam.py",
            kicker="Exam conditions",
            title="Mock exam",
            icon="timer",
            text="Either paper or both, against the real clock, with an estimated "
            "scale score at the end.",
            cta="Sit a mock exam",
        )

    st.markdown("### Review")
    review = st.columns(3, gap="medium")
    with review[0]:
        style.nav_card(
            "progress",
            "pages/4_Progress.py",
            kicker="Your record",
            title="Progress",
            icon="chart",
            text="Accuracy by part, recurring mistakes and your coach's written "
            "review of where the marks are going.",
            cta="See progress",
        )
    with review[1]:
        style.nav_card(
            "coach",
            "pages/5_Talk_with_your_coach.py",
            kicker="Ask anything",
            title="Talk with your coach",
            icon="chat",
            text="Questions about your results, an item you got wrong, or English "
            "grammar and vocabulary.",
            cta="Open the chat",
        )
    with review[2]:
        style.nav_card(
            "account",
            "pages/6_Account.py",
            kicker="Settings",
            title="Account",
            icon="user",
            text="Change your password. Your record stays exactly as it is.",
            cta="Manage account",
        )


user = current_user()
if user is None:
    render_sign_in()
else:
    render_home(user)
