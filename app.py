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
from cae.services import progress  # noqa: E402
from cae.ui.shell import (  # noqa: E402
    APP_TITLE,
    DATA_NOTICE,
    api_key_banner,
    current_user,
    page_setup,
    sidebar_account,
)

page_setup("Home")


def render_sign_in() -> None:
    _, centre, _ = st.columns([1, 2, 1])
    with centre:
        st.title(APP_TITLE)
        st.markdown(
            "Practise the **Reading and Use of English** paper and both **Writing** "
            "tasks of the Cambridge C1 Advanced exam. Every exercise is written for "
            "you, marked item by item, and added to a record of what you keep "
            "getting wrong."
        )

        sign_in_tab, register_tab = st.tabs(["Sign in", "Create an account"])

        with sign_in_tab:
            with st.form("sign_in"):
                username = st.text_input("Username", key="login_username")
                password = st.text_input(
                    "Password", type="password", key="login_password"
                )
                submitted = st.form_submit_button("Sign in", width="stretch")
            if submitted:
                try:
                    st.session_state["user"] = auth.login(username, password)
                    st.rerun()
                except auth.AuthError as exc:
                    st.error(str(exc))

        with register_tab:
            with st.form("register"):
                new_username = st.text_input("Username", key="reg_username")
                new_password = st.text_input(
                    "Password", type="password", key="reg_password"
                )
                st.caption("At least 8 characters.")
                st.caption(DATA_NOTICE)
                created = st.form_submit_button("Create account", width="stretch")
            if created:
                try:
                    st.session_state["user"] = auth.register(new_username, new_password)
                    st.rerun()
                except auth.AuthError as exc:
                    st.error(str(exc))


def render_home(user: dict) -> None:
    sidebar_account(user)
    st.title(f"Welcome back, {user['username']}")
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

    st.divider()
    st.markdown("#### Where to go")
    nav = st.columns(5)
    with nav[0]:
        st.page_link(
            "pages/1_Reading_and_Use_of_English.py",
            label="Reading and Use of English",
            icon=":material/edit:",
        )
    with nav[1]:
        st.page_link("pages/2_Writing.py", label="Writing", icon=":material/draw:")
    with nav[2]:
        st.page_link("pages/3_Mock_exam.py", label="Mock exam", icon=":material/timer:")
    with nav[3]:
        st.page_link("pages/4_Progress.py", label="Progress", icon=":material/insights:")
    with nav[4]:
        st.page_link(
            "pages/5_Talk_with_your_coach.py",
            label="Talk with your coach",
            icon=":material/forum:",
        )


user = current_user()
if user is None:
    render_sign_in()
else:
    render_home(user)
