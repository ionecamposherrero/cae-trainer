"""Write a Part 1 essay or a Part 2 task and have it marked on the four subscales."""

from __future__ import annotations

import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st  # noqa: E402

from cae.domain.parts import (  # noqa: E402
    WRITING_GENRES,
    WRITING_PART2_GENRES,
    WRITING_TIME_MINUTES,
    WRITING_WORD_RANGE,
)
from cae.graphs.writing_task import WritingTaskFailed  # noqa: E402
from cae.llm import LLMUnavailableError  # noqa: E402
from cae.services import next_writing_task, submit_writing  # noqa: E402
from cae.ui import style  # noqa: E402
from cae.ui.feedback import render_writing_result  # noqa: E402
from cae.ui.renderers import render_word_count, render_writing_task  # noqa: E402
from cae.ui.shell import (  # noqa: E402
    api_key_banner,
    llm_error,
    page_header,
    page_setup,
    require_login,
    sidebar_account,
)

page_setup("Writing", icon=":material/edit_note:")
user = require_login()
sidebar_account(user)

MIN_WORDS, MAX_WORDS = WRITING_WORD_RANGE
STATE = "writing_task"
RESULT = "writing_result"
SCRIPT_KEY = "writing_script"

page_header(
    "Writing",
    f"Two tasks, {MIN_WORDS}-{MAX_WORDS} words each, {WRITING_TIME_MINUTES} minutes "
    "in the real exam. Marked 0-5 on Content, Communicative Achievement, "
    "Organisation and Language, exactly as Cambridge does.",
    eyebrow="Paper 2",
)

has_key = api_key_banner()


with st.container(key="cae_panel_task_choice"):
    controls = st.columns([1, 2, 1], vertical_alignment="bottom")
    with controls[0]:
        part = st.radio(
            "Which task?",
            options=[1, 2],
            format_func=lambda p: (
                "Part 1 — essay" if p == 1 else "Part 2 — choose a genre"
            ),
            key="writing_part",
        )
    with controls[1]:
        if part == 2:
            genre = st.selectbox(
                "Genre",
                options=WRITING_PART2_GENRES,
                format_func=lambda g: WRITING_GENRES[g].name,
                key="writing_genre",
            )
        else:
            genre = "essay"
            st.caption("Part 1 is always a compulsory essay.")
    with controls[2]:
        new_task = st.button(
            "New task",
            type="primary",
            width="stretch",
            disabled=not has_key,
            icon=":material/add:",
        )
    style.chips(
        [
            f"{MIN_WORDS}-{MAX_WORDS} words",
            "Content · Communicative Achievement · Organisation · Language",
            "0-5 on each subscale, 20 marks a task",
        ]
    )

if new_task:
    with st.spinner("Writing a task for you."):
        try:
            task_id, payload = next_writing_task(part=part, genre=genre)
        except LLMUnavailableError as exc:
            llm_error(exc)
            st.stop()
        except WritingTaskFailed as exc:
            st.error(
                f"{exc}\n\nThe model could not satisfy the format rules for this "
                "task. Try again.",
                icon=":material/error:",
            )
            st.stop()
    st.session_state[STATE] = {
        "task_id": task_id,
        "part": part,
        "genre": payload["genre"],
        "payload": payload,
        "started_at": time.time(),
    }
    st.session_state.pop(RESULT, None)
    st.session_state[SCRIPT_KEY] = ""
    st.rerun()

current = st.session_state.get(STATE)
if current is None:
    st.info(
        "Choose Part 1 or Part 2 and press **New task**.", icon=":material/play_arrow:"
    )
    st.stop()

st.divider()
result = st.session_state.get(RESULT)


if result is None:
    task_col, write_col = st.columns([2, 3], gap="large")
    with task_col:
        render_writing_task(current["payload"])

    with write_col:
        st.markdown("### Your answer")
        script = st.text_area(
            "Write your answer here",
            height=460,
            key=SCRIPT_KEY,
            label_visibility="collapsed",
            placeholder="Start writing...",
        )
        in_range = render_word_count(script)

        if st.button(
            "Submit for marking",
            type="primary",
            disabled=not in_range or not has_key,
            icon=":material/grading:",
        ):
            with st.spinner(
                "Four examiners are marking your script, one per subscale."
            ):
                st.session_state[RESULT] = submit_writing(
                    user_id=user["id"],
                    task_id=current["task_id"],
                    part=current["part"],
                    genre=current["genre"],
                    task=current["payload"],
                    script=script,
                    duration_seconds=int(time.time() - current["started_at"]),
                )
            st.rerun()


else:
    with st.expander("The task you answered", expanded=False):
        render_writing_task(current["payload"])
    render_writing_result(result, st.session_state.get(SCRIPT_KEY, ""))

    st.divider()
    actions = st.columns([1, 1, 3])
    with actions[0]:
        if st.button(
            "Another task",
            type="primary",
            width="stretch",
            icon=":material/arrow_forward:",
        ):
            st.session_state.pop(STATE, None)
            st.session_state.pop(RESULT, None)
            st.session_state[SCRIPT_KEY] = ""
            st.rerun()
    with actions[1]:
        st.page_link(
            "pages/4_Progress.py", label="See progress", icon=":material/insights:"
        )
