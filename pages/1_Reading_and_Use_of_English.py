"""The Reading and Use of English paper, practised one part at a time."""

from __future__ import annotations

import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st  # noqa: E402

from cae.domain.parts import (  # noqa: E402
    PART_SPECS,
    READING_MAX_MARKS,
    READING_TIME_MINUTES,
)
from cae.graphs.rue_evaluator import MarkingFailed  # noqa: E402
from cae.graphs.rue_task import GenerationFailed  # noqa: E402
from cae.llm import LLMUnavailableError  # noqa: E402
from cae.services import next_exercise, part_label, submit_attempt  # noqa: E402
from cae.ui.feedback import render_reading_results, render_score_header  # noqa: E402
from cae.ui import style  # noqa: E402
from cae.ui.renderers import render_exercise  # noqa: E402
from cae.ui.shell import (  # noqa: E402
    api_key_banner,
    clear_widget_state,
    llm_error,
    page_header,
    page_setup,
    require_login,
    sidebar_account,
)

page_setup("Reading and Use of English", icon=":material/menu_book:")
user = require_login()
sidebar_account(user)

page_header(
    "Reading and Use of English",
    f"Eight parts, {sum(s.item_count for s in PART_SPECS.values())} questions, "
    f"{READING_MAX_MARKS} marks, {READING_TIME_MINUTES} minutes in the real exam. "
    "Practise one part at a time, marked item by item, with every mistake filed "
    "against a weakness your Progress page tracks.",
    eyebrow="Paper 1",
)

STATE = "practice_exercise"
RESULT = "practice_result"


def load(part: int) -> None:
    with st.spinner(f"Writing a new exercise for {part_label(part)}. This can take a minute."):
        try:
            exercise_id, payload = next_exercise(user["id"], part)
        except LLMUnavailableError as exc:
            llm_error(exc)
            return
        except GenerationFailed as exc:
            st.error(
                f"{exc}\n\nThis usually means the model could not satisfy the format "
                "rules for this part. Try again.",
                icon=":material/error:",
            )
            return
    previous = st.session_state.get(STATE)
    if previous:
        clear_widget_state(f"p{previous['part']}_e{previous['exercise_id']}")
    st.session_state[STATE] = {
        "part": part,
        "exercise_id": exercise_id,
        "payload": payload,
        "started_at": time.time(),
    }
    st.session_state.pop(RESULT, None)


has_key = api_key_banner()

with st.container(key="cae_panel_part_choice"):
    controls = st.columns([3, 1], vertical_alignment="bottom")
    with controls[0]:
        chosen = st.selectbox(
            "Which part?",
            options=sorted(PART_SPECS),
            format_func=part_label,
            key="practice_part",
        )
    spec = PART_SPECS[chosen]
    with controls[1]:
        start = st.button(
            "Start",
            type="primary",
            width="stretch",
            disabled=not has_key,
            icon=":material/play_arrow:",
        )

    style.chips(
        [
            f"{spec.item_count} questions",
            f"{spec.max_marks} marks",
            f"about {spec.suggested_minutes} minutes in the real exam",
        ]
    )
    if spec.focus:
        st.markdown("**What it tests:** " + " · ".join(spec.focus))

if start:
    load(chosen)

current = st.session_state.get(STATE)
if current is None:
    st.info(
        "Choose a part and press **Start**. A new exercise is written for you, "
        "aimed at the mistakes on your record.",
        icon=":material/play_arrow:",
    )
    st.stop()

st.divider()

part = current["part"]
prefix = f"p{part}_e{current['exercise_id']}"
result = st.session_state.get(RESULT)


if result is None:
    with st.form(f"form_{prefix}", border=False):
        answers = render_exercise(part, current["payload"], prefix)
        st.divider()
        submitted = st.form_submit_button(
            "Submit answers", type="primary", icon=":material/task_alt:"
        )
    if submitted:
        marked = False
        with st.spinner("Marking your answers."):
            try:
                st.session_state[RESULT] = submit_attempt(
                    user_id=user["id"],
                    exercise_id=current["exercise_id"],
                    part=part,
                    exercise=current["payload"],
                    responses=answers,
                    duration_seconds=int(time.time() - current["started_at"]),
                    mode="practice",
                )
                marked = True
            except LLMUnavailableError as exc:
                llm_error(exc)
            except MarkingFailed as exc:
                st.error(
                    f"{exc}\n\nYour answers have not been saved. Submit them again to "
                    "have another go.",
                    icon=":material/error:",
                )
        if marked:
            st.rerun()


else:
    st.markdown(f"## Results · {part_label(part)}")
    render_score_header(result["marks"], result["max_marks"])
    st.divider()
    render_reading_results(part, current["payload"], result["rows"])

    st.divider()
    again = st.columns([1, 1, 3])
    with again[0]:
        if st.button(
            "Next exercise",
            type="primary",
            width="stretch",
            icon=":material/arrow_forward:",
        ):
            load(part)
            st.rerun()
    with again[1]:
        st.page_link(
            "pages/4_Progress.py", label="See progress", icon=":material/insights:"
        )
