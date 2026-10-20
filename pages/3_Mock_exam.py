"""A timed mock exam: either paper on its own, or both back to back.

Each paper runs against its own clock and is prepared just before it starts, so
the Writing tasks are not generated while the Reading clock is already running.
When a clock reaches zero the paper is handed in as it stands.

A Reading part or a Writing task is saved whole or not at all: only when the
learner confirms it with "Next part"/"Next task", "Back" or "Finish". When time
runs out, whatever was changed on the open one since it was last confirmed is
lost; one never confirmed scores zero, even if some of it was answered.

The handed-in Reading paper is marked whole or not at all, too: if the model
cannot mark one part, no part is saved, and the exam waits with its clock
stopped until the learner asks for it to be marked again.
"""

from __future__ import annotations

import random
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
    READING_PART_NUMBERS,
    READING_TIME_MINUTES,
    WRITING_GENRES,
    WRITING_MAX_MARKS_PER_TASK,
    WRITING_PART2_GENRES,
    WRITING_TIME_MINUTES,
    WRITING_WORD_RANGE,
)
from cae.domain.scoring import combined_summary, summarise_reading  # noqa: E402
from cae.graphs.rue_evaluator import MarkingFailed  # noqa: E402
from cae.graphs.rue_task import GenerationFailed  # noqa: E402
from cae.graphs.writing_task import WritingTaskFailed  # noqa: E402
from cae.llm import LLMUnavailableError  # noqa: E402
from cae.services import (  # noqa: E402
    complete_mock_exam,
    next_exercise,
    next_writing_task,
    part_label,
    start_mock_exam,
    submit_mock_reading,
    submit_writing,
)
from cae.ui.feedback import (  # noqa: E402
    SCALE_CAVEAT,
    render_reading_results,
    render_writing_result,
)
from cae.ui.renderers import (  # noqa: E402
    render_exercise,
    render_word_count,
    render_writing_task,
)
from cae.ui.shell import (  # noqa: E402
    api_key_banner,
    countdown,
    llm_error,
    page_header,
    page_setup,
    require_login,
    sidebar_account,
)

page_setup("Mock exam", icon="⏱️")
user = require_login()
sidebar_account(user)

STATE = "mock"
MIN_WORDS, MAX_WORDS = WRITING_WORD_RANGE
WRITING_TASK_COUNT = 2

PAPER_CHOICES: dict[str, tuple[str, ...]] = {
    "Reading and Use of English": ("reading",),
    "Writing": ("writing",),
    "Both papers": ("reading", "writing"),
}


def paper_minutes(papers: tuple[str, ...]) -> int:
    return (READING_TIME_MINUTES if "reading" in papers else 0) + (
        WRITING_TIME_MINUTES if "writing" in papers else 0
    )


def mock() -> dict | None:
    return st.session_state.get(STATE)


def abandon() -> None:
    st.session_state.pop(STATE, None)


def time_is_up(state: dict) -> bool:
    return state["phase"] in ("reading", "writing") and time.time() >= state["deadline"]


def describe_reading() -> None:
    st.markdown(
        f"**Reading and Use of English** — 8 parts, "
        f"{sum(s.item_count for s in PART_SPECS.values())} questions, "
        f"{READING_MAX_MARKS} marks, {READING_TIME_MINUTES} minutes."
    )
    for number, spec in PART_SPECS.items():
        st.markdown(
            f"- **Part {number}** · {spec.name} — {spec.item_count} questions, "
            f"{spec.max_marks} marks, about {spec.suggested_minutes} minutes"
        )


def describe_writing() -> None:
    st.markdown(
        f"**Writing** — {WRITING_TASK_COUNT} tasks of {MIN_WORDS}-{MAX_WORDS} words, "
        f"{WRITING_TASK_COUNT * WRITING_MAX_MARKS_PER_TASK} marks, "
        f"{WRITING_TIME_MINUTES} minutes. A task outside {MIN_WORDS}-{MAX_WORDS} "
        "words cannot be handed in."
    )
    st.markdown("- **Part 1** · Essay — compulsory")
    st.markdown(
        "- **Part 2** · one of "
        + ", ".join(WRITING_GENRES[g].name.lower() for g in WRITING_PART2_GENRES)
        + ", chosen for you"
    )


def prepare_reading() -> dict[int, dict] | None:
    """Generate all eight exercises. Returns None if something went wrong."""
    exercises: dict[int, dict] = {}
    progress = st.progress(0.0, text="Preparing Part 1...")
    for index, part in enumerate(READING_PART_NUMBERS, start=1):
        progress.progress(
            (index - 1) / len(READING_PART_NUMBERS),
            text=f"Preparing {part_label(part)} ({index} of {len(READING_PART_NUMBERS)})",
        )
        try:
            exercise_id, payload = next_exercise(user["id"], part)
        except LLMUnavailableError as exc:
            progress.empty()
            llm_error(exc)
            return None
        except GenerationFailed as exc:
            progress.empty()
            st.error(f"Part {part} could not be prepared: {exc}", icon=":material/error:")
            return None
        exercises[part] = {"exercise_id": exercise_id, "payload": payload}
    progress.progress(1.0, text="Ready.")
    return exercises


def prepare_writing() -> list[dict] | None:
    """Generate the two writing tasks. Returns None if something went wrong."""
    tasks: list[dict] = []
    try:
        for part, genre in ((1, "essay"), (2, random.choice(WRITING_PART2_GENRES))):
            task_id, payload = next_writing_task(part=part, genre=genre)
            tasks.append(
                {
                    "task_id": task_id,
                    "part": part,
                    "genre": payload["genre"],
                    "payload": payload,
                    "script": "",
                    "draft": "",
                }
            )
    except LLMUnavailableError as exc:
        llm_error(exc)
        return None
    except WritingTaskFailed as exc:
        st.error(f"The Writing paper could not be prepared: {exc}", icon=":material/error:")
        return None
    return tasks


def render_setup() -> None:
    page_header(
        "Mock exam",
        "Sit a paper against the clock, exactly as long as the real one.",
    )
    has_key = api_key_banner()

    choice = st.radio(
        "Which paper do you want to sit?",
        options=list(PAPER_CHOICES),
        format_func=lambda label: f"{label} — {paper_minutes(PAPER_CHOICES[label])} minutes",
        key="mock_paper_choice",
    )
    papers = PAPER_CHOICES[choice]

    st.divider()
    st.markdown("#### What you are about to sit")
    if "reading" in papers:
        describe_reading()
    if "writing" in papers:
        if "reading" in papers:
            st.markdown("")
        describe_writing()
    if len(papers) > 1:
        st.caption(
            f"The two papers are timed separately: {READING_TIME_MINUTES} minutes, then "
            f"{WRITING_TIME_MINUTES} minutes. The Writing tasks are written while the "
            "Reading clock is stopped."
        )

    st.warning(
        "Everything is prepared before the clock starts, which takes a few minutes. "
        "Do not close the tab while it is being written. When the clock reaches "
        "zero the paper is handed in and marked, and only the parts and tasks you "
        "have confirmed with Next part or Next task count.",
        icon=":material/hourglass:",
    )

    if not st.button("Start the exam", type="primary", disabled=not has_key):
        return

    state = {
        "exam_id": start_mock_exam(user["id"]),
        "papers": papers,
        "exercises": {},
        "answers": {},
        "index": 0,
        "results": {},
        "writing": [],
        "writing_index": 0,
    }

    if "reading" in papers:
        exercises = prepare_reading()
        if exercises is None:
            return
        state["exercises"] = exercises
        state["phase"] = "reading"
        state["deadline"] = time.time() + READING_TIME_MINUTES * 60
    else:
        with st.spinner("Preparing the Writing paper."):
            tasks = prepare_writing()
        if tasks is None:
            return
        state["writing"] = tasks
        state["phase"] = "writing"
        state["deadline"] = time.time() + WRITING_TIME_MINUTES * 60

    st.session_state[STATE] = state
    st.rerun()


def finish_exam(state: dict) -> None:
    """Record the whole exam once, over whichever papers were actually sat."""
    reading_marks = None
    marks = 0.0
    max_marks = 0.0
    if state["results"]:
        reading_marks = sum(r["marks"] for r in state["results"].values())
        marks += reading_marks
        max_marks += READING_MAX_MARKS

    written = [t for t in state["writing"] if t.get("result")]
    bands = [t["result"].get("bands", {}) for t in written]
    if written:
        marks += sum(t["result"].get("total", 0) for t in written)
        max_marks += WRITING_MAX_MARKS_PER_TASK * len(written)

    overall = combined_summary(reading_marks, bands or None)
    complete_mock_exam(
        exam_id=state["exam_id"],
        marks=marks,
        max_marks=max_marks,
        scale_score=overall.scale_score if overall else 0,
    )
    state["overall"] = overall
    state["phase"] = "done"


def grade_reading(state: dict) -> None:
    """Mark the handed-in Reading paper, all of it or none of it.

    If a part cannot be marked nothing is saved, and the exam waits in the
    ``unmarked`` phase, clock stopped and answers kept, for another try.
    """
    progress = st.progress(0.0, text="Marking Part 1...")

    def on_part(part: int) -> None:
        position = READING_PART_NUMBERS.index(part)
        progress.progress(position / len(READING_PART_NUMBERS), text=f"Marking Part {part}...")

    try:
        results = submit_mock_reading(
            user_id=user["id"],
            mock_exam_id=state["exam_id"],
            exercises={part: state["exercises"][part] for part in READING_PART_NUMBERS},
            answers=state["answers"],
            on_part=on_part,
        )
    except (LLMUnavailableError, MarkingFailed) as exc:
        progress.empty()
        state["phase"] = "unmarked"
        state["marking_error"] = exc
        return
    progress.empty()

    state.pop("marking_error", None)
    state["results"] = results
    state["reading_summary"] = summarise_reading(sum(r["marks"] for r in results.values()))

    if "writing" not in state["papers"]:
        finish_exam(state)
        return

    with st.spinner("Preparing the Writing paper."):
        tasks = prepare_writing()
    if tasks is None:
        finish_exam(state)
        return
    state["writing"] = tasks
    state["phase"] = "writing"
    state["deadline"] = time.time() + WRITING_TIME_MINUTES * 60


def render_reading(state: dict) -> None:
    part = READING_PART_NUMBERS[state["index"]]
    total = len(READING_PART_NUMBERS)
    entry = state["exercises"][part]

    page_header(f"Mock exam · {part_label(part)}", f"Part {state['index'] + 1} of {total}")
    countdown(state["deadline"], READING_TIME_MINUTES, "Reading and Use of English")

    prefix = f"mock{state['exam_id']}_p{part}"
    is_last = state["index"] == total - 1

    for number, answer in state["answers"].get(part, {}).items():
        if answer:
            st.session_state.setdefault(f"{prefix}_q{number}", answer)

    with st.form(f"mockform_{prefix}"):
        answers = render_exercise(part, entry["payload"], prefix)
        cols = st.columns([1, 1, 4])
        with cols[0]:
            back = st.form_submit_button(
                "Back", width="stretch", disabled=state["index"] == 0
            )
        with cols[1]:
            forward = st.form_submit_button(
                "Finish paper" if is_last else "Next part",
                type="primary",
                width="stretch",
            )

    if back or forward:
        state["answers"][part] = answers
        if back:
            state["index"] -= 1
            st.rerun()
        elif is_last:
            grade_reading(state)
            st.rerun()
        else:
            state["index"] += 1
            st.rerun()

    st.divider()
    with st.expander("Leave the exam"):
        st.caption("Your answers so far will not be marked.")
        if st.button("Abandon this exam"):
            abandon()
            st.rerun()


def render_unmarked(state: dict) -> None:
    """The Reading paper was handed in but could not be marked."""
    page_header("Mock exam · Reading not marked yet")
    error = state.get("marking_error")
    if isinstance(error, LLMUnavailableError):
        llm_error(error)
    else:
        st.error(str(error), icon=":material/error:")
    st.info(
        "Your answers are kept exactly as you handed them in, nothing has been "
        "saved yet, and the clock has stopped. Mark the paper again when the model "
        "is back.",
        icon=":material/pause_circle:",
    )
    cols = st.columns([1, 1, 3])
    with cols[0]:
        if st.button("Mark it again", type="primary", width="stretch"):
            grade_reading(state)
            st.rerun()
    with cols[1]:
        if st.button("Abandon this exam", width="stretch"):
            abandon()
            st.rerun()


def mark_writing(state: dict) -> None:
    tasks = state["writing"]
    progress = st.progress(0.0, text="Marking your writing...")
    for position, entry in enumerate(tasks, start=1):
        progress.progress((position - 1) / len(tasks), text=f"Marking task {position}...")
        entry["result"] = submit_writing(
            user_id=user["id"],
            task_id=entry["task_id"],
            part=entry["part"],
            genre=entry["genre"],
            task=entry["payload"],
            script=entry["script"],
            mock_exam_id=state["exam_id"],
        )
    progress.empty()
    finish_exam(state)


def render_writing(state: dict) -> None:
    tasks = state["writing"]
    index = state["writing_index"]
    task = tasks[index]

    page_header(
        f"Mock exam · Writing Part {task['part']}", f"Task {index + 1} of {len(tasks)}"
    )
    countdown(state["deadline"], WRITING_TIME_MINUTES, "Writing")

    task_col, write_col = st.columns([2, 3], gap="large")
    with task_col:
        render_writing_task(task["payload"])
    with write_col:
        st.markdown("### Your answer")
        script = st.text_area(
            "Your answer",
            value=task["draft"],
            height=460,
            key=f"mockwrite_{state['exam_id']}_{index}",
            label_visibility="collapsed",
        )
        in_range = render_word_count(script)

        is_last = index == len(tasks) - 1
        cols = st.columns([1, 1, 3])
        with cols[0]:
            advance = st.button(
                "Finish the paper" if is_last else "Next task",
                type="primary",
                width="stretch",
                disabled=not in_range,
            )
        with cols[1]:
            back = st.button(
                "Back", width="stretch", disabled=index == 0
            )

        if advance or back:
            task["draft"] = script
            if in_range:
                task["script"] = script
            if back:
                state["writing_index"] -= 1
            elif is_last:
                mark_writing(state)
            else:
                state["writing_index"] += 1
            st.rerun()


def hand_in(state: dict) -> None:
    """Collect the paper whose clock has run out, as it stands, and mark it."""
    paper = state["phase"]
    state.setdefault("timed_out", []).append(paper)
    page_header("Time is up")
    st.info("Handing in what you have and marking it.", icon=":material/alarm:")
    if paper == "reading":
        grade_reading(state)
    else:
        mark_writing(state)


def render_results(state: dict) -> None:
    page_header("Mock exam results")
    notes = {
        "reading": (
            "Time ran out on Reading and Use of English. Only the parts you had "
            "confirmed with Next part were handed in; changes to the part you had "
            "open were not saved, and parts never confirmed score zero."
        ),
        "writing": (
            "Time ran out on Writing. Only the tasks you had confirmed with Next "
            "task were handed in; changes to the task you had open were not saved, "
            "and tasks never confirmed score zero."
        ),
    }
    for paper in state.get("timed_out", []):
        st.info(notes[paper], icon=":material/alarm:")

    reading = state.get("reading_summary")
    written = [t for t in state["writing"] if t.get("result")]
    overall = state.get("overall")

    metrics: list[tuple[str, object]] = []
    if reading is not None:
        metrics.append(
            ("Reading & UoE", f"{reading.marks:g} / {reading.max_marks:g}")
        )
        metrics.append(("Accuracy", f"{reading.percentage:g}%"))
    if written:
        earned = sum(t["result"].get("total", 0) for t in written)
        metrics.append(
            ("Writing", f"{earned} / {WRITING_MAX_MARKS_PER_TASK * len(written)}")
        )
    metrics.append(("Estimated scale score", overall.scale_score if overall else 0))
    metrics.append(("Estimated grade", overall.grade if overall else "-"))

    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)
    caveat = SCALE_CAVEAT
    if len(state["papers"]) == 1:
        caveat += " This mock covered one paper, so the estimate rests on that alone."
    st.caption(
        caveat
        + " The real certificate also weighs Listening and Speaking, which this app "
        "does not cover."
    )

    if reading is not None:
        st.divider()
        st.markdown("#### Marks by part")
        st.dataframe(
            [
                {
                    "Part": part_label(part),
                    "Marks": f"{result['marks']:g} / {result['max_marks']:g}",
                    "Accuracy": f"{100 * result['marks'] / result['max_marks']:.0f}%",
                }
                for part, result in state["results"].items()
            ],
            hide_index=True,
            width="stretch",
        )
        st.markdown("#### Item feedback")
        for part, result in state["results"].items():
            with st.expander(
                f"{part_label(part)} — {result['marks']:g}/{result['max_marks']:g}"
            ):
                render_reading_results(
                    part, state["exercises"][part]["payload"], result["rows"]
                )

    if written:
        st.divider()
        st.markdown("#### Writing")
        for task in written:
            result = task["result"]
            genre = WRITING_GENRES[task["genre"]].name
            with st.expander(
                f"Part {task['part']} · {genre} — {result.get('total', 0)}/20"
            ):
                render_writing_result(result, task["script"])

    st.divider()
    actions = st.columns([1, 1, 3])
    with actions[0]:
        if st.button("New mock exam", type="primary", width="stretch"):
            abandon()
            st.rerun()
    with actions[1]:
        st.page_link("pages/4_Progress.py", label="See progress", icon=":material/insights:")


state = mock()
if state is not None and time_is_up(state):
    hand_in(state)
    st.rerun()

if state is None:
    render_setup()
elif state["phase"] == "reading":
    render_reading(state)
elif state["phase"] == "unmarked":
    render_unmarked(state)
elif state["phase"] == "writing":
    render_writing(state)
else:
    render_results(state)
