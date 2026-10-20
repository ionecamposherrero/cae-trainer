"""One renderer per exam part.

Every renderer is called inside a ``st.form`` and returns ``{item_number: answer}``
from the widgets' return values.
"""

from __future__ import annotations

import re
from typing import Any, Callable

import streamlit as st

from cae.domain.parts import WRITING_WORD_RANGE, get_part_spec
from cae.graphs.writing_evaluator import count_words

_GAP_RE = re.compile(r"\{\{(\d+)\}\}")


def _gap_label(number: int, stem: str = "") -> str:
    inner = f"{number}" if not stem else f"{number} · {stem.upper()}"
    return f" **&nbsp;({inner})&nbsp;** "


def render_passage(text: str, stems: dict[int, str] | None = None) -> None:
    """Show a passage with its gaps marked as (1), (2), ..."""
    stems = stems or {}

    def replace(match: re.Match[str]) -> str:
        number = int(match.group(1))
        return _gap_label(number, stems.get(number, ""))

    body = _GAP_RE.sub(replace, text)
    for paragraph in body.split("\n"):
        if paragraph.strip():
            st.markdown(paragraph.strip())


def _letter_options(options: list[str], letters: tuple[str, ...]) -> Callable[[str], str]:
    lookup = dict(zip(letters, options))

    def fmt(letter: str) -> str:
        return f"{letter}  ·  {lookup.get(letter, '')}"

    return fmt


def _blank_choice(letters: tuple[str, ...]) -> list[str]:
    return ["", *letters]


def render_part_1(exercise: dict[str, Any], prefix: str) -> dict[int, str]:
    spec = get_part_spec(1)
    answers: dict[int, str] = {}
    text_col, answer_col = st.columns([3, 2], gap="large")
    with text_col:
        render_passage(exercise["text"])
    with answer_col:
        for item in exercise["items"]:
            number = item["number"]
            answers[number] = (
                st.radio(
                    f"Gap {number}",
                    options=spec.option_letters,
                    format_func=_letter_options(item["options"], spec.option_letters),
                    key=f"{prefix}_q{number}",
                    index=None,
                )
                or ""
            )
    return answers


def render_part_2(exercise: dict[str, Any], prefix: str) -> dict[int, str]:
    answers: dict[int, str] = {}
    text_col, answer_col = st.columns([3, 2], gap="large")
    with text_col:
        render_passage(exercise["text"])
    with answer_col:
        st.caption("Use exactly ONE word in each gap.")
        for item in exercise["items"]:
            number = item["number"]
            answers[number] = st.text_input(f"Gap {number}", key=f"{prefix}_q{number}")
    return answers


def render_part_3(exercise: dict[str, Any], prefix: str) -> dict[int, str]:
    stems = {item["number"]: item["stem"] for item in exercise["items"]}
    answers: dict[int, str] = {}
    text_col, answer_col = st.columns([3, 2], gap="large")
    with text_col:
        render_passage(exercise["text"], stems=stems)
    with answer_col:
        st.caption("Form a word from the word in capitals.")
        for item in exercise["items"]:
            number = item["number"]
            answers[number] = st.text_input(
                f"Gap {number} — {item['stem'].upper()}", key=f"{prefix}_q{number}"
            )
    return answers


def render_part_4(exercise: dict[str, Any], prefix: str) -> dict[int, str]:
    answers: dict[int, str] = {}
    for item in exercise["items"]:
        number = item["number"]
        st.markdown(f"**{number}.**  {item['lead_in']}")
        st.markdown(f"### `{item['key_word'].upper()}`")
        left, middle, right = st.columns([2, 3, 2], vertical_alignment="center")
        with left:
            st.markdown(f"{item['sentence_start']}")
        with middle:
            answers[number] = st.text_input(
                "Answer",
                key=f"{prefix}_q{number}",
                label_visibility="collapsed",
                placeholder="three to six words",
            )
        with right:
            st.markdown(f"{item['sentence_end']}")
        st.divider()
    return answers


def render_part_5(exercise: dict[str, Any], prefix: str) -> dict[int, str]:
    spec = get_part_spec(5)
    answers: dict[int, str] = {}
    text_col, answer_col = st.columns([3, 2], gap="large")
    with text_col:
        for paragraph in exercise["text"].split("\n"):
            if paragraph.strip():
                st.markdown(paragraph.strip())
    with answer_col:
        for item in exercise["items"]:
            number = item["number"]
            st.markdown(f"**{number}.**  {item['question']}")
            answers[number] = (
                st.radio(
                    f"Question {number}",
                    options=spec.option_letters,
                    format_func=_letter_options(item["options"], spec.option_letters),
                    key=f"{prefix}_q{number}",
                    index=None,
                    label_visibility="collapsed",
                )
                or ""
            )
            st.divider()
    return answers


def render_part_6(exercise: dict[str, Any], prefix: str) -> dict[int, str]:
    letters = tuple(t["letter"] for t in exercise["texts"])
    answers: dict[int, str] = {}
    tabs = st.tabs([f"Text {t['letter']} — {t['author']}" for t in exercise["texts"]])
    for tab, text in zip(tabs, exercise["texts"]):
        with tab:
            st.markdown(text["content"])
    st.divider()
    for item in exercise["items"]:
        number = item["number"]
        left, right = st.columns([4, 1], vertical_alignment="center")
        with left:
            st.markdown(f"**{number}.**  {item['prompt']}")
        with right:
            answers[number] = st.selectbox(
                f"Answer {number}",
                options=_blank_choice(letters),
                key=f"{prefix}_q{number}",
                label_visibility="collapsed",
            )
    return answers


def render_part_7(exercise: dict[str, Any], prefix: str) -> dict[int, str]:
    letters = tuple(o["letter"] for o in exercise["options"])
    answers: dict[int, str] = {}
    text_col, answer_col = st.columns([3, 2], gap="large")
    with text_col:
        render_passage(exercise["text"])
    with answer_col:
        st.markdown("#### Missing paragraphs")
        for option in exercise["options"]:
            with st.expander(f"Paragraph {option['letter']}"):
                st.markdown(option["paragraph"])
        st.divider()
        st.caption("One paragraph is not used.")
        for item in exercise["items"]:
            number = item["number"]
            answers[number] = st.selectbox(
                f"Gap {number}",
                options=_blank_choice(letters),
                key=f"{prefix}_q{number}",
            )
    return answers


def render_part_8(exercise: dict[str, Any], prefix: str) -> dict[int, str]:
    letters = tuple(s["letter"] for s in exercise["sections"])
    answers: dict[int, str] = {}
    text_col, answer_col = st.columns([3, 2], gap="large")
    with text_col:
        for section in exercise["sections"]:
            st.markdown(f"#### {section['letter']}  ·  {section['heading']}")
            st.markdown(section["content"])
    with answer_col:
        st.caption("In which section is the following mentioned?")
        for item in exercise["items"]:
            number = item["number"]
            st.markdown(f"**{number}.**  {item['prompt']}")
            answers[number] = st.selectbox(
                f"Answer {number}",
                options=_blank_choice(letters),
                key=f"{prefix}_q{number}",
                label_visibility="collapsed",
            )
    return answers


RENDERERS: dict[int, Callable[[dict[str, Any], str], dict[int, str]]] = {
    1: render_part_1,
    2: render_part_2,
    3: render_part_3,
    4: render_part_4,
    5: render_part_5,
    6: render_part_6,
    7: render_part_7,
    8: render_part_8,
}


def render_exercise(part: int, exercise: dict[str, Any], prefix: str) -> dict[int, str]:
    """Render one exercise and return the candidate's answers."""
    spec = get_part_spec(part)
    st.markdown(f"### {exercise.get('title', spec.name)}")
    st.info(spec.instructions, icon=":material/info:")
    return RENDERERS[part](exercise, prefix)


def render_word_count(script: str) -> bool:
    """Show the script's length against the required range.

    Returns True only when it is within the range: a script outside it cannot be
    handed in, and the message says so, because it is the reason the button is
    disabled. Counted the way the marker counts, so the two cannot disagree.
    """
    low, high = WRITING_WORD_RANGE
    words = count_words(script)
    if low <= words <= high:
        st.success(f"{words} words — within the required range.")
        return True
    if words == 0:
        st.caption(f"0 words · write {low}-{high} words to hand it in.")
    elif words < low:
        st.warning(
            f"{words} words — {low - words} short of the minimum. You can hand it "
            f"in once it is {low}-{high} words."
        )
    else:
        st.warning(
            f"{words} words — {words - high} over the maximum. You can hand it in "
            f"once it is {low}-{high} words."
        )
    return False


def render_writing_task(task: dict[str, Any]) -> None:
    """Show a writing task the way the question paper presents it."""
    from cae.domain.parts import get_genre

    genre = get_genre(task.get("genre", "essay"))
    st.markdown(f"### {task.get('title', genre.name)}")
    st.caption(f"Writing Part {task.get('part', 1)} · {genre.name}")

    with st.container(border=True):
        st.markdown(task.get("scenario", ""))
        notes = task.get("notes") or []
        if notes:
            st.markdown("**Notes**")
            for note in notes:
                st.markdown(f"- {note}")
        opinions = task.get("opinions") or []
        if opinions:
            st.markdown("**Some opinions expressed in the discussion**")
            for opinion in opinions:
                st.markdown(f"> *“{opinion}”*")
        st.markdown(f"**{task.get('instruction', '')}**")
