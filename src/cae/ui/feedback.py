"""Rendering of graded results: item feedback, writing bands, annotated scripts."""

from __future__ import annotations

import html
import re
from typing import Any

import streamlit as st

from cae.domain.parts import SUBSCALE_LABELS, SUBSCALES, get_part_spec
from cae.domain.scoring import ScoreSummary, summarise
from cae.domain.taxonomy import label as category_label

SCALE_CAVEAT = (
    "Cambridge does not publish raw-to-scale conversion tables, so the scale score "
    "shown here is an estimate based on the published grade boundaries."
)

_SEVERITY_COLOUR = {
    "slip": "rgba(250,178,25,0.28)",
    "error": "rgba(236,131,90,0.28)",
    "impedes": "rgba(208,59,59,0.28)",
}
_CORRECT_WASH = "rgba(12,163,12,0.20)"
_WRONG_WASH = "rgba(208,59,59,0.20)"

PARTS_WITH_MATERIAL = frozenset({1, 2, 3, 5, 6, 7, 8})
GAPPED_PARTS = frozenset({1, 2, 3, 7})


def render_score_header(marks: float, max_marks: float, *, caption: str = "") -> ScoreSummary:
    summary = summarise(marks, max_marks)
    cols = st.columns(4)
    cols[0].metric("Marks", f"{marks:g} / {max_marks:g}")
    cols[1].metric("Accuracy", f"{summary.percentage:g}%")
    cols[2].metric("Estimated scale score", summary.scale_score)
    cols[3].metric("Estimated grade", summary.grade)
    st.caption(caption or SCALE_CAVEAT)
    return summary


def _items_by_number(exercise: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {item["number"]: item for item in exercise.get("items", [])}


def _readable_answer(part: int, item: dict[str, Any] | None, value: str) -> str:
    """A stored answer as display text; Part 1 option letters become the option."""
    value = (value or "").strip()
    if not value:
        return ""
    if part == 1 and item:
        options = item.get("options") or []
        index = "ABCD".find(value.upper())
        if 0 <= index < len(options):
            return str(options[index])
    return value


def _gap_badge(part: int, item: dict[str, Any] | None, row: dict[str, Any]) -> str:
    number = row["item_number"]
    correct = _readable_answer(part, item, str(row["correct_answer"]).split(" / ")[0])
    given = _readable_answer(part, item, row["user_answer"])
    wash = _CORRECT_WASH if row["is_correct"] else _WRONG_WASH
    if row["is_correct"]:
        body = html.escape(given or correct)
    elif given:
        body = (
            f'<s style="opacity:.65">{html.escape(given)}</s> '
            f'<b>{html.escape(correct)}</b>'
        )
    else:
        body = f'<i style="opacity:.65">blank</i> <b>{html.escape(correct)}</b>'
    return (
        f'<span style="background:{wash};color:inherit;border-radius:4px;'
        f'padding:1px 6px;margin:0 2px;display:inline-block">'
        f'<sup style="opacity:.7">{number}</sup> {body}</span>'
    )


def render_reviewed_passage(
    part: int, exercise: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    """The passage with every gap filled in and marked right or wrong."""
    items = _items_by_number(exercise)
    badges = {
        row["item_number"]: _gap_badge(part, items.get(row["item_number"]), row)
        for row in rows
    }
    body = html.escape(exercise.get("text", ""))
    for number, badge in badges.items():
        body = body.replace("{{" + str(number) + "}}", badge)
    paragraphs = "".join(
        f'<p style="margin:0 0 .8em 0">{line.strip()}</p>'
        for line in body.split("\n")
        if line.strip()
    )
    st.markdown(
        f'<div style="line-height:2.1">{paragraphs}</div>', unsafe_allow_html=True
    )
    st.caption("Green is what you got right; red shows your answer struck through, "
               "then the key in bold.")


def _render_plain_text(text: str) -> None:
    for paragraph in text.split("\n"):
        if paragraph.strip():
            st.markdown(paragraph.strip())


def render_exercise_material(
    part: int, exercise: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    """Show the material the questions referred to, next to the feedback."""
    st.markdown(f"#### {exercise.get('title', get_part_spec(part).name)}")

    if part in GAPPED_PARTS:
        render_reviewed_passage(part, exercise, rows)
        if part == 7:
            st.markdown("**The paragraphs you were choosing from**")
            used = {str(row["correct_answer"]).split(" / ")[0] for row in rows}
            for option in exercise.get("options", []):
                letter = option["letter"]
                mark = "" if letter in used else "  ·  not used"
                with st.expander(f"Paragraph {letter}{mark}"):
                    st.markdown(option["paragraph"])
        return

    if part == 5:
        _render_plain_text(exercise.get("text", ""))
        return

    if part == 6:
        texts = exercise.get("texts", [])
        for tab, text in zip(
            st.tabs([f"Text {t['letter']} — {t['author']}" for t in texts]), texts
        ):
            with tab:
                st.markdown(text["content"])
        return

    if part == 8:
        for section in exercise.get("sections", []):
            st.markdown(f"##### {section['letter']}  ·  {section['heading']}")
            st.markdown(section["content"])


def _item_context(part: int, item: dict[str, Any] | None, row: dict[str, Any]) -> str:
    """A one-line reminder of what this item actually asked."""
    if item is None:
        return ""
    if part == 4:
        given = row["user_answer"] or "..."
        return (
            f"{item['lead_in']}\n\n"
            f"**{item['key_word'].upper()}**\n\n"
            f"{item['sentence_start']} **{given}** {item['sentence_end']}"
        )
    if part in (5, 6, 8):
        return str(item.get("question") or item.get("prompt") or "")
    return ""


def render_item_results(
    part: int, rows: list[dict[str, Any]], exercise: dict[str, Any] | None = None
) -> None:
    """One expandable card per item, wrong answers open by default."""
    spec = get_part_spec(part)
    items = _items_by_number(exercise or {})
    st.markdown(f"#### Item feedback · Part {part}: {spec.name}")
    for row in rows:
        correct = bool(row["is_correct"])
        partial = 0 < row["marks_awarded"] < row["max_marks"]
        icon = "✅" if correct else ("🟡" if partial else "❌")
        given = row["user_answer"] or "(blank)"
        header = (
            f"{icon}  Item {row['item_number']} — {row['marks_awarded']:g}/"
            f"{row['max_marks']:g}   ·   you wrote: {given}"
        )
        item = items.get(row["item_number"])
        with st.expander(header, expanded=not correct):
            context = _item_context(part, item, row)
            if context:
                st.markdown(context)
                st.divider()
            if not correct:
                st.markdown(f"**Answer key:** {row['correct_answer']}")
            if row.get("reason"):
                st.caption(row["reason"])
            if row.get("adjudicated"):
                st.caption("This answer was reviewed individually against the key.")
            if item and item.get("evidence"):
                st.markdown(f"**In the text:** *“{item['evidence']}”*")
            if row.get("category"):
                st.markdown(f"**Weakness:** {category_label(row['category'])}")
            if row.get("explanation"):
                st.markdown(row["explanation"])


def render_reading_results(
    part: int, exercise: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    """The exercise and its feedback side by side."""
    if part not in PARTS_WITH_MATERIAL:
        render_item_results(part, rows, exercise)
        return

    material_col, feedback_col = st.columns([3, 2], gap="large")
    with material_col:
        if part in (5, 6, 7, 8):
            with st.container(height=620):
                render_exercise_material(part, exercise, rows)
        else:
            render_exercise_material(part, exercise, rows)
    with feedback_col:
        render_item_results(part, rows, exercise)


def render_bands(result: dict[str, Any]) -> None:
    bands = result.get("bands", {})
    cols = st.columns(4)
    for col, subscale in zip(cols, SUBSCALES):
        col.metric(SUBSCALE_LABELS[subscale], f"{bands.get(subscale, 0)} / 5")
    total = result.get("total", 0)
    st.progress(min(1.0, total / 20), text=f"Total {total} / 20")


def render_subscale_detail(result: dict[str, Any]) -> None:
    for subscale in SUBSCALES:
        detail = result.get(subscale) or {}
        band = detail.get("band", 0)
        with st.expander(f"{SUBSCALE_LABELS[subscale]} — band {band}/5", expanded=False):
            if detail.get("failed"):
                st.warning(detail.get("justification", "This subscale could not be assessed."))
                continue
            st.markdown(detail.get("justification", ""))
            evidence = detail.get("evidence") or []
            if evidence:
                st.markdown("**From your script:**")
                for quote in evidence:
                    st.markdown(f"> {quote}")
            if detail.get("to_improve"):
                st.success(f"**To reach the next band:** {detail['to_improve']}")


def highlight_script(script: str, annotations: list[dict[str, Any]]) -> str:
    """Wrap each locatable error span in a coloured mark with a tooltip."""
    spans: list[tuple[int, int, dict[str, Any]]] = []
    cursor_by_span: dict[str, int] = {}
    for annotation in annotations:
        span = annotation.get("span", "")
        if not annotation.get("found") or not span:
            continue
        start = script.find(span, cursor_by_span.get(span, 0))
        if start < 0:
            continue
        cursor_by_span[span] = start + len(span)
        spans.append((start, start + len(span), annotation))

    spans.sort(key=lambda s: s[0])
    out: list[str] = []
    position = 0
    for start, end, annotation in spans:
        if start < position:
            continue
        out.append(html.escape(script[position:start]))
        colour = _SEVERITY_COLOUR.get(annotation.get("severity", "error"), "#f2a1a1")
        tooltip = html.escape(
            f"{category_label(annotation['category'])}: {annotation.get('correction', '')}"
        )
        out.append(
            f'<mark style="background:{colour};color:inherit;border-radius:3px;'
            f'padding:0 2px" title="{tooltip}">{html.escape(script[start:end])}</mark>'
        )
        position = end
    out.append(html.escape(script[position:]))
    body = "".join(out).replace("\n", "<br>")
    return f'<div style="line-height:1.8">{body}</div>'


def render_annotated_script(script: str, annotations: list[dict[str, Any]]) -> None:
    st.markdown("#### Your script, marked up")
    located = [a for a in annotations if a.get("found")]
    if located:
        st.caption("Hover a highlight to see the correction. Colour shows severity.")
        st.markdown(highlight_script(script, located), unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="line-height:1.8">{html.escape(script)}</div>',
                    unsafe_allow_html=True)

    unlocated = [a for a in annotations if not a.get("found")]
    if annotations:
        st.markdown("#### Corrections")
        for annotation in annotations:
            st.markdown(
                f"- ~~{annotation['span']}~~ → **{annotation.get('correction', '')}**  \n"
                f"  *{category_label(annotation['category'])}* — "
                f"{annotation.get('explanation', '')}"
            )
    if unlocated:
        st.caption(
            f"{len(unlocated)} correction(s) could not be located in your text exactly, "
            "so they are listed but not highlighted."
        )


_MARKDOWN_HEADING_RE = re.compile(r"^ {0,3}#{1,6}(?:\s+(.*?))?[\s#]*$")


def plain_headings(text: str) -> str:
    """Turn Markdown headings into bold lines of body size.

    A report or proposal has headings, and the model tends to write them as
    ``## Introduction``, which ``st.markdown`` would draw larger than the section
    title above it. Each becomes a paragraph of its own, so a heading followed by
    a single newline does not run into the text under it.
    """
    lines: list[str] = []
    for line in text.splitlines():
        match = _MARKDOWN_HEADING_RE.match(line)
        if match is None:
            lines.append(line)
        elif match.group(1):
            lines += ["", f"**{match.group(1)}**", ""]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def render_improved_version(improved: dict[str, Any]) -> None:
    if not improved:
        return
    st.markdown("#### Your text at band 5")
    st.caption("Your ideas and structure, rewritten with stronger language.")
    st.markdown(plain_headings(improved.get("text", "")))
    upgrades = improved.get("upgrades") or []
    if upgrades:
        st.markdown("**What changed and why**")
        for upgrade in upgrades:
            st.markdown(
                f"- *{upgrade.get('original', '')}* → **{upgrade.get('improved', '')}**  \n"
                f"  {upgrade.get('why', '')}"
            )


def render_writing_result(result: dict[str, Any], script: str) -> None:
    if result.get("status") == "out_of_range":
        st.error(result.get("message", "This answer was outside the word range."))
        return

    cols = st.columns(3)
    cols[0].metric("Total", f"{result.get('total', 0)} / 20")
    cols[1].metric("Estimated scale score", result.get("scale_score", 0))
    cols[2].metric("Estimated grade", result.get("grade", "-"))
    st.caption(SCALE_CAVEAT)

    render_bands(result)
    strengths = result.get("strengths") or []
    if strengths:
        st.markdown("#### What you did well")
        for strength in strengths:
            st.markdown(f"- {strength}")

    st.markdown("#### The four subscales")
    render_subscale_detail(result)

    render_annotated_script(script, result.get("annotations") or [])
    render_improved_version(result.get("improved") or {})
