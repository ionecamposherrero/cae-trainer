"""Rendering the exercise material alongside its corrections."""

from __future__ import annotations

import pytest

from cae.graphs.rue_evaluator import grade_attempt
from cae.schemas.rue_task import answer_key
from cae.ui import feedback
from tests import factories

MODEL_CALLS: list[type] = []


def graded(part: int, wrong: set[int] | None = None) -> tuple[dict, list[dict]]:
    """Grade an exercise deterministically, getting the given items wrong."""
    exercise = factories.build(part)
    key = answer_key(part, exercise)
    wrong = wrong or set()
    responses = {}
    for item in key:
        if item.number in wrong:
            responses[item.number] = "" if item.number % 2 else _other_answer(part, item)
        else:
            responses[item.number] = item.answer
    MODEL_CALLS.clear()
    result = grade_attempt(part, exercise.model_dump(), responses)
    if not wrong:
        assert MODEL_CALLS == [], "a correct script must be marked without the model"
    return exercise.model_dump(), result["rows"]


def _other_answer(part: int, item) -> str:
    if len(item.answer) == 1 and item.answer.isupper():
        return "A" if item.answer != "A" else "B"
    return "nonsense"


@pytest.fixture(autouse=True)
def stubbed_model(monkeypatch: pytest.MonkeyPatch):
    """A model that adds nothing: no verdicts, no tags, so the marks stay deterministic.

    The grader catches a failed call, so a stub that raised would be swallowed;
    recording the calls is what lets ``graded`` check them.
    """
    from cae.graphs import rue_evaluator

    def fake(schema, system, user, **kwargs):
        MODEL_CALLS.append(schema)
        return factories.marking_reply(schema, user)

    monkeypatch.setattr(rue_evaluator, "structured_call", fake)


@pytest.mark.parametrize("part", sorted(feedback.GAPPED_PARTS))
def test_every_gap_is_filled_in_the_reviewed_passage(part: int) -> None:
    exercise, rows = graded(part)
    items = feedback._items_by_number(exercise)
    body = exercise["text"]
    for row in rows:
        badge = feedback._gap_badge(part, items.get(row["item_number"]), row)
        body = body.replace("{{" + str(row["item_number"]) + "}}", badge)

    assert "{{" not in body, "a gap marker was left unfilled in the review"


def test_a_correct_gap_is_washed_green_and_shows_the_answer() -> None:
    exercise, rows = graded(2)
    items = feedback._items_by_number(exercise)
    badge = feedback._gap_badge(2, items[1], rows[0])

    assert rows[0]["is_correct"]
    assert feedback._CORRECT_WASH in badge
    assert "which" in badge
    assert "<s " not in badge, "nothing to strike through when the answer was right"


def test_a_wrong_gap_strikes_the_answer_and_bolds_the_key() -> None:
    exercise, rows = graded(2, wrong={2})
    items = feedback._items_by_number(exercise)
    row = next(r for r in rows if r["item_number"] == 2)
    badge = feedback._gap_badge(2, items[2], row)

    assert feedback._WRONG_WASH in badge
    assert "<s " in badge and "nonsense" in badge
    assert "<b>which</b>" in badge


def test_a_blank_gap_says_so_rather_than_showing_nothing() -> None:
    exercise, rows = graded(2, wrong={1})
    items = feedback._items_by_number(exercise)
    badge = feedback._gap_badge(2, items[1], rows[0])

    assert "blank" in badge
    assert "<b>which</b>" in badge


def test_part_1_shows_the_option_text_not_the_bare_letter() -> None:
    """A letter on its own tells the learner nothing when they reread the text."""
    exercise, rows = graded(1, wrong={1})
    items = feedback._items_by_number(exercise)
    badge = feedback._gap_badge(1, items[1], rows[0])

    correct_letter = exercise["items"][0]["answer"]
    expected = exercise["items"][0]["options"]["ABCD".index(correct_letter)]
    assert expected in badge
    assert f"<b>{correct_letter}</b>" not in badge


def test_badges_escape_the_learners_own_text() -> None:
    exercise, rows = graded(2, wrong={2})
    items = feedback._items_by_number(exercise)
    row = next(r for r in rows if r["item_number"] == 2)
    row["user_answer"] = "<script>alert(1)</script>"
    badge = feedback._gap_badge(2, items[2], row)

    assert "<script>" not in badge
    assert "&lt;script&gt;" in badge


def test_part_4_cards_rebuild_the_sentence_with_the_learners_answer() -> None:
    """Part 4 has no passage, so the card has to carry its own context."""
    exercise, rows = graded(4, wrong={1})
    items = feedback._items_by_number(exercise)
    context = feedback._item_context(4, items[1], rows[0])

    assert exercise["items"][0]["lead_in"] in context
    assert "**WISH**" in context
    assert exercise["items"][0]["sentence_end"] in context


@pytest.mark.parametrize(("part", "field"), [(5, "question"), (6, "prompt"), (8, "prompt")])
def test_comprehension_cards_repeat_the_question(part: int, field: str) -> None:
    exercise, rows = graded(part)
    items = feedback._items_by_number(exercise)
    context = feedback._item_context(part, items[1], rows[0])

    assert context == exercise["items"][0][field]


def test_gapped_parts_need_no_item_context() -> None:
    """The passage beside the cards already shows where each gap sits."""
    exercise, rows = graded(3)
    items = feedback._items_by_number(exercise)
    assert feedback._item_context(3, items[1], rows[0]) == ""


def test_part_4_is_the_only_part_without_material_to_show() -> None:
    assert feedback.PARTS_WITH_MATERIAL == {1, 2, 3, 5, 6, 7, 8}


def test_a_rewritten_reports_headings_are_drawn_at_body_size() -> None:
    """``## Introduction`` from the model must not outgrow the section's own title."""
    text = (
        "# Report on Arts Education\n"
        "## Introduction\n"
        "The aim of this report is to evaluate the arts programmes.\n"
        "\n"
        "## Recommendations ##\n"
        "The college should allocate greater resources.\n"
        "##"
    )
    shown = feedback.plain_headings(text)

    assert "#" not in shown
    assert shown == (
        "**Report on Arts Education**\n\n"
        "**Introduction**\n\n"
        "The aim of this report is to evaluate the arts programmes.\n\n"
        "**Recommendations**\n\n"
        "The college should allocate greater resources."
    )
    assert feedback.plain_headings("Item #3 was hard.\nSo was #4.") == (
        "Item #3 was hard.\nSo was #4."
    )
