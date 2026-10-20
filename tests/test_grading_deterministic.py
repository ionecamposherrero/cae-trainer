"""Deterministic marking: everything that happens before the model is asked."""

from __future__ import annotations

import pytest

from cae.domain.answers import (
    apply_adjudication,
    grade_deterministically,
    match_item,
    normalise,
    tokens,
)
from cae.schemas.rue_task import KeyItem, answer_key
from tests import factories


def key(answer: str, *, accepted: list[str] | None = None, key_word: str = "") -> KeyItem:
    return KeyItem(number=1, answer=answer, accepted=accepted or [], key_word=key_word)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Which  ", "which"),
        ("WHICH.", "which"),
        ("it’s", "it's"),
        ("“quoted”", "quoted"),
        ("", ""),
    ],
)
def test_normalise(raw: str, expected: str) -> None:
    assert normalise(raw) == expected


def test_tokens_splits_on_punctuation() -> None:
    assert tokens("wish I had worked!") == ["wish", "i", "had", "worked"]


@pytest.mark.parametrize("part", [1, 5, 6, 7, 8])
def test_letter_answers_are_never_adjudicated(part: int) -> None:
    wrong = match_item(part, key("B"), "C")
    assert not wrong.is_correct
    assert not wrong.needs_adjudication
    assert wrong.marks == 0


def test_letter_answers_are_case_insensitive() -> None:
    assert match_item(1, key("B"), "b").is_correct


@pytest.mark.parametrize(("part", "marks"), [(1, 1), (5, 2), (6, 2), (7, 2), (8, 1)])
def test_each_part_awards_its_own_marks_per_item(part: int, marks: int) -> None:
    assert match_item(part, key("B"), "B").marks == marks


def test_an_accepted_alternative_scores_full_marks_without_the_model() -> None:
    result = match_item(2, key("which", accepted=["that"]), "that")
    assert result.is_correct
    assert not result.needs_adjudication


def test_a_plausible_single_word_goes_to_adjudication() -> None:
    result = match_item(2, key("which"), "whose")
    assert not result.is_correct
    assert result.needs_adjudication


def test_open_cloze_rejects_multi_word_answers_outright() -> None:
    result = match_item(2, key("which"), "in which")
    assert not result.is_correct
    assert result.marks == 0
    assert not result.needs_adjudication
    assert "exactly one word" in result.reason


def test_a_blank_answer_is_never_adjudicated() -> None:
    result = match_item(2, key("which"), "   ")
    assert not result.needs_adjudication
    assert result.marks == 0


def test_part4_exact_answer_scores_both_marks() -> None:
    item = key("wish I had worked", key_word="WISH")
    result = match_item(4, item, "Wish I had worked")
    assert result.is_correct
    assert result.marks == 2


@pytest.mark.parametrize("answer", ["wish worked", "wish I had really been working much"])
def test_part4_word_limit_violations_score_zero(answer: str) -> None:
    result = match_item(4, key("wish I had worked", key_word="WISH"), answer)
    assert result.marks == 0
    assert not result.needs_adjudication
    assert "between 3 and 6 words" in result.reason


def test_part4_changing_the_key_word_scores_zero() -> None:
    result = match_item(4, key("wish I had worked", key_word="WISH"), "wished I had worked")
    assert result.marks == 0
    assert not result.needs_adjudication
    assert "key word" in result.reason


def test_part4_near_miss_earns_one_mark_and_a_second_opinion() -> None:
    result = match_item(4, key("wish I had worked", key_word="WISH"), "wish I have worked")
    assert result.marks == 1
    assert result.is_partial
    assert result.needs_adjudication


def test_part4_unrelated_answer_still_goes_to_adjudication() -> None:
    """It obeys the rules, so an examiner -- not a string match -- should decide."""
    result = match_item(4, key("wish I had worked", key_word="WISH"), "wish for better luck")
    assert result.marks == 0
    assert result.needs_adjudication


def test_accept_awards_full_marks() -> None:
    result = apply_adjudication(match_item(2, key("which"), "whose"), "accept")
    assert result.is_correct
    assert result.marks == result.max_marks
    assert result.adjudicated


def test_partial_awards_half_marks_only_where_half_marks_exist() -> None:
    part4 = apply_adjudication(
        match_item(4, key("wish I had worked", key_word="WISH"), "wish I have worked"),
        "partial",
    )
    assert part4.marks == 1

    part2 = apply_adjudication(match_item(2, key("which"), "whose"), "partial")
    assert part2.marks == 0


def test_reject_removes_the_provisional_half_mark() -> None:
    result = match_item(4, key("wish I had worked", key_word="WISH"), "wish I have worked")
    assert result.marks == 1
    assert apply_adjudication(result, "reject").marks == 0


def test_a_perfect_script_needs_no_adjudication() -> None:
    exercise = factories.part1()
    exercise_key = answer_key(1, exercise)
    responses = {item.number: item.answer for item in exercise_key}
    results = grade_deterministically(1, exercise_key, responses)
    assert sum(r.marks for r in results) == 8
    assert not any(r.needs_adjudication for r in results)


def test_unanswered_items_are_graded_as_zero() -> None:
    exercise = factories.part1()
    results = grade_deterministically(1, answer_key(1, exercise), {})
    assert sum(r.marks for r in results) == 0
    assert all(r.reason == "No answer given." for r in results)
