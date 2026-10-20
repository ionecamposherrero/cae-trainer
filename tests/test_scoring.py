"""Marks, percentages and the estimated Cambridge Scale mapping."""

from __future__ import annotations

import pytest

from cae.domain.parts import PART_SPECS, READING_MAX_MARKS
from cae.domain.scoring import (
    SCALE_MAX,
    SCALE_MIN,
    combined_summary,
    grade_for,
    percentage,
    scale_score,
    summarise_reading,
    summarise_writing,
    writing_total,
)


def test_the_paper_is_worth_78_marks() -> None:
    """The real Reading & Use of English paper totals 78 marks."""
    assert READING_MAX_MARKS == 78
    assert sum(s.item_count for s in PART_SPECS.values()) == 56


def test_percentage_handles_a_zero_denominator() -> None:
    assert percentage(0, 0) == 0.0


@pytest.mark.parametrize(
    ("pct", "expected"),
    [(0, 142), (40, 160), (60, 180), (75, 193), (80, 200), (100, 210)],
)
def test_scale_anchors(pct: float, expected: int) -> None:
    assert scale_score(pct) == expected


def test_scale_is_monotonic_and_bounded() -> None:
    scores = [scale_score(p) for p in range(0, 101)]
    assert scores == sorted(scores)
    assert min(scores) >= SCALE_MIN
    assert max(scores) <= SCALE_MAX


def test_out_of_range_percentages_are_clamped() -> None:
    assert scale_score(-20) == SCALE_MIN
    assert scale_score(140) == SCALE_MAX


@pytest.mark.parametrize(
    ("score", "grade", "cefr"),
    [
        (210, "Grade A", "C2"),
        (200, "Grade A", "C2"),
        (199, "Grade B", "C1"),
        (193, "Grade B", "C1"),
        (192, "Grade C", "C1"),
        (180, "Grade C", "C1"),
        (179, "Level B2", "B2"),
        (160, "Level B2", "B2"),
        (159, "Below B2", "-"),
    ],
)
def test_grade_boundaries(score: int, grade: str, cefr: str) -> None:
    assert grade_for(score) == (grade, cefr)


def test_60_percent_is_the_pass_mark() -> None:
    """180 on the scale is a C1 certificate; just below it is not."""
    assert summarise_reading(READING_MAX_MARKS * 0.60).passed
    assert not summarise_reading(READING_MAX_MARKS * 0.59).passed


def test_writing_total_sums_the_four_subscales() -> None:
    bands = {
        "content": 4,
        "communicative_achievement": 3,
        "organisation": 5,
        "language": 3,
    }
    assert writing_total(bands) == 15
    assert summarise_writing(bands).max_marks == 20


def test_missing_subscales_count_as_zero() -> None:
    assert writing_total({"content": 5}) == 5


def test_combined_summary_weights_each_paper_equally() -> None:
    """A perfect reading paper and a blank writing task average to 50%."""
    result = combined_summary(READING_MAX_MARKS, [dict.fromkeys(
        ["content", "communicative_achievement", "organisation", "language"], 0
    )])
    assert result is not None
    assert result.percentage == 50.0


def test_combined_summary_needs_something_to_summarise() -> None:
    assert combined_summary(None, None) is None
