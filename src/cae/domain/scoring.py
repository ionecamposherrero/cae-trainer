"""Marks, percentages and the estimated Cambridge Scale score.

Cambridge does not publish raw-to-scale tables, so scale scores are estimates
interpolated between the published grade boundaries (142-210):
    200-210  Grade A  (certificate at C2)
    193-199  Grade B  (certificate at C1)
    180-192  Grade C  (certificate at C1)
    160-179  Level B2 (certificate at B2)
    142-159  below B2 (no certificate)
"""

from __future__ import annotations

from dataclasses import dataclass

from cae.domain.parts import (
    READING_MAX_MARKS,
    SUBSCALES,
    WRITING_MAX_MARKS_PER_TASK,
)

SCALE_MIN = 142
SCALE_MAX = 210

_SCALE_ANCHORS: tuple[tuple[float, float], ...] = (
    (0.0, 142.0),
    (40.0, 160.0),
    (60.0, 180.0),
    (75.0, 193.0),
    (80.0, 200.0),
    (100.0, 210.0),
)

_GRADE_BANDS: tuple[tuple[int, str, str], ...] = (
    (200, "Grade A", "C2"),
    (193, "Grade B", "C1"),
    (180, "Grade C", "C1"),
    (160, "Level B2", "B2"),
    (SCALE_MIN, "Below B2", "-"),
)

GRADE_THRESHOLDS: tuple[int, ...] = (SCALE_MAX, *(threshold for threshold, _, _ in _GRADE_BANDS))


@dataclass(frozen=True)
class ScoreSummary:
    marks: int
    max_marks: int
    percentage: float
    scale_score: int
    grade: str
    cefr: str

    @property
    def passed(self) -> bool:
        """A C1 certificate needs 180 on the scale."""
        return self.scale_score >= 180


def percentage(marks: float, max_marks: float) -> float:
    if max_marks <= 0:
        return 0.0
    return round(100.0 * marks / max_marks, 1)


def scale_score(pct: float) -> int:
    """Interpolate a percentage onto the Cambridge Scale. Estimate only."""
    pct = max(0.0, min(100.0, pct))
    for (lo_pct, lo_score), (hi_pct, hi_score) in zip(_SCALE_ANCHORS, _SCALE_ANCHORS[1:]):
        if pct <= hi_pct:
            span = hi_pct - lo_pct
            ratio = 0.0 if span == 0 else (pct - lo_pct) / span
            return int(round(lo_score + ratio * (hi_score - lo_score)))
    return SCALE_MAX


def grade_for(score: int) -> tuple[str, str]:
    """Return ``(grade, cefr)`` for a scale score."""
    for threshold, grade, cefr in _GRADE_BANDS:
        if score >= threshold:
            return grade, cefr
    return "Below B2", "-"


def summarise(marks: float, max_marks: float) -> ScoreSummary:
    pct = percentage(marks, max_marks)
    score = scale_score(pct)
    grade, cefr = grade_for(score)
    return ScoreSummary(
        marks=int(round(marks)),
        max_marks=int(round(max_marks)),
        percentage=pct,
        scale_score=score,
        grade=grade,
        cefr=cefr,
    )


def summarise_reading(marks: float) -> ScoreSummary:
    """Summarise a full Reading & Use of English paper (78 marks)."""
    return summarise(marks, READING_MAX_MARKS)


def writing_total(bands: dict[str, int]) -> int:
    """Sum the four subscale bands (0-5 each) into a mark out of 20."""
    return sum(int(bands.get(name, 0)) for name in SUBSCALES)


def summarise_writing(bands: dict[str, int]) -> ScoreSummary:
    return summarise(writing_total(bands), WRITING_MAX_MARKS_PER_TASK)


def combined_summary(
    reading_marks: float | None,
    writing_bands: list[dict[str, int]] | None,
) -> ScoreSummary | None:
    """Estimate an overall score from the papers this app covers.

    The real certificate also weighs Listening and Speaking, so this is a
    partial estimate over Reading & Use of English and Writing only.
    """
    parts: list[tuple[float, float]] = []
    if reading_marks is not None:
        parts.append((reading_marks, READING_MAX_MARKS))
    if writing_bands:
        earned = sum(writing_total(b) for b in writing_bands)
        parts.append((earned, WRITING_MAX_MARKS_PER_TASK * len(writing_bands)))
    if not parts:
        return None
    pct = sum(percentage(m, mx) for m, mx in parts) / len(parts)
    score = scale_score(pct)
    grade, cefr = grade_for(score)
    total_marks = sum(m for m, _ in parts)
    total_max = sum(mx for _, mx in parts)
    return ScoreSummary(
        marks=int(round(total_marks)),
        max_marks=int(round(total_max)),
        percentage=round(pct, 1),
        scale_score=score,
        grade=grade,
        cefr=cefr,
    )
