"""Deterministic answer matching.

Only near-miss answers in Parts 2-4 are left for the model to adjudicate. A
Part 4 half mark approximates Cambridge's unpublished split: key word used
correctly and most of the required wording present.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from cae.domain.parts import get_part_spec
from cae.schemas.rue_task import KeyItem

_SMART = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})
_STRIP_EDGE = " \t\n\r.,;:!?\"'()[]"

LETTER_PARTS: frozenset[int] = frozenset({1, 5, 6, 7, 8})
ADJUDICATED_PARTS: frozenset[int] = frozenset({2, 3, 4})

PART4_MIN_WORDS = 3
PART4_MAX_WORDS = 6


def normalise(text: str) -> str:
    """Lowercase, straighten quotes, drop edge punctuation, collapse spaces."""
    text = (text or "").translate(_SMART)
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(_STRIP_EDGE).lower()


def tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9']+", normalise(text)) if t]


def contains_word(haystack: str, word: str) -> bool:
    """Whole-word, case-insensitive containment. Used for the Part 4 key word."""
    if not word:
        return True
    return re.search(rf"\b{re.escape(word)}\b", haystack or "", flags=re.IGNORECASE) is not None


def _overlap(a: list[str], b: list[str]) -> float:
    """Fraction of ``b``'s tokens present in ``a``, order-insensitive."""
    if not b:
        return 0.0
    remaining = list(a)
    hits = 0
    for token in b:
        if token in remaining:
            remaining.remove(token)
            hits += 1
    return hits / len(b)


@dataclass
class MatchResult:
    number: int
    user_answer: str
    correct_answer: str
    marks: float
    max_marks: float
    is_correct: bool
    needs_adjudication: bool
    reason: str = ""
    adjudicated: bool = False

    @property
    def is_partial(self) -> bool:
        return 0 < self.marks < self.max_marks


def match_item(part: int, key: KeyItem, user_answer: str) -> MatchResult:
    """Score one item without calling the model."""
    max_marks = float(get_part_spec(part).marks_per_item)
    given = normalise(user_answer)
    base = MatchResult(
        number=key.number,
        user_answer=(user_answer or "").strip(),
        correct_answer=key.answer,
        marks=0.0,
        max_marks=max_marks,
        is_correct=False,
        needs_adjudication=False,
    )

    if not given:
        base.reason = "No answer given."
        return base

    variants = [normalise(v) for v in key.variants if v]
    if given in variants:
        base.marks = max_marks
        base.is_correct = True
        base.reason = "Matches the key."
        return base

    if part in LETTER_PARTS:
        base.reason = "Wrong option."
        return base

    if part == 4:
        return _match_part4(key, user_answer, base)

    if len(tokens(given)) == 1:
        base.needs_adjudication = True
        base.reason = "Not in the key; sent for adjudication."
    else:
        base.reason = "Answers must be exactly one word."
    return base


def _match_part4(key: KeyItem, user_answer: str, base: MatchResult) -> MatchResult:
    """Key word transformations: rule violations score zero outright."""
    given_tokens = tokens(user_answer)
    if not PART4_MIN_WORDS <= len(given_tokens) <= PART4_MAX_WORDS:
        base.reason = (
            f"{len(given_tokens)} words: the answer must use between "
            f"{PART4_MIN_WORDS} and {PART4_MAX_WORDS} words."
        )
        return base
    if key.key_word and not contains_word(user_answer, key.key_word):
        base.reason = f"The key word '{key.key_word.upper()}' is missing or was changed."
        return base

    best = max(
        (_overlap(given_tokens, tokens(variant)) for variant in key.variants if variant),
        default=0.0,
    )
    if best >= 0.5:
        base.marks = base.max_marks / 2
        base.needs_adjudication = True
        base.reason = "Close to the key; sent for adjudication."
    else:
        base.needs_adjudication = True
        base.reason = "Different from the key; sent for adjudication."
    return base


def grade_deterministically(
    part: int, key: list[KeyItem], responses: dict[int, str]
) -> list[MatchResult]:
    """Score every item. Items needing a second opinion are flagged, not judged."""
    return [match_item(part, item, responses.get(item.number, "")) for item in key]


def apply_adjudication(result: MatchResult, verdict: str) -> MatchResult:
    """Fold the model's verdict ('accept' / 'partial' / 'reject') into a result."""
    result.adjudicated = True
    if verdict == "accept":
        result.marks = result.max_marks
        result.is_correct = True
        result.reason = "Accepted as a valid alternative to the key."
    elif verdict == "partial" and result.max_marks >= 2:
        result.marks = result.max_marks / 2
        result.is_correct = False
        result.reason = "Partly correct: one mark of two."
    else:
        result.marks = 0.0
        result.is_correct = False
        result.reason = "Not accepted as a valid answer."
    return result
