"""The closed taxonomy's shape: what each part may be tagged with, and why."""

from __future__ import annotations

import pytest

from cae.domain.parts import PART_SPECS
from cae.domain.taxonomy import (
    CATEGORY_ADVICE,
    CATEGORY_FAMILY,
    CATEGORY_GUIDE,
    CATEGORY_LABELS,
    LANGUAGE_CATEGORIES,
    PART_CATEGORIES,
    WRITING_CATEGORIES,
    ErrorCategory,
    Family,
    categories_for_part,
    family,
    label,
)
from cae.schemas.rue_evaluator import error_tag_batch

USE_OF_ENGLISH = (1, 2, 3, 4)
READING = (5, 6, 7, 8)


def test_every_part_has_a_category_set() -> None:
    assert set(PART_CATEGORIES) == set(PART_SPECS)


def test_use_of_english_parts_share_one_set() -> None:
    """The four parts test the same language, so a weakness must aggregate across them."""
    for part in USE_OF_ENGLISH:
        assert categories_for_part(part) == LANGUAGE_CATEGORIES


def test_each_reading_part_has_one_category_of_its_own() -> None:
    """The category is the task itself: there is nothing for the model to choose."""
    seen: set[ErrorCategory] = set()
    for part in READING:
        (only,) = categories_for_part(part)
        assert CATEGORY_FAMILY[only] is Family.READING
        assert only not in seen
        seen.add(only)


def test_writing_adds_register_and_punctuation_to_the_shared_set() -> None:
    assert WRITING_CATEGORIES == (
        *LANGUAGE_CATEGORIES,
        ErrorCategory.REGISTER,
        ErrorCategory.PUNCTUATION,
    )


def test_language_categories_never_reach_reading_and_back() -> None:
    reading = {c for part in READING for c in categories_for_part(part)}
    assert reading.isdisjoint(WRITING_CATEGORIES)


@pytest.mark.parametrize("category", list(ErrorCategory))
def test_every_category_is_described(category: ErrorCategory) -> None:
    assert CATEGORY_LABELS[category]
    assert CATEGORY_ADVICE[category]
    assert category in CATEGORY_FAMILY


def test_every_category_a_model_chooses_from_says_when_to_use_it() -> None:
    """The boundaries between neighbours live in the guide, so none may be missing."""
    assert set(CATEGORY_GUIDE) == set(WRITING_CATEGORIES)
    assert all(CATEGORY_GUIDE[c] for c in WRITING_CATEGORIES)


def test_every_category_is_used_somewhere() -> None:
    """A member no part or genre can produce is dead weight in every chart."""
    used = set(WRITING_CATEGORIES) | {c for cats in PART_CATEGORIES.values() for c in cats}
    assert used == set(ErrorCategory)


@pytest.mark.parametrize("part", sorted(PART_SPECS))
def test_the_tag_schema_allows_exactly_the_parts_categories(part: int) -> None:
    """What the model decodes against, not just what the prompt lists."""
    schema = error_tag_batch(part).model_json_schema()
    (enum,) = [d["enum"] for name, d in schema["$defs"].items() if name.endswith("Category")]
    assert enum == [c.value for c in categories_for_part(part)]


def test_an_unknown_code_still_reads_as_words() -> None:
    """A stored code outside the enum still renders, with no family."""
    assert label("GRAMMAR_TENSE") == "Grammar Tense"
    assert family("GRAMMAR_TENSE") == ""
