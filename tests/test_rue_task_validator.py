"""The structural validators must accept good exercises and reject bad ones."""

from __future__ import annotations

import pytest

from cae.domain.parts import PART_SPECS, gap_marker
from cae.validators import count_words
from cae.validators.rue_task import validate_exercise
from tests import factories


@pytest.mark.parametrize("part", sorted(PART_SPECS))
def test_well_formed_exercise_passes(part: int) -> None:
    issues = validate_exercise(part, factories.build(part))
    assert issues == [], f"Part {part} should validate cleanly, got: {issues}"


@pytest.mark.parametrize("part", [1, 2, 3, 7])
def test_missing_gap_marker_is_caught(part: int) -> None:
    exercise = factories.build(part)
    exercise.text = exercise.text.replace(gap_marker(2), "")
    issues = validate_exercise(part, exercise)
    assert any("{{2}}" in issue for issue in issues)


@pytest.mark.parametrize("part", [1, 2, 3, 7])
def test_duplicated_gap_marker_is_caught(part: int) -> None:
    exercise = factories.build(part)
    exercise.text = exercise.text + " " + gap_marker(1)
    issues = validate_exercise(part, exercise)
    assert any("appears 2 times" in issue for issue in issues)


@pytest.mark.parametrize("part", sorted(PART_SPECS))
def test_wrong_item_count_is_caught(part: int) -> None:
    exercise = factories.build(part)
    exercise.items = exercise.items[:-1]
    issues = validate_exercise(part, exercise)
    expected = PART_SPECS[part].item_count
    assert any(f"exactly {expected} items" in issue for issue in issues)


def test_passage_length_is_left_to_the_model() -> None:
    exercise = factories.part1()
    exercise.text = factories.gapped_text(60, 8)
    assert validate_exercise(1, exercise) == []


def test_part1_rejects_identical_options() -> None:
    exercise = factories.part1()
    exercise.items[0].options[1] = exercise.items[0].options[0]
    issues = validate_exercise(1, exercise)
    assert any("not all different" in issue for issue in issues)


def test_part1_rejects_an_option_of_more_than_one_word() -> None:
    exercise = factories.part1()
    exercise.items[0].options[0] = "came up"
    issues = validate_exercise(1, exercise)
    assert any("must be a single word" in issue for issue in issues)


def test_part2_rejects_multi_word_answer() -> None:
    exercise = factories.part2()
    exercise.items[3].answer = "in order"
    issues = validate_exercise(2, exercise)
    assert any("more than one word" in issue for issue in issues)


def test_part3_rejects_answer_equal_to_stem() -> None:
    exercise = factories.part3()
    exercise.items[0].answer = "rely"
    issues = validate_exercise(3, exercise)
    assert any("identical to the stem" in issue for issue in issues)


@pytest.mark.parametrize(
    ("stem", "answer"),
    [
        ("RELY", "unreliable"),
        ("WILLING", "unwilling"),
        ("LEGAL", "illegal"),
        ("RESPONSIBLE", "irresponsible"),
        ("POSSIBLE", "impossible"),
        ("SUCCEED", "unsuccessful"),
        ("LONG", "length"),
        ("HIGH", "height"),
        ("STRONG", "strength"),
        ("AGREE", "disagreement"),
        ("STAND", "understandable"),
    ],
)
def test_part3_accepts_real_derivations(stem: str, answer: str) -> None:
    """A negative prefix changes the first letter; that is not a broken answer."""
    exercise = factories.part3()
    exercise.items[0].stem = stem
    exercise.items[0].answer = answer
    issues = validate_exercise(3, exercise)
    assert not any("does not look like a form" in issue for issue in issues), issues


def test_part3_rejects_an_unrelated_answer() -> None:
    exercise = factories.part3()
    exercise.items[0].stem = "RELY"
    exercise.items[0].answer = "bicycle"
    issues = validate_exercise(3, exercise)
    assert any("does not look like a form" in issue for issue in issues)


def test_part3_rejects_lowercase_stem() -> None:
    exercise = factories.part3()
    exercise.items[0].stem = "Rely"
    issues = validate_exercise(3, exercise)
    assert any("written in capitals" in issue for issue in issues)


@pytest.mark.parametrize(
    ("answer", "fragment"),
    [
        ("wish worked", "words"),
        ("wish I had really been working much", "words"),
        ("hoped I had worked", "does not contain the key word"),
    ],
)
def test_part4_rejects_bad_answers(answer: str, fragment: str) -> None:
    exercise = factories.part4()
    exercise.items[0].answer = answer
    issues = validate_exercise(4, exercise)
    assert any(fragment in issue for issue in issues), issues


def test_part4_rejects_key_word_in_lead_in() -> None:
    exercise = factories.part4()
    exercise.items[0].lead_in = "I wish things had gone differently."
    issues = validate_exercise(4, exercise)
    assert any("appears in the lead-in" in issue for issue in issues)


@pytest.mark.parametrize("part", [5, 6, 8])
def test_evidence_must_come_from_the_text(part: int) -> None:
    exercise = factories.build(part)
    exercise.items[0].evidence = "a sentence that was never written anywhere"
    issues = validate_exercise(part, exercise)
    assert any("does not appear" in issue for issue in issues)


def test_part7_requires_exactly_one_unused_paragraph() -> None:
    exercise = factories.part7()
    exercise.items[1].answer = exercise.items[0].answer
    issues = validate_exercise(7, exercise)
    assert any("only one gap" in issue for issue in issues)


def test_part7_flags_mislabelled_distractor() -> None:
    exercise = factories.part7()
    exercise.distractor_letter = "A"
    issues = validate_exercise(7, exercise)
    assert any("distractor_letter" in issue for issue in issues)


def test_part8_requires_every_section_to_be_used() -> None:
    exercise = factories.part8()
    for item in exercise.items:
        if item.answer == "F":
            item.answer = "A"
            item.evidence = factories.quote_from(exercise.sections[0].content, start=2)
    issues = validate_exercise(8, exercise)
    assert any("unused: ['F']" in issue for issue in issues)


def test_part8_rejects_prompts_lifted_from_the_text() -> None:
    exercise = factories.part8()
    section = exercise.sections[0]
    exercise.items[0].answer = "A"
    exercise.items[0].prompt = factories.quote_from(section.content, start=0, length=8)
    exercise.items[0].evidence = factories.quote_from(section.content, start=0, length=6)
    issues = validate_exercise(8, exercise)
    assert any("copies" in issue for issue in issues)


def test_count_words_ignores_gap_markers() -> None:
    assert count_words("one {{1}} two {{2}} three") == 3


def test_the_two_validators_are_siblings_not_a_chain() -> None:
    """The shared text helpers live in the package, not in the Reading validator."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "cae" / "validators"
    for module in ("rue_task", "writing_task"):
        tree = ast.parse((root / f"{module}.py").read_text(encoding="utf-8"))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        siblings = {m for m in imported if m.startswith("cae.validators.")}
        assert not siblings, f"{module} imports from {siblings}"
        assert "cae.validators" in imported, f"{module} should use the shared helpers"
