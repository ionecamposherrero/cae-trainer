"""Graph wiring, exercised end to end with the model stubbed out."""

from __future__ import annotations

from typing import Any

import pytest

from cae.domain.parts import PART_SPECS, SUBSCALES, gap_marker
from cae.graphs import rue_evaluator, rue_task, writing_evaluator
from cae.schemas.rue_task import ModerationResult, Part1Exercise, answer_key
from cae.schemas.rue_evaluator import (
    AdjudicationBatch,
    AdjudicationDecision,
    ErrorTag,
    ErrorTagBatch,
    error_tag_batch,
)
from cae.schemas.writing_evaluator import AnnotationSet, ImprovedVersion, SubscaleAssessment
from cae.validators.rue_task import validate_exercise
from tests import factories


class Recorder:
    """A stand-in for ``structured_call`` that answers by schema."""

    def __init__(self, responses: dict[type, Any]):
        self.responses = responses
        self.calls: list[type] = []

    def __call__(self, schema, system, user, **kwargs):
        self.calls.append(schema)
        value = self.responses[schema]
        return value(system, user) if callable(value) and not isinstance(value, type) else value


def test_generation_repairs_an_invalid_draft(monkeypatch: pytest.MonkeyPatch) -> None:
    broken = factories.part1()
    broken.items = broken.items[:6]
    fixed = factories.part1()
    drafts = iter([broken, fixed])

    def fake_call(schema, system, user, **kwargs):
        assert schema is Part1Exercise
        return next(drafts)

    monkeypatch.setattr(rue_task, "structured_call", fake_call)

    result = rue_task.generate_exercise(1, persist_result=False)

    assert result["attempts"] == 2, "the first draft should have been repaired, not accepted"
    assert result["issues"] == []
    assert validate_exercise(1, Part1Exercise.model_validate(result["exercise"])) == []


def test_generation_gives_up_after_the_retry_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    def always_broken(schema, system, user, **kwargs):
        broken = factories.part2()
        broken.items[0].answer = "two words"
        return broken

    monkeypatch.setattr(rue_task, "structured_call", always_broken)

    with pytest.raises(rue_task.GenerationFailed) as excinfo:
        rue_task.generate_exercise(2, persist_result=False, max_attempts=2)

    assert "more than one word" in str(excinfo.value)


def test_comprehension_parts_are_moderated(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder(
        {
            type(factories.part5()): factories.part5(),
            ModerationResult: ModerationResult(issues=[], verdict="accept"),
        }
    )
    monkeypatch.setattr(rue_task, "structured_call", recorder)

    result = rue_task.generate_exercise(5, persist_result=False)

    assert ModerationResult in recorder.calls, "Part 5 must go through the moderator"
    assert result["status"] == "ok"


def test_cloze_parts_skip_moderation(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder({type(factories.part3()): factories.part3()})
    monkeypatch.setattr(rue_task, "structured_call", recorder)

    rue_task.generate_exercise(3, persist_result=False)

    assert ModerationResult not in recorder.calls


def test_a_moderator_objection_triggers_a_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    verdicts = iter(
        [
            ModerationResult(issues=["Item 3: two options are defensible."], verdict="revise"),
            ModerationResult(issues=[], verdict="accept"),
        ]
    )

    def fake_call(schema, system, user, **kwargs):
        if schema is ModerationResult:
            return next(verdicts)
        return factories.part6()

    monkeypatch.setattr(rue_task, "structured_call", fake_call)

    result = rue_task.generate_exercise(6, persist_result=False)

    assert result["attempts"] == 2
    assert result["status"] == "ok"


@pytest.mark.parametrize("part", sorted(PART_SPECS))
def test_a_perfect_script_never_calls_the_model(
    monkeypatch: pytest.MonkeyPatch, part: int
) -> None:
    calls: list[type] = []

    def record(schema, *args, **kwargs):
        calls.append(schema)
        raise RuntimeError("no model in this test")

    monkeypatch.setattr(rue_evaluator, "structured_call", record)

    exercise = factories.build(part)
    key = answer_key(part, exercise)
    result = rue_evaluator.grade_attempt(
        part, exercise.model_dump(), {k.number: k.answer for k in key}
    )

    assert calls == [], "grading a perfect script should need no API call"
    assert result["marks"] == result["max_marks"]
    assert all(row["is_correct"] for row in result["rows"])


def test_a_near_miss_is_adjudicated_and_can_be_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exercise = factories.part2()
    key = answer_key(2, exercise)
    responses = {k.number: k.answer for k in key}
    responses[1] = "whose"

    def fake_call(schema, system, user, **kwargs):
        if schema is AdjudicationBatch:
            return AdjudicationBatch(
                decisions=[
                    AdjudicationDecision(
                        number=1, verdict="accept", reason="Also correct here."
                    )
                ]
            )
        return ErrorTagBatch(tags=[])

    monkeypatch.setattr(rue_evaluator, "structured_call", fake_call)

    result = rue_evaluator.grade_attempt(2, exercise.model_dump(), responses)

    first = next(row for row in result["rows"] if row["item_number"] == 1)
    assert first["is_correct"]
    assert first["adjudicated"]
    assert result["marks"] == 8


def test_wrong_answers_are_tagged_with_a_category(monkeypatch: pytest.MonkeyPatch) -> None:
    exercise = factories.part1()
    key = answer_key(1, exercise)
    responses = {k.number: ("A" if k.answer != "A" else "B") for k in key}
    calls: list[type] = []

    def fake_call(schema, system, user, **kwargs):
        calls.append(schema)
        return ErrorTagBatch(
            tags=[
                ErrorTag(
                    number=k.number,
                    category="FIXED_PHRASES",
                    diagnosis="You chose a word that means the right thing but does not collocate.",
                )
                for k in key
            ]
        )

    monkeypatch.setattr(rue_evaluator, "structured_call", fake_call)

    result = rue_evaluator.grade_attempt(1, exercise.model_dump(), responses)

    assert calls == [error_tag_batch(1)], "letter answers must not be adjudicated"
    assert result["marks"] == 0
    assert all(row["category"] == "FIXED_PHRASES" for row in result["rows"])
    assert all("collocate" in row["explanation"] for row in result["rows"])


def _one_wrong_letter(part: int, exercise) -> tuple[dict[int, str], str]:
    """Every item right except item 1, which gets the first letter that is not its key."""
    key = answer_key(part, exercise)
    letters = PART_SPECS[part].option_letters
    responses = {k.number: k.answer for k in key}
    wrong = next(letter for letter in letters if letter != responses[1])
    responses[1] = wrong
    return responses, wrong


def test_a_category_outside_the_part_fails_the_marking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The schema rules it out; a reply that slips past it is not filed under a guess."""
    exercise = factories.part1()
    responses, _ = _one_wrong_letter(1, exercise)
    monkeypatch.setattr(
        rue_evaluator,
        "structured_call",
        lambda *a, **k: ErrorTagBatch(
            tags=[ErrorTag(number=1, category="TEXT_STRUCTURE", diagnosis="Off the map.")]
        ),
    )

    with pytest.raises(rue_evaluator.MarkingFailed, match="category Part 1 does not use"):
        rue_evaluator.grade_attempt(1, exercise.model_dump(), responses)


def _no_model(*args, **kwargs):
    raise AssertionError("a Reading part must be marked without the model")


def _graded_row(part: int, responses: dict[int, str], number: int = 1) -> dict[str, Any]:
    result = rue_evaluator.grade_attempt(part, factories.build(part).model_dump(), responses)
    return next(r for r in result["rows"] if r["item_number"] == number)


@pytest.mark.parametrize("part", [5, 6, 7, 8])
def test_a_reading_part_is_tagged_and_diagnosed_without_the_model(
    part: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Its one category leaves nothing to choose, and the diagnosis needs only the exercise."""
    from cae.domain.taxonomy import PART_CATEGORIES

    monkeypatch.setattr(rue_evaluator, "structured_call", _no_model)
    responses, _ = _one_wrong_letter(part, factories.build(part))

    row = _graded_row(part, responses)

    assert row["category"] == PART_CATEGORIES[part][0].value
    assert "You chose" in row["explanation"]


def test_a_part5_diagnosis_quotes_both_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rue_evaluator, "structured_call", _no_model)
    responses, wrong = _one_wrong_letter(5, factories.part5())

    row = _graded_row(5, responses)

    key = factories.part5().items[0].answer
    assert f'You chose option {wrong}: "claim1{wrong.lower()}".' in row["explanation"]
    assert f'The answer is option {key}: "claim1{key.lower()}".' in row["explanation"]


def test_a_part7_paragraph_that_belongs_elsewhere_says_where(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rue_evaluator, "structured_call", _no_model)
    row = _graded_row(7, {1: "A", 2: "A", 3: "F", 4: "B", 5: "G", 6: "E"})
    assert "You chose paragraph A" in row["explanation"]
    assert "It belongs in gap 2." in row["explanation"]


def test_a_part7_distractor_is_named_as_the_paragraph_that_fits_nowhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rue_evaluator, "structured_call", _no_model)
    row = _graded_row(7, {1: "D", 2: "A", 3: "F", 4: "B", 5: "G", 6: "E"})
    assert "fits none of the gaps" in row["explanation"]


def test_a_part8_section_that_answers_other_questions_names_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rue_evaluator, "structured_call", _no_model)
    row = _graded_row(8, {1: "B"})
    assert "You chose section B (Case B). It is the answer to questions 2 and 8." in row["explanation"]
    assert "The answer is section A (Case A)." in row["explanation"]


def test_a_blank_reading_answer_is_still_diagnosed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rue_evaluator, "structured_call", _no_model)
    row = _graded_row(6, {})
    assert row["category"] == "COMPARING_VIEWS"
    assert "You left this one blank. The answer is text B (Writer B)." in row["explanation"]


def test_the_part1_tagger_sees_what_the_chosen_letter_said(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare "C" cannot be diagnosed: the prompt must carry the option's words."""
    exercise = factories.part1()
    responses, wrong = _one_wrong_letter(1, exercise)
    chosen = exercise.items[0].options["ABCD".index(wrong)]
    prompts: list[str] = []

    def fake_call(schema, system, user, **kwargs):
        prompts.append(user)
        return factories.marking_reply(schema, user)

    monkeypatch.setattr(rue_evaluator, "structured_call", fake_call)
    rue_evaluator.grade_attempt(1, exercise.model_dump(), responses)

    assert f"Candidate chose: {wrong} — {chosen}" in prompts[0]


@pytest.mark.parametrize("part", [1, 2])
def test_a_model_that_cannot_be_reached_leaves_the_attempt_unmarked(
    part: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No deterministic stand-in: Part 1 needs the tagger, Part 2 the adjudicator."""
    from cae.llm import ModelUnreachableError

    def down(*args, **kwargs):
        raise ModelUnreachableError("overloaded")

    monkeypatch.setattr(rue_evaluator, "structured_call", down)
    exercise = factories.build(part)
    key = answer_key(part, exercise)
    if part == 1:
        wrong = {k.number: "A" if k.answer != "A" else "B" for k in key}
    else:
        wrong = {k.number: "zzz" for k in key}

    with pytest.raises(ModelUnreachableError):
        rue_evaluator.grade_attempt(part, exercise.model_dump(), wrong)


def test_an_adjudicator_that_skips_an_item_fails_the_marking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exercise = factories.part2()
    responses = {k.number: k.answer for k in answer_key(2, exercise)}
    responses[1], responses[2] = "whose", "whom"
    monkeypatch.setattr(
        rue_evaluator,
        "structured_call",
        lambda *a, **k: AdjudicationBatch(
            decisions=[AdjudicationDecision(number=1, verdict="reject", reason="No.")]
        ),
    )

    with pytest.raises(rue_evaluator.MarkingFailed, match="no verdict on items 2"):
        rue_evaluator.grade_attempt(2, exercise.model_dump(), responses)


def test_a_tagger_that_skips_an_item_fails_the_marking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exercise = factories.part1()
    key = answer_key(1, exercise)
    monkeypatch.setattr(rue_evaluator, "structured_call", lambda *a, **k: ErrorTagBatch(tags=[]))

    with pytest.raises(rue_evaluator.MarkingFailed, match="no error category for items"):
        rue_evaluator.grade_attempt(
            1, exercise.model_dump(), {k.number: ("A" if k.answer != "A" else "B") for k in key}
        )


def test_a_perfect_script_needs_no_model_at_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """A right script must never wait on a model."""
    monkeypatch.setattr(rue_evaluator, "structured_call", _no_model)
    for part in range(1, 9):
        exercise = factories.build(part)
        key = answer_key(part, exercise)
        result = rue_evaluator.grade_attempt(
            part, exercise.model_dump(), {k.number: k.answer for k in key}
        )
        assert result["marks"] == result["max_marks"]


def _graded_rows(part: int, responses: dict[int, str], exercise) -> list[dict]:
    return rue_evaluator.grade_attempt(part, exercise.model_dump(), responses)["rows"]


def test_evidence_carries_the_answer_the_key_and_the_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rue_evaluator,
        "structured_call",
        lambda schema, system, user, **k: factories.marking_reply(schema, user),
    )
    exercise = factories.part1()
    key = {k.number: k for k in answer_key(1, exercise)}
    wrong = {n: ("A" if k.answer != "A" else "B") for n, k in key.items()}

    for row in _graded_rows(1, wrong, exercise):
        evidence = row["evidence"]
        assert f'wrote "{wrong[row["item_number"]]}"' in evidence
        assert f'key "{key[row["item_number"]].answer}"' in evidence
        assert gap_marker(row["item_number"]) in evidence


def test_a_blank_answer_says_blank_rather_than_going_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The evidence of a blank answer must not be the examiner's prose."""
    monkeypatch.setattr(
        rue_evaluator,
        "structured_call",
        lambda schema, system, user, **k: factories.marking_reply(schema, user),
    )
    exercise = factories.part1()
    rows = _graded_rows(1, {k.number: "" for k in answer_key(1, exercise)}, exercise)

    for row in rows:
        assert '(blank)' in row["evidence"]
        assert row["explanation"] not in row["evidence"]


def test_evidence_stays_inside_the_column_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """repo.record_attempt truncates at 500; nothing should reach that."""
    monkeypatch.setattr(
        rue_evaluator,
        "structured_call",
        lambda schema, system, user, **k: factories.marking_reply(schema, user),
    )
    for part, build in ((1, factories.part1), (4, factories.part4), (8, factories.part8)):
        exercise = build()
        rows = _graded_rows(part, {k.number: "zzz" for k in answer_key(part, exercise)}, exercise)
        assert rows, f"part {part} produced no rows"
        assert all(len(row["evidence"]) < 500 for row in rows)


SCRIPT = " ".join(["Museums should remain free to enter for everyone."] * 30)


def _writing_stub(bands: dict[str, int]):
    order = iter(
        [bands[name] for name in
         ("content", "communicative_achievement", "organisation", "language")]
    )

    def fake_call(schema, system, user, **kwargs):
        if schema is SubscaleAssessment:
            for name, band in bands.items():
                if f"subscale only: {name.replace('_', ' ').title()}" in system:
                    return SubscaleAssessment(
                        band=band, justification="Because.", evidence=[], to_improve="Do more."
                    )
            return SubscaleAssessment(
                band=next(order), justification="", evidence=[], to_improve=""
            )
        if schema is AnnotationSet:
            return AnnotationSet(
                annotations=[
                    {
                        "span": "Museums should remain free",
                        "category": "REGISTER",
                        "correction": "Museums ought to remain free",
                        "explanation": "Stronger register for an essay.",
                        "severity": "slip",
                    }
                ],
                strengths=["Clear position."],
            )
        return ImprovedVersion(text="An improved essay.", upgrades=[])

    return fake_call


def test_all_four_subscales_are_assessed_and_aggregated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bands = {
        "content": 4,
        "communicative_achievement": 3,
        "organisation": 5,
        "language": 3,
    }
    monkeypatch.setattr(writing_evaluator, "structured_call", _writing_stub(bands))

    result = writing_evaluator.assess_writing(
        part=1, genre="essay", task={"notes": [], "opinions": []}, script=SCRIPT
    )

    assert result["bands"] == bands
    assert result["total"] == 15
    assert result["status"] == "assessed"
    assert result["improved"]["text"] == "An improved essay."


def test_annotations_are_marked_as_locatable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(writing_evaluator, "structured_call", _writing_stub(dict.fromkeys(
        ["content", "communicative_achievement", "organisation", "language"], 3
    )))

    result = writing_evaluator.assess_writing(
        part=1, genre="essay", task={"notes": [], "opinions": []}, script=SCRIPT
    )

    annotation = result["annotations"][0]
    assert annotation["found"] is True
    assert annotation["category"] == "REGISTER"


@pytest.mark.parametrize(
    "script",
    [
        "",
        "Too short.",
        " ".join(["Museums should remain free to enter for everyone."] * 27),
        " ".join(["Museums should remain free to enter for everyone."] * 33),
    ],
)
def test_a_script_outside_the_word_range_is_not_sent_to_the_examiners(
    monkeypatch: pytest.MonkeyPatch, script: str
) -> None:
    def explode(*args, **kwargs):  # pragma: no cover
        raise AssertionError("a script outside the range must not reach the model")

    monkeypatch.setattr(writing_evaluator, "structured_call", explode)

    result = writing_evaluator.assess_writing(
        part=1, genre="essay", task={}, script=script
    )

    assert result["status"] == "out_of_range"
    assert result["total"] == 0
    assert result["bands"] == dict.fromkeys(SUBSCALES, 0)


def test_one_failed_subscale_does_not_lose_the_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_call(schema, system, user, **kwargs):
        if schema is SubscaleAssessment:
            if "Organisation" in system:
                raise RuntimeError("timed out")
            return SubscaleAssessment(
                band=4, justification="", evidence=[], to_improve=""
            )
        if schema is AnnotationSet:
            return AnnotationSet(annotations=[], strengths=[])
        return ImprovedVersion(text="", upgrades=[])

    monkeypatch.setattr(writing_evaluator, "structured_call", fake_call)

    result = writing_evaluator.assess_writing(
        part=1, genre="essay", task={"notes": []}, script=SCRIPT
    )

    assert result["bands"]["organisation"] == 0
    assert result["bands"]["content"] == 4
    assert result["organisation"]["failed"] is True
