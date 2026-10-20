"""The Writing task validator, and the generation graph's repair loop that uses it."""

from __future__ import annotations

import pytest

from cae.graphs import writing_task as task_module
from cae.validators.writing_task import (
    MAX_NOTE_WORDS,
    MAX_OPINION_WORDS,
    MIN_SCENARIO_WORDS,
    validate_writing_task,
)
from tests import factories


def part1(**changes) -> dict:
    return {**factories.writing_task(1), **changes}


def part2(**changes) -> dict:
    return {**factories.writing_task(2, "proposal"), **changes}


def test_a_well_formed_part_1_task_passes() -> None:
    assert validate_writing_task(1, "essay", factories.writing_task(1)) == []


@pytest.mark.parametrize("genre", ["letter_email", "proposal", "report", "review"])
def test_a_well_formed_part_2_task_passes(genre: str) -> None:
    assert validate_writing_task(2, genre, factories.writing_task(2, genre)) == []


def test_an_unknown_part_is_refused() -> None:
    with pytest.raises(ValueError, match="Part 1 or Part 2"):
        validate_writing_task(3, "essay", factories.writing_task(1))


def test_an_unknown_genre_is_refused() -> None:
    with pytest.raises(ValueError, match="Unknown writing genre"):
        validate_writing_task(2, "haiku", factories.writing_task(2))


def test_a_missing_title_is_caught() -> None:
    issues = validate_writing_task(1, "essay", part1(title="  "))
    assert any("no title" in i for i in issues)


def test_a_stub_scenario_is_caught() -> None:
    issues = validate_writing_task(1, "essay", part1(scenario="You went somewhere."))
    assert any(f"at least {MIN_SCENARIO_WORDS} words" in i for i in issues)


def test_an_empty_instruction_is_caught() -> None:
    issues = validate_writing_task(1, "essay", part1(instruction=""))
    assert any("instruction is empty" in i for i in issues)


@pytest.mark.parametrize(
    "instruction",
    [
        "Write an essay for your tutor discussing two of the points.",
        "Write an essay discussing two points in about 300 words.",
        "Write an essay discussing two points in 220 words or so.",
    ],
)
def test_an_instruction_without_the_word_count_is_caught(instruction: str) -> None:
    issues = validate_writing_task(1, "essay", part1(instruction=instruction))
    assert any("does not state the length" in i for i in issues)


def test_a_missing_reader_is_caught() -> None:
    issues = validate_writing_task(1, "essay", part1(reader=""))
    assert any("reader is empty" in i for i in issues)


def test_a_content_point_written_as_a_sentence_is_caught() -> None:
    long_note = " ".join(["word"] * (MAX_NOTE_WORDS + 1))
    issues = validate_writing_task(1, "essay", part1(notes=[long_note, "cost", "access"]))
    assert any("short phrases" in i for i in issues)


def test_duplicated_content_points_are_caught() -> None:
    issues = validate_writing_task(1, "essay", part1(notes=["cost", "Cost.", "access"]))
    assert any("the same" in i for i in issues)


def test_an_empty_content_point_is_caught() -> None:
    issues = validate_writing_task(1, "essay", part1(notes=["cost", "  ", "access"]))
    assert any("content points is empty" in i for i in issues)


@pytest.mark.parametrize("notes", [["a", "b"], ["a", "b", "c", "d"]])
def test_part_1_needs_exactly_three_content_points(notes: list[str]) -> None:
    issues = validate_writing_task(1, "essay", part1(notes=notes))
    assert any("exactly 3 content points" in i for i in issues)


def test_part_1_needs_exactly_three_opinions() -> None:
    issues = validate_writing_task(1, "essay", part1(opinions=["Only one remark."]))
    assert any("exactly 3 quoted opinions" in i for i in issues)


def test_a_speech_instead_of_an_overheard_remark_is_caught() -> None:
    speech = " ".join(["word"] * (MAX_OPINION_WORDS + 1))
    issues = validate_writing_task(1, "essay", part1(opinions=[speech, "Two.", "Three."]))
    assert any("one overheard remark" in i for i in issues)


def test_part_1_instruction_must_name_the_essay() -> None:
    issues = validate_writing_task(
        1, "essay", part1(instruction="Write something for your tutor in 220-260 words, two points.")
    )
    assert any("writes an essay" in i for i in issues)


def test_part_1_instruction_must_ask_for_two_of_the_three_points() -> None:
    issues = validate_writing_task(
        1,
        "essay",
        part1(instruction="Write an essay for your tutor in 220-260 words."),
    )
    assert any("discuss TWO" in i for i in issues)


def test_the_digit_2_is_accepted_as_well_as_the_word() -> None:
    task = part1(
        instruction="Write an essay discussing 2 of the points, in 220-260 words."
    )
    assert not any("discuss TWO" in i for i in validate_writing_task(1, "essay", task))


def test_part_2_must_not_carry_opinions() -> None:
    """Quoted opinions belong to the Part 1 essay and nowhere else."""
    issues = validate_writing_task(2, "proposal", part2(opinions=["A remark."]))
    assert any("Only Part 1 essays use" in i for i in issues)


@pytest.mark.parametrize("notes", [["one"], ["a", "b", "c", "d", "e", "f"]])
def test_part_2_content_points_must_be_within_range(notes: list[str]) -> None:
    issues = validate_writing_task(2, "proposal", part2(notes=notes))
    assert any("between 2 and 5 content points" in i for i in issues)


def test_part_2_must_name_its_genre() -> None:
    task = part2(
        scenario="The council has money available to improve local sports facilities.",
        instruction="Write something for the council about the facilities. Write 220-260 words.",
    )
    issues = validate_writing_task(2, "proposal", task)
    assert any("names the genre" in i for i in issues)


def test_naming_the_genre_in_the_scenario_is_enough() -> None:
    task = part2(
        scenario=(
            "The council has invited residents to submit a proposal about how to "
            "improve the local sports facilities this year."
        ),
        instruction="Write it for the council, explaining your case. Write 220-260 words.",
    )
    assert not any("names the genre" in i for i in validate_writing_task(2, "proposal", task))


def test_an_email_task_may_be_called_a_letter_or_an_email() -> None:
    for word in ("letter", "email"):
        task = part2(
            instruction=f"Write a {word} to the council explaining your case. 220-260 words."
        )
        issues = validate_writing_task(2, "letter_email", task)
        assert not any("names the genre" in i for i in issues), word


def test_the_task_graph_mirrors_the_exercise_graph() -> None:
    """Same shape as its sibling, minus the one node that would buy nothing."""
    from cae.graphs.rue_task import build_generation_graph
    from cae.graphs.writing_task import build_writing_task_graph

    def nodes(graph):
        return {n for n in graph.get_graph().nodes if not n.startswith("__")}

    exercises = nodes(build_generation_graph())
    tasks = nodes(build_writing_task_graph())

    assert tasks == {"plan_task", "generate", "validate", "repair", "persist", "give_up"}
    assert exercises - tasks == {"critique"}


def test_part_1_is_forced_to_be_an_essay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        task_module, "structured_call", lambda *a, **k: factories.writing_task_spec()
    )
    result = task_module.generate_writing_task(
        part=1, genre="review", persist_result=False
    )
    assert result["genre"] == "essay"


def test_a_part_2_task_without_a_genre_is_refused() -> None:
    with pytest.raises(ValueError, match="needs a genre"):
        task_module.generate_writing_task(part=2, persist_result=False)


def test_the_topic_is_carried_into_the_stored_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []

    def fake_call(schema, system, user, **kwargs):
        prompts.append(user)
        return factories.writing_task_spec()

    monkeypatch.setattr(task_module, "structured_call", fake_call)

    result = task_module.generate_writing_task(
        part=1, topic="urban beekeeping", persist_result=False
    )

    assert result["task"]["topic"] == "urban beekeeping"
    assert "urban beekeeping" in prompts[0]


def test_a_generated_task_is_stored_and_reachable(user: dict, monkeypatch) -> None:
    from cae.db import repo
    from cae.db.session import session_scope

    monkeypatch.setattr(
        task_module, "structured_call", lambda *a, **k: factories.writing_task_spec()
    )
    result = task_module.generate_writing_task(part=1)

    assert result["task_id"] is not None
    with session_scope() as session:
        stored = repo.get_writing_task(session, result["task_id"])
    assert stored is not None
    assert stored.genre == "essay"
    assert stored.payload["generation_attempts"] == 1


def test_an_invalid_task_is_repaired_rather_than_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drafts = iter([
        factories.writing_task_spec(opinions=["Only one remark."]),
        factories.writing_task_spec(),
    ])
    prompts: list[str] = []

    def fake_call(schema, system, user, **kwargs):
        prompts.append(user)
        return next(drafts)

    monkeypatch.setattr(task_module, "structured_call", fake_call)

    result = task_module.generate_writing_task(part=1, persist_result=False)
    payload = result["task"]

    assert result["genre"] == "essay"
    assert payload["generation_attempts"] == 2
    assert len(result["issue_history"]) == 2 and result["issue_history"][-1] == []
    assert validate_writing_task(1, "essay", payload) == []
    assert "exactly 3 quoted opinions" in prompts[1]


def test_generation_gives_up_after_the_retry_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        task_module,
        "structured_call",
        lambda *a, **k: factories.writing_task_spec(notes=["only", "two"]),
    )

    with pytest.raises(task_module.WritingTaskFailed) as excinfo:
        task_module.generate_writing_task(
            part=1, max_attempts=2, persist_result=False
        )

    assert "exactly 3 content points" in str(excinfo.value)
    assert excinfo.value.part == 1
    assert excinfo.value.attempts == 2
    assert len(excinfo.value.issue_history) == 2 and all(excinfo.value.issue_history)


def test_a_valid_first_draft_costs_a_single_call(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_call(schema, system, user, **kwargs):
        calls.append(kwargs.get("temperature"))
        return factories.writing_task_spec()

    monkeypatch.setattr(task_module, "structured_call", fake_call)

    payload = task_module.generate_writing_task(part=1, persist_result=False)["task"]

    assert payload["generation_attempts"] == 1
    assert calls == [0.9], "the first draft is generated with variety, not obedience"
