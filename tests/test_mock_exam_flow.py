"""Sitting a mock exam on one paper or both."""

from __future__ import annotations

import re
import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from cae.db import repo
from cae.db.session import session_scope
from cae.domain.parts import (
    READING_MAX_MARKS,
    READING_PART_NUMBERS,
    WRITING_GENRES,
    WRITING_PART2_GENRES,
)
from cae.graphs import rue_evaluator, rue_task, writing_evaluator
from cae.graphs import writing_task as task_module
from cae.schemas.rue_task import ModerationResult, answer_key
from cae.schemas.rue_evaluator import AdjudicationBatch, ErrorTagBatch
from cae.schemas.writing_task import WritingTaskSpec
from cae.schemas.writing_evaluator import AnnotationSet, ImprovedVersion, SubscaleAssessment
from tests import factories

ENTRYPOINT = str(Path(__file__).resolve().parents[1] / "app.py")
SCRIPT = " ".join(["Museums ought to remain free for every visitor."] * 30)
SHORT_SCRIPT = " ".join(["Museums ought to remain free for every visitor."] * 10)


@pytest.fixture
def stubbed_model(monkeypatch: pytest.MonkeyPatch, free_providers):
    unexpected: list[type] = []

    def fake_generation(schema, system, user, **kwargs):
        if schema is ModerationResult:
            return ModerationResult(issues=[], verdict="accept")
        for part in range(1, 9):
            if schema is type(factories.build(part)):
                return factories.build(part)
        unexpected.append(schema)
        raise RuntimeError(f"unexpected schema {schema}")

    def fake_grading(schema, system, user, **kwargs):
        if not issubclass(schema, (ErrorTagBatch, AdjudicationBatch)):
            unexpected.append(schema)
            raise RuntimeError(f"unexpected schema {schema}")
        return factories.marking_reply(schema, user)

    def fake_writing(schema, system, user, **kwargs):
        if schema is WritingTaskSpec:
            if "Part 1 task" in user:
                return factories.writing_task_spec(1)
            genre = next(
                g for g in WRITING_PART2_GENRES if WRITING_GENRES[g].name in user
            )
            return factories.writing_task_spec(2, genre)
        if schema is SubscaleAssessment:
            return SubscaleAssessment(
                band=4, justification="Solid.", evidence=[], to_improve="Vary linking."
            )
        if schema is AnnotationSet:
            return AnnotationSet(annotations=[], strengths=["A clear position."])
        return ImprovedVersion(text="Improved.", upgrades=[])

    monkeypatch.setattr(rue_task, "structured_call", fake_generation)
    monkeypatch.setattr(rue_evaluator, "structured_call", fake_grading)
    monkeypatch.setattr(writing_evaluator, "structured_call", fake_writing)
    monkeypatch.setattr(task_module, "structured_call", fake_writing)
    yield
    assert not unexpected, f"the model was asked for {unexpected}"


def open_mock(user: dict) -> AppTest:
    app = AppTest.from_file(ENTRYPOINT, default_timeout=120)
    app.session_state["user"] = user
    app.run()
    app.switch_page("pages/3_Mock_exam.py")
    return app.run()


def start(app: AppTest, choice: str) -> AppTest:
    app.radio(key="mock_paper_choice").set_value(choice)
    app.run()
    next(b for b in app.button if b.label == "Start the exam").click()
    return app.run()


def click(app: AppTest, label: str) -> AppTest:
    next(b for b in app.button if b.label == label).click()
    return app.run()


_ANSWERS: dict[int, dict[int, str]] = {
    part: {item.number: item.answer for item in answer_key(part, factories.build(part))}
    for part in READING_PART_NUMBERS
}
_KEY_RE = re.compile(r"^mock\d+_p(\d+)_q(\d+)$")


def fill_answers(app: AppTest) -> None:
    """Answer every question widget in the tree from its own answer key.

    AppTest keeps widgets emitted before an ``st.rerun()``, so all of them must
    hold a value or the next run raises ``KeyError``.
    """
    for widgets in (app.radio, app.selectbox, app.text_input):
        for widget in list(widgets):
            match = _KEY_RE.match(widget.key or "")
            if match is None:
                continue
            part, number = int(match.group(1)), int(match.group(2))
            answer = _ANSWERS.get(part, {}).get(number)
            if answer is not None:
                widget.set_value(answer)


def saved_exam(user_id: int):
    with session_scope() as session:
        exams = repo.list_mock_exams(session, user_id)
        papers = repo.mock_exam_papers(session, [e.id for e in exams])
        return exams[0], papers[exams[0].id]


def test_a_writing_only_mock_never_enters_the_reading_paper(
    user: dict, stubbed_model
) -> None:
    app = start(open_mock(user), "Writing")
    assert not app.exception

    state = app.session_state["mock"]
    assert state["papers"] == ("writing",)
    assert state["phase"] == "writing"
    assert state["exercises"] == {}
    assert len(state["writing"]) == 2

    for index, label in enumerate(["Next task", "Finish the paper"]):
        app.text_area(key=f"mockwrite_{state['exam_id']}_{index}").set_value(SCRIPT)
        app = click(app.run(), label)
        assert not app.exception

    state = app.session_state["mock"]
    assert state["phase"] == "done"
    assert all(task.get("result") for task in state["writing"])

    exam, papers = saved_exam(user["id"])
    assert papers == "Writing"
    assert exam.max_marks == 40, "two writing tasks are worth 20 each"
    assert exam.marks == 32, "band 4 on all four subscales is 16 per task"
    assert exam.scale_score > 0


def button(app: AppTest, label: str):
    return next(b for b in app.button if b.label == label)


def test_a_task_outside_the_word_range_cannot_be_handed_in(
    user: dict, stubbed_model
) -> None:
    app = start(open_mock(user), "Writing")
    box = f"mockwrite_{app.session_state['mock']['exam_id']}_0"

    app.text_area(key=box).set_value(SHORT_SCRIPT)
    app = app.run()
    assert button(app, "Next task").disabled
    assert any("short of the minimum" in w.value for w in app.warning)

    app.text_area(key=box).set_value(SCRIPT + " " + SCRIPT)
    app = app.run()
    assert button(app, "Next task").disabled
    assert any("over the maximum" in w.value for w in app.warning)

    app.text_area(key=box).set_value(SCRIPT)
    app = app.run()
    assert not button(app, "Next task").disabled


def test_back_keeps_a_short_task_without_handing_it_in(
    user: dict, stubbed_model
) -> None:
    app = start(open_mock(user), "Writing")
    exam_id = app.session_state["mock"]["exam_id"]
    app.text_area(key=f"mockwrite_{exam_id}_0").set_value(SCRIPT)
    app = click(app.run(), "Next task")
    app.text_area(key=f"mockwrite_{exam_id}_1").set_value(SHORT_SCRIPT)
    app = click(app.run(), "Back")

    assert not app.exception
    second = app.session_state["mock"]["writing"][1]
    assert second["draft"] == SHORT_SCRIPT, "the learner's text is not thrown away"
    assert second["script"] == "", "but it was never confirmed"

    app = click(app, "Next task")
    assert app.text_area(key=f"mockwrite_{exam_id}_1").value == SHORT_SCRIPT
    app = expire(app)
    assert app.session_state["mock"]["writing"][1]["result"]["status"] == "out_of_range"


def test_writing_only_results_report_no_reading_marks(user: dict, stubbed_model) -> None:
    app = start(open_mock(user), "Writing")
    state = app.session_state["mock"]
    for index, label in enumerate(["Next task", "Finish the paper"]):
        app.text_area(key=f"mockwrite_{state['exam_id']}_{index}").set_value(SCRIPT)
        app = click(app.run(), label)

    assert not app.exception
    labels = [metric.label for metric in app.metric]
    assert "Writing" in labels
    assert "Reading & UoE" not in labels
    assert "Estimated scale score" in labels
    assert any("rests on that alone" in caption.value for caption in app.caption)


def test_a_reading_only_mock_runs_all_eight_parts_and_is_recorded(
    user: dict, stubbed_model
) -> None:
    app = start(open_mock(user), "Reading and Use of English")
    assert not app.exception

    state = app.session_state["mock"]
    assert state["papers"] == ("reading",)
    assert set(state["exercises"]) == set(READING_PART_NUMBERS)

    for position, part in enumerate(READING_PART_NUMBERS):
        assert app.session_state["mock"]["index"] == position
        fill_answers(app)
        is_last = part == READING_PART_NUMBERS[-1]
        app = click(app, "Finish paper" if is_last else "Next part")
        assert not app.exception, f"Part {part} raised: {app.exception}"

    state = app.session_state["mock"]
    assert state["phase"] == "done", "a reading-only mock must not ask for writing"
    assert not state["writing"]

    exam, papers = saved_exam(user["id"])
    assert papers == "Reading & UoE"
    assert exam.max_marks == READING_MAX_MARKS
    assert exam.marks == READING_MAX_MARKS, "every answer came from the key"
    assert exam.scale_score == 210


def test_going_back_a_part_keeps_the_answers(user: dict, stubbed_model) -> None:
    app = start(open_mock(user), "Reading and Use of English")
    exam_id = app.session_state["mock"]["exam_id"]

    app.radio(key=f"mock{exam_id}_p1_q1").set_value("A")
    app = click(app, "Next part")
    assert app.session_state["mock"]["index"] == 1

    app = click(app, "Back")
    assert not app.exception
    assert app.session_state["mock"]["index"] == 0
    assert app.session_state["mock"]["answers"][1][1] == "A"
    assert app.radio(key=f"mock{exam_id}_p1_q1").value == "A"


def expire(app: AppTest) -> AppTest:
    """Move the deadline into the past, as if the clock had just reached zero."""
    state = app.session_state["mock"]
    state["deadline"] = time.time() - 1
    app.session_state["mock"] = state
    return app.run()


def test_time_up_hands_in_only_the_confirmed_reading_parts(
    user: dict, stubbed_model
) -> None:
    app = start(open_mock(user), "Reading and Use of English")
    fill_answers(app)
    app = click(app, "Next part")
    fill_answers(app)
    app = expire(app.run())

    assert not app.exception
    state = app.session_state["mock"]
    assert state["phase"] == "done"
    assert state["timed_out"] == ["reading"]
    assert 2 not in state["answers"]
    assert any("Time ran out" in info.value for info in app.info)

    exam, _ = saved_exam(user["id"])
    assert exam.marks == state["results"][1]["max_marks"], "only part 1 counts"
    assert exam.max_marks == READING_MAX_MARKS


def test_time_up_on_reading_starts_the_writing_clock(user: dict, stubbed_model) -> None:
    app = expire(start(open_mock(user), "Both papers"))

    assert not app.exception
    state = app.session_state["mock"]
    assert state["phase"] == "writing"
    assert state["results"], "the reading paper was marked"
    assert state["deadline"] > time.time() + 60, "writing gets a fresh clock"


def test_time_up_hands_in_only_the_confirmed_writing_tasks(
    user: dict, stubbed_model
) -> None:
    app = start(open_mock(user), "Writing")
    exam_id = app.session_state["mock"]["exam_id"]
    app.text_area(key=f"mockwrite_{exam_id}_0").set_value(SCRIPT)
    app = click(app.run(), "Next task")
    app.text_area(key=f"mockwrite_{exam_id}_1").set_value(SCRIPT)
    app = expire(app.run())

    assert not app.exception
    state = app.session_state["mock"]
    assert state["phase"] == "done"
    assert state["timed_out"] == ["writing"]
    assert state["writing"][0]["script"] == SCRIPT
    assert state["writing"][1]["script"] == ""
    assert state["writing"][1]["result"]["status"] == "out_of_range"

    exam, _ = saved_exam(user["id"])
    assert exam.max_marks == 40
    assert exam.marks == 16, "the confirmed task is marked; the open one scores 0"


def _attempts_of(exam_id: int) -> int:
    from cae.db.models import Attempt

    with session_scope() as session:
        return session.query(Attempt).filter_by(mock_exam_id=exam_id).count()


def test_a_reading_paper_the_model_cannot_mark_waits_unsaved_and_is_marked_once(
    user: dict, stubbed_model, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whole or not at all: no part is saved until every part is marked, so marking
    it again later records each part exactly once."""
    from cae.graphs import rue_evaluator
    from cae.llm import ModelUnreachableError

    app = start(open_mock(user), "Reading and Use of English")
    exam_id = app.session_state["mock"]["exam_id"]
    for part in READING_PART_NUMBERS:
        fill_answers(app)
        if part == 1:
            widget = next(
                w for w in list(app.radio) + list(app.selectbox)
                if re.match(r"^mock\d+_p1_q1$", w.key or "")
            )
            widget.set_value(next(letter for letter in "ABCD" if letter != _ANSWERS[1][1]))
        if part == READING_PART_NUMBERS[-1]:
            def down(*args, **kwargs):
                raise ModelUnreachableError("overloaded")

            monkeypatch.setattr(rue_evaluator, "structured_call", down)
        app = click(app, "Finish paper" if part == READING_PART_NUMBERS[-1] else "Next part")
        assert not app.exception, f"Part {part} raised: {app.exception}"

    state = app.session_state["mock"]
    assert state["phase"] == "unmarked"
    assert app.error, "the learner must be told the paper was not marked"
    assert _attempts_of(exam_id) == 0, "no part may be saved before all are marked"

    monkeypatch.setattr(
        rue_evaluator,
        "structured_call",
        lambda schema, system, user_prompt, **k: factories.marking_reply(schema, user_prompt),
    )
    fill_answers(app)
    app = click(app, "Mark it again")

    assert not app.exception
    assert app.session_state["mock"]["phase"] == "done"
    assert _attempts_of(exam_id) == len(READING_PART_NUMBERS)
    exam, _ = saved_exam(user["id"])
    assert exam.marks == READING_MAX_MARKS - 1
