"""The practice page, driven through the real UI.

Answering every part from its own key and scoring full marks proves the
renderers and the grader agree.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from cae.graphs import rue_evaluator, rue_task
from cae.schemas.rue_task import ModerationResult, answer_key
from cae.schemas.rue_evaluator import AdjudicationBatch, ErrorTagBatch
from tests import factories

ENTRYPOINT = str(Path(__file__).resolve().parents[1] / "app.py")

ALL_PARTS = list(range(1, 9))


@pytest.fixture
def stubbed_model(monkeypatch: pytest.MonkeyPatch, free_providers):
    """Generation returns a known-good exercise, moderation accepts it, tagging
    returns nothing.

    The graphs catch a failed model call, so an error raised inside a stub
    never reaches the test. Unexpected requests are recorded instead and
    checked once the test is over.
    """
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

    monkeypatch.setattr(rue_task, "structured_call", fake_generation)
    monkeypatch.setattr(rue_evaluator, "structured_call", fake_grading)
    yield
    assert not unexpected, f"the model was asked for {unexpected}"


def open_practice(user: dict) -> AppTest:
    app = AppTest.from_file(ENTRYPOINT, default_timeout=60)
    app.session_state["user"] = user
    app.run()
    app.switch_page("pages/1_Reading_and_Use_of_English.py")
    return app.run()


def start(app: AppTest, part: int) -> AppTest:
    app.selectbox(key="practice_part").set_value(part)
    app.run()
    next(b for b in app.button if b.label == "Start").click()
    return app.run()


def _is_letter(value: str) -> bool:
    return len(value) == 1 and value.isalpha() and value.isupper()


def answer_everything(app: AppTest, part: int, correct: bool) -> None:
    """Fill every answer widget on the page from the exercise's own key."""
    payload = app.session_state["practice_exercise"]["payload"]
    exercise_id = app.session_state["practice_exercise"]["exercise_id"]
    prefix = f"p{part}_e{exercise_id}"

    for item in answer_key(part, factories.build(part)):
        key = f"{prefix}_q{item.number}"
        value = item.answer
        if not correct:
            value = ("A" if value != "A" else "B") if _is_letter(value) else "not this"
        for widgets in (app.radio, app.selectbox, app.text_input):
            try:
                widgets(key=key).set_value(value)
                break
            except (KeyError, ValueError):
                continue
    assert payload["items"], "the exercise should have items to answer"


def submit(app: AppTest) -> AppTest:
    next(b for b in app.button if "Submit answers" in b.label).click()
    return app.run()


@pytest.mark.parametrize("part", ALL_PARTS)
def test_a_correct_script_scores_full_marks_through_the_ui(
    part: int, user: dict, stubbed_model
) -> None:
    app = start(open_practice(user), part)
    assert not app.exception
    assert app.session_state["practice_exercise"]["part"] == part

    answer_everything(app, part, correct=True)
    app = submit(app)

    assert not app.exception, f"Part {part} raised: {app.exception}"
    result = app.session_state["practice_result"]
    assert result["marks"] == result["max_marks"], (
        f"Part {part}: the renderer and the answer key disagree — "
        f"scored {result['marks']} of {result['max_marks']}"
    )


@pytest.mark.parametrize("part", ALL_PARTS)
def test_the_exercise_stays_on_screen_with_the_corrections(
    part: int, user: dict, stubbed_model
) -> None:
    """A correction is useless if the text it refers to has disappeared."""
    app = start(open_practice(user), part)
    answer_everything(app, part, correct=False)
    app = submit(app)
    assert not app.exception

    body = " ".join(block.value for block in app.markdown)
    payload = app.session_state["practice_exercise"]["payload"]

    if part == 4:
        assert payload["items"][0]["lead_in"] in body
    else:
        sample = factories.quote_from(
            payload.get("text") or _first_body(payload), start=2, length=6
        )
        assert sample in body, f"Part {part}: the material is missing from the results"

    assert "Item feedback" in body


def _first_body(payload: dict) -> str:
    """The first block of prose for the parts that have no single 'text'."""
    for key in ("texts", "sections"):
        blocks = payload.get(key)
        if blocks:
            return blocks[0].get("content", "")
    return ""


def test_a_wrong_script_scores_zero_and_shows_the_key(user: dict, stubbed_model) -> None:
    app = start(open_practice(user), 1)
    answer_everything(app, 1, correct=False)
    app = submit(app)

    assert not app.exception
    result = app.session_state["practice_result"]
    assert result["marks"] == 0
    assert all(row["correct_answer"] for row in result["rows"])
    assert any("Results" in block.value for block in app.markdown)


def test_starting_again_clears_the_previous_answers(user: dict, stubbed_model) -> None:
    app = start(open_practice(user), 2)
    answer_everything(app, 2, correct=True)
    app = submit(app)
    first_id = app.session_state["practice_exercise"]["exercise_id"]

    next(b for b in app.button if b.label == "Start").click()
    app.run()

    assert not app.exception
    assert app.session_state["practice_exercise"]["exercise_id"] != first_id
    assert "practice_result" not in app.session_state
    with pytest.raises(KeyError):
        app.text_input(key=f"p2_e{first_id}_q1")


def test_a_part_the_model_cannot_mark_is_not_saved_and_can_be_sent_again(
    user: dict, stubbed_model, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No fallback marks: the learner is told, nothing is stored, the answers stay."""
    from cae.db import repo
    from cae.db.session import session_scope

    app = start(open_practice(user), 1)
    answer_everything(app, 1, correct=False)
    monkeypatch.setattr(rue_evaluator, "structured_call", lambda *a, **k: ErrorTagBatch(tags=[]))
    app = submit(app)

    assert not app.exception
    assert "practice_result" not in app.session_state
    assert any("could not be marked" in e.value for e in app.error)
    with session_scope() as session:
        assert repo.attempt_history(session, user["id"]) == []

    monkeypatch.setattr(
        rue_evaluator,
        "structured_call",
        lambda schema, system, user_prompt, **k: factories.marking_reply(schema, user_prompt),
    )
    app = submit(app)

    assert not app.exception
    result = app.session_state["practice_result"]
    assert all(row["user_answer"] for row in result["rows"]), "the answers were kept"
    with session_scope() as session:
        assert len(repo.attempt_history(session, user["id"])) == 1
