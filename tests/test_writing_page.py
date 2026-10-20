"""The Writing practice page: a script is handed in only within the word range."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from cae.graphs import writing_evaluator
from cae.graphs import writing_task as task_module
from cae.schemas.writing_evaluator import AnnotationSet, ImprovedVersion, SubscaleAssessment
from cae.schemas.writing_task import WritingTaskSpec
from tests import factories

ENTRYPOINT = str(Path(__file__).resolve().parents[1] / "app.py")
SENTENCE = "Museums ought to remain free for every visitor."
SCRIPT = " ".join([SENTENCE] * 30)


@pytest.fixture
def stubbed_model(monkeypatch: pytest.MonkeyPatch, free_providers):
    def fake(schema, system, user, **kwargs):
        if schema is WritingTaskSpec:
            return factories.writing_task_spec(1)
        if schema is SubscaleAssessment:
            return SubscaleAssessment(
                band=4, justification="Solid.", evidence=[], to_improve="Vary linking."
            )
        if schema is AnnotationSet:
            return AnnotationSet(annotations=[], strengths=["A clear position."])
        return ImprovedVersion(text="Improved.", upgrades=[])

    monkeypatch.setattr(task_module, "structured_call", fake)
    monkeypatch.setattr(writing_evaluator, "structured_call", fake)


def open_task(user: dict) -> AppTest:
    app = AppTest.from_file(ENTRYPOINT, default_timeout=120)
    app.session_state["user"] = user
    app.run()
    app.switch_page("pages/2_Writing.py")
    app.run()
    next(b for b in app.button if b.label == "New task").click()
    return app.run()


def submit_button(app: AppTest):
    return next(b for b in app.button if b.label == "Submit for marking")


@pytest.mark.parametrize(
    "script, message",
    [
        ("", "to hand it in"),
        (" ".join([SENTENCE] * 27), "short of the minimum"),
        (" ".join([SENTENCE] * 33), "over the maximum"),
    ],
)
def test_a_script_outside_the_word_range_cannot_be_submitted(
    user: dict, stubbed_model, script: str, message: str
) -> None:
    app = open_task(user)
    app.text_area(key="writing_script").set_value(script)
    app = app.run()

    assert not app.exception
    assert submit_button(app).disabled
    shown = [w.value for w in app.warning] + [c.value for c in app.caption]
    assert any(message in text for text in shown)


def test_a_script_within_the_range_is_submitted_and_marked(
    user: dict, stubbed_model
) -> None:
    app = open_task(user)
    app.text_area(key="writing_script").set_value(SCRIPT)
    app = app.run()
    assert not submit_button(app).disabled

    submit_button(app).click()
    app = app.run()
    assert not app.exception
    assert app.session_state["writing_result"]["total"] == 16
