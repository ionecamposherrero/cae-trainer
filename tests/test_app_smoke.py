"""Every Streamlit page renders without raising.

Pages are reached with ``switch_page`` from ``app.py``, since page links only
resolve against the real entry point.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ENTRYPOINT = str(Path(__file__).resolve().parents[1] / "app.py")
PAGES = [
    "pages/1_Reading_and_Use_of_English.py",
    "pages/2_Writing.py",
    "pages/3_Mock_exam.py",
    "pages/4_Progress.py",
    "pages/5_Talk_with_your_coach.py",
    "pages/6_Account.py",
]


def home(session_state: dict | None = None) -> AppTest:
    app = AppTest.from_file(ENTRYPOINT, default_timeout=30)
    for key, value in (session_state or {}).items():
        app.session_state[key] = value
    return app.run()


def open_page(script: str, session_state: dict | None = None) -> AppTest:
    app = home(session_state)
    app.switch_page(script)
    return app.run()


def signed_in(app: AppTest) -> bool:
    return "user" in app.session_state


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("app.py", "pages", "src")
DEPRECATED = ("use_container_width",)


def source_files() -> list[Path]:
    files = [PROJECT_ROOT / "app.py"]
    for directory in SOURCE_DIRS[1:]:
        files.extend((PROJECT_ROOT / directory).rglob("*.py"))
    return files


@pytest.mark.parametrize("name", DEPRECATED)
def test_no_deprecated_streamlit_arguments(name: str) -> None:
    offenders = [
        f"{path.relative_to(PROJECT_ROOT)}:{number}"
        for path in source_files()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if name in line
    ]
    assert not offenders, f"`{name}` is deprecated; found at {offenders}"


def test_home_renders_the_sign_in_screen() -> None:
    app = home()
    assert not app.exception
    assert "C1 Advanced Trainer" in app.title[0].value


def test_a_visitor_can_register_and_lands_on_the_dashboard() -> None:
    import uuid

    username = f"smoke-{uuid.uuid4().hex[:8]}"
    app = home()
    app.text_input(key="reg_username").set_value(username)
    app.text_input(key="reg_password").set_value("correct-horse-battery")
    app.button[1].click().run()

    assert not app.exception
    assert app.session_state["user"]["username"] == username
    assert "Welcome back" in app.title[0].value


def test_registration_shows_its_errors() -> None:
    app = home()
    app.text_input(key="reg_username").set_value("ab")
    app.text_input(key="reg_password").set_value("short")
    app.button[1].click().run()

    assert not app.exception
    assert not signed_in(app)
    assert any("3-32 characters" in error.value for error in app.error)


def test_signing_in_with_a_bad_password_is_refused(user: dict) -> None:
    app = home()
    app.text_input(key="login_username").set_value(user["username"])
    app.text_input(key="login_password").set_value("not-the-password")
    app.button[0].click().run()

    assert not app.exception
    assert not signed_in(app)
    assert any("Incorrect username or password" in error.value for error in app.error)


@pytest.mark.parametrize("script", PAGES)
def test_pages_require_a_signed_in_user(script: str) -> None:
    app = open_page(script)
    assert not app.exception
    assert any("sign in" in warning.value.lower() for warning in app.warning)


@pytest.mark.parametrize("script", PAGES)
def test_pages_render_for_a_signed_in_user(script: str, user: dict) -> None:
    app = open_page(script, {"user": user})
    assert not app.exception, f"{script} raised: {app.exception}"


def test_progress_page_guides_a_learner_with_no_history(user: dict) -> None:
    app = open_page("pages/4_Progress.py", {"user": user})
    assert not app.exception
    assert any("Nothing to show yet" in info.value for info in app.info)


def test_reading_page_offers_all_eight_parts(user: dict) -> None:
    app = open_page("pages/1_Reading_and_Use_of_English.py", {"user": user})
    assert not app.exception
    options = app.selectbox(key="practice_part").options
    assert len(options) == 8
    assert options[0].startswith("Part 1")
    assert options[-1].endswith("Multiple matching")


def test_mock_exam_offers_a_choice_of_paper(user: dict) -> None:
    app = open_page("pages/3_Mock_exam.py", {"user": user})
    assert not app.exception
    options = app.radio(key="mock_paper_choice").options
    assert options == [
        "Reading and Use of English — 90 minutes",
        "Writing — 90 minutes",
        "Both papers — 180 minutes",
    ]
    assert any("Start the exam" in button.label for button in app.button)


def test_mock_exam_describes_only_the_chosen_paper(user: dict) -> None:
    app = open_page("pages/3_Mock_exam.py", {"user": user})

    body = " ".join(block.value for block in app.markdown)
    assert "Part 8" in body and "Multiple matching" in body
    assert "compulsory" not in body

    app.radio(key="mock_paper_choice").set_value("Writing").run()
    assert not app.exception
    body = " ".join(block.value for block in app.markdown)
    assert "compulsory" in body and "220-260 words" in body
    assert "Multiple matching" not in body


def test_an_overloaded_model_shows_a_message_not_a_traceback(
    user: dict, free_providers, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 503 from the provider reaches the learner as a message, not a traceback.

    The stand-in sits where the real client does, so the whole path runs for
    real: retry, move along the chain, exhaust every model, translate, and
    surface through the page's own handler. Only the backoff sleep is stubbed out
    (by ``free_providers``).
    """
    from cae import llm

    class Overloaded:
        """A chat model that fails the way Gemini does under load."""

        def with_structured_output(self, _schema):
            return self

        def invoke(self, *_args, **_kwargs):
            raise RuntimeError(
                "google.genai.errors.ServerError: 503 UNAVAILABLE. {'error': "
                "{'code': 503, 'message': 'This model is currently experiencing "
                "high demand.', 'status': 'UNAVAILABLE'}}"
            )

    monkeypatch.setattr(llm, "build_llm", lambda *_args, **_kwargs: Overloaded())

    app = open_page("pages/1_Reading_and_Use_of_English.py", {"user": user})
    next(b for b in app.button if b.label == "Start").click().run()

    assert not app.exception, "the page must handle this, not raise"
    errors = " ".join(error.value for error in app.error)
    assert "try again" in errors.lower()
    for leak in ("503", "UNAVAILABLE", "google.genai", "Traceback"):
        assert leak not in errors, f"{leak!r} was shown to the learner"


def stub_coach(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the Progress page offline once the learner has work on record.

    Above two graded pieces the page writes a coaching report, which is a real
    model call. Stubbing ``narrate``'s call leaves the rest of the graph — the
    SQL, the trend arithmetic, the rendering — running for real.
    """
    from cae.graphs import coach
    from cae.schemas.coach import CoachNarrative, FocusArea

    narrative = CoachNarrative(
        headline="Steady, with Part 1 to fix.",
        summary="You are marking around 40%, short of the 180 pass mark.",
        strengths=["Open cloze is your best part at 50%."],
        common_errors=["Collocation: you pick a near-synonym that does not partner."],
        trend="Too early to call a direction.",
        focus_areas=[
            FocusArea(title="Collocation", why="Your commonest error.", action="Ten a day."),
        ],
        next_session="Another Part 1.",
    )
    monkeypatch.setattr(coach, "structured_call", lambda *a, **k: narrative)


def test_progress_groups_the_reading_only_sections_under_their_own_heading(
    user: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Accuracy and the part rankings come from reading attempts alone.

    The recurring-mistakes table does not: the taxonomy is shared, so it
    aggregates both papers and must sit outside either paper's heading.
    """
    from cae.db import repo
    from cae.db.session import session_scope
    from tests import factories

    stub_coach(monkeypatch)
    with session_scope() as session:
        exercise = repo.save_exercise(
            session, part=1, payload=factories.build(1).model_dump()
        )
        repo.record_attempt(
            session,
            user_id=user["id"],
            exercise_id=exercise.id,
            part=1,
            responses={"1": "A"},
            item_rows=[
                {
                    "item_number": 1,
                    "user_answer": "A",
                    "correct_answer": "B",
                    "is_correct": False,
                    "marks_awarded": 0.0,
                    "max_marks": 1.0,
                    "category": "FIXED_PHRASES",
                    "evidence": "...gap {{1}}... — wrote \"A\", key \"B\"",
                }
            ],
            marks=5.0,
            max_marks=8.0,
        )
        task = repo.save_writing_task(
            session, part=1, genre="essay", payload={"title": "T"}
        )
        submission = repo.save_submission(
            session,
            user_id=user["id"],
            task_id=task.id,
            text="word " * 250,
            word_count=250,
        )
        repo.save_writing_score(
            session,
            submission_id=submission.id,
            user_id=user["id"],
            bands={
                "content": 4,
                "communicative_achievement": 3,
                "organisation": 3,
                "language": 3,
            },
            total=13,
            scale_score=180,
            feedback={},
            annotations=[],
        )

    app = open_page("pages/4_Progress.py", {"user": user})
    assert not app.exception

    blocks = [block.value for block in app.markdown]

    assert "### Reading and Use of English" in blocks
    for nested in ("#### Accuracy", "#### Strongest parts", "#### Weakest parts"):
        assert nested in blocks, f"{nested} should be nested under Reading"
        assert f"#{nested}" not in blocks, f"{nested} should not also be top level"

    assert "### Writing" in blocks

    assert blocks.index("### Your recurring mistakes") > blocks.index("### Writing")
    assert blocks.index("### Your recurring mistakes") > blocks.index(
        "### Reading and Use of English"
    )


def test_the_mock_exam_section_is_there_before_the_first_one_is_sat(
    user: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The heading must appear with practice on record but no mock exam yet."""
    from cae.db import repo
    from cae.db.session import session_scope
    from tests import factories

    stub_coach(monkeypatch)
    with session_scope() as session:
        exercise = repo.save_exercise(
            session, part=1, payload=factories.build(1).model_dump()
        )
        repo.record_attempt(
            session,
            user_id=user["id"],
            exercise_id=exercise.id,
            part=1,
            responses={"1": "A"},
            item_rows=[],
            marks=5.0,
            max_marks=8.0,
        )

    app = open_page("pages/4_Progress.py", {"user": user})
    assert not app.exception

    assert "Mock exams" in " ".join(block.value for block in app.markdown)
    assert any("not finished a mock exam yet" in info.value for info in app.info)


def test_progress_charts_each_kind_of_mock_exam_sitting(
    user: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reading-only, Writing-only and complete sittings each get their own line."""
    from cae.db import repo
    from cae.db.session import session_scope
    from tests import factories

    stub_coach(monkeypatch)

    def sit(*, reading: bool, writing: bool, score: int) -> None:
        with session_scope() as session:
            exam = repo.create_mock_exam(session, user["id"])
            if reading:
                exercise = repo.save_exercise(
                    session, part=1, payload=factories.build(1).model_dump()
                )
                repo.record_attempt(
                    session,
                    user_id=user["id"],
                    exercise_id=exercise.id,
                    part=1,
                    responses={"1": "A"},
                    item_rows=[],
                    marks=5.0,
                    max_marks=8.0,
                    mode="mock",
                    mock_exam_id=exam.id,
                )
            if writing:
                task = repo.save_writing_task(
                    session, part=1, genre="essay", payload={"title": "T"}
                )
                repo.save_submission(
                    session,
                    user_id=user["id"],
                    task_id=task.id,
                    text="word " * 250,
                    word_count=250,
                    mock_exam_id=exam.id,
                )
            repo.finish_mock_exam(
                session, exam_id=exam.id, marks=50.0, max_marks=78.0, scale_score=score
            )

    sit(reading=True, writing=True, score=168)
    sit(reading=True, writing=False, score=174)
    sit(reading=False, writing=True, score=161)

    app = open_page("pages/4_Progress.py", {"user": user})
    assert not app.exception, f"Progress raised: {app.exception}"

    body = " ".join(block.value for block in app.markdown)
    assert "Mock exams" in body
    captions = " ".join(caption.value for caption in app.caption)
    assert "separate line for each kind of sitting" in captions


def test_choosing_both_papers_shows_both_and_their_separate_clocks(user: dict) -> None:
    app = open_page("pages/3_Mock_exam.py", {"user": user})
    app.radio(key="mock_paper_choice").set_value("Both papers").run()
    assert not app.exception
    body = " ".join(block.value for block in app.markdown)
    assert "Multiple matching" in body and "compulsory" in body
    assert any("timed separately" in caption.value for caption in app.caption)
