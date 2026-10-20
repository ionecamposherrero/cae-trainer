"""The whole loop against a real database: generate, answer, mark, aggregate.

Only the model is stubbed. Everything else -- the graphs, the repository, the
analytics -- is the code the app runs.
"""

from __future__ import annotations

import pytest

from cae import auth, services
from cae.db import repo
from cae.db.session import session_scope
from cae.graphs import rue_evaluator, rue_task, writing_evaluator
from cae.graphs import writing_task as task_module
from cae.schemas.rue_task import ModerationResult, answer_key, parse_exercise
from cae.schemas.rue_evaluator import ErrorTag, ErrorTagBatch
from cae.schemas.writing_evaluator import AnnotationSet, ImprovedVersion, SubscaleAssessment
from tests import factories


@pytest.fixture
def stub_generation(monkeypatch: pytest.MonkeyPatch):
    """Stub the generator and return a list that records every call to it."""
    calls: list[int] = []
    unexpected: list[type] = []

    def fake_call(schema, system, user, **kwargs):
        if schema is ModerationResult:
            return ModerationResult(issues=[], verdict="accept")
        for part in range(1, 9):
            if schema is type(factories.build(part)):
                calls.append(part)
                return factories.build(part)
        unexpected.append(schema)
        raise RuntimeError(f"unexpected schema {schema}")

    monkeypatch.setattr(rue_task, "structured_call", fake_call)
    yield calls
    assert not unexpected, f"the generator was asked for {unexpected}"


@pytest.fixture
def stub_tagging(monkeypatch: pytest.MonkeyPatch):
    unexpected: list[type] = []

    def fake_call(schema, system, user, **kwargs):
        if not issubclass(schema, ErrorTagBatch):
            unexpected.append(schema)
            raise RuntimeError(f"unexpected schema {schema}")
        numbers = [int(line.split()[1]) for line in user.splitlines() if line.startswith("Item ")]
        return ErrorTagBatch(
            tags=[
                ErrorTag(
                    number=n,
                    category="FIXED_PHRASES",
                    diagnosis="This word does not collocate here.",
                )
                for n in numbers
            ]
        )

    monkeypatch.setattr(rue_evaluator, "structured_call", fake_call)
    yield
    assert not unexpected, f"the grader was asked for {unexpected}"


def test_registration_rejects_a_duplicate_username(user: dict) -> None:
    with pytest.raises(auth.AuthError, match="already taken"):
        auth.register(user["username"], "another-password-1")


def test_login_round_trips(user: dict) -> None:
    assert auth.login(user["username"], "correct-horse-battery")["id"] == user["id"]
    with pytest.raises(auth.AuthError):
        auth.login(user["username"], "wrong-password")


def test_a_failed_login_costs_the_same_whether_or_not_the_user_exists(
    user: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One bcrypt check in both cases, so timing does not reveal existing accounts."""
    import uuid

    checks: list[str] = []
    real_verify = auth.verify_password

    def counting_verify(password: str, password_hash: str) -> bool:
        checks.append(password_hash)
        return real_verify(password, password_hash)

    def no_hashing(password: str) -> str:
        raise AssertionError("login must not compute a new hash")

    monkeypatch.setattr(auth, "verify_password", counting_verify)
    monkeypatch.setattr(auth, "hash_password", no_hashing)

    for username in (user["username"], f"ghost{uuid.uuid4().hex[:8]}"):
        with pytest.raises(auth.AuthError, match="Incorrect username or password"):
            auth.login(username, "wrong-password")

    assert len(checks) == 2


def test_every_request_generates_a_new_exercise_and_none_is_reused(
    user: dict, stub_generation: list[int]
) -> None:
    import uuid

    first_id, payload = services.next_exercise(user["id"], 3)
    assert payload["items"]
    assert stub_generation == [3], "one exercise should have been generated"
    with session_scope() as session:
        assert repo.get_exercise(session, first_id) is not None, "it is stored"

    other = auth.register(f"learner{uuid.uuid4().hex[:8]}", "correct-horse-battery")
    other_id, _ = services.next_exercise(other["id"], 3)
    again_id, _ = services.next_exercise(user["id"], 3)
    assert stub_generation == [3, 3, 3]
    assert len({first_id, other_id, again_id}) == 3


def test_a_wrong_answer_becomes_an_error_event(
    user: dict, stub_generation, stub_tagging
) -> None:
    exercise_id, payload = services.next_exercise(user["id"], 1)
    key = answer_key(1, parse_exercise(1, payload))

    responses = {k.number: k.answer for k in key}
    responses[1] = "A" if key[0].answer != "A" else "B"

    result = services.submit_attempt(
        user_id=user["id"],
        exercise_id=exercise_id,
        part=1,
        exercise=payload,
        responses=responses,
        duration_seconds=42,
    )

    assert result["marks"] == 7
    assert result["max_marks"] == 8

    with session_scope() as session:
        attempt = repo.get_attempt(session, result["attempt_id"])
        assert attempt is not None
        assert attempt.duration_seconds == 42
        assert len(attempt.items) == 8
        events = repo.top_error_categories(session, user["id"])

    assert events == [
        {"category": "FIXED_PHRASES", "label": "Collocations, phrasal verbs and idioms", "count": 1, "share": 100.0}
    ]


def test_progress_snapshot_aggregates_across_parts(
    user: dict, stub_generation, stub_tagging
) -> None:
    for part in (1, 2, 3):
        exercise_id, payload = services.next_exercise(user["id"], part)
        key = answer_key(part, parse_exercise(part, payload))
        if part == 1:
            responses = {k.number: k.answer for k in key}
        elif part == 2:
            responses = {
                k.number: (k.answer if k.number <= 4 else "") for k in key
            }
        else:
            responses = {k.number: "" for k in key}
        services.submit_attempt(
            user_id=user["id"],
            exercise_id=exercise_id,
            part=part,
            exercise=payload,
            responses=responses,
        )

    snapshot = services.progress(user["id"])

    assert snapshot.reading_attempts == 3
    assert snapshot.activity_count == 3
    by_part = {stat.part: stat for stat in snapshot.part_stats}
    assert by_part[1].accuracy == 100.0
    assert by_part[2].accuracy == 50.0
    assert by_part[3].accuracy == 0.0
    assert by_part[4].attempts == 0

    assert snapshot.strengths[0].part == 1
    assert snapshot.weaknesses[0].part == 3
    assert snapshot.overall_accuracy == pytest.approx(50.0)


def test_writing_submission_is_scored_and_recorded(
    user: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = " ".join(["Museums ought to remain free for every visitor."] * 30)

    def fake_task(schema, system, user_prompt, **kwargs):
        return factories.writing_task_spec()

    def fake_assess(schema, system, user_prompt, **kwargs):
        if schema is SubscaleAssessment:
            return SubscaleAssessment(
                band=4, justification="Solid.", evidence=[], to_improve="Vary the linking."
            )
        if schema is AnnotationSet:
            return AnnotationSet(
                annotations=[
                    {
                        "span": "Museums ought to remain free",
                        "category": "LINKING",
                        "correction": "Museums should remain free",
                        "explanation": "Simpler modal reads better here.",
                        "severity": "slip",
                    }
                ],
                strengths=["A clear position."],
            )
        return ImprovedVersion(text="Improved.", upgrades=[])

    monkeypatch.setattr(task_module, "structured_call", fake_task)
    task_id, payload = services.next_writing_task(part=1)
    assert payload["genre"] == "essay"

    monkeypatch.setattr(writing_evaluator, "structured_call", fake_assess)
    result = services.submit_writing(
        user_id=user["id"],
        task_id=task_id,
        part=1,
        genre="essay",
        task=payload,
        script=script,
    )

    assert result["total"] == 16
    assert result["bands"]["language"] == 4

    snapshot = services.progress(user["id"])
    assert snapshot.writing_submissions == 1
    assert snapshot.writing_averages["content"] == 4.0
    assert any(row["category"] == "LINKING" for row in snapshot.top_errors)


def test_mock_exam_is_recorded(user: dict) -> None:
    exam_id = services.start_mock_exam(user["id"])
    services.complete_mock_exam(exam_id=exam_id, marks=52, max_marks=78, scale_score=182)

    with session_scope() as session:
        exams = repo.list_mock_exams(session, user["id"])

    assert len(exams) == 1
    assert exams[0].scale_score == 182
    assert exams[0].status == "finished"
