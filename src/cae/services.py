"""Service layer: the only module the Streamlit pages call."""

from __future__ import annotations

from typing import Any, Callable

from cae.db import repo
from cae.db.session import session_scope
from cae.domain.parts import get_part_spec
from cae.graphs.coach import build_report
from cae.graphs.coach_chat import answer as coach_answer
from cae.graphs.rue_task import generate_exercise
from cae.graphs.rue_evaluator import grade_attempt
from cae.graphs.writing_evaluator import assess_writing, count_words
from cae.graphs.writing_task import generate_writing_task
from cae.schemas.writing_task import WritingTaskSpec


def next_exercise(user_id: int, part: int) -> tuple[int, dict[str, Any]]:
    """A freshly generated exercise for this user, aimed at their weaknesses.

    Stored exercises are never served again, to this learner or another: each
    one was written for the weaknesses of whoever asked for it.
    Returns ``(exercise_id, payload)``.
    """
    with session_scope() as session:
        weaknesses = repo.weakness_categories(session, user_id, part)
    result = generate_exercise(
        part, weaknesses=weaknesses, trace_metadata={"user_id": user_id}
    )
    return int(result["exercise_id"]), result["exercise"]


def load_exercise(exercise_id: int) -> tuple[int, dict[str, Any]]:
    with session_scope() as session:
        exercise = repo.get_exercise(session, exercise_id)
        if exercise is None:
            raise ValueError(f"No exercise with id {exercise_id}")
        return exercise.part, exercise.payload


def submit_attempt(
    *,
    user_id: int,
    exercise_id: int,
    part: int,
    exercise: dict[str, Any],
    responses: dict[int, str],
    duration_seconds: int = 0,
    mode: str = "practice",
    mock_exam_id: int | None = None,
) -> dict[str, Any]:
    """Grade an attempt, store it, and return the graded result.

    Graded first, stored second: an attempt the model could not mark raises
    (``LLMUnavailableError`` or ``MarkingFailed``) and nothing is saved.
    """
    result = grade_attempt(
        part,
        exercise,
        responses,
        trace_metadata={
            "user_id": user_id,
            "exercise_id": exercise_id,
            "mode": mode,
            "mock_exam_id": mock_exam_id,
        },
    )
    with session_scope() as session:
        attempt_id = _record(
            session,
            user_id=user_id,
            exercise_id=exercise_id,
            part=part,
            responses=responses,
            result=result,
            duration_seconds=duration_seconds,
            mode=mode,
            mock_exam_id=mock_exam_id,
        )
    return {**result, "attempt_id": attempt_id, "part": part}


def submit_mock_reading(
    *,
    user_id: int,
    mock_exam_id: int,
    exercises: dict[int, dict[str, Any]],
    answers: dict[int, dict[int, str]],
    on_part: Callable[[int], None] | None = None,
) -> dict[int, dict[str, Any]]:
    """Grade every part of a mock Reading paper, then store them all, or none.

    ``exercises`` maps each part to ``{"exercise_id", "payload"}``. Every part is
    marked before any is saved: if one cannot be marked, the call raises and the
    paper can be marked again later without a part being recorded twice.
    ``on_part`` is told which part is about to be marked, for a progress bar.
    """
    graded: dict[int, dict[str, Any]] = {}
    for part in sorted(exercises):
        if on_part:
            on_part(part)
        graded[part] = grade_attempt(
            part,
            exercises[part]["payload"],
            answers.get(part, {}),
            trace_metadata={
                "user_id": user_id,
                "exercise_id": exercises[part]["exercise_id"],
                "mode": "mock",
                "mock_exam_id": mock_exam_id,
            },
        )
    results: dict[int, dict[str, Any]] = {}
    with session_scope() as session:
        for part, result in graded.items():
            attempt_id = _record(
                session,
                user_id=user_id,
                exercise_id=exercises[part]["exercise_id"],
                part=part,
                responses=answers.get(part, {}),
                result=result,
                mode="mock",
                mock_exam_id=mock_exam_id,
            )
            results[part] = {**result, "attempt_id": attempt_id, "part": part}
    return results


def _record(
    session,
    *,
    user_id: int,
    exercise_id: int,
    part: int,
    responses: dict[int, str],
    result: dict[str, Any],
    duration_seconds: int = 0,
    mode: str,
    mock_exam_id: int | None,
) -> int:
    attempt = repo.record_attempt(
        session,
        user_id=user_id,
        exercise_id=exercise_id,
        part=part,
        responses={str(k): v for k, v in responses.items()},
        item_rows=result["rows"],
        marks=result["marks"],
        max_marks=result["max_marks"],
        duration_seconds=duration_seconds,
        mode=mode,
        mock_exam_id=mock_exam_id,
    )
    return attempt.id


def next_writing_task(*, part: int, genre: str = "") -> tuple[int, dict[str, Any]]:
    """Generate and store a writing task. Returns ``(task_id, payload)``."""
    result = generate_writing_task(part=part, genre=genre)
    return int(result["task_id"]), result["task"]


def load_writing_task(task_id: int) -> tuple[int, str, dict[str, Any]]:
    with session_scope() as session:
        task = repo.get_writing_task(session, task_id)
        if task is None:
            raise ValueError(f"No writing task with id {task_id}")
        return task.part, task.genre, task.payload


def submit_writing(
    *,
    user_id: int,
    task_id: int,
    part: int,
    genre: str,
    task: dict[str, Any],
    script: str,
    duration_seconds: int = 0,
    mock_exam_id: int | None = None,
) -> dict[str, Any]:
    """Assess a script against the four subscales, store it, return the result."""
    result = assess_writing(
        part=part,
        genre=genre,
        task=task,
        script=script,
        trace_metadata={
            "user_id": user_id,
            "task_id": task_id,
            "mock_exam_id": mock_exam_id,
        },
    )
    with session_scope() as session:
        submission = repo.save_submission(
            session,
            user_id=user_id,
            task_id=task_id,
            text=script,
            word_count=count_words(script),
            duration_seconds=duration_seconds,
            mock_exam_id=mock_exam_id,
        )
        feedback = {
            key: result.get(key)
            for key in (
                "content",
                "communicative_achievement",
                "organisation",
                "language",
                "annotations",
                "strengths",
                "improved",
                "status",
                "message",
            )
        }
        repo.save_writing_score(
            session,
            submission_id=submission.id,
            user_id=user_id,
            bands=result.get("bands", {}),
            total=result.get("total", 0),
            scale_score=result.get("scale_score", 0),
            feedback=feedback,
            annotations=result.get("annotations", []),
        )
        submission_id = submission.id
    return {**result, "submission_id": submission_id}


def start_mock_exam(user_id: int) -> int:
    with session_scope() as session:
        return repo.create_mock_exam(session, user_id).id


def complete_mock_exam(
    *, exam_id: int, marks: float, max_marks: float, scale_score: int
) -> None:
    with session_scope() as session:
        repo.finish_mock_exam(
            session,
            exam_id=exam_id,
            marks=marks,
            max_marks=max_marks,
            scale_score=scale_score,
        )


def progress(user_id: int):
    return repo.snapshot_for(user_id)


def coaching_report(user_id: int, username: str, *, refresh: bool = False) -> dict[str, Any]:
    return build_report(user_id, username, use_cache=not refresh)


def coach_reply(
    user_id: int,
    username: str,
    history: list[tuple[str, str]],
    question: str,
) -> dict[str, Any]:
    """One turn of the coach conversation.

    ``history`` is ``(role, text)`` pairs, oldest first. Returns only ``reply``:
    the retrieved passages and tool outputs are for the eval harness, which
    calls the graph directly, and are never shown to the learner.
    """
    result = coach_answer(user_id, username, history, question)
    return {"reply": result["reply"]}


def part_label(part: int) -> str:
    spec = get_part_spec(part)
    return f"Part {part} — {spec.name}"


__all__ = [
    "WritingTaskSpec",
    "coach_reply",
    "coaching_report",
    "complete_mock_exam",
    "load_exercise",
    "load_writing_task",
    "next_exercise",
    "next_writing_task",
    "part_label",
    "progress",
    "start_mock_exam",
    "submit_attempt",
    "submit_writing",
]
