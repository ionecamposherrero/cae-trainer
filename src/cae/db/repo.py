"""Persistence and analytics queries."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from cae.db.models import (
    Attempt,
    CoachReport,
    ErrorEvent,
    Exercise,
    ItemResult,
    MockExam,
    User,
    WritingScore,
    WritingSubmission,
    WritingTask,
)
from cae.db.session import session_scope
from cae.domain.parts import PART_SPECS, SUBSCALES, get_part_spec
from cae.domain.taxonomy import categories_for_part, label as category_label


def get_user_by_username(session: Session, username: str) -> User | None:
    """Look a user up by their exact username. Case is significant.

    Sign-in must be case-sensitive, so this is a plain equality check. Use
    ``username_is_taken`` for the registration check, which deliberately is not.
    """
    return session.scalar(select(User).where(User.username == username))


def username_is_taken(session: Session, username: str) -> bool:
    """Whether the name is claimed, ignoring case.

    Registration is stricter than sign-in on purpose: ``learner`` and ``LEARNER`` would
    be two accounts nobody could tell apart, so only the first may have the name.
    """
    return (
        session.scalar(
            select(User.id).where(func.lower(User.username) == username.lower()).limit(1)
        )
        is not None
    )


def create_user(session: Session, username: str, password_hash: str) -> User:
    user = User(username=username, password_hash=password_hash)
    session.add(user)
    session.flush()
    return user


def touch_login(session: Session, user_id: int) -> None:
    user = session.get(User, user_id)
    if user is not None:
        user.last_login_at = datetime.now(UTC)


def save_exercise(
    session: Session,
    *,
    part: int,
    payload: dict[str, Any],
    topic: str = "",
    focus: str = "",
    attempts: int = 1,
) -> Exercise:
    exercise = Exercise(
        part=part,
        title=str(payload.get("title", ""))[:255],
        topic=(topic or str(payload.get("topic", "")))[:120],
        focus=focus[:255],
        payload=payload,
        generation_attempts=attempts,
    )
    session.add(exercise)
    session.flush()
    return exercise


def get_exercise(session: Session, exercise_id: int) -> Exercise | None:
    return session.get(Exercise, exercise_id)


def record_attempt(
    session: Session,
    *,
    user_id: int,
    exercise_id: int,
    part: int,
    responses: dict[str, Any],
    item_rows: list[dict[str, Any]],
    marks: float,
    max_marks: float,
    duration_seconds: int = 0,
    mode: str = "practice",
    mock_exam_id: int | None = None,
) -> Attempt:
    """Persist a graded attempt, its per-item results and its error events."""
    attempt = Attempt(
        user_id=user_id,
        exercise_id=exercise_id,
        part=part,
        mode=mode,
        mock_exam_id=mock_exam_id,
        responses=responses,
        marks=marks,
        max_marks=max_marks,
        duration_seconds=duration_seconds,
    )
    session.add(attempt)
    session.flush()

    for row in item_rows:
        session.add(
            ItemResult(
                attempt_id=attempt.id,
                item_number=int(row["item_number"]),
                user_answer=str(row.get("user_answer", "")),
                correct_answer=str(row.get("correct_answer", "")),
                is_correct=bool(row.get("is_correct", False)),
                marks_awarded=float(row.get("marks_awarded", 0.0)),
                max_marks=float(row.get("max_marks", 1.0)),
                category=row.get("category"),
                explanation=str(row.get("explanation", "")),
                adjudicated=bool(row.get("adjudicated", False)),
            )
        )
        if not row.get("is_correct") and row.get("category"):
            session.add(
                ErrorEvent(
                    user_id=user_id,
                    source_type="reading",
                    source_id=attempt.id,
                    part=part,
                    category=str(row["category"]),
                    evidence=str(row.get("evidence", ""))[:500],
                )
            )
    session.flush()
    return attempt


def get_attempt(session: Session, attempt_id: int) -> Attempt | None:
    return session.get(Attempt, attempt_id)


def save_writing_task(
    session: Session, *, part: int, genre: str, payload: dict[str, Any]
) -> WritingTask:
    task = WritingTask(
        part=part,
        genre=genre,
        title=str(payload.get("title", ""))[:255],
        payload=payload,
    )
    session.add(task)
    session.flush()
    return task


def get_writing_task(session: Session, task_id: int) -> WritingTask | None:
    return session.get(WritingTask, task_id)


def save_submission(
    session: Session,
    *,
    user_id: int,
    task_id: int,
    text: str,
    word_count: int,
    duration_seconds: int = 0,
    mock_exam_id: int | None = None,
) -> WritingSubmission:
    submission = WritingSubmission(
        user_id=user_id,
        task_id=task_id,
        text=text,
        word_count=word_count,
        duration_seconds=duration_seconds,
        mock_exam_id=mock_exam_id,
    )
    session.add(submission)
    session.flush()
    return submission


def save_writing_score(
    session: Session,
    *,
    submission_id: int,
    user_id: int,
    bands: dict[str, int],
    total: int,
    scale_score: int,
    feedback: dict[str, Any],
    annotations: list[dict[str, Any]],
) -> WritingScore:
    score = WritingScore(
        submission_id=submission_id,
        content=int(bands.get("content", 0)),
        communicative_achievement=int(bands.get("communicative_achievement", 0)),
        organisation=int(bands.get("organisation", 0)),
        language=int(bands.get("language", 0)),
        total=total,
        scale_score=scale_score,
        feedback=feedback,
    )
    session.add(score)
    for annotation in annotations:
        if not annotation.get("category"):
            continue
        session.add(
            ErrorEvent(
                user_id=user_id,
                source_type="writing",
                source_id=submission_id,
                part=None,
                category=str(annotation["category"]),
                evidence=str(annotation.get("span", ""))[:500],
            )
        )
    session.flush()
    return score


def get_submission(session: Session, submission_id: int) -> WritingSubmission | None:
    return session.get(WritingSubmission, submission_id)


def create_mock_exam(session: Session, user_id: int) -> MockExam:
    exam = MockExam(user_id=user_id)
    session.add(exam)
    session.flush()
    return exam


def finish_mock_exam(
    session: Session, *, exam_id: int, marks: float, max_marks: float, scale_score: int
) -> MockExam | None:
    exam = session.get(MockExam, exam_id)
    if exam is None:
        return None
    exam.finished_at = datetime.now(UTC)
    exam.status = "finished"
    exam.marks = marks
    exam.max_marks = max_marks
    exam.scale_score = scale_score
    return exam


@dataclass(frozen=True)
class PaperCoverage:
    """Which papers a mock exam actually produced marked work for.

    A learner can sit either paper alone, so a mock's score is only comparable
    with others that covered the same ground. ``series`` is that grouping.
    """

    reading: bool
    writing: bool

    @property
    def label(self) -> str:
        """For the results table: what this sitting covered."""
        covered = []
        if self.reading:
            covered.append("Reading & UoE")
        if self.writing:
            covered.append("Writing")
        return " + ".join(covered) or "—"

    @property
    def series(self) -> str:
        """For the chart: which of the three lines this mock belongs on."""
        if self.reading and self.writing:
            return "Complete exam"
        if self.reading:
            return "Reading & UoE only"
        if self.writing:
            return "Writing only"
        return ""

    def __bool__(self) -> bool:
        """False for a sitting that was abandoned before anything was marked."""
        return self.reading or self.writing


def mock_exam_coverage(session: Session, exam_ids: list[int]) -> dict[int, PaperCoverage]:
    """Which papers each mock exam covered, derived from the rows it produced.

    There is no column for it: an exam's coverage is whatever work it left
    behind, which is also what makes it right for an exam abandoned part-way.
    """
    if not exam_ids:
        return {}
    with_reading = set(
        session.scalars(
            select(Attempt.mock_exam_id)
            .where(Attempt.mock_exam_id.in_(exam_ids))
            .distinct()
        )
    )
    with_writing = set(
        session.scalars(
            select(WritingSubmission.mock_exam_id)
            .where(WritingSubmission.mock_exam_id.in_(exam_ids))
            .distinct()
        )
    )
    return {
        exam_id: PaperCoverage(
            reading=exam_id in with_reading, writing=exam_id in with_writing
        )
        for exam_id in exam_ids
    }


def mock_exam_papers(session: Session, exam_ids: list[int]) -> dict[int, str]:
    """The coverage of each mock exam as a label, for the results table."""
    return {
        exam_id: coverage.label
        for exam_id, coverage in mock_exam_coverage(session, exam_ids).items()
    }


def list_mock_exams(session: Session, user_id: int) -> list[MockExam]:
    return list(
        session.scalars(
            select(MockExam)
            .where(MockExam.user_id == user_id, MockExam.status == "finished")
            .order_by(MockExam.finished_at.desc())
        )
    )


@dataclass
class PartStat:
    part: int
    name: str
    attempts: int
    marks: float
    max_marks: float
    accuracy: float
    recent_accuracy: float
    trend: float

    @property
    def delta(self) -> float:
        return round(self.recent_accuracy - self.accuracy, 1)


@dataclass
class ProgressSnapshot:
    """Everything the dashboard and the coach graph need, in one query pass."""

    activity_count: int
    reading_attempts: int
    writing_submissions: int
    overall_accuracy: float
    part_stats: list[PartStat] = field(default_factory=list)
    strengths: list[PartStat] = field(default_factory=list)
    weaknesses: list[PartStat] = field(default_factory=list)
    top_errors: list[dict[str, Any]] = field(default_factory=list)
    attempt_history: list[dict[str, Any]] = field(default_factory=list)
    writing_history: list[dict[str, Any]] = field(default_factory=list)
    writing_averages: dict[str, float] = field(default_factory=dict)


def _linear_trend(values: list[float]) -> float:
    """Least-squares slope in percentage points per attempt."""
    n = len(values)
    if n < 3:
        return 0.0
    mean_x = (n - 1) / 2
    mean_y = sum(values) / n
    denominator = sum((i - mean_x) ** 2 for i in range(n))
    if denominator == 0:
        return 0.0
    numerator = sum((i - mean_x) * (v - mean_y) for i, v in enumerate(values))
    return round(numerator / denominator, 2)


def attempt_history(session: Session, user_id: int, limit: int = 200) -> list[dict[str, Any]]:
    rows = session.execute(
        select(
            Attempt.id,
            Attempt.part,
            Attempt.marks,
            Attempt.max_marks,
            Attempt.mode,
            Attempt.submitted_at,
        )
        .where(Attempt.user_id == user_id)
        .order_by(Attempt.submitted_at)
        .limit(limit)
    ).all()
    return [
        {
            "attempt_id": r.id,
            "part": r.part,
            "part_name": get_part_spec(r.part).name,
            "marks": r.marks,
            "max_marks": r.max_marks,
            "accuracy": round(100.0 * r.marks / r.max_marks, 1) if r.max_marks else 0.0,
            "mode": r.mode,
            "submitted_at": r.submitted_at,
        }
        for r in rows
    ]


def part_stats(session: Session, user_id: int, recent_window: int = 5) -> list[PartStat]:
    history = attempt_history(session, user_id)
    by_part: dict[int, list[dict[str, Any]]] = {}
    for row in history:
        by_part.setdefault(row["part"], []).append(row)

    stats: list[PartStat] = []
    for part, spec in PART_SPECS.items():
        rows = by_part.get(part, [])
        if not rows:
            stats.append(
                PartStat(part, spec.name, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
            )
            continue
        marks = sum(r["marks"] for r in rows)
        max_marks = sum(r["max_marks"] for r in rows)
        accuracy = round(100.0 * marks / max_marks, 1) if max_marks else 0.0
        recent = rows[-recent_window:]
        recent_marks = sum(r["marks"] for r in recent)
        recent_max = sum(r["max_marks"] for r in recent)
        stats.append(
            PartStat(
                part=part,
                name=spec.name,
                attempts=len(rows),
                marks=marks,
                max_marks=max_marks,
                accuracy=accuracy,
                recent_accuracy=round(100.0 * recent_marks / recent_max, 1) if recent_max else 0.0,
                trend=_linear_trend([r["accuracy"] for r in rows]),
            )
        )
    return stats


def top_error_categories(
    session: Session, user_id: int, limit: int = 8
) -> list[dict[str, Any]]:
    rows = session.execute(
        select(ErrorEvent.category, func.count(ErrorEvent.id).label("n"))
        .where(ErrorEvent.user_id == user_id)
        .group_by(ErrorEvent.category)
        .order_by(func.count(ErrorEvent.id).desc())
        .limit(limit)
    ).all()
    total = session.scalar(
        select(func.count(ErrorEvent.id)).where(ErrorEvent.user_id == user_id)
    ) or 0
    return [
        {
            "category": r.category,
            "label": category_label(r.category),
            "count": int(r.n),
            "share": round(100.0 * r.n / total, 1) if total else 0.0,
        }
        for r in rows
    ]


def error_examples(
    session: Session, user_id: int, category: str, limit: int = 3
) -> list[str]:
    rows = session.scalars(
        select(ErrorEvent.evidence)
        .where(ErrorEvent.user_id == user_id, ErrorEvent.category == category)
        .order_by(ErrorEvent.created_at.desc())
        .limit(limit)
    ).all()
    return [r for r in rows if r]


def writing_history(session: Session, user_id: int) -> list[dict[str, Any]]:
    rows = session.execute(
        select(
            WritingSubmission.id,
            WritingSubmission.submitted_at,
            WritingSubmission.word_count,
            WritingTask.genre,
            WritingTask.part,
            WritingScore.content,
            WritingScore.communicative_achievement,
            WritingScore.organisation,
            WritingScore.language,
            WritingScore.total,
            WritingScore.scale_score,
        )
        .join(WritingTask, WritingTask.id == WritingSubmission.task_id)
        .join(WritingScore, WritingScore.submission_id == WritingSubmission.id)
        .where(WritingSubmission.user_id == user_id)
        .order_by(WritingSubmission.submitted_at)
    ).all()
    return [
        {
            "submission_id": r.id,
            "submitted_at": r.submitted_at,
            "word_count": r.word_count,
            "genre": r.genre,
            "part": r.part,
            "content": r.content,
            "communicative_achievement": r.communicative_achievement,
            "organisation": r.organisation,
            "language": r.language,
            "total": r.total,
            "scale_score": r.scale_score,
        }
        for r in rows
    ]


def progress_snapshot(session: Session, user_id: int) -> ProgressSnapshot:
    """One consolidated read of the learner's record."""
    history = attempt_history(session, user_id)
    stats = part_stats(session, user_id)
    writing = writing_history(session, user_id)

    attempted = [s for s in stats if s.attempts > 0]
    ranked = sorted(attempted, key=lambda s: s.accuracy, reverse=True)

    total_marks = sum(r["marks"] for r in history)
    total_max = sum(r["max_marks"] for r in history)

    averages: dict[str, float] = {}
    if writing:
        for subscale in SUBSCALES:
            averages[subscale] = round(
                sum(w[subscale] for w in writing) / len(writing), 2
            )

    return ProgressSnapshot(
        activity_count=len(history) + len(writing),
        reading_attempts=len(history),
        writing_submissions=len(writing),
        overall_accuracy=round(100.0 * total_marks / total_max, 1) if total_max else 0.0,
        part_stats=stats,
        strengths=ranked[:3],
        weaknesses=list(reversed(ranked[-3:])) if len(ranked) > 3 else list(reversed(ranked)),
        top_errors=top_error_categories(session, user_id),
        attempt_history=history,
        writing_history=writing,
        writing_averages=averages,
    )


def weakness_categories(
    session: Session, user_id: int, part: int, limit: int = 4
) -> list[str]:
    """The user's most frequent error categories that ``part`` can test, for
    biasing its generation.

    Another part's categories and unknown codes are left out, and so is a Reading
    part: its one category is the task itself, so there is nothing to aim at.
    """
    allowed = [c.value for c in categories_for_part(part)]
    if len(allowed) < 2:
        return []
    return list(
        session.scalars(
            select(ErrorEvent.category)
            .where(ErrorEvent.user_id == user_id, ErrorEvent.category.in_(allowed))
            .group_by(ErrorEvent.category)
            .order_by(func.count(ErrorEvent.id).desc())
            .limit(limit)
        ).all()
    )


TASK_FIELDS = ("scenario", "instruction", "notes", "opinions", "reader")


def learner_export(session: Session, user_id: int) -> dict[str, list[dict[str, Any]]]:
    """One learner's record, table by table, in the shape of ``sandbox.SCHEMA``.

    ``item_results`` rows come without ``question``, ``key_word`` and ``options``:
    those are derived from the exercise, which is the chat graph's job.
    """
    attempts = session.execute(
        select(Attempt, Exercise.topic, Exercise.title)
        .join(Exercise, Exercise.id == Attempt.exercise_id)
        .where(Attempt.user_id == user_id)
        .order_by(Attempt.submitted_at)
    ).all()

    items = session.scalars(
        select(ItemResult)
        .join(Attempt, Attempt.id == ItemResult.attempt_id)
        .where(Attempt.user_id == user_id)
        .order_by(ItemResult.attempt_id, ItemResult.item_number)
    ).all()

    errors = session.scalars(
        select(ErrorEvent)
        .where(ErrorEvent.user_id == user_id)
        .order_by(ErrorEvent.created_at)
    ).all()

    submissions = session.execute(
        select(WritingSubmission, WritingTask)
        .join(WritingTask, WritingTask.id == WritingSubmission.task_id)
        .where(WritingSubmission.user_id == user_id)
        .order_by(WritingSubmission.submitted_at)
    ).all()

    scores = session.scalars(
        select(WritingScore)
        .join(WritingSubmission, WritingSubmission.id == WritingScore.submission_id)
        .where(WritingSubmission.user_id == user_id)
    ).all()

    exams = session.scalars(
        select(MockExam).where(MockExam.user_id == user_id).order_by(MockExam.started_at)
    ).all()

    return {
        "attempts": [
            {
                "id": a.id,
                "part": a.part,
                "part_name": get_part_spec(a.part).name,
                "mode": a.mode,
                "marks": a.marks,
                "max_marks": a.max_marks,
                "accuracy": round(100.0 * a.marks / a.max_marks, 1) if a.max_marks else 0.0,
                "duration_seconds": a.duration_seconds,
                "submitted_at": a.submitted_at,
                "mock_exam_id": a.mock_exam_id,
                "exercise_id": a.exercise_id,
                "topic": topic,
                "title": title,
            }
            for a, topic, title in attempts
        ],
        "item_results": [
            {
                "attempt_id": i.attempt_id,
                "item_number": i.item_number,
                "user_answer": i.user_answer,
                "correct_answer": i.correct_answer,
                "is_correct": i.is_correct,
                "marks_awarded": i.marks_awarded,
                "max_marks": i.max_marks,
                "category": i.category,
                "explanation": i.explanation,
                "adjudicated": i.adjudicated,
            }
            for i in items
        ],
        "error_events": [
            {
                "id": e.id,
                "source_type": e.source_type,
                "source_id": e.source_id,
                "part": e.part,
                "category": e.category,
                "evidence": e.evidence,
                "created_at": e.created_at,
            }
            for e in errors
        ],
        "writing_submissions": [
            {
                "id": s.id,
                "part": t.part,
                "genre": t.genre,
                "title": t.title,
                **{field: (t.payload or {}).get(field) or "" for field in TASK_FIELDS},
                "word_count": s.word_count,
                "duration_seconds": s.duration_seconds,
                "submitted_at": s.submitted_at,
                "mock_exam_id": s.mock_exam_id,
                "text": s.text,
            }
            for s, t in submissions
        ],
        "writing_scores": [
            {
                "submission_id": sc.submission_id,
                **sc.bands,
                "total": sc.total,
                "scale_score": sc.scale_score,
                "feedback": sc.feedback,
            }
            for sc in scores
        ],
        "mock_exams": [
            {
                "id": m.id,
                "started_at": m.started_at,
                "finished_at": m.finished_at,
                "status": m.status,
                "marks": m.marks,
                "max_marks": m.max_marks,
                "scale_score": m.scale_score,
            }
            for m in exams
        ],
    }


def attempted_exercises(session: Session, user_id: int) -> dict[int, dict[str, Any]]:
    """The payload of every exercise this learner has been marked on, by id."""
    rows = session.execute(
        select(Exercise.id, Exercise.payload)
        .join(Attempt, Attempt.exercise_id == Exercise.id)
        .where(Attempt.user_id == user_id)
    ).all()
    return {row.id: row.payload or {} for row in rows}


def get_cached_report(
    session: Session, user_id: int, activity_count: int
) -> CoachReport | None:
    report = session.scalar(
        select(CoachReport)
        .where(CoachReport.user_id == user_id)
        .order_by(CoachReport.generated_at.desc())
        .limit(1)
    )
    if report is not None and report.activity_count == activity_count:
        return report
    return None


def save_report(
    session: Session, *, user_id: int, activity_count: int, report: dict[str, Any]
) -> CoachReport:
    row = CoachReport(user_id=user_id, activity_count=activity_count, report=report)
    session.add(row)
    session.flush()
    return row


def snapshot_for(user_id: int) -> ProgressSnapshot:
    with session_scope() as session:
        return progress_snapshot(session, user_id)
