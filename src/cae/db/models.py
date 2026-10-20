"""SQLAlchemy models.

``error_events`` is the single table Reading and Writing mistakes are aggregated
from, all tagged with the closed ``ErrorCategory`` vocabulary.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    attempts: Mapped[list[Attempt]] = relationship(back_populates="user")
    submissions: Mapped[list[WritingSubmission]] = relationship(back_populates="user")


class Exercise(Base):
    """A generated, validated exercise. Kept for the attempts that answer it;
    never served again."""

    __tablename__ = "exercises"

    id: Mapped[int] = mapped_column(primary_key=True)
    part: Mapped[int] = mapped_column(Integer, index=True)
    title: Mapped[str] = mapped_column(String(255), default="")
    topic: Mapped[str] = mapped_column(String(120), default="", index=True)
    focus: Mapped[str] = mapped_column(String(255), default="")
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    generation_attempts: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    attempts: Mapped[list[Attempt]] = relationship(back_populates="exercise")


class MockExam(Base):
    __tablename__ = "mock_exams"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(16), default="in_progress")
    marks: Mapped[float] = mapped_column(Float, default=0.0)
    max_marks: Mapped[float] = mapped_column(Float, default=0.0)
    scale_score: Mapped[int | None] = mapped_column(Integer)


class Attempt(Base):
    """One graded go at one exercise."""

    __tablename__ = "attempts"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    exercise_id: Mapped[int] = mapped_column(ForeignKey("exercises.id"))
    mock_exam_id: Mapped[int | None] = mapped_column(ForeignKey("mock_exams.id"), index=True)
    part: Mapped[int] = mapped_column(Integer, index=True)
    mode: Mapped[str] = mapped_column(String(16), default="practice")
    responses: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    marks: Mapped[float] = mapped_column(Float, default=0.0)
    max_marks: Mapped[float] = mapped_column(Float, default=0.0)
    duration_seconds: Mapped[int] = mapped_column(Integer, default=0)
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    user: Mapped[User] = relationship(back_populates="attempts")
    exercise: Mapped[Exercise] = relationship(back_populates="attempts")
    items: Mapped[list[ItemResult]] = relationship(
        back_populates="attempt", cascade="all, delete-orphan"
    )

    @property
    def percentage(self) -> float:
        return 100.0 * self.marks / self.max_marks if self.max_marks else 0.0


Index("ix_attempts_user_part_time", Attempt.user_id, Attempt.part, Attempt.submitted_at)


class ItemResult(Base):
    __tablename__ = "item_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    attempt_id: Mapped[int] = mapped_column(ForeignKey("attempts.id"), index=True)
    item_number: Mapped[int] = mapped_column(Integer)
    user_answer: Mapped[str] = mapped_column(Text, default="")
    correct_answer: Mapped[str] = mapped_column(Text, default="")
    is_correct: Mapped[bool] = mapped_column(Boolean, default=False)
    marks_awarded: Mapped[float] = mapped_column(Float, default=0.0)
    max_marks: Mapped[float] = mapped_column(Float, default=1.0)
    category: Mapped[str | None] = mapped_column(String(48))
    explanation: Mapped[str] = mapped_column(Text, default="")
    adjudicated: Mapped[bool] = mapped_column(Boolean, default=False)

    attempt: Mapped[Attempt] = relationship(back_populates="items")


class WritingTask(Base):
    __tablename__ = "writing_tasks"

    id: Mapped[int] = mapped_column(primary_key=True)
    part: Mapped[int] = mapped_column(Integer)
    genre: Mapped[str] = mapped_column(String(32), index=True)
    title: Mapped[str] = mapped_column(String(255), default="")
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class WritingSubmission(Base):
    __tablename__ = "writing_submissions"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("writing_tasks.id"))
    mock_exam_id: Mapped[int | None] = mapped_column(ForeignKey("mock_exams.id"), index=True)
    text: Mapped[str] = mapped_column(Text)
    word_count: Mapped[int] = mapped_column(Integer, default=0)
    duration_seconds: Mapped[int] = mapped_column(Integer, default=0)
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    user: Mapped[User] = relationship(back_populates="submissions")
    task: Mapped[WritingTask] = relationship()
    score: Mapped[WritingScore | None] = relationship(
        back_populates="submission", cascade="all, delete-orphan", uselist=False
    )


class WritingScore(Base):
    __tablename__ = "writing_scores"

    id: Mapped[int] = mapped_column(primary_key=True)
    submission_id: Mapped[int] = mapped_column(
        ForeignKey("writing_submissions.id"), unique=True, index=True
    )
    content: Mapped[int] = mapped_column(Integer, default=0)
    communicative_achievement: Mapped[int] = mapped_column(Integer, default=0)
    organisation: Mapped[int] = mapped_column(Integer, default=0)
    language: Mapped[int] = mapped_column(Integer, default=0)
    total: Mapped[int] = mapped_column(Integer, default=0)
    scale_score: Mapped[int] = mapped_column(Integer, default=0)
    feedback: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    submission: Mapped[WritingSubmission] = relationship(back_populates="score")

    @property
    def bands(self) -> dict[str, int]:
        return {
            "content": self.content,
            "communicative_achievement": self.communicative_achievement,
            "organisation": self.organisation,
            "language": self.language,
        }


class ErrorEvent(Base):
    """One tagged mistake. The single source the analytics layer aggregates."""

    __tablename__ = "error_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    source_type: Mapped[str] = mapped_column(String(16))
    source_id: Mapped[int] = mapped_column(Integer)
    part: Mapped[int | None] = mapped_column(Integer, index=True)
    category: Mapped[str] = mapped_column(String(48), index=True)
    evidence: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


Index("ix_error_events_user_cat_time", ErrorEvent.user_id, ErrorEvent.category, ErrorEvent.created_at)


class CoachReport(Base):
    """Cached narrative feedback, invalidated when the user does more work."""

    __tablename__ = "coach_reports"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    activity_count: Mapped[int] = mapped_column(Integer, default=0)
    report: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
