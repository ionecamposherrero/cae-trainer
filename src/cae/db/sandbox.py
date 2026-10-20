"""A throwaway, read-only SQLite database holding one learner's record.

The coach chat lets the model write its own SELECT. That query never runs against
the application database: :func:`load` builds an in-memory copy holding only the
rows it is handed, and locks it so nothing but a read gets through. Which rows go
in is the caller's business (``repo.learner_export`` filters on ``user_id``), and
that is what makes another learner's work unreachable rather than merely filtered
out -- there is no ``WHERE`` clause for a query to get round.

:data:`SCHEMA` is the single source for both the tables that are created and the
description the model reads, so the two cannot drift apart.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    note: str = ""


@dataclass(frozen=True)
class Table:
    name: str
    note: str
    columns: tuple[Column, ...]

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)


SCHEMA: tuple[Table, ...] = (
    Table(
        "attempts",
        "One marked Reading & Use of English exercise.",
        (
            Column("id", "INTEGER"),
            Column("part", "INTEGER", "Reading & Use of English part, 1-8"),
            Column("part_name", "TEXT", "e.g. 'Multiple-choice cloze'"),
            Column("mode", "TEXT", "'practice' or 'mock'"),
            Column("marks", "REAL"),
            Column("max_marks", "REAL"),
            Column("accuracy", "REAL", "100 * marks / max_marks"),
            Column("duration_seconds", "INTEGER", "time taken; 0 = not recorded"),
            Column("submitted_at", "TEXT"),
            Column("mock_exam_id", "INTEGER", "mock_exams.id; NULL for practice"),
            Column("exercise_id", "INTEGER"),
            Column("topic", "TEXT", "subject of the text"),
            Column("title", "TEXT"),
        ),
    ),
    Table(
        "item_results",
        "One item (question) of an attempt.",
        (
            Column("attempt_id", "INTEGER", "attempts.id"),
            Column("item_number", "INTEGER", "as numbered on the paper"),
            Column("question", "TEXT", "what the item asked, or the text around its gap"),
            Column("key_word", "TEXT", "Part 4 key word; '' otherwise"),
            Column("options", "TEXT", "JSON {letter: option} for multiple choice; ''"),
            Column("user_answer", "TEXT"),
            Column("correct_answer", "TEXT"),
            Column("is_correct", "INTEGER", "1 or 0"),
            Column("marks_awarded", "REAL"),
            Column("max_marks", "REAL"),
            Column("category", "TEXT", "error_categories.code when marks were lost"),
            Column("explanation", "TEXT", "why the key is right"),
            Column("adjudicated", "INTEGER", "1 = an answer off the key was accepted"),
        ),
    ),
    Table(
        "error_events",
        "Every tagged mistake, Reading and Writing alike.",
        (
            Column("id", "INTEGER"),
            Column("source_type", "TEXT", "'reading' or 'writing'"),
            Column("source_id", "INTEGER", "attempts.id or writing_submissions.id"),
            Column("part", "INTEGER", "Reading part; NULL for writing"),
            Column("category", "TEXT", "error_categories.code"),
            Column("evidence", "TEXT", "what the student wrote"),
            Column("created_at", "TEXT"),
        ),
    ),
    Table(
        "writing_submissions",
        "One piece of writing the student handed in.",
        (
            Column("id", "INTEGER"),
            Column("part", "INTEGER", "Writing part, 1 or 2"),
            Column("genre", "TEXT", "essay, report, proposal, review, letter, email"),
            Column("title", "TEXT"),
            Column("scenario", "TEXT", "the task as set"),
            Column("instruction", "TEXT"),
            Column("notes", "TEXT", "JSON list of the task's content points"),
            Column("opinions", "TEXT", "JSON list"),
            Column("reader", "TEXT", "target reader"),
            Column("word_count", "INTEGER"),
            Column("duration_seconds", "INTEGER", "0 = not recorded"),
            Column("submitted_at", "TEXT"),
            Column("mock_exam_id", "INTEGER", "NULL for practice"),
            Column("text", "TEXT", "the student's script"),
        ),
    ),
    Table(
        "writing_scores",
        "The assessment of a submission; unmarked ones have no row.",
        (
            Column("submission_id", "INTEGER", "writing_submissions.id"),
            Column("content", "INTEGER", "band 0-5"),
            Column("communicative_achievement", "INTEGER", "band 0-5"),
            Column("organisation", "INTEGER", "band 0-5"),
            Column("language", "INTEGER", "band 0-5"),
            Column("total", "INTEGER", "0-20"),
            Column("scale_score", "INTEGER", "Cambridge scale, 180 = C1 pass"),
            Column(
                "feedback",
                "TEXT",
                "JSON: content, communicative_achievement, organisation, language "
                "(each {band, justification, evidence, to_improve}); annotations "
                "[{span, category, correction, explanation, severity}]; strengths; "
                "improved {text, upgrades}",
            ),
        ),
    ),
    Table(
        "mock_exams",
        "A full mock exam; its parts are attempts/submissions with this mock_exam_id.",
        (
            Column("id", "INTEGER"),
            Column("started_at", "TEXT"),
            Column("finished_at", "TEXT", "NULL if abandoned"),
            Column("status", "TEXT", "'in_progress' or 'finished'"),
            Column("marks", "REAL"),
            Column("max_marks", "REAL"),
            Column("scale_score", "INTEGER"),
        ),
    ),
    Table(
        "error_categories",
        "The closed error taxonomy.",
        (
            Column("code", "TEXT", "e.g. FIXED_PHRASES"),
            Column("label", "TEXT"),
            Column("family", "TEXT", "GRAMMAR, VOCABULARY, TEXT or READING"),
            Column("advice", "TEXT", "standard advice for the category"),
        ),
    ),
)

TABLES: dict[str, Table] = {table.name: table for table in SCHEMA}

MAX_ROWS = 50
MAX_CELL_CHARS = 2000
MAX_RESULT_CHARS = 12_000
QUERY_SECONDS = 2.0
PROGRESS_INTERVAL = 10_000
MAX_VALUE_BYTES = 1_000_000

_ALLOWED_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        getattr(sqlite3, "SQLITE_RECURSIVE", 33),
    }
)


def _ddl(table: Table) -> str:
    columns = ", ".join(f"{c.name} {c.type}" for c in table.columns)
    return f"CREATE TABLE {table.name} ({columns})"


def describe() -> str:
    """The schema as the model reads it, one table per paragraph."""
    blocks = []
    for table in SCHEMA:
        lines = [f"{table.name} -- {table.note}"]
        for column in table.columns:
            note = f" -- {column.note}" if column.note else ""
            lines.append(f"  {column.name} {column.type}{note}")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def _cell(value: Any) -> Any:
    """A Python value as SQLite should store it."""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False) if value else ""
    return value


def _authorize(action: int, *_args: Any) -> int:
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def load(tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> sqlite3.Connection:
    """An in-memory database holding exactly these rows, open for reading only.

    Every table in :data:`SCHEMA` is created, empty if no rows are given. A table
    or column not in the schema raises: the export and the description the model
    reads have to describe the same database.
    """
    unknown = set(tables) - set(TABLES)
    if unknown:
        raise ValueError(f"Not in the sandbox schema: {sorted(unknown)}")

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    for table in SCHEMA:
        conn.execute(_ddl(table))
        rows = tables.get(table.name) or []
        names = table.column_names
        for row in rows:
            extra = set(row) - set(names)
            if extra:
                raise ValueError(f"{table.name} has no column(s) {sorted(extra)}")
        placeholders = ", ".join("?" for _ in names)
        conn.executemany(
            f"INSERT INTO {table.name} ({', '.join(names)}) VALUES ({placeholders})",
            [tuple(_cell(row.get(name)) for name in names) for row in rows],
        )
    conn.commit()

    conn.execute("PRAGMA query_only = ON")
    conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_VALUE_BYTES)
    conn.set_authorizer(_authorize)
    return conn


def _clip(value: Any) -> Any:
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if isinstance(value, str) and len(value) > MAX_CELL_CHARS:
        return value[:MAX_CELL_CHARS] + " [...cut]"
    return value


def run_select(conn: sqlite3.Connection, sql: str) -> dict[str, Any] | str:
    """Run one read-only query. Failures come back as text the model can act on."""
    deadline = time.monotonic() + QUERY_SECONDS
    conn.set_progress_handler(lambda: time.monotonic() > deadline, PROGRESS_INTERVAL)
    try:
        cursor = conn.execute(sql)
        if cursor.description is None:
            return "That statement returned no result set. Write a single SELECT."
        columns = [d[0] for d in cursor.description]
        fetched = cursor.fetchmany(MAX_ROWS + 1)
    except sqlite3.DatabaseError as exc:
        return _explain(exc)
    except (sqlite3.Warning, ValueError) as exc:
        return f"SQL error: {exc}. Send exactly one SELECT statement."
    finally:
        conn.set_progress_handler(None, 0)

    rows: list[list[Any]] = []
    size = 0
    truncated = len(fetched) > MAX_ROWS
    for raw in fetched[:MAX_ROWS]:
        row = [_clip(value) for value in raw]
        size += len(json.dumps(row, ensure_ascii=False, default=str))
        if rows and size > MAX_RESULT_CHARS:
            truncated = True
            break
        rows.append(row)

    result: dict[str, Any] = {"columns": columns, "rows": rows}
    if not rows:
        result["note"] = (
            "No rows: their record has nothing matching this. Say so; do not guess."
        )
    elif truncated:
        result["note"] = (
            f"Only the first {len(rows)} rows are shown. Aggregate in SQL "
            "(COUNT, AVG, GROUP BY) or add a LIMIT rather than reading them all."
        )
    return result


def _explain(exc: sqlite3.DatabaseError) -> str:
    message = str(exc).rstrip(".")
    if "one statement at a time" in message:
        return "SQL error: send exactly one statement, a single SELECT."
    if "not authorized" in message or "readonly" in message or "query_only" in message:
        return (
            "Not allowed: the record is read-only and only a SELECT over the "
            "tables in the tool description can run."
        )
    if "interrupted" in message:
        return (
            "The query took too long and was stopped. Simplify it, and make sure "
            "any recursive CTE has a condition that ends it."
        )
    return (
        f"SQL error: {message}. Check the table and column names against the "
        "schema in the tool description and try again."
    )


__all__ = [
    "MAX_ROWS",
    "SCHEMA",
    "TABLES",
    "Column",
    "Table",
    "describe",
    "load",
    "run_select",
]
