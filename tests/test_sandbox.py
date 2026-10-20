"""The read-only database the coach chat's model queries, on its own.

Whose rows go in is tested in ``test_coach_chat.py``; this is the lock on the door.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import UTC, datetime, timedelta

import pytest

from cae.db import sandbox


def attempt(n: int, **fields) -> dict:
    return {"id": n, "part": 1, "marks": 1.0, "max_marks": 8.0, **fields}


def test_every_described_table_exists_even_when_empty() -> None:
    conn = sandbox.load({})

    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}

    assert names == set(sandbox.TABLES)


def test_a_column_outside_the_schema_is_refused() -> None:
    """The description is the model's only map; the data must not go off it."""
    with pytest.raises(ValueError, match="user_id"):
        sandbox.load({"attempts": [attempt(1, user_id=7)]})


def test_a_table_outside_the_schema_is_refused() -> None:
    with pytest.raises(ValueError, match="users"):
        sandbox.load({"users": [{"id": 1}]})


def test_values_are_stored_so_sql_functions_work_on_them() -> None:
    """Dates must work with date(), JSON with json_extract, booleans as 1/0."""
    yesterday = datetime.now(UTC) - timedelta(days=1)
    conn = sandbox.load(
        {
            "attempts": [attempt(1, submitted_at=yesterday)],
            "item_results": [
                {"attempt_id": 1, "item_number": 1, "is_correct": True,
                 "options": {"A": "light", "B": "glow"}},
            ],
        }
    )

    recent = sandbox.run_select(
        conn, "SELECT COUNT(*) FROM attempts WHERE submitted_at >= date('now', '-7 days')"
    )
    option = sandbox.run_select(
        conn, "SELECT json_extract(options, '$.B'), is_correct FROM item_results"
    )

    assert recent["rows"] == [[1]]
    assert option["rows"] == [["glow", 1]]


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO attempts (id) VALUES (2)",
        "UPDATE attempts SET marks = 8",
        "DELETE FROM attempts",
        "DROP TABLE attempts",
        "CREATE TABLE loot (x)",
        "CREATE TEMP VIEW v AS SELECT 1",
        "PRAGMA query_only = OFF",
        "BEGIN",
    ],
)
def test_anything_but_a_read_is_refused(statement: str) -> None:
    conn = sandbox.load({"attempts": [attempt(1)]})

    refused = sandbox.run_select(conn, statement)

    assert isinstance(refused, str) and "Not allowed" in refused
    assert conn.execute("SELECT marks FROM attempts").fetchall() == [(1.0,)]


def test_another_database_cannot_be_attached(tmp_path) -> None:
    """ATTACH would reach the real file, and every learner in it."""
    other = tmp_path / "other.db"
    sqlite3.connect(other).execute("CREATE TABLE users (password_hash)").connection.commit()
    conn = sandbox.load({})

    refused = sandbox.run_select(conn, f"ATTACH DATABASE '{other.as_posix()}' AS real")

    assert isinstance(refused, str) and "Not allowed" in refused


def test_only_one_statement_runs() -> None:
    conn = sandbox.load({"attempts": [attempt(1)]})

    refused = sandbox.run_select(conn, "SELECT 1; DELETE FROM attempts")

    assert isinstance(refused, str) and "one statement" in refused
    assert conn.execute("SELECT COUNT(*) FROM attempts").fetchone() == (1,)


def test_a_runaway_query_is_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "QUERY_SECONDS", 0.2)
    conn = sandbox.load({})

    started = time.monotonic()
    result = sandbox.run_select(
        conn,
        "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n) "
        "SELECT COUNT(*) FROM n",
    )

    assert isinstance(result, str) and "too long" in result
    assert time.monotonic() - started < 5


def test_the_connection_is_usable_after_a_stopped_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sandbox, "QUERY_SECONDS", 0.1)
    conn = sandbox.load({"attempts": [attempt(1)]})
    sandbox.run_select(
        conn, "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x FROM n) SELECT * FROM n"
    )

    assert sandbox.run_select(conn, "SELECT COUNT(*) FROM attempts")["rows"] == [[1]]


def test_a_huge_value_cannot_be_built() -> None:
    conn = sandbox.load({})

    result = sandbox.run_select(conn, "SELECT length(zeroblob(50000000))")

    assert isinstance(result, str) and "too big" in result


def test_rows_are_capped_and_the_model_told_to_aggregate() -> None:
    conn = sandbox.load({"attempts": [attempt(n) for n in range(1, 200)]})

    result = sandbox.run_select(conn, "SELECT id FROM attempts")

    assert len(result["rows"]) == sandbox.MAX_ROWS
    assert "Aggregate" in result["note"]


def test_a_long_cell_is_cut() -> None:
    script = "word " * 2000
    conn = sandbox.load({"writing_submissions": [{"id": 1, "text": script}]})

    [[text]] = sandbox.run_select(conn, "SELECT text FROM writing_submissions")["rows"]

    assert len(text) < len(script)
    assert text.endswith("[...cut]")


def test_the_whole_result_is_bounded() -> None:
    """Fifty long cells would still be a very long tool message."""
    conn = sandbox.load(
        {"writing_submissions": [{"id": n, "text": "x" * 1500} for n in range(1, 40)]}
    )

    result = sandbox.run_select(conn, "SELECT text FROM writing_submissions")

    assert len(result["rows"]) < 39
    assert "note" in result


def test_no_rows_says_so() -> None:
    conn = sandbox.load({})

    result = sandbox.run_select(conn, "SELECT * FROM mock_exams")

    assert result["rows"] == []
    assert "do not guess" in result["note"]


def test_a_mistyped_query_explains_itself() -> None:
    conn = sandbox.load({})

    result = sandbox.run_select(conn, "SELECT accurracy FROM attempts")

    assert isinstance(result, str)
    assert "accurracy" in result and "schema" in result


def test_the_description_matches_the_database() -> None:
    """The model writes SQL from the description, so it must be the real schema."""
    conn = sandbox.load({})
    description = sandbox.describe()

    for table in sandbox.SCHEMA:
        cursor = conn.execute(f"SELECT * FROM {table.name} LIMIT 0")
        assert [d[0] for d in cursor.description] == list(table.column_names)
        assert f"{table.name} --" in description
