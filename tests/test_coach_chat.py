"""The coach chat graph, with the model stubbed."""

from __future__ import annotations

import json
import uuid
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from cae import auth
from cae.config import get_settings
from cae.db import repo, sandbox
from cae.db.session import session_scope
from cae.graphs import coach_chat
from tests import factories


def seed_attempt(user_id: int, *, part: int = 1, wrong_answer: str = "glow") -> int:
    """One marked attempt with a single wrong item.

    The payload must be valid: the shared session database serves stored
    exercises to other tests.
    """
    with session_scope() as session:
        exercise = repo.save_exercise(
            session, part=part, payload=factories.build(part).model_dump()
        )
        attempt = repo.record_attempt(
            session,
            user_id=user_id,
            exercise_id=exercise.id,
            part=part,
            responses={"1": wrong_answer},
            item_rows=[
                {
                    "item_number": 1,
                    "user_answer": wrong_answer,
                    "correct_answer": "light",
                    "is_correct": False,
                    "marks_awarded": 0.0,
                    "max_marks": 1.0,
                    "category": "FIXED_PHRASES",
                    "explanation": "'Cast light on' is a fixed collocation.",
                }
            ],
            marks=0.0,
            max_marks=1.0,
        )
        return attempt.id


def query(user_id: int, sql: str) -> dict[str, Any] | str:
    """Run one query through the tool, exactly as the model would."""
    tools = {t.name: t for t in coach_chat.build_tools(user_id, [])}
    return tools["query_record"].invoke({"sql": sql})


def rows(result: dict[str, Any] | str) -> list[dict[str, Any]]:
    assert isinstance(result, dict), f"the query failed: {result}"
    return [dict(zip(result["columns"], row)) for row in result["rows"]]


def seed_writing(user_id: int) -> int:
    with session_scope() as session:
        task = repo.save_writing_task(
            session,
            part=1,
            genre="essay",
            payload={
                "title": "Tourism essay",
                "scenario": "Your class has attended a panel discussion.",
                "instruction": "Discuss two of the points in your notes.",
                "notes": ["local business revenue", "infrastructure", "culture"],
                "reader": "your tutor",
            },
        )
        submission = repo.save_submission(
            session, user_id=user_id, task_id=task.id, text="An essay.", word_count=2
        )
        repo.save_writing_score(
            session,
            submission_id=submission.id,
            user_id=user_id,
            bands={"content": 4, "communicative_achievement": 4,
                   "organisation": 4, "language": 4},
            total=16,
            scale_score=190,
            feedback={"content": {"band": 4, "justification": "An element is missing."}},
            annotations=[{"category": "REGISTER", "span": "gonna"}],
        )
        return submission.id


def reply(text: str = "Here is what I can see.") -> AIMessage:
    return AIMessage(content=text)


def tool_reply(name: str, args: dict[str, Any], *more: dict[str, Any]) -> AIMessage:
    """One tool call, or several in the same round."""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": a, "id": f"call_{name}_{i}"}
            for i, a in enumerate([args, *more])
        ],
    )


@pytest.fixture
def no_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """No reference corpus, which is the state of a fresh clone."""
    monkeypatch.setattr(coach_chat.reference_store, "index_exists", lambda: False)
    monkeypatch.setattr(coach_chat.reference_store, "search", lambda *a, **k: [])


def test_a_plain_question_is_answered_in_one_call(
    user: dict, no_index: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The snapshot is preloaded, so a simple question needs no tool round-trip."""
    calls: list[dict] = []

    def fake_chat(messages, **kwargs):
        calls.append({"messages": messages, "tools": kwargs.get("tools")})
        return reply("You are averaging 62%.")

    monkeypatch.setattr(coach_chat, "chat_call", fake_chat)
    seed_attempt(user["id"])

    result = coach_chat.answer(user["id"], user["username"], [], "How am I doing?")

    assert len(calls) == 1
    assert result["reply"] == "You are averaging 62%."
    assert result["tool_rounds"] == 0
    assert "Accuracy by part" in calls[0]["messages"][0].content


def test_a_tool_result_reaches_the_next_call(
    user: dict, no_index: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt_id = seed_attempt(user["id"], wrong_answer="glow")
    seen: list[list] = []

    def fake_chat(messages, **kwargs):
        seen.append(messages)
        if len(seen) == 1:
            return tool_reply(
                "query_record",
                {
                    "sql": "SELECT user_answer, explanation FROM item_results "
                    f"WHERE attempt_id = {attempt_id}"
                },
            )
        return reply("You wrote 'glow'; the collocation is 'cast light on'.")

    monkeypatch.setattr(coach_chat, "chat_call", fake_chat)

    result = coach_chat.answer(
        user["id"], user["username"], [], "Why was item 1 wrong?"
    )

    assert len(seen) == 2
    assert result["tool_rounds"] == 1
    transcript = " ".join(str(m.content) for m in seen[1])
    assert "glow" in transcript
    assert "cast light on" in transcript.lower()


def test_the_tools_a_turn_called_are_returned_in_order(
    user: dict, no_index: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The evals score which tools a turn needed; the page never sees them."""
    turns = iter(
        [
            tool_reply("query_record", {"sql": "SELECT COUNT(*) FROM attempts"}),
            tool_reply("reference_lookup", {"query": "Part 6 marks"}),
            reply("Done."),
        ]
    )
    monkeypatch.setattr(coach_chat, "chat_call", lambda messages, **kwargs: next(turns))

    result = coach_chat.answer(user["id"], user["username"], [], "Anything.")

    assert result["tool_calls"] == [
        {"name": "query_record", "args": {"sql": "SELECT COUNT(*) FROM attempts"}},
        {"name": "reference_lookup", "args": {"query": "Part 6 marks"}},
    ]
    assert result["tool_rounds"] == 2 and len(result["tool_outputs"]) == 2


def test_a_reply_in_content_blocks_reaches_the_learner_as_text(
    user: dict, no_index: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gemini 3.x replies in blocks; the learner must see text, not their repr."""
    blocks = AIMessage(
        content=[
            {"type": "text", "text": "Yes, Organisation ", "extras": {"signature": "x"}},
            {"type": "text", "text": "was your lowest."},
        ]
    )
    monkeypatch.setattr(coach_chat, "chat_call", lambda messages, **kwargs: blocks)
    result = coach_chat.answer(user["id"], user["username"], [], "Weakest?")
    assert result["reply"] == "Yes, Organisation was your lowest."


def test_a_plain_answer_called_no_tools(
    user: dict, no_index: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(coach_chat, "chat_call", lambda messages, **kwargs: reply("Fine."))
    assert coach_chat.answer(user["id"], user["username"], [], "Hi")["tool_calls"] == []


def test_the_tool_budget_stops_a_runaway_loop(
    user: dict, no_index: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model that only ever asks for tools must still terminate."""
    seed_attempt(user["id"])
    rounds: list[Any] = []

    def fake_chat(messages, **kwargs):
        rounds.append(kwargs.get("tools"))
        if not kwargs.get("tools"):
            return reply("Here is what I found.")
        return tool_reply("query_record", {"sql": "SELECT COUNT(*) FROM attempts"})

    monkeypatch.setattr(coach_chat, "chat_call", fake_chat)

    result = coach_chat.answer(user["id"], user["username"], [], "Tell me everything")

    assert result["tool_rounds"] == coach_chat.MAX_TOOL_ROUNDS
    assert len(rounds) == coach_chat.MAX_TOOL_ROUNDS + 1
    assert rounds[-1] == (), "the final call must be made with no tools bound"
    assert result["reply"] == "Here is what I found."


def test_a_tool_cannot_read_another_learners_attempt(user: dict) -> None:
    """The model writes the SQL, so no clause in it can be what grants access."""
    intruder = auth.register(f"nosy{uuid.uuid4().hex[:8]}", "correct-horse-battery")
    victim_attempt = seed_attempt(user["id"], wrong_answer="secret")

    direct = query(
        intruder["id"], f"SELECT * FROM item_results WHERE attempt_id = {victim_attempt}"
    )
    everything = query(intruder["id"], "SELECT user_answer FROM item_results")

    assert rows(direct) == []
    assert "secret" not in json.dumps(everything)


def test_the_record_holds_only_this_learners_work(user: dict) -> None:
    other = auth.register(f"other{uuid.uuid4().hex[:8]}", "correct-horse-battery")
    seed_attempt(user["id"])
    seed_attempt(user["id"], part=2)
    seed_attempt(other["id"])

    [count] = rows(query(user["id"], "SELECT COUNT(*) AS n FROM attempts"))

    assert count["n"] == 2


def test_the_record_has_no_accounts_and_no_owner_column(user: dict) -> None:
    """Nothing to filter on means nothing a query could filter wrongly."""
    seed_attempt(user["id"])

    tables = rows(query(user["id"], "SELECT name, sql FROM sqlite_master"))

    assert {t["name"] for t in tables} == set(sandbox.TABLES)
    for table in tables:
        assert "user_id" not in table["sql"], table["name"]
        assert "password" not in table["sql"], table["name"]
        assert "payload" not in table["sql"], table["name"]


@pytest.mark.parametrize(
    "statement",
    [
        "DELETE FROM attempts",
        "UPDATE attempts SET marks = 99",
        "INSERT INTO attempts (id) VALUES (999999)",
        "DROP TABLE attempts",
        "PRAGMA table_info(attempts)",
    ],
)
def test_the_record_is_read_only(user: dict, statement: str) -> None:
    seed_attempt(user["id"])
    tools = {t.name: t for t in coach_chat.build_tools(user["id"], [])}

    refused = tools["query_record"].invoke({"sql": statement})
    after = tools["query_record"].invoke({"sql": "SELECT marks FROM attempts"})

    assert isinstance(refused, str) and "Not allowed" in refused
    assert rows(after) == [{"marks": 0.0}]


def test_the_real_database_cannot_be_attached(user: dict) -> None:
    """ATTACH is the one way out of the copy and into every learner's record."""
    path = get_settings().database_url.removeprefix("sqlite:///")

    refused = query(user["id"], f"ATTACH DATABASE '{path}' AS real")

    assert isinstance(refused, str) and "Not allowed" in refused


def test_a_bad_query_is_answered_not_raised(user: dict) -> None:
    """The error goes back to the model, which can fix its query next round."""
    result = query(user["id"], "SELECT nope FROM attempts")

    assert isinstance(result, str)
    assert "no such column: nope" in result


def test_an_opened_item_carries_the_question_the_learner_saw(user: dict) -> None:
    """Without the question, "why was I wrong?" rests on the marker's prose alone."""
    attempt_id = seed_attempt(user["id"], part=1)

    [item] = rows(
        query(
            user["id"],
            f"SELECT question, options FROM item_results WHERE attempt_id = {attempt_id}",
        )
    )

    assert item["question"], "no context was attached to the item"
    options = json.loads(item["options"])
    assert set(options) == {"A", "B", "C", "D"}
    assert options["A"]


def test_opening_an_item_does_not_return_the_whole_passage(user: dict) -> None:
    """The excerpt is the point: a Part 5 text is 700-800 words per lookup."""
    attempt_id = seed_attempt(user["id"], part=1)

    [item] = rows(
        query(user["id"], f"SELECT * FROM item_results WHERE attempt_id = {attempt_id}")
    )

    assert "payload" not in item, "the raw exercise must not reach the model"
    assert len(item["question"].split()) < 100, "the context should be an excerpt"


def test_an_unparsable_exercise_still_returns_the_marks(
    user: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exercise payload that does not parse must degrade, not break the tool."""

    def boom(*args, **kwargs):
        raise ValueError("payload does not match the current schema")

    monkeypatch.setattr(coach_chat, "parse_exercise", boom)
    attempt_id = seed_attempt(user["id"], part=1)

    [item] = rows(
        query(user["id"], f"SELECT * FROM item_results WHERE attempt_id = {attempt_id}")
    )

    assert item["correct_answer"], "the marks and key are still worth returning"
    assert item["question"] == ""


def test_writing_feedback_carries_the_task_that_was_set(user: dict) -> None:
    """Explaining a Content band needs the task's instruction and notes."""
    seed_writing(user["id"])

    [row] = rows(
        query(
            user["id"],
            "SELECT s.instruction, s.notes, sc.content, "
            "json_extract(sc.feedback, '$.content.justification') AS why "
            "FROM writing_submissions s "
            "JOIN writing_scores sc ON sc.submission_id = s.id",
        )
    )

    assert row["instruction"]
    assert json.loads(row["notes"]) == ["local business revenue", "infrastructure", "culture"]
    assert row["content"] == 4
    assert row["why"] == "An element is missing."


def test_writing_errors_join_back_to_their_script(user: dict) -> None:
    submission_id = seed_writing(user["id"])

    [event] = rows(
        query(
            user["id"],
            "SELECT e.category, e.evidence, c.label FROM error_events e "
            "JOIN error_categories c ON c.code = e.category "
            f"WHERE e.source_type = 'writing' AND e.source_id = {submission_id}",
        )
    )

    assert event == {"category": "REGISTER", "evidence": "gonna", "label": "Register and tone"}


def test_the_export_fills_every_described_column(user: dict) -> None:
    """A described column the export leaves out reads as NULL, and the model
    would report it as missing data rather than as a bug."""
    seed_attempt(user["id"])
    seed_writing(user["id"])
    with session_scope() as session:
        exam = repo.create_mock_exam(session, user["id"])
        repo.finish_mock_exam(
            session, exam_id=exam.id, marks=10, max_marks=20, scale_score=170
        )
        tables = repo.learner_export(session, user["id"])

    derived = {"item_results": {"question", "key_word", "options"}}
    for name, table in sandbox.TABLES.items():
        if name == "error_categories":
            continue
        assert tables[name], f"nothing was seeded into {name}"
        expected = set(table.column_names) - derived.get(name, set())
        assert set(tables[name][0]) == expected, name


def test_the_accuracy_column_agrees_with_the_dashboard(user: dict) -> None:
    seed_attempt(user["id"])
    with session_scope() as session:
        [history] = repo.attempt_history(session, user["id"])

    [row] = rows(query(user["id"], "SELECT accuracy FROM attempts"))

    assert row["accuracy"] == history["accuracy"]


def test_a_plain_question_builds_no_record(
    user: dict, no_index: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Binding the tool schemas must not copy the learner's record."""
    built: list[int] = []
    monkeypatch.setattr(coach_chat, "learner_record", lambda uid: built.append(uid))
    monkeypatch.setattr(coach_chat, "chat_call", lambda messages, **kw: reply())
    seed_attempt(user["id"])

    coach_chat.answer(user["id"], user["username"], [], "How am I doing?")

    assert built == []


def test_parallel_queries_share_one_record(
    user: dict, no_index: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tool node runs parallel calls on threads; they must share one copy."""
    built: list[int] = []
    real = coach_chat.learner_record

    def counting(uid: int):
        built.append(uid)
        return real(uid)

    monkeypatch.setattr(coach_chat, "learner_record", counting)
    seed_attempt(user["id"])
    turns: list[Any] = []

    def fake_chat(messages, **kwargs):
        turns.append(messages)
        if len(turns) == 1:
            return tool_reply(
                "query_record",
                {"sql": "SELECT COUNT(*) FROM attempts"},
                {"sql": "SELECT COUNT(*) FROM item_results"},
            )
        return reply("Done.")

    monkeypatch.setattr(coach_chat, "chat_call", fake_chat)

    result = coach_chat.answer(user["id"], user["username"], [], "Count my work")

    assert built == [user["id"]]
    assert len(result["tool_outputs"]) == 2
    assert all('"rows"' in output for output in result["tool_outputs"])


def test_no_tool_exposes_user_id(user: dict) -> None:
    """The authenticated learner is closed over, never a model-supplied argument."""
    for tool in coach_chat.build_tools(user["id"], []):
        fields = tool.args_schema.model_fields
        assert "user_id" not in fields, f"{tool.name} exposes user_id to the model"


def test_every_tool_declares_a_schema_and_a_description(user: dict) -> None:
    for tool in coach_chat.build_tools(user["id"], []):
        assert tool.args_schema is not None, f"{tool.name} has no argument schema"
        assert len(tool.description) > 40, f"{tool.name} is under-described"


def test_the_query_tool_describes_every_table_and_column(user: dict) -> None:
    tools = {t.name: t for t in coach_chat.build_tools(user["id"], [])}
    description = tools["query_record"].description

    for table in sandbox.SCHEMA:
        assert f"{table.name} --" in description
        for column in table.columns:
            assert f"  {column.name} {column.type}" in description


def test_retrieved_passages_reach_the_caller(
    user: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The eval harness scores retrieval, so the passages must survive the round-trip."""
    hits = [
        {
            "text": "After a negative adverbial the subject and auxiliary invert.",
            "source": "grammar.md",
            "section": "Inversion",
            "citation": "grammar.md — Inversion",
        }
    ]
    monkeypatch.setattr(coach_chat.reference_store, "index_exists", lambda: True)
    monkeypatch.setattr(coach_chat.reference_store, "search", lambda *a, **k: hits)

    turns: list[int] = []

    def fake_chat(messages, **kwargs):
        turns.append(1)
        if len(turns) == 1:
            return tool_reply("reference_lookup", {"query": "inversion"})
        return reply("Inversion follows a negative adverbial.")

    monkeypatch.setattr(coach_chat, "chat_call", fake_chat)

    result = coach_chat.answer(user["id"], user["username"], [], "When do I invert?")

    assert result["sources"] == hits


def test_the_page_never_receives_the_retrieved_passages(
    user: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retrieval is for the evals; the learner sees the reply and nothing else."""
    from cae import services

    hits = [{"text": "Subject and auxiliary invert.", "citation": "grammar.md — Inversion"}]
    monkeypatch.setattr(coach_chat.reference_store, "index_exists", lambda: True)
    monkeypatch.setattr(coach_chat.reference_store, "search", lambda *a, **k: hits)

    turns: list[int] = []

    def fake_chat(messages, **kwargs):
        turns.append(1)
        if len(turns) == 1:
            return tool_reply("reference_lookup", {"query": "inversion"})
        return reply("Inversion follows a negative adverbial.")

    monkeypatch.setattr(coach_chat, "chat_call", fake_chat)

    result = services.coach_reply(user["id"], user["username"], [], "When do I invert?")

    assert result == {"reply": "Inversion follows a negative adverbial."}


def test_a_missing_corpus_is_reported_not_hidden(
    user: dict, no_index: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without an index the coach must say so rather than invent a citation."""
    prompts: list[str] = []

    def fake_chat(messages, **kwargs):
        prompts.append(messages[0].content)
        return reply("Not covered by my reference material, but: ...")

    monkeypatch.setattr(coach_chat, "chat_call", fake_chat)

    coach_chat.answer(user["id"], user["username"], [], "When do I invert?")

    assert "no reference material has been indexed" in prompts[0]

    tools = {t.name: t for t in coach_chat.build_tools(user["id"], [])}
    lookup = tools["reference_lookup"].invoke({"query": "inversion"})
    assert "not covered by the reference material" in lookup


def test_a_learner_with_no_record_is_described_as_such(
    user: dict, no_index: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A page of zeroes reads as a bad result; "nothing yet" does not."""
    prompts: list[str] = []

    def fake_chat(messages, **kwargs):
        prompts.append(messages[0].content)
        return reply("You have not been marked on anything yet.")

    monkeypatch.setattr(coach_chat, "chat_call", fake_chat)

    coach_chat.answer(user["id"], user["username"], [], "How am I doing?")

    assert "have not been marked on anything yet" in prompts[0]
    assert "Accuracy by part" not in prompts[0]


def test_practice_requests_are_sent_to_the_exercise_pages(
    user: dict, no_index: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Items the coach invented would be unchecked and never reach the record."""
    prompts: list[str] = []

    def fake_chat(messages, **kwargs):
        prompts.append(messages[0].content)
        return reply("Try a Part 1 exercise on the Reading and Use of English page.")

    monkeypatch.setattr(coach_chat, "chat_call", fake_chat)

    coach_chat.answer(user["id"], user["username"], [], "Give me five Part 1 questions")

    assert "Never write exercises" in prompts[0]
    for page in ("Reading and Use of English", "Writing", "Mock exam"):
        assert f"{page} page" in prompts[0]


def test_the_thread_is_replayed_to_the_model(
    user: dict, no_index: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The graph is stateless between turns; the page owns the thread."""
    seen: list[list] = []

    def fake_chat(messages, **kwargs):
        seen.append(messages)
        return reply("Yes.")

    monkeypatch.setattr(coach_chat, "chat_call", fake_chat)

    coach_chat.answer(
        user["id"],
        user["username"],
        [("user", "Is inversion tested?"), ("assistant", "Yes, in Part 2.")],
        "Where else?",
    )

    texts = [str(m.content) for m in seen[0]]
    assert "Is inversion tested?" in texts
    assert "Yes, in Part 2." in texts
    assert texts[-1] == "Where else?"


@pytest.mark.parametrize(
    "sql", [sql for _, sql in coach_chat.QUERY_EXAMPLES], ids=lambda s: s[:40]
)
def test_query_examples_run_against_the_sandbox(sql: str) -> None:
    """An example that does not match the schema would teach the model an error."""
    result = sandbox.run_select(sandbox.load({}), sql)
    assert isinstance(result, dict), result
    assert result["columns"]


def test_the_examples_are_in_the_tool_description() -> None:
    for _, sql in coach_chat.QUERY_EXAMPLES:
        assert sql in coach_chat.QUERY_RECORD_DESCRIPTION
