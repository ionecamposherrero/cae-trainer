"""Coach chat graph.

    load_context -> respond -> [tools] -> respond -> ... -> END

The progress snapshot is preloaded into the system message. For anything else
about the learner the model writes its own SELECT, which ``query_record`` runs
against a read-only copy of that learner's record and nothing else
(``cae.db.sandbox``). ``reference_lookup`` searches the reference corpus.
"""

from __future__ import annotations

import operator
import sqlite3
import threading
from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from cae.db import repo, sandbox
from cae.db.session import session_scope
from cae.domain.scoring import grade_for, scale_score
from cae.domain.taxonomy import ErrorCategory, advice, family, label as category_label
from cae.graphs import coach
from cae.graphs.rue_evaluator import item_context
from cae.llm import chat_call
from cae.observability import graph_config, note_degraded
from cae.prompts.coach_chat import chat_context, chat_system, empty_context
from cae.rag import store as reference_store
from cae.schemas.rue_task import answer_key, options_by_item, parse_exercise
from cae.schemas.coach_chat import RecordQueryArgs, ReferenceLookupArgs

MAX_TOOL_ROUNDS = 4


class ChatState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: int
    username: str
    system: str
    sources: Annotated[list[dict[str, Any]], operator.add]


def _item_columns(part: int, payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Each item's question, key word and options, by item number.

    Only the excerpt around each gap is kept, never the whole passage.
    """
    try:
        exercise = parse_exercise(part, payload)
        keys = answer_key(part, exercise)
        options = options_by_item(exercise)
    except Exception as exc:
        note_degraded("coach_chat.item_columns", exc, part=part)
        return {}
    return {
        key.number: {
            "question": item_context(payload, key) or "",
            "key_word": key.key_word or "",
            "options": options.get(key.number) or "",
        }
        for key in keys
    }


def learner_record(user_id: int) -> sqlite3.Connection:
    """The database ``query_record`` reads: this learner's rows and nothing else."""
    with session_scope() as session:
        tables = repo.learner_export(session, user_id)
        payloads = repo.attempted_exercises(session, user_id)

    by_exercise: dict[int, dict[int, dict[str, Any]]] = {}
    for attempt in tables["attempts"]:
        exercise_id = attempt["exercise_id"]
        if exercise_id not in by_exercise:
            by_exercise[exercise_id] = _item_columns(
                attempt["part"], payloads.get(exercise_id, {})
            )
    by_attempt = {a["id"]: by_exercise[a["exercise_id"]] for a in tables["attempts"]}

    blank = {"question": "", "key_word": "", "options": ""}
    for row in tables["item_results"]:
        row.update(by_attempt.get(row["attempt_id"], {}).get(row["item_number"], blank))

    tables["error_categories"] = [
        {"code": c.value, "label": category_label(c), "family": family(c), "advice": advice(c)}
        for c in ErrorCategory
    ]
    return sandbox.load(tables)


QUERY_EXAMPLES: tuple[tuple[str, str], ...] = (
    (
        "Part 4 items they got wrong, most recent first",
        "SELECT a.submitted_at, i.item_number, i.key_word, i.user_answer, "
        "i.correct_answer, i.category FROM item_results i JOIN attempts a ON "
        "a.id = i.attempt_id WHERE a.part = 4 AND i.is_correct = 0 "
        "ORDER BY a.submitted_at DESC LIMIT 10",
    ),
    (
        "Mistakes by category over the last 30 days",
        "SELECT category, COUNT(*) AS mistakes FROM error_events WHERE "
        "created_at >= datetime('now', '-30 days') GROUP BY category "
        "ORDER BY mistakes DESC",
    ),
    (
        "How their Organisation band has moved, with the examiner's advice",
        "SELECT s.submitted_at, s.genre, w.organisation, "
        "json_extract(w.feedback, '$.organisation.to_improve') AS advice FROM "
        "writing_scores w JOIN writing_submissions s ON s.id = w.submission_id "
        "ORDER BY s.submitted_at",
    ),
)

QUERY_RECORD_DESCRIPTION = (
    "Run one read-only SQLite SELECT over this student's record and get the rows "
    "back. The database holds their work and nobody else's, so there is no student "
    "column to filter on. Use it for anything about them that the figures you were "
    "given do not cover: a particular exercise or item and why it was marked wrong, "
    "their own examples of a mistake, time taken, topics, mock exams, writing bands "
    "and the examiner's feedback. Times are UTC text 'YYYY-MM-DD HH:MM:SS', so "
    "date('now', '-7 days') works; JSON columns work with json_extract and json_each."
    "\n\nTables:\n"
    + sandbox.describe()
    + "\n\nExamples:\n"
    + "\n".join(f"-- {question}\n{sql}" for question, sql in QUERY_EXAMPLES)
)


def build_tools(user_id: int, sources: list[dict[str, Any]]) -> list[BaseTool]:
    """The tools for one conversation, bound to one authenticated learner.

    ``user_id`` is closed over, never a tool argument. ``query_record`` runs
    against a copy of this learner's record, built on first use so that binding
    the schemas costs nothing. ``reference_lookup`` appends its hits to ``sources``.
    """
    lock = threading.Lock()
    record: list[sqlite3.Connection] = []

    def query_record(sql: str) -> dict[str, Any] | str:
        with lock:
            if not record:
                record.append(learner_record(user_id))
            return sandbox.run_select(record[0], sql)

    def reference_lookup(query: str) -> list[dict[str, Any]] | str:
        hits = reference_store.search(query)
        if not hits:
            return (
                "Nothing in the reference material matches that. Answer from your "
                "own knowledge and say it is not covered by the reference material."
            )
        sources.extend(hits)
        return hits

    return [
        StructuredTool.from_function(
            func=query_record,
            name="query_record",
            description=QUERY_RECORD_DESCRIPTION,
            args_schema=RecordQueryArgs,
        ),
        StructuredTool.from_function(
            func=reference_lookup,
            name="reference_lookup",
            description=(
                "Search the Cambridge reference material for a point of grammar, "
                "vocabulary, exam format or assessment criteria. Use this before "
                "answering any question about the English language itself, and "
                "quote the source it returns."
            ),
            args_schema=ReferenceLookupArgs,
        ),
    ]


def load_context(state: ChatState) -> dict[str, Any]:
    """Build the system message from the coach report's own figures."""
    username = state.get("username", "the student")
    inner: dict[str, Any] = {"user_id": state["user_id"], "username": username}
    inner.update(coach.load_stats(inner))
    snapshot = inner["snapshot"]

    if snapshot.activity_count == 0:
        context = empty_context(username)
    else:
        inner.update(coach.detect_trend(inner))
        tables = inner["tables"]
        score = scale_score(snapshot.overall_accuracy)
        grade, _ = grade_for(score)
        context = chat_context(
            username=username,
            reading_attempts=snapshot.reading_attempts,
            writing_count=snapshot.writing_submissions,
            overall=snapshot.overall_accuracy,
            scale=score,
            grade=grade,
            part_table=tables["parts"],
            error_table=tables["errors"],
            examples_block=tables["examples"],
            writing_block=tables["writing"],
        )

    system = chat_system(index_available=reference_store.index_exists())
    return {"system": f"{system}\n\n{context}"}


def _tool_rounds(messages: Sequence[AnyMessage]) -> int:
    return sum(
        1
        for message in messages
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None)
    )


def respond(state: ChatState) -> dict[str, Any]:
    """One model turn, with the tool schemas bound."""
    tools = build_tools(state["user_id"], [])
    messages = [SystemMessage(content=state["system"]), *state["messages"]]

    exhausted = _tool_rounds(state["messages"]) >= MAX_TOOL_ROUNDS
    try:
        reply = chat_call(
            messages,
            tools=() if exhausted else tools,
            tier="chat",
            temperature=0.3,
            label="coach_chat.respond",
        )
    except Exception as exc:
        note_degraded("coach_chat.respond", exc, user_id=state.get("user_id"))
        raise
    return {"messages": [reply]}


def _after_respond(state: ChatState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return END


def dispatch_tools(state: ChatState) -> dict[str, Any]:
    """Run the requested tools, built per call because they close over ``user_id``."""
    retrieved: list[dict[str, Any]] = []
    tools = build_tools(state["user_id"], retrieved)
    result = ToolNode(tools).invoke({"messages": state["messages"]})
    return {"messages": result["messages"], "sources": retrieved}


def build_chat_graph():
    graph = StateGraph(ChatState)
    graph.add_node("load_context", load_context)
    graph.add_node("respond", respond)
    graph.add_node("tools", dispatch_tools)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "respond")
    graph.add_conditional_edges("respond", _after_respond, {"tools": "tools", END: END})
    graph.add_edge("tools", "respond")
    return graph.compile()


_GRAPH = None


def get_chat_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_chat_graph()
    return _GRAPH


def answer(
    user_id: int,
    username: str,
    history: Sequence[tuple[str, str]],
    question: str,
    *,
    trace_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Answer one question in context. Returns the reply and its cited sources.

    ``history`` is ``(role, text)`` pairs, oldest first. ``tool_outputs`` is what
    the tools returned this turn, which the evals count as figures handed over;
    ``tool_calls`` is what the model asked for, in order, so the evals can tell a
    turn that needed a query from one that spent a round it did not need. The
    page sees neither: ``services.coach_reply`` passes on the reply alone.
    """
    messages: list[AnyMessage] = []
    for role, text in history:
        messages.append(
            HumanMessage(content=text) if role == "user" else AIMessage(content=text)
        )
    messages.append(HumanMessage(content=question))

    config = graph_config(
        {"graph": "coach_chat", "user_id": user_id, **(trace_metadata or {})},
        run_name="coach_chat.answer",
    )
    final = get_chat_graph().invoke(
        {
            "user_id": user_id,
            "username": username,
            "messages": messages,
            "sources": [],
        },
        config,
    )

    reply = ""
    for message in reversed(final["messages"]):
        if isinstance(message, AIMessage) and message.text:
            reply = message.text
            break

    return {
        "reply": reply,
        "sources": _unique_sources(final.get("sources") or []),
        "tool_rounds": _tool_rounds(final["messages"]),
        "tool_calls": [
            {"name": call["name"], "args": dict(call.get("args") or {})}
            for m in final["messages"]
            if isinstance(m, AIMessage)
            for call in getattr(m, "tool_calls", None) or []
        ],
        "tool_outputs": [
            str(m.content) for m in final["messages"] if isinstance(m, ToolMessage)
        ],
    }


def _unique_sources(sources: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """De-duplicate retrieved passages, keeping the order they were found in."""
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for source in sources:
        marker = source.get("text", "")[:200]
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(source)
    return unique
