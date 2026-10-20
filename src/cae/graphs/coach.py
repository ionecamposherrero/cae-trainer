"""Coach report graph.

    load_stats -> detect_trend -> narrate -> assemble -> cache -> END

Statistics come from SQL and Python; the model only writes prose about them.
Reports are cached against the learner's activity count.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session

from cae.db import repo
from cae.db.session import session_scope
from cae.domain.parts import SUBSCALE_LABELS, SUBSCALES
from cae.domain.scoring import grade_for, scale_score
from cae.domain.taxonomy import advice, label as category_label
from cae.llm import structured_call
from cae.observability import graph_config, note_degraded
from cae.prompts.coach import coach_prompt
from cae.schemas.coach import CoachNarrative

MIN_ACTIVITY_FOR_REPORT = 2


class CoachState(TypedDict, total=False):
    user_id: int
    username: str
    snapshot: Any
    examples: dict[str, list[str]]
    tables: dict[str, str]
    narrative: dict[str, Any]
    status: str
    message: str
    error: str
    report: dict[str, Any]


def stats_from(session: Session, user_id: int) -> dict[str, Any]:
    """The learner's snapshot and recent error examples, read through ``session``.

    ``load_stats`` reads ``cae.db``; the coach eval suite reads a learner it
    seeded in memory, through this same function.
    """
    snapshot = repo.progress_snapshot(session, user_id)
    examples = {
        row["category"]: repo.error_examples(session, user_id, row["category"])
        for row in snapshot.top_errors[:3]
    }
    return {"snapshot": snapshot, "examples": examples}


def load_stats(state: CoachState) -> dict[str, Any]:
    with session_scope() as session:
        return stats_from(session, state["user_id"])


def detect_trend(state: CoachState) -> dict[str, Any]:
    """Render the figures the model will write about, exactly as computed."""
    snapshot = state["snapshot"]
    examples: dict[str, list[str]] = state.get("examples", {})  # type: ignore[assignment]

    part_rows = ["part | name | attempts | accuracy | last 5 | trend"]
    for stat in snapshot.part_stats:
        if stat.attempts == 0:
            part_rows.append(f"{stat.part} | {stat.name} | 0 | not attempted yet | - | -")
            continue
        trend = f"{stat.trend:+.1f}" if abs(stat.trend) > 0.01 else ""
        part_rows.append(
            f"{stat.part} | {stat.name} | {stat.attempts} | {stat.accuracy}% | "
            f"{stat.recent_accuracy}% | {trend}"
        )

    if snapshot.top_errors:
        error_rows = ["category | label | times | share of all errors"]
        error_rows += [
            f"{row['category']} | {row['label']} | {row['count']} | {row['share']}%"
            for row in snapshot.top_errors
        ]
    else:
        error_rows = ["No errors recorded yet."]

    example_lines: list[str] = []
    for category, spans in examples.items():
        cleaned = [s for s in spans if s][:3]
        if cleaned:
            example_lines.append(
                f"{category_label(category)}: " + "; ".join(f'"{s}"' for s in cleaned)
            )
    examples_block = ""
    if example_lines:
        examples_block = (
            "Recent examples of those errors\n"
            "-----------------------------------------\n" + "\n".join(example_lines) + "\n"
        )

    if snapshot.writing_averages:
        writing_rows = [
            f"{SUBSCALE_LABELS[s]}: {snapshot.writing_averages.get(s, 0)}" for s in SUBSCALES
        ]
        writing_block = "\n".join(writing_rows)
    else:
        writing_block = "No writing tasks submitted yet."

    return {
        "tables": {
            "parts": "\n".join(part_rows),
            "errors": "\n".join(error_rows),
            "examples": examples_block,
            "writing": writing_block,
        }
    }


def prompt_figures(snapshot: Any) -> dict[str, Any]:
    """The figures the prompt states above the tables, exactly as ``narrate`` sends them."""
    score = scale_score(snapshot.overall_accuracy)
    grade, _ = grade_for(score)
    return {
        "reading_attempts": snapshot.reading_attempts,
        "writing_count": snapshot.writing_submissions,
        "overall": snapshot.overall_accuracy,
        "scale": score,
        "grade": grade,
    }


def narrate(state: CoachState) -> dict[str, Any]:
    snapshot = state["snapshot"]
    if snapshot.activity_count < MIN_ACTIVITY_FOR_REPORT:
        return {"narrative": {}, "status": "not_enough_data"}

    tables = state["tables"]
    system, user = coach_prompt(
        username=state.get("username", "the student"),
        **prompt_figures(snapshot),
        part_table=tables["parts"],
        error_table=tables["errors"],
        examples_block=tables["examples"],
        writing_block=tables["writing"],
    )
    try:
        result = structured_call(
            CoachNarrative,
            system,
            user,
            tier="coach",
            temperature=0.4,
            label="coach.narrate",
        )
        return {"narrative": result.model_dump(), "status": "ok"}
    except Exception as exc:
        note_degraded("coach.narrate", exc, user_id=state.get("user_id"))
        return {
            "narrative": {},
            "status": "failed",
            "message": str(exc),
            "error": type(exc).__name__,
        }


def assemble(state: CoachState) -> dict[str, Any]:
    """Combine the narrative with the hard numbers and the taxonomy's own advice."""
    snapshot = state["snapshot"]
    report = {
        "status": state.get("status", "ok"),
        "narrative": state.get("narrative", {}),
        "overall_accuracy": snapshot.overall_accuracy,
        "scale_score": scale_score(snapshot.overall_accuracy),
        "reading_attempts": snapshot.reading_attempts,
        "writing_submissions": snapshot.writing_submissions,
        "strengths": [
            {"part": s.part, "name": s.name, "accuracy": s.accuracy}
            for s in snapshot.strengths
        ],
        "weaknesses": [
            {"part": s.part, "name": s.name, "accuracy": s.accuracy, "trend": s.trend}
            for s in snapshot.weaknesses
        ],
        "top_errors": [
            {**row, "advice": advice(row["category"])} for row in snapshot.top_errors
        ],
        "writing_averages": snapshot.writing_averages,
    }
    return {"report": report}


def cache(state: CoachState) -> dict[str, Any]:
    snapshot = state["snapshot"]
    with session_scope() as session:
        repo.save_report(
            session,
            user_id=state["user_id"],
            activity_count=snapshot.activity_count,
            report=state["report"],
        )
    return {}


def build_coach_graph():
    graph = StateGraph(CoachState)
    graph.add_node("load_stats", load_stats)
    graph.add_node("detect_trend", detect_trend)
    graph.add_node("narrate", narrate)
    graph.add_node("assemble", assemble)
    graph.add_node("cache", cache)

    graph.add_edge(START, "load_stats")
    graph.add_edge("load_stats", "detect_trend")
    graph.add_edge("detect_trend", "narrate")
    graph.add_edge("narrate", "assemble")
    graph.add_edge("assemble", "cache")
    graph.add_edge("cache", END)
    return graph.compile()


_GRAPH = None


def get_coach_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_coach_graph()
    return _GRAPH


def build_report(
    user_id: int,
    username: str,
    *,
    use_cache: bool = True,
    trace_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the learner's coaching report, reusing the cached one if still current."""
    with session_scope() as session:
        snapshot = repo.progress_snapshot(session, user_id)
        if use_cache:
            cached = repo.get_cached_report(session, user_id, snapshot.activity_count)
            if cached is not None:
                return {**cached.report, "cached": True}

    config = graph_config(
        {"graph": "coach", "user_id": user_id, **(trace_metadata or {})},
        run_name="coach.report",
    )
    final = get_coach_graph().invoke({"user_id": user_id, "username": username}, config)
    return {**final["report"], "cached": False}
