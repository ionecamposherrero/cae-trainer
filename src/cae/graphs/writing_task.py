"""Writing task generation graph.

    plan_task -> generate -> validate --ok--> persist -> END
                     ^            |
                     +-- repair <-+
                                  |
                             give_up -> END

The same loop as :mod:`cae.graphs.rue_task`, without a critique step: a writing
task has no answer key to moderate.
"""

from __future__ import annotations

import json
import random
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from cae.config import get_settings
from cae.db import repo
from cae.db.session import session_scope
from cae.domain.parts import WRITING_PART1_GENRES, get_genre
from cae.llm import structured_call
from cae.observability import graph_config
from cae.prompts.writing_task import TASK_TOPICS, task_prompt, task_repair_prompt
from cae.schemas.writing_task import WritingTaskSpec
from cae.validators.writing_task import validate_writing_task


class WritingTaskState(TypedDict, total=False):
    part: int
    genre: str
    topic: str
    task: dict[str, Any]
    issues: list[str]
    issue_history: list[list[str]]
    attempts: int
    max_attempts: int
    status: str
    task_id: int | None
    persist: bool


class WritingTaskFailed(RuntimeError):
    """The model could not produce a usable task within the retry budget."""

    def __init__(
        self,
        part: int,
        genre: str,
        issues: list[str],
        *,
        attempts: int = 0,
        issue_history: list[list[str]] | None = None,
    ):
        self.part = part
        self.genre = genre
        self.issues = issues
        self.attempts = attempts
        self.issue_history = issue_history or []
        super().__init__(
            f"Could not generate a valid Writing Part {part} task ({genre}). "
            "Remaining problems: " + "; ".join(issues[:5])
        )


def plan_task(state: WritingTaskState) -> dict[str, Any]:
    """Settle the genre and the topic before anything is written."""
    part = state["part"]
    genre = state.get("genre") or ""
    if part == 1:
        genre = WRITING_PART1_GENRES[0]
    elif not genre:
        raise ValueError("A Part 2 task needs a genre.")
    get_genre(genre)
    return {
        "genre": genre,
        "topic": state.get("topic") or random.choice(TASK_TOPICS),
        "attempts": 0,
        "issues": [],
        "issue_history": [],
        "status": "drafting",
    }


def generate(state: WritingTaskState) -> dict[str, Any]:
    system, user = task_prompt(state["part"], state["genre"], state["topic"])
    spec = structured_call(
        WritingTaskSpec,
        system,
        user,
        tier="writing_task",
        temperature=0.9,
        label="writing_task.generate",
    )
    return {"task": spec.model_dump(), "attempts": state.get("attempts", 0) + 1}


def validate(state: WritingTaskState) -> dict[str, Any]:
    issues = validate_writing_task(state["part"], state["genre"], state["task"])
    return {"issues": issues, "issue_history": [*state.get("issue_history", []), issues]}


def repair(state: WritingTaskState) -> dict[str, Any]:
    system, _ = task_prompt(state["part"], state["genre"], state["topic"])
    previous = json.dumps(state["task"], ensure_ascii=False, indent=2)
    user = task_repair_prompt(state.get("issues", []), previous)
    spec = structured_call(
        WritingTaskSpec,
        system,
        user,
        tier="writing_task",
        temperature=0.4,
        label="writing_task.repair",
    )
    return {"task": spec.model_dump(), "attempts": state.get("attempts", 0) + 1}


def persist(state: WritingTaskState) -> dict[str, Any]:
    task = {
        **state["task"],
        "genre": state["genre"],
        "part": state["part"],
        "topic": state["topic"],
        "generation_attempts": state.get("attempts", 1),
    }
    if not state.get("persist", True):
        return {"task": task, "status": "ok"}
    with session_scope() as session:
        row = repo.save_writing_task(
            session, part=state["part"], genre=state["genre"], payload=task
        )
        return {"task": task, "task_id": row.id, "status": "ok"}


def give_up(state: WritingTaskState) -> dict[str, Any]:
    return {"status": "failed"}


def _after_validate(state: WritingTaskState) -> str:
    if not state.get("issues"):
        return "persist"
    return "repair" if state.get("attempts", 0) < state["max_attempts"] else "give_up"


def build_writing_task_graph():
    graph = StateGraph(WritingTaskState)
    graph.add_node("plan_task", plan_task)
    graph.add_node("generate", generate)
    graph.add_node("validate", validate)
    graph.add_node("repair", repair)
    graph.add_node("persist", persist)
    graph.add_node("give_up", give_up)

    graph.add_edge(START, "plan_task")
    graph.add_edge("plan_task", "generate")
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges(
        "validate",
        _after_validate,
        {"repair": "repair", "persist": "persist", "give_up": "give_up"},
    )
    graph.add_edge("repair", "validate")
    graph.add_edge("persist", END)
    graph.add_edge("give_up", END)
    return graph.compile()


_GRAPH = None


def get_writing_task_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_writing_task_graph()
    return _GRAPH


def generate_writing_task(
    *,
    part: int,
    genre: str = "",
    topic: str = "",
    persist_result: bool = True,
    max_attempts: int | None = None,
    trace_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate one validated writing task.

    Returns the final state, including ``task`` (the payload), ``genre``,
    ``task_id`` when persisted, ``attempts`` and ``issue_history``.
    """
    initial: WritingTaskState = {
        "part": part,
        "genre": genre,
        "topic": topic,
        "max_attempts": max_attempts or get_settings().cae_max_generation_attempts,
        "persist": persist_result,
    }
    config = graph_config(
        {"graph": "writing_task", "part": part, "genre": genre, **(trace_metadata or {})},
        recursion_limit=50,
        run_name=f"writing_task.part{part}",
    )
    final = get_writing_task_graph().invoke(initial, config)
    if final.get("status") == "failed":
        raise WritingTaskFailed(
            part,
            final.get("genre", genre),
            final.get("issues", []),
            attempts=final.get("attempts", 0),
            issue_history=final.get("issue_history", []),
        )
    return final
