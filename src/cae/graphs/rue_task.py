"""Exercise generation graph.

    plan_task -> generate -> validate --ok--> critique --clean--> persist -> END
                     ^                 |            |
                     |                 v            v
                     +----------- repair <----------+

``validate`` is deterministic (:mod:`cae.validators.rue_task`). ``critique`` is a
second model pass for Parts 5-8, where a unique answer cannot be checked in code.

Every model call here (generate, critique, repair) goes to the ``rue_task`` tier
(the generation chain, ``CAE_GENERATION_MODEL``, unless ``CAE_RUE_TASK_MODEL``
overrides it), so the whole graph runs on one chain.
"""

from __future__ import annotations

import json
import random
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from cae.config import get_settings
from cae.db import repo
from cae.db.session import session_scope
from cae.domain.parts import get_part_spec
from cae.llm import structured_call
from cae.observability import graph_config, note_degraded
from cae.prompts.rue_task import (
    TOPICS,
    critique_prompt,
    generation_prompt,
    repair_prompt,
)
from cae.schemas.rue_task import ModerationResult, parse_exercise, schema_for
from cae.validators.rue_task import validate_exercise

CRITIQUE_PARTS: frozenset[int] = frozenset({5, 6, 7, 8})


class GenerationState(TypedDict, total=False):
    part: int
    topic: str
    focus: str
    weaknesses: list[str]
    exercise: dict[str, Any]
    issues: list[str]
    attempts: int
    max_attempts: int
    critique_rounds: int
    status: str
    exercise_id: int | None
    persist: bool


def plan_task(state: GenerationState) -> dict[str, Any]:
    """Pick topic and testing focus for this exercise."""
    spec = get_part_spec(state["part"])
    focus = state.get("focus") or ""
    if not focus:
        picks = random.sample(spec.focus, k=min(2, len(spec.focus)))
        focus = "; ".join(picks)
    return {
        "topic": state.get("topic") or random.choice(TOPICS),
        "focus": focus,
        "attempts": 0,
        "critique_rounds": 0,
        "issues": [],
        "status": "drafting",
    }


def generate(state: GenerationState) -> dict[str, Any]:
    part = state["part"]
    system, user = generation_prompt(
        part,
        topic=state["topic"],
        focus=state["focus"],
        weaknesses=state.get("weaknesses") or None,
    )
    draft = structured_call(
        schema_for(part),
        system,
        user,
        tier="rue_task",
        temperature=0.9,
        label="rue_task.generate",
    )
    return {"exercise": draft.model_dump(), "attempts": state.get("attempts", 0) + 1}


def validate(state: GenerationState) -> dict[str, Any]:
    part = state["part"]
    try:
        exercise = parse_exercise(part, state["exercise"])
    except Exception as exc:
        return {"issues": [f"The task does not match the required format: {exc}"]}
    return {"issues": validate_exercise(part, exercise)}


def critique(state: GenerationState) -> dict[str, Any]:
    """Ask the model to moderate its own draft. Only for Parts 5-8."""
    part = state["part"]
    task_json = json.dumps(state["exercise"], ensure_ascii=False, indent=2)
    system, user = critique_prompt(part, task_json)
    try:
        result = structured_call(
            ModerationResult,
            system,
            user,
            tier="rue_task",
            temperature=0.2,
            label="rue_task.critique",
        )
    except Exception as exc:
        note_degraded("rue_task.critique", exc, part=part)
        return {"critique_rounds": state.get("critique_rounds", 0) + 1, "issues": []}
    issues = list(result.issues) if result.verdict == "revise" else []
    return {"critique_rounds": state.get("critique_rounds", 0) + 1, "issues": issues}


def repair(state: GenerationState) -> dict[str, Any]:
    part = state["part"]
    system, _ = generation_prompt(
        part,
        topic=state["topic"],
        focus=state["focus"],
    )
    previous = json.dumps(state["exercise"], ensure_ascii=False, indent=2)
    user = repair_prompt(state.get("issues", []), previous)
    fixed = structured_call(
        schema_for(part),
        system,
        user,
        tier="rue_task",
        temperature=0.4,
        label="rue_task.repair",
    )
    return {"exercise": fixed.model_dump(), "attempts": state.get("attempts", 0) + 1}


def persist(state: GenerationState) -> dict[str, Any]:
    status = "ok_with_warnings" if state.get("issues") else "ok"
    if not state.get("persist", True):
        return {"status": status}
    with session_scope() as session:
        exercise = repo.save_exercise(
            session,
            part=state["part"],
            payload=state["exercise"],
            topic=state.get("topic", ""),
            focus=state.get("focus", ""),
            attempts=state.get("attempts", 1),
        )
        return {"exercise_id": exercise.id, "status": status}


def _after_validate(state: GenerationState) -> str:
    issues = state.get("issues") or []
    if issues:
        return "repair" if state.get("attempts", 0) < state["max_attempts"] else "give_up"
    if state["part"] in CRITIQUE_PARTS and state.get("critique_rounds", 0) == 0:
        return "critique"
    return "persist"


def _after_critique(state: GenerationState) -> str:
    issues = state.get("issues") or []
    if not issues:
        return "persist"
    if state.get("attempts", 0) < state["max_attempts"]:
        return "repair"
    return "persist"


def give_up(state: GenerationState) -> dict[str, Any]:
    return {"status": "failed"}


def build_generation_graph():
    graph = StateGraph(GenerationState)
    graph.add_node("plan_task", plan_task)
    graph.add_node("generate", generate)
    graph.add_node("validate", validate)
    graph.add_node("critique", critique)
    graph.add_node("repair", repair)
    graph.add_node("persist", persist)
    graph.add_node("give_up", give_up)

    graph.add_edge(START, "plan_task")
    graph.add_edge("plan_task", "generate")
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges(
        "validate",
        _after_validate,
        {"repair": "repair", "critique": "critique", "persist": "persist", "give_up": "give_up"},
    )
    graph.add_conditional_edges(
        "critique", _after_critique, {"repair": "repair", "persist": "persist"}
    )
    graph.add_edge("repair", "validate")
    graph.add_edge("persist", END)
    graph.add_edge("give_up", END)
    return graph.compile()


_GRAPH = None


def get_generation_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_generation_graph()
    return _GRAPH


class GenerationFailed(RuntimeError):
    def __init__(self, part: int, issues: list[str]):
        self.part = part
        self.issues = issues
        super().__init__(
            f"Could not generate a valid Part {part} exercise. Remaining problems: "
            + "; ".join(issues[:5])
        )


def generate_exercise(
    part: int,
    *,
    topic: str = "",
    focus: str = "",
    weaknesses: list[str] | None = None,
    persist_result: bool = True,
    max_attempts: int | None = None,
    trace_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate one validated exercise.

    Returns the final state, including ``exercise`` (the payload),
    ``exercise_id`` when persisted, ``attempts`` and any residual ``issues``
    the moderator raised.
    """
    settings = get_settings()
    initial: GenerationState = {
        "part": part,
        "topic": topic,
        "focus": focus,
        "weaknesses": weaknesses or [],
        "max_attempts": max_attempts or settings.cae_max_generation_attempts,
        "persist": persist_result,
    }
    config = graph_config(
        {"graph": "rue_task", "part": part, **(trace_metadata or {})},
        recursion_limit=50,
        run_name=f"rue_task.part{part}",
    )
    final = get_generation_graph().invoke(initial, config)
    if final.get("status") == "failed":
        raise GenerationFailed(part, final.get("issues", []))
    return final
