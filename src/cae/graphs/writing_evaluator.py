"""Writing assessment graph.

                 +-> assess_content ------------+
                 +-> assess_communicative ------+
    START -------+-> assess_organisation -------+--> aggregate -> improve -> END
       |         +-> assess_language -----------+
       |         +-> annotate ------------------+
       +-> out_of_range -> END

Each subscale is marked by its own concurrent call, so strength in one cannot
inflate another. Every model node runs on its own ``writing_evaluator`` tier
(the marking chain, ``CAE_MARKING_MODEL``, shared with Reading marking).

Only a script within the required word range is assessed. The app never hands
in anything else (except a blank task when the mock exam's clock runs out), and
within the range the length says nothing about any subscale, so the examiners
are not told it. A script outside the range scores zero with no model call.
"""

from __future__ import annotations

import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from cae.domain.parts import SUBSCALES, WRITING_WORD_RANGE
from cae.domain.scoring import scale_score, summarise_writing, writing_total
from cae.llm import structured_call
from cae.observability import graph_config, note_degraded
from cae.prompts.writing_evaluator import annotate_prompt, assess_prompt, improve_prompt
from cae.schemas.writing_evaluator import AnnotationSet, ImprovedVersion, SubscaleAssessment

MIN_WORDS, MAX_WORDS = WRITING_WORD_RANGE

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


class WritingState(TypedDict, total=False):
    part: int
    genre: str
    task: dict[str, Any]
    script: str
    content: dict[str, Any]
    communicative_achievement: dict[str, Any]
    organisation: dict[str, Any]
    language: dict[str, Any]
    annotations: list[dict[str, Any]]
    annotate_failed: bool
    strengths: list[str]
    bands: dict[str, int]
    total: int
    scale_score: int
    grade: str
    improved: dict[str, Any]
    status: str
    message: str


def _assess(subscale: str):
    """Build the node that marks one subscale."""

    def node(state: WritingState) -> dict[str, Any]:
        system, user = assess_prompt(
            subscale,
            part=state["part"],
            genre=state["genre"],
            task=state["task"],
            script=state["script"],
        )
        try:
            result = structured_call(
                SubscaleAssessment,
                system,
                user,
                tier="writing_evaluator",
                temperature=0.2,
                label=f"writing_evaluator.assess.{subscale}",
            )
            return {subscale: result.model_dump()}
        except Exception as exc:
            note_degraded(f"writing_evaluator.assess.{subscale}", exc)
            return {
                subscale: {
                    "band": 0,
                    "justification": f"This subscale could not be assessed: {exc}",
                    "evidence": [],
                    "to_improve": "",
                    "failed": True,
                }
            }

    node.__name__ = f"assess_{subscale}"
    return node


def annotate(state: WritingState) -> dict[str, Any]:
    system, user = annotate_prompt(state["genre"], state["script"])
    try:
        result = structured_call(
            AnnotationSet,
            system,
            user,
            tier="writing_evaluator",
            temperature=0.1,
            label="writing_evaluator.annotate",
        )
    except Exception as exc:
        note_degraded("writing_evaluator.annotate", exc, genre=state["genre"])
        return {"annotations": [], "strengths": [], "annotate_failed": True}

    script = state["script"]
    annotations: list[dict[str, Any]] = []
    for item in result.annotations:
        row = item.model_dump()
        row["category"] = str(item.category)
        row["found"] = item.span.strip() != "" and item.span in script
        annotations.append(row)
    return {"annotations": annotations, "strengths": list(result.strengths)}


def out_of_range(state: WritingState) -> dict[str, Any]:
    words = count_words(state["script"])
    zero = {
        "band": 0,
        "justification": (
            f"Only answers of {MIN_WORDS}-{MAX_WORDS} words are assessed; this one "
            f"has {words}."
        ),
        "evidence": [],
        "to_improve": f"Write a full answer of {MIN_WORDS}-{MAX_WORDS} words.",
    }
    bands = dict.fromkeys(SUBSCALES, 0)
    return {
        **{s: dict(zero) for s in SUBSCALES},
        "annotations": [],
        "strengths": [],
        "bands": bands,
        "total": 0,
        "scale_score": scale_score(0.0),
        "grade": "Below B2",
        "improved": {},
        "status": "out_of_range",
        "message": (
            f"This answer has {words} words, outside the required {MIN_WORDS}-"
            f"{MAX_WORDS}, so it was not marked and scores zero."
        ),
    }


def aggregate(state: WritingState) -> dict[str, Any]:
    bands = {s: int(state.get(s, {}).get("band", 0)) for s in SUBSCALES}
    summary = summarise_writing(bands)
    return {
        "bands": bands,
        "total": writing_total(bands),
        "scale_score": summary.scale_score,
        "grade": summary.grade,
        "status": "assessed",
    }


def improve(state: WritingState) -> dict[str, Any]:
    bands = state["bands"]
    weakest = min(SUBSCALES, key=lambda s: bands.get(s, 0))
    system, user = improve_prompt(state["genre"], state["script"], weakest)
    try:
        result = structured_call(
            ImprovedVersion,
            system,
            user,
            tier="writing_evaluator",
            temperature=0.5,
            label="writing_evaluator.improve",
        )
        return {"improved": result.model_dump()}
    except Exception as exc:
        note_degraded("writing_evaluator.improve", exc, weakest=weakest)
        return {"improved": {}}


_ASSESSOR_NODES = [f"assess_{s}" for s in SUBSCALES] + ["annotate"]


def _route(state: WritingState) -> list[str]:
    if not MIN_WORDS <= count_words(state["script"]) <= MAX_WORDS:
        return ["out_of_range"]
    return list(_ASSESSOR_NODES)


def build_writing_graph():
    graph = StateGraph(WritingState)
    for subscale in SUBSCALES:
        graph.add_node(f"assess_{subscale}", _assess(subscale))
    graph.add_node("annotate", annotate)
    graph.add_node("out_of_range", out_of_range)
    graph.add_node("aggregate", aggregate)
    graph.add_node("improve", improve)

    graph.add_conditional_edges(START, _route, _ASSESSOR_NODES + ["out_of_range"])
    for node in _ASSESSOR_NODES:
        graph.add_edge(node, "aggregate")
    graph.add_edge("aggregate", "improve")
    graph.add_edge("improve", END)
    graph.add_edge("out_of_range", END)
    return graph.compile()


_GRAPH = None


def get_writing_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_writing_graph()
    return _GRAPH


def assess_writing(
    *,
    part: int,
    genre: str,
    task: dict[str, Any],
    script: str,
    trace_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assess one writing script against the four official subscales."""
    config = graph_config(
        {"graph": "writing_evaluator", "part": part, "genre": genre, **(trace_metadata or {})},
        run_name=f"writing_evaluator.part{part}",
    )
    return get_writing_graph().invoke(
        {"part": part, "genre": genre, "task": task, "script": script}, config
    )
