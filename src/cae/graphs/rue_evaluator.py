"""Grading graph for Reading & Use of English.

    normalise -> score -> [adjudicate] -> [tag_errors] -> summarise -> END

Both model nodes run on their own ``rue_evaluator`` tier
(the marking chain, ``CAE_MARKING_MODEL``: the Flash models, most capable first).

Scoring is deterministic. The model only adjudicates near misses in Parts 2-4
and categorises dropped marks in Parts 1-4, so a perfect script needs no API
calls, and neither does any Reading part (5-8): each has one fixed category, and
its diagnosis is built from the exercise itself.

There is no fallback. When the model is needed and cannot answer, or answers
for only some of the items, the attempt is not marked at all -- the way
``rue_task`` gives up rather than hand out an exercise it could not repair. A
deterministic stand-in would score a near miss as wrong and file the mistake
under a guessed category, and nothing downstream could tell it from a judgement.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from cae.domain.answers import (
    ADJUDICATED_PARTS,
    LETTER_PARTS,
    MatchResult,
    apply_adjudication,
    grade_deterministically,
)
from cae.domain.parts import gap_marker, get_part_spec
from cae.domain.taxonomy import ErrorCategory, categories_for_part
from cae.llm import structured_call
from cae.observability import graph_config
from cae.prompts.rue_evaluator import adjudicate_prompt, tag_prompt
from cae.schemas.rue_task import AnyExercise, KeyItem, answer_key, choice_text, parse_exercise
from cae.schemas.rue_evaluator import AdjudicationBatch, error_tag_batch


class MarkingFailed(RuntimeError):
    """The model answered, but not for every item it was asked about.

    A model that could not be reached raises ``LLMUnavailableError`` instead; both
    leave the attempt unmarked and unsaved.
    """

    def __init__(self, part: int, problem: str):
        self.part = part
        self.problem = problem
        super().__init__(f"Part {part} could not be marked: {problem}")


class GradingState(TypedDict, total=False):
    part: int
    exercise: dict[str, Any]
    responses: dict[int, str]
    key: list[KeyItem]
    results: list[MatchResult]
    diagnoses: dict[int, dict[str, str]]
    marks: float
    max_marks: float
    rows: list[dict[str, Any]]


def gap_context(text: str, number: int, window: int = 140) -> str:
    """The stretch of text around a gap, so the adjudicator can see the structure."""
    marker = gap_marker(number)
    index = text.find(marker)
    if index < 0:
        return ""
    start = max(0, index - window)
    end = min(len(text), index + len(marker) + window)
    snippet = text[start:end].replace("\n", " ")
    return ("..." if start > 0 else "") + snippet.strip() + ("..." if end < len(text) else "")


def item_context(exercise: dict[str, Any], key: KeyItem, *, window: int = 140) -> str:
    """The question an item asked: its own context, or the text around its gap."""
    if key.context:
        return key.context
    return gap_context(str(exercise.get("text", "")), key.number, window=window)


EVIDENCE_WINDOW = 90
EVIDENCE_CONTEXT_LIMIT = 220
LETTER_TEXT_LIMIT = 160
DIAGNOSIS_QUOTE_LIMIT = 100


def _shorten(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def error_evidence(exercise: dict[str, Any], key: KeyItem, result: MatchResult) -> str:
    """The mistake in enough context to still mean something months later.

    ``user_answer`` alone cannot carry an error event: in the letter parts it is a
    bare "C", and it is empty whenever the item was left blank. The coach reads
    these back through :func:`cae.db.repo.error_examples`, so pair the answer with
    the question that was asked and the key that was missed. The writing side
    stores the offending span of the learner's own text for the same reason.
    """
    written = result.user_answer.strip() or "(blank)"
    answers = f'wrote "{written}", key "{key.answer}"'
    context = _shorten(item_context(exercise, key, window=EVIDENCE_WINDOW), EVIDENCE_CONTEXT_LIMIT)
    return f"{context} — {answers}" if context else answers


def _listed(numbers: list[int]) -> str:
    return ", ".join(str(n) for n in numbers)


def _lettered(part: int, exercise: AnyExercise, number: int, letter: str) -> str:
    """A letter answer with what it stood for: ``C — arrive``."""
    text = choice_text(part, exercise, number, letter)
    return f"{letter} — {_shorten(text, LETTER_TEXT_LIMIT)}" if text else letter


def _items_block(
    part: int,
    exercise: dict[str, Any],
    key_by_number: dict[int, KeyItem],
    results: list[MatchResult],
    *,
    include_verdict: bool = False,
) -> str:
    parsed = parse_exercise(part, exercise) if part in LETTER_PARTS else None
    blocks: list[str] = []
    for result in results:
        key = key_by_number[result.number]
        lines = [f"Item {result.number}"]
        context = item_context(exercise, key)
        if context:
            lines.append(f"  Context: {context}")
        if key.key_word:
            lines.append(f"  Key word: {key.key_word.upper()}")
        if parsed is not None:
            lines.append(f"  Published key: {_lettered(part, parsed, key.number, key.answer)}")
            chosen = result.user_answer
            lines.append(
                f"  Candidate chose: {_lettered(part, parsed, key.number, chosen) if chosen else '(blank)'}"
            )
        else:
            lines.append(f"  Published key: {', '.join(key.variants)}")
            lines.append(f"  Candidate wrote: {result.user_answer or '(blank)'}")
        if include_verdict and result.reason:
            lines.append(f"  Automatic marking said: {result.reason}")
        if key.explanation:
            lines.append(f"  Why the key is right: {key.explanation}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


_CHOICE_NOUNS = {5: "option", 6: "text", 7: "paragraph", 8: "section"}


def _numbers(values: list[int]) -> str:
    """``3``, ``3 and 5``, ``2, 3 and 5``."""
    words = [str(v) for v in values]
    return words[0] if len(words) == 1 else f"{', '.join(words[:-1])} and {words[-1]}"


def choice_diagnosis(
    part: int,
    exercise: AnyExercise,
    key: KeyItem,
    chosen: str,
    answers: dict[int, str],
) -> str:
    """Why a Reading answer lost its marks, from the exercise alone.

    The feedback card already shows the key's letter, its evidence and its
    explanation; what it cannot show is what the learner's own letter stood for,
    or that it is the answer to another item -- the commonest Part 6-8 slip.
    ``answers`` is every item's key, by number.
    """
    noun = _CHOICE_NOUNS[part]

    def named(letter: str) -> str:
        text = choice_text(part, exercise, key.number, letter).strip().rstrip(".;:")
        text = _shorten(text, DIAGNOSIS_QUOTE_LIMIT)
        if not text:
            return letter
        return f'{letter}: "{text}"' if part in (5, 7) else f"{letter} ({text})"

    chosen = (chosen or "").strip().upper()
    if not chosen:
        opening = "You left this one blank."
    else:
        opening = f"You chose {noun} {named(chosen)}."
        elsewhere = sorted(
            n for n, letter in answers.items() if letter == chosen and n != key.number
        )
        if part == 7 and chosen == exercise.distractor_letter:
            opening += " That is the one paragraph that fits none of the gaps."
        elif part == 7 and elsewhere:
            opening += f" It belongs in gap {_numbers(elsewhere)}."
        elif part in (6, 8) and elsewhere:
            questions = "question" if len(elsewhere) == 1 else "questions"
            opening += f" It is the answer to {questions} {_numbers(elsewhere)}."
    return f"{opening} The answer is {noun} {named(key.answer)}."


def normalise(state: GradingState) -> dict[str, Any]:
    part = state["part"]
    exercise = parse_exercise(part, state["exercise"])
    responses: dict[int, str] = {}
    for raw_number, value in (state.get("responses") or {}).items():
        try:
            responses[int(raw_number)] = "" if value is None else str(value)
        except (TypeError, ValueError):
            continue
    return {"key": answer_key(part, exercise), "responses": responses}


def score(state: GradingState) -> dict[str, Any]:
    results = grade_deterministically(state["part"], state["key"], state["responses"])
    return {"results": results}


def adjudicate(state: GradingState) -> dict[str, Any]:
    """Ask the examiner model about answers the key does not cover."""
    part = state["part"]
    results = state["results"]
    pending = [r for r in results if r.needs_adjudication]
    if not pending:
        return {}
    key_by_number = {k.number: k for k in state["key"]}
    system, user = adjudicate_prompt(
        part, _items_block(part, state["exercise"], key_by_number, pending)
    )
    batch = structured_call(
        AdjudicationBatch,
        system,
        user,
        tier="rue_evaluator",
        temperature=0.0,
        label="rue_evaluator.adjudicate",
    )

    verdicts = {d.number: d for d in batch.decisions}
    skipped = [r.number for r in pending if r.number not in verdicts]
    if skipped:
        raise MarkingFailed(part, f"the examiner gave no verdict on items {_listed(skipped)}")
    for result in pending:
        decision = verdicts[result.number]
        result.needs_adjudication = False
        apply_adjudication(result, decision.verdict)
        if decision.reason:
            result.reason = decision.reason
    return {"results": results}


def tag_errors(state: GradingState) -> dict[str, Any]:
    """Give every dropped mark a category from the closed taxonomy."""
    part = state["part"]
    results = state["results"]
    wrong = [r for r in results if r.marks < r.max_marks]
    if not wrong:
        return {"diagnoses": {}}
    key_by_number = {k.number: k for k in state["key"]}

    categories = categories_for_part(part)
    if len(categories) == 1:
        (only,) = categories
        exercise = parse_exercise(part, state["exercise"])
        answers = {k.number: k.answer for k in state["key"]}
        return {
            "diagnoses": {
                result.number: {
                    "category": only.value,
                    "diagnosis": choice_diagnosis(
                        part, exercise, key_by_number[result.number], result.user_answer, answers
                    ),
                }
                for result in wrong
            }
        }

    system, user = tag_prompt(
        part,
        _items_block(part, state["exercise"], key_by_number, wrong, include_verdict=True),
    )
    batch = structured_call(
        error_tag_batch(part),
        system,
        user,
        tier="rue_evaluator",
        temperature=0.1,
        label="rue_evaluator.tag_errors",
    )
    tags = {t.number: t for t in batch.tags}

    untagged = [r.number for r in wrong if r.number not in tags]
    if untagged:
        raise MarkingFailed(part, f"no error category for items {_listed(untagged)}")
    allowed = {c.value for c in categories}
    outside = [r.number for r in wrong if str(tags[r.number].category) not in allowed]
    if outside:
        raise MarkingFailed(part, f"items {_listed(outside)} were given a category Part {part} does not use")
    return {
        "diagnoses": {
            result.number: {
                "category": ErrorCategory(str(tags[result.number].category)).value,
                "diagnosis": tags[result.number].diagnosis,
            }
            for result in wrong
        }
    }


def summarise(state: GradingState) -> dict[str, Any]:
    """Assemble the rows the repository stores and the UI renders."""
    part = state["part"]
    spec = get_part_spec(part)
    key_by_number = {k.number: k for k in state["key"]}
    diagnoses = state.get("diagnoses") or {}

    rows: list[dict[str, Any]] = []
    for result in state["results"]:
        key = key_by_number[result.number]
        diagnosis = diagnoses.get(result.number, {})
        explanation = key.explanation
        if diagnosis.get("diagnosis"):
            explanation = f"{key.explanation}\n\n{diagnosis['diagnosis']}".strip()
        rows.append(
            {
                "item_number": result.number,
                "user_answer": result.user_answer,
                "correct_answer": " / ".join(key.variants),
                "is_correct": result.is_correct,
                "marks_awarded": result.marks,
                "max_marks": result.max_marks,
                "category": diagnosis.get("category"),
                "explanation": explanation,
                "adjudicated": result.adjudicated,
                "reason": result.reason,
                "evidence": error_evidence(state["exercise"], key, result),
            }
        )
    marks = sum(r["marks_awarded"] for r in rows)
    return {"rows": rows, "marks": marks, "max_marks": float(spec.max_marks)}


def _after_score(state: GradingState) -> str:
    if state["part"] in ADJUDICATED_PARTS and any(
        r.needs_adjudication for r in state["results"]
    ):
        return "adjudicate"
    return "tag_errors"


def build_grading_graph():
    graph = StateGraph(GradingState)
    graph.add_node("normalise", normalise)
    graph.add_node("score", score)
    graph.add_node("adjudicate", adjudicate)
    graph.add_node("tag_errors", tag_errors)
    graph.add_node("summarise", summarise)

    graph.add_edge(START, "normalise")
    graph.add_edge("normalise", "score")
    graph.add_conditional_edges(
        "score", _after_score, {"adjudicate": "adjudicate", "tag_errors": "tag_errors"}
    )
    graph.add_edge("adjudicate", "tag_errors")
    graph.add_edge("tag_errors", "summarise")
    graph.add_edge("summarise", END)
    return graph.compile()


_GRAPH = None


def get_grading_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_grading_graph()
    return _GRAPH


def grade_attempt(
    part: int,
    exercise: dict[str, Any],
    responses: dict[int | str, str],
    *,
    trace_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Grade one exercise. Returns ``rows``, ``marks`` and ``max_marks``.

    Raises ``LLMUnavailableError`` when the model is needed and cannot be
    reached, and :class:`MarkingFailed` when it leaves an item unanswered.
    """
    config = graph_config(
        {"graph": "rue_evaluator", "part": part, **(trace_metadata or {})},
        run_name=f"rue_evaluator.part{part}",
    )
    return get_grading_graph().invoke(
        {"part": part, "exercise": exercise, "responses": dict(responses)}, config
    )
