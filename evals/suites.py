"""Evaluation suites. Each returns a plain dict, ready for JSON or LangSmith.

Generation and writing_task use fixed topic seeds; the writing_evaluator suite marks every
script against the task fixed in its dataset, and the coach suite seeds each
synthetic learner into an in-memory database and drives the graph's nodes
directly, so neither ``cae.db`` nor the report cache is touched. The chat suite
records its synthetic learners into a scratch database the whole app is pointed
at, and runs the chat graph unchanged against it. Both are marked by the same
paid judge (``evals.judge``); for the coach's report it is the only measure.

Every case row carries what the model actually wrote — the exercise, the
examiner's feedback, the chat reply — next to its metrics, so a person can judge
it in LangSmith's comparison view. With tracing on, each case is also one trace
(``eval.<suite>``) holding all its model calls, and the row keeps its id.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Mapping, Sequence

from cae.config import TIER_SETTINGS, get_settings
from cae.domain.parts import SUBSCALES
from cae.observability import tracing_enabled

from evals import load_dataset
from evals import metrics


def _timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    value = fn()
    return value, round(time.perf_counter() - started, 3)


def _case_id(row: Mapping[str, Any]) -> str:
    return str(row.get("id") or f"part{row.get('part')}")


@contextmanager
def _case_trace(
    suite: str, row: dict[str, Any], inputs: Mapping[str, Any]
) -> Iterator[None]:
    """Group one case's model calls under a single trace, and link ``row`` to it.

    LangGraph runs started inside nest under this one, so the per-node prompts
    and replies are one click from the row. On the way out the row becomes the
    trace's outputs and gains ``trace_id``. Without tracing it does nothing.

    Only the id is kept: resolving a URL costs a request per case, which is slow
    against an unreachable LangSmith. ``upload`` turns ids into links.
    """
    if not tracing_enabled():
        yield
        return

    import langsmith as ls

    settings = get_settings()
    metadata = {
        "eval_suite": suite,
        "eval_case": _case_id(row),
        **{f"cae_{tier}_chain": settings.chain(tier) for tier in TIER_SETTINGS},
    }
    with ls.trace(
        f"eval.{suite}",
        inputs={"case": _case_id(row), **inputs},
        metadata=metadata,
        tags=["eval", f"suite:{suite}"],
    ) as run:
        try:
            yield
        finally:
            run.add_outputs(dict(row))
            row["trace_id"] = str(run.trace_id)


def _verdict_from_row(row: dict[str, Any]) -> str:
    """The adjudication verdict implied by a graded row's marks."""
    marks = float(row.get("marks_awarded", 0.0))
    maximum = float(row.get("max_marks", 0.0) or 0.0)
    if maximum and marks >= maximum:
        return "accept"
    if marks > 0:
        return "partial"
    return "reject"


def _marked_items(
    rows: Sequence[dict[str, Any]],
    verdicts: Mapping[int, str],
    expected_verdicts: Mapping[int, str],
    expected_categories: Mapping[int, str],
    routes: Mapping[int, str] | None = None,
) -> list[dict[str, Any]]:
    """The items the model had a say in, beside what a person expected of it.

    A right answer the deterministic pass settled is left out: there is nothing
    of the model's in it to judge. ``routes`` says how each item reached the
    adjudicator (see :func:`adjudication_routes`).
    """
    items: list[dict[str, Any]] = []
    for r in rows:
        number = r["item_number"]
        if not (r.get("adjudicated") or r.get("category") or number in expected_verdicts):
            continue
        items.append(
            {
                "item": number,
                "answer": r.get("user_answer"),
                "key": r.get("correct_answer"),
                "verdict": verdicts.get(number),
                "expected_verdict": expected_verdicts.get(number),
                "reason": r.get("reason"),
                "category": r.get("category"),
                "expected_category": expected_categories.get(number),
                "explanation": r.get("explanation"),
                **({"route": routes[number]} if routes and number in routes else {}),
            }
        )
    return items


def adjudication_routes(
    part: int, exercise: Mapping[str, Any], responses: Mapping[int, str]
) -> dict[int, str]:
    """How each Part 4 answer reached the adjudicator: ``close`` to the key, with a
    provisional half mark, or ``different`` from it, starting from zero.

    The two start from different marks, so a failed or timid adjudicator errs
    differently on each, and the suite reports them apart. Pure Python: it re-runs
    the deterministic pass, which costs nothing.
    """
    if part != 4:
        return {}
    from cae.domain.answers import grade_deterministically
    from cae.schemas.rue_task import answer_key, parse_exercise

    key = answer_key(part, parse_exercise(part, dict(exercise)))
    return {
        r.number: "close" if r.marks > 0 else "different"
        for r in grade_deterministically(part, key, dict(responses))
        if r.needs_adjudication
    }


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def generation_groups() -> list[int]:
    """The seed groups ``generation_seeds.json`` defines, for ``--group``."""
    return sorted({s["group"] for s in load_dataset("generation_seeds")["seeds"]})


def generation_case_id(part: int, group: int) -> str:
    return f"part{part}-g{group}"


def run_generation(
    parts: Sequence[int] | None = None, group: int | None = None
) -> dict[str, Any]:
    """Generate one exercise per seed; check validity, attempts and key consistency.

    ``group`` runs one group of seeds (one per part); ``None`` runs them all.
    """
    from cae.graphs.rue_task import generate_exercise

    if group is not None and group not in generation_groups():
        raise ValueError(f"No seed group {group}; the groups are {generation_groups()}.")
    dataset = load_dataset("generation_seeds")
    seeds = [
        s
        for s in dataset["seeds"]
        if (parts is None or s["part"] in parts) and (group is None or s["group"] == group)
    ]

    cases: list[dict[str, Any]] = []
    for seed in seeds:
        part = seed["part"]
        focus = "; ".join(seed["focus"])
        row: dict[str, Any] = {
            "id": generation_case_id(part, seed["group"]),
            "part": part,
            "group": seed["group"],
            "topic": seed["topic"],
            "focus": focus,
        }
        inputs = {"part": part, "group": seed["group"], "topic": seed["topic"], "focus": focus}
        with _case_trace("generation", row, inputs):
            try:
                result, elapsed = _timed(
                    lambda: generate_exercise(
                        part,
                        topic=seed["topic"],
                        focus=focus,
                        persist_result=False,
                    )
                )
            except Exception as exc:
                row.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
                cases.append(row)
                continue

            payload = result["exercise"]
            row.update(
                {
                    "ok": True,
                    "seconds": elapsed,
                    "attempts": result.get("attempts", 1),
                    "status": result.get("status"),
                    "title": payload.get("title"),
                    "moderator_issues": result.get("issues") or [],
                    "exercise": payload,
                }
            )
            try:
                row["self_consistency"] = metrics.key_self_consistency(part, payload)
            except Exception as exc:
                row["self_consistency"] = {"error": f"{type(exc).__name__}: {exc}"}
        cases.append(row)

    return {
        "suite": "generation",
        "group": group,
        "cases": cases,
        "metrics": generation_metrics(cases),
    }


def generation_metrics(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The suite's metrics over any set of cases: one run, or several combined."""
    good = [c for c in cases if c.get("ok")]
    consistent = [
        c for c in good if (c.get("self_consistency") or {}).get("consistent")
    ]
    return {
        **metrics.validity_rate(cases),
        "mean_attempts": metrics.mean_attempts(good),
        "mean_seconds": _mean([c["seconds"] for c in good]),
        "key_self_consistent": len(consistent),
        "key_self_consistency_rate": (
            round(len(consistent) / len(good), 3) if good else 0.0
        ),
    }


def writing_task_groups() -> list[int]:
    """The seed groups ``writing_task_seeds.json`` defines, for ``--group``."""
    return sorted({s["group"] for s in load_dataset("writing_task_seeds")["seeds"]})


def writing_task_case_id(genre: str, group: int) -> str:
    return f"{genre}-g{group}"


def run_writing_task(
    genres: Sequence[str] | None = None, group: int | None = None
) -> dict[str, Any]:
    """Generate one Writing task per seed; count the repairs each one needed.

    ``group`` runs one group of seeds (one per genre); ``None`` runs them all.

    Each row keeps what ``validate`` said on every pass (``issue_history``), so a
    repair can be traced to the rule that asked for it. A task the graph could
    not repair within its budget is ``failed`` and still carries its repairs; one
    no model was there for (``LLMUnavailableError``) is ``unavailable`` and
    measures the provider, so it is left out of the rates.
    """
    from cae.graphs.writing_task import WritingTaskFailed, generate_writing_task
    from cae.llm import LLMUnavailableError

    if group is not None and group not in writing_task_groups():
        raise ValueError(f"No seed group {group}; the groups are {writing_task_groups()}.")
    seeds = [
        s
        for s in load_dataset("writing_task_seeds")["seeds"]
        if (genres is None or s["genre"] in genres) and (group is None or s["group"] == group)
    ]

    cases: list[dict[str, Any]] = []
    for seed in seeds:
        row: dict[str, Any] = {
            "id": writing_task_case_id(seed["genre"], seed["group"]),
            "part": seed["part"],
            "genre": seed["genre"],
            "group": seed["group"],
            "topic": seed["topic"],
        }
        inputs = {k: seed[k] for k in ("part", "genre", "group", "topic")}
        with _case_trace("writing_task", row, inputs):
            try:
                result, elapsed = _timed(
                    lambda: generate_writing_task(
                        part=seed["part"],
                        genre=seed["genre"],
                        topic=seed["topic"],
                        persist_result=False,
                    )
                )
            except WritingTaskFailed as exc:
                row.update(
                    {
                        "ok": False,
                        "failed": True,
                        "attempts": exc.attempts,
                        "repairs": max(exc.attempts - 1, 0),
                        "issue_history": exc.issue_history,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                cases.append(row)
                continue
            except Exception as exc:
                status = "unavailable" if isinstance(exc, LLMUnavailableError) else "failed"
                row.update({"ok": False, status: True, "error": f"{type(exc).__name__}: {exc}"})
                cases.append(row)
                continue

            attempts = result.get("attempts", 1)
            row.update(
                {
                    "ok": True,
                    "seconds": elapsed,
                    "attempts": attempts,
                    "repairs": attempts - 1,
                    "issue_history": result.get("issue_history") or [],
                    "title": result["task"].get("title"),
                    "task": result["task"],
                }
            )
        cases.append(row)

    return {
        "suite": "writing_task",
        "group": group,
        "cases": cases,
        "metrics": writing_task_metrics(cases),
    }


def writing_task_metrics(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The suite's metrics over any set of cases: one run, or several combined.

    Rates are over the cases a model answered; ``unavailable`` ones are only
    counted. ``mean_repairs`` includes the tasks that ran out of budget, which
    used every repair they had, so a model that fails often cannot look frugal.
    """
    answered = [c for c in cases if not c.get("unavailable")]
    good = [c for c in answered if c.get("ok")]
    counted = [c for c in answered if "repairs" in c]
    first_try = [c for c in good if c["repairs"] == 0]

    by_genre: dict[str, dict[str, Any]] = {}
    for genre in sorted({c["genre"] for c in answered}):
        tried = [c for c in answered if c["genre"] == genre]
        with_repairs = [c["repairs"] for c in tried if "repairs" in c]
        by_genre[genre] = {
            "valid": sum(1 for c in tried if c.get("ok")),
            "tried": len(tried),
            "mean_repairs": _mean(with_repairs) if with_repairs else None,
        }

    return {
        **metrics.validity_rate(answered),
        "unavailable": len(cases) - len(answered),
        "first_try": len(first_try),
        "first_try_rate": round(len(first_try) / len(answered), 3) if answered else 0.0,
        "mean_repairs": _mean([c["repairs"] for c in counted]) if counted else None,
        "total_repairs": sum(c["repairs"] for c in counted),
        "repairs_histogram": metrics.repairs_histogram(answered),
        "mean_attempts": metrics.mean_attempts(good),
        "mean_seconds": _mean([c["seconds"] for c in good]),
        "by_genre": by_genre,
        "repair_reasons": metrics.repair_reasons(c.get("issue_history") or [] for c in answered),
    }


def marking_groups() -> list[int]:
    """The case groups ``reading_cases.json`` defines, for ``--group``."""
    return sorted({c["group"] for c in load_dataset("reading_cases")["cases"]})


def run_marking(
    case_ids: Sequence[str] | None = None, group: int | None = None
) -> dict[str, Any]:
    """Mark the hand-authored near-miss papers; score adjudication and tagging.

    ``group`` runs one group of papers (four, one per part); ``None`` runs them
    all.

    The grader has no fallback, so a paper the model did not mark has nothing to
    score. One it was not there for (``LLMUnavailableError``) is ``unavailable``:
    that measures the provider. One it answered unusably -- items left without a
    verdict or a category, or a reply no schema could read -- is ``failed``: that
    is the model's doing, and it is counted, though it has no figures either.
    """
    from cae.graphs.rue_evaluator import grade_attempt
    from cae.llm import LLMUnavailableError

    if group is not None and group not in marking_groups():
        raise ValueError(f"No case group {group}; the groups are {marking_groups()}.")
    dataset = load_dataset("reading_cases")
    selected = [
        c
        for c in dataset["cases"]
        if (case_ids is None or c["id"] in case_ids) and (group is None or c["group"] == group)
    ]

    cases: list[dict[str, Any]] = []
    for case in selected:
        part = case["part"]
        responses = {int(k): v for k, v in case["responses"].items()}
        row: dict[str, Any] = {"id": case["id"], "part": part, "group": case["group"]}
        with _case_trace("marking", row, {"part": part, "responses": case["responses"]}):
            try:
                graded, elapsed = _timed(
                    lambda: grade_attempt(part, case["exercise"], responses)
                )
            except Exception as exc:
                status = "unavailable" if isinstance(exc, LLMUnavailableError) else "failed"
                row.update({"ok": False, status: True, "error": f"{type(exc).__name__}: {exc}"})
                cases.append(row)
                continue

            verdicts = {r["item_number"]: _verdict_from_row(r) for r in graded["rows"]}
            categories = {
                r["item_number"]: r.get("category") or "" for r in graded["rows"]
            }
            expected_verdicts = {
                int(k): v
                for k, v in (case.get("expected_adjudication") or {}).items()
                if v
            }
            expected_categories = {
                int(k): v for k, v in (case.get("expected_category") or {}).items() if v
            }
            row.update(
                {
                    "ok": True,
                    "seconds": elapsed,
                    "marks": graded["marks"],
                    "max_marks": graded["max_marks"],
                    "adjudication": metrics.adjudication_accuracy(
                        verdicts, expected_verdicts
                    ),
                    "tagging": metrics.category_accuracy(
                        categories, expected_categories
                    ),
                    "items": _marked_items(
                        graded["rows"],
                        verdicts,
                        expected_verdicts,
                        expected_categories,
                        adjudication_routes(part, case["exercise"], responses),
                    ),
                }
            )
        cases.append(row)

    return {
        "suite": "marking",
        "group": group,
        "cases": cases,
        "metrics": marking_metrics(cases),
    }


def _pooled_rate(scores: Sequence[Mapping[str, Any]], key: str = "items") -> float | None:
    """Agreement over every item of several cases' scores, not a mean of rates."""
    items = sum(s[key] for s in scores)
    missed = sum(len(s["disagreements"]) for s in scores)
    return round((items - missed) / items, 3) if items else None


def _route_rates(cases: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Adjudication agreement per Part 4 route, pooled over every item on it."""
    tally: dict[str, list[int]] = {}
    for case in cases:
        for item in case.get("items") or []:
            route, expected = item.get("route"), item.get("expected_verdict")
            if route and expected:
                agreed, total = tally.setdefault(route, [0, 0])
                tally[route] = [agreed + (item["verdict"] == expected), total + 1]
    return {route: round(agreed / total, 3) for route, (agreed, total) in sorted(tally.items())}


def marking_metrics(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The suite's metrics. A paper with nothing to adjudicate (Part 1) or nothing
    labelled is left out of that average rather than scored zero, and so is one
    the model did not mark. With nothing left, a rate is None, not 0."""
    good = [c for c in cases if c.get("ok")]
    adjudicated = [c for c in good if c["adjudication"]["items"]]
    tagged = [c for c in good if c["tagging"]["items"]]
    parts = sorted({c["part"] for c in good})

    def mean(rows: Sequence[dict[str, Any]], score: str, key: str) -> float | None:
        return _mean([c[score][key] for c in rows]) if rows else None

    balanced_by_part = {
        p: rate
        for p in parts
        if (
            rate := metrics.balanced_accuracy(
                metrics.pooled_by_verdict(c["adjudication"] for c in adjudicated if c["part"] == p)
            )
        )
        is not None
    }
    by_verdict = metrics.pooled_by_verdict(c["adjudication"] for c in adjudicated)

    return {
        "papers": len(good),
        "unavailable_papers": sum(1 for c in cases if c.get("unavailable")),
        "failed_papers": sum(1 for c in cases if c.get("failed")),
        "adjudication_accuracy": mean(adjudicated, "adjudication", "accuracy"),
        "adjudication_balanced": (
            _mean(list(balanced_by_part.values())) if balanced_by_part else None
        ),
        "adjudication_balanced_by_part": balanced_by_part,
        "adjudication_by_verdict": {
            v: round(t["agreed"] / t["items"], 3) for v, t in by_verdict.items() if t["items"]
        },
        "adjudicated_items": sum(c["adjudication"]["items"] for c in adjudicated),
        "over_accepted": sum(c["adjudication"]["over_accepted"] for c in adjudicated),
        "over_rejected": sum(c["adjudication"]["over_rejected"] for c in adjudicated),
        "category_exact": mean(tagged, "tagging", "exact"),
        "category_family": mean(tagged, "tagging", "family"),
        "labelled_items": sum(c["tagging"]["items"] for c in tagged),
        "adjudication_by_part": {
            p: _pooled_rate([c["adjudication"] for c in adjudicated if c["part"] == p])
            for p in parts
            if any(c["part"] == p for c in adjudicated)
        },
        "part4_adjudication_by_route": _route_rates(adjudicated),
        "category_exact_by_part": {
            p: _pooled_rate([c["tagging"] for c in tagged if c["part"] == p])
            for p in parts
            if any(c["part"] == p for c in tagged)
        },
        "mean_seconds": _mean([c["seconds"] for c in good]) if good else None,
    }


def writing_evaluator_groups() -> list[int]:
    """The script groups ``writing_scripts.json`` defines, for ``--group``."""
    return sorted({c["group"] for c in load_dataset("writing_scripts")["cases"]})


def _mark_script(
    row: dict[str, Any],
    case: dict[str, Any],
    task: dict[str, Any],
    part: int,
    genre: str,
    repeat: int,
) -> None:
    """Mark one reference script ``repeat`` times and fill ``row`` in place."""
    from cae.graphs.writing_evaluator import assess_writing

    runs: list[dict[str, Any]] = []
    for _ in range(max(1, repeat)):
        try:
            assessed, elapsed = _timed(
                lambda: assess_writing(
                    part=part,
                    genre=genre,
                    task=task,
                    script=case["script"],
                )
            )
        except Exception as exc:
            runs.append({"error": f"{type(exc).__name__}: {exc}"})
            continue
        failed = [s for s in SUBSCALES if (assessed.get(s) or {}).get("failed")]
        runs.append(
            {
                "seconds": elapsed,
                "bands": assessed.get("bands", {}),
                "total": assessed.get("total", 0),
                "scale_score": assessed.get("scale_score", 0),
                "status": assessed.get("status"),
                "degraded_subscales": failed,
                "annotate_degraded": bool(assessed.get("annotate_failed")),
                "feedback": {s: assessed.get(s) or {} for s in SUBSCALES},
                "annotation_list": assessed.get("annotations") or [],
                "strengths": assessed.get("strengths") or [],
                "improved": assessed.get("improved") or {},
            }
        )

    usable = [r for r in runs if "error" not in r]
    if not usable:
        row.update({"ok": False, "runs": runs, "error": runs[-1]["error"]})
        return

    first = next((r for r in usable if not r["degraded_subscales"]), usable[0])
    row.update(
        {
            "ok": True,
            "bands": first["bands"],
            "total": first["total"],
            "scale_score": first["scale_score"],
            "seconds": _mean([r["seconds"] for r in usable]),
            "degraded_subscales": first["degraded_subscales"],
            "annotate_degraded": first["annotate_degraded"],
            **{k: first[k] for k in ("feedback", "annotation_list", "strengths", "improved")},
        }
    )
    if repeat > 1:
        row["consistency"] = metrics.consistency([r["total"] for r in usable])
    if not first["annotate_degraded"]:
        row["annotation_scores"] = metrics.annotation_scores(
            first["annotation_list"],
            case["script"],
            case["expected_annotations"],
            case.get("tolerated_annotations") or [],
        )
    if not first["degraded_subscales"]:
        row["expected_bands"] = case["expected_bands"]
        row["band_errors"] = metrics.band_errors(first["bands"], case["expected_bands"])
        row["band_accuracy"] = metrics.band_accuracy(first["bands"], case["expected_bands"])


def run_writing_evaluator(
    case_ids: Sequence[str] | None = None,
    group: int | None = None,
    genres: Sequence[str] | None = None,
    repeat: int = 1,
) -> dict[str, Any]:
    """Mark the reference scripts; score their bands and their annotations.

    ``group`` runs one group of five scripts, one per genre; ``None`` runs them
    all. ``genres`` keeps only those genres. Each script is marked against its
    task, fixed in the dataset, so groups run on different days compare.
    """
    if group is not None and group not in writing_evaluator_groups():
        raise ValueError(f"No script group {group}; the groups are {writing_evaluator_groups()}.")
    dataset = load_dataset("writing_scripts")
    tasks = dataset["tasks"]

    selected = [
        c
        for c in dataset["cases"]
        if (case_ids is None or c["id"] in case_ids)
        and (group is None or c["group"] == group)
        and (genres is None or tasks[c["task"]]["genre"] in genres)
    ]
    cases: list[dict[str, Any]] = []
    for case in selected:
        task = tasks[case["task"]]
        row: dict[str, Any] = {
            "id": case["id"],
            "group": case["group"],
            "part": task["part"],
            "genre": task["genre"],
            "stresses": case.get("stresses"),
            "task": {"id": case["task"], "title": task["spec"].get("title"), "spec": task["spec"]},
        }
        with _case_trace("writing_evaluator", row, {"script": case["script"], "repeat": repeat}):
            _mark_script(row, case, task["spec"], task["part"], task["genre"], repeat)
        cases.append(row)

    return {
        "suite": "writing_evaluator",
        "group": group,
        "cases": cases,
        "metrics": writing_evaluator_metrics(cases),
    }


def _scored(cases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The marked cases with no degraded subscale: bands the model really awarded."""
    return [c for c in cases if c.get("ok") and not c.get("degraded_subscales")]


def _writing_scores(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Band accuracy and annotation scores over ``cases``.

    Band accuracy is the mean over the scripts whose bands were scored; recall,
    precision and category accuracy are pooled over every labelled error, not a
    mean of the scripts' rates. With nothing to score, a figure is None, not 0.
    """
    scored = _scored(cases)
    pooled = metrics.pooled_annotation_scores(
        c["annotation_scores"] for c in cases if c.get("ok") and "annotation_scores" in c
    )

    def accuracy(key: str) -> float | None:
        return _mean([c["band_accuracy"][key] for c in scored]) if scored else None

    return {
        "scripts": len(scored),
        **{f"accuracy_{s}": accuracy(s) for s in SUBSCALES},
        "accuracy_total": accuracy("total"),
        "recall": pooled["recall"],
        "precision": pooled["precision"],
        "accuracy_category": pooled["category_accuracy"],
        "expected_errors": pooled["expected_errors"],
        "annotations": pooled["annotations"],
    }


def writing_evaluator_metrics(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The suite's metrics, over every script and per genre.

    A script with a degraded subscale has a band 0 that no examiner awarded, so
    its bands are left out of the accuracy; one whose annotate call failed has
    no annotations to score. Both are counted.
    """
    from cae.domain.parts import WRITING_PART1_GENRES, WRITING_PART2_GENRES

    good = [c for c in cases if c.get("ok")]
    genres = [
        g for g in (*WRITING_PART1_GENRES, *WRITING_PART2_GENRES) if any(c["genre"] == g for c in cases)
    ]
    pooled = metrics.pooled_annotation_scores(
        c["annotation_scores"] for c in good if "annotation_scores" in c
    )
    return {
        "failed_scripts": len(cases) - len(good),
        "degraded_scripts": len(good) - len(_scored(cases)),
        "degraded_annotations": sum(1 for c in good if c.get("annotate_degraded")),
        **_writing_scores(cases),
        "by_genre": {g: _writing_scores([c for c in cases if c["genre"] == g]) for g in genres},
        "annotation_by_category": pooled["by_category"],
        "mean_seconds": _mean([c["seconds"] for c in good]) if good else None,
    }


ARCHETYPES: tuple[str, ...] = ("threshold", "improving", "slipping", "writing_led", "strong")


def coach_groups() -> list[int]:
    """The profile groups ``coach_profiles.json`` defines, for ``--group``."""
    return sorted({p["group"] for p in load_dataset("coach_profiles")["profiles"]})


def coach_case_id(archetype: str, group: int) -> str:
    return f"{archetype}-g{group}"


@contextmanager
def seeded_learner(profile: Mapping[str, Any]) -> Iterator[tuple[Any, int]]:
    """A session over an in-memory database holding one learner: ``profile``.

    The learner's work goes in as the app records it -- attempts in order, the
    tagged errors, the marked Writing -- so ``coach.stats_from`` reads it with the
    app's own SQL. ``cae.db`` is never opened, and nothing outlives the block.
    """
    from datetime import datetime, timedelta, timezone

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from sqlalchemy.pool import StaticPool

    from cae.db.models import (
        Attempt,
        Base,
        ErrorEvent,
        Exercise,
        User,
        WritingScore,
        WritingSubmission,
        WritingTask,
    )
    from cae.domain.parts import PART_SPECS

    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    clock = datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc)

    def tick() -> datetime:
        nonlocal clock
        clock += timedelta(hours=1)
        return clock

    session = Session(engine, expire_on_commit=False)
    try:
        user = User(username=str(profile["id"]), password_hash="")
        session.add(user)
        session.flush()

        last_attempt: dict[int, int] = {}
        for part_key, marks in profile.get("attempts", {}).items():
            part = int(part_key)
            exercise = Exercise(part=part, title=f"eval part {part}", payload={})
            session.add(exercise)
            session.flush()
            for mark in marks:
                attempt = Attempt(
                    user_id=user.id,
                    exercise_id=exercise.id,
                    part=part,
                    marks=float(mark),
                    max_marks=float(PART_SPECS[part].max_marks),
                    submitted_at=tick(),
                )
                session.add(attempt)
                session.flush()
                last_attempt[part] = attempt.id

        last_submission = 0
        for piece in profile.get("writing", []):
            task = WritingTask(part=piece["part"], genre=piece["genre"], payload={})
            session.add(task)
            session.flush()
            submission = WritingSubmission(
                user_id=user.id, task_id=task.id, text="", submitted_at=tick()
            )
            session.add(submission)
            session.flush()
            bands = piece["bands"]
            session.add(WritingScore(submission_id=submission.id, total=sum(bands.values()), **bands))
            last_submission = submission.id

        for error in profile.get("errors", []):
            part = error.get("part")
            evidence = error.get("evidence") or [""]
            for i in range(error["count"]):
                session.add(
                    ErrorEvent(
                        user_id=user.id,
                        source_type="reading" if part else "writing",
                        source_id=last_attempt.get(part, 0) if part else last_submission,
                        part=part,
                        category=error["category"],
                        evidence=evidence[i % len(evidence)],
                        created_at=tick(),
                    )
                )
        session.commit()
        yield session, user.id
    finally:
        session.close()
        engine.dispose()


def coach_facts(
    snapshot: Any, examples: Mapping[str, Sequence[str]], figures: Mapping[str, Any]
) -> dict[str, Any]:
    """The learner's figures as the coach's metrics read them (see ``evals.metrics``)."""
    attempted = [s for s in snapshot.part_stats if s.attempts]
    return {
        "parts": {
            s.part: {
                "name": s.name,
                "attempts": s.attempts,
                "accuracy": s.accuracy,
                "recent": s.recent_accuracy,
                "trend": s.trend,
            }
            for s in snapshot.part_stats
        },
        "errors": [dict(row) for row in snapshot.top_errors],
        "examples": {c: [s for s in spans if s][:3] for c, spans in examples.items()},
        "writing": dict(snapshot.writing_averages),
        "header": dict(figures),
        "strengths": [s.part for s in snapshot.strengths],
        "weakest": min(attempted, key=lambda s: s.accuracy).part if attempted else None,
    }


def _unavailable_names() -> set[str]:
    """``LLMUnavailableError`` and every subclass, by the name ``narrate`` records."""
    from cae.llm import LLMUnavailableError

    names, pending = set(), [LLMUnavailableError]
    while pending:
        cls = pending.pop()
        names.add(cls.__name__)
        pending.extend(cls.__subclasses__())
    return names


def _narrate_profile(row: dict[str, Any], state: dict[str, Any]) -> None:
    """Write the report once and fill ``row`` in place."""
    from cae.graphs import coach

    narrated, elapsed = _timed(lambda: coach.narrate(state))
    if narrated.get("status") != "ok":
        error = f"{narrated.get('error', narrated.get('status'))}: {narrated.get('message', '')}"
        status = "unavailable" if narrated.get("error") in _unavailable_names() else "failed"
        row.update({"ok": False, status: True, "error": error})
        return
    row.update({"ok": True, "seconds": elapsed, "narrative": narrated["narrative"]})


def run_coach(
    case_ids: Sequence[str] | None = None,
    group: int | None = None,
    archetypes: Sequence[str] | None = None,
    judge: bool = True,
) -> dict[str, Any]:
    """Write the report for each synthetic learner; have the judge mark it.

    ``group`` runs one group of five learners, one per archetype; ``None`` runs
    them all. ``archetypes`` keeps only those. Each learner is seeded into an
    in-memory database and read with the graph's own nodes (``stats_from``,
    ``detect_trend``, ``narrate``), so the report cache and ``cae.db`` are never
    touched. The paid judge (``evals.judge``) marks every written report against
    the profile's ``reference`` and the figures and tables ``narrate`` was
    handed -- it is the only measure of the report -- unless ``judge`` is False,
    which leaves the reports to be judged later (``judge_coach_report.py``).
    """
    from cae.graphs import coach

    if group is not None and group not in coach_groups():
        raise ValueError(f"No profile group {group}; the groups are {coach_groups()}.")
    selected = [
        p
        for p in load_dataset("coach_profiles")["profiles"]
        if (case_ids is None or p["id"] in case_ids)
        and (group is None or p["group"] == group)
        and (archetypes is None or p["archetype"] in archetypes)
    ]
    judge_model = None
    if judge:
        from evals.judge import CoachVerdict, build_judge

        judge_model = build_judge(schema=CoachVerdict)

    cases: list[dict[str, Any]] = []
    for profile in selected:
        with seeded_learner(profile) as (session, user_id):
            state: dict[str, Any] = {
                "user_id": user_id,
                "username": "the student",
                **coach.stats_from(session, user_id),
            }
        state.update(coach.detect_trend(state))
        figures = coach.prompt_figures(state["snapshot"])
        row: dict[str, Any] = {
            "id": profile["id"],
            "group": profile["group"],
            "archetype": profile["archetype"],
            "figures": figures,
            "tables": state["tables"],
            "facts": coach_facts(state["snapshot"], state["examples"], figures),
            "reference": profile["reference"],
        }
        with _case_trace("coach", row, {"profile": profile["id"]}):
            _narrate_profile(row, state)
        cases.append(row)

    if judge_model is not None:
        from evals.judge import judge_coach_cases

        judge_coach_cases(cases, judge_model)

    return {
        "suite": "coach",
        "group": group,
        "cases": cases,
        "metrics": coach_metrics(cases),
    }


def _rate(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 3) if denominator else None


def _judged(cases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [c for c in cases if isinstance((c.get("judge") or {}).get("correctness"), int)]


def _judge_scores(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The judge's figures over ``cases``, the same in the chat and coach suites.

    Its verdicts' means, on their own scales (0-2, 1-5), and the share of cases
    that got the top mark. Empty when no case was judged, so a figure with
    nothing behind it is absent, not 0.
    """
    judged = _judged(cases)
    if not judged:
        return {}
    return {
        "judged": len(judged),
        "judge_correctness_mean": _mean([c["judge"]["correctness"] for c in judged]),
        "judge_faithfulness_mean": _mean([c["judge"]["faithfulness"] for c in judged]),
        "judge_helpfulness": _mean([c["judge"]["helpfulness"] for c in judged]),
        "judge_correct_rate": _rate(
            sum(c["judge"]["correctness"] == 2 for c in judged), len(judged)
        ),
        "judge_faithful_rate": _rate(
            sum(c["judge"]["faithfulness"] == 2 for c in judged), len(judged)
        ),
    }


def _coach_scores(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The coach's figures over ``cases``: how many reports were written, and the
    judge's verdicts on them. Nothing else scores the report."""
    good = [c for c in cases if c.get("ok")]
    return {"profiles": len(good), **_judge_scores(good)}


def coach_metrics(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The suite's metrics, over every learner and per archetype.

    A report no model was there for (``unavailable``) measures the provider and
    one it answered unusably (``failed``) has nothing to judge: both are only
    counted.
    """
    good = [c for c in cases if c.get("ok")]
    archetypes = [a for a in ARCHETYPES if any(c["archetype"] == a for c in cases)]
    return {
        "unavailable": sum(1 for c in cases if c.get("unavailable")),
        "failed": sum(1 for c in cases if c.get("failed")),
        **_coach_scores(cases),
        "by_archetype": {
            a: _coach_scores([c for c in cases if c["archetype"] == a]) for a in archetypes
        },
        "mean_seconds": _mean([c["seconds"] for c in good]) if good else None,
    }


CHAT_KINDS: tuple[str, ...] = (
    "snapshot",
    "weakest",
    "item",
    "error_example",
    "time",
    "recent",
    "writing_best",
    "empty",
    "exam",
    "dialogue",
)

PRELOADED_KINDS = frozenset({"snapshot", "weakest"})

CHAT_USERNAME = "the student"


def chat_groups() -> list[int]:
    """The learner groups ``chat_learners.json`` defines, for ``--group``."""
    return sorted({learner["group"] for learner in load_dataset("chat_learners")["learners"]})


def chat_turns(learner: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every turn put to ``learner``, in order.

    A dialogue is unrolled into its turns, each keeping its own id and expectation
    and gaining ``dialogue`` (the conversation's id) and ``turn`` (1, 2, 3).
    """
    turns: list[dict[str, Any]] = []
    for question in learner["questions"]:
        if question["kind"] != "dialogue":
            turns.append(dict(question))
            continue
        for number, turn in enumerate(question["turns"], 1):
            turns.append({**turn, "kind": "dialogue", "dialogue": question["id"], "turn": number})
    return turns


def chat_exercises() -> dict[str, tuple[int, dict[str, Any]]]:
    """Every exercise a chat learner sits, by id, as ``(part, payload)``.

    Parts 1-4 are the hand-made papers of ``reading_cases.json``, referenced
    rather than copied; Parts 5-8 are frozen in ``chat_learners.json``, each one
    an exercise the generator wrote and the validator passed. Real payloads are
    what let ``learner_record`` fill each item's question and options.
    """
    papers = {
        case["id"]: (case["part"], case["exercise"])
        for case in load_dataset("reading_cases")["cases"]
    }
    frozen = load_dataset("chat_learners")["exercises"]
    return {**papers, **{eid: (e["part"], e["exercise"]) for eid, e in frozen.items()}}


@contextmanager
def isolated_database() -> Iterator[None]:
    """Point the whole app at a scratch SQLite file for the length of the block.

    The chat graph opens its own sessions -- ``load_context`` through
    ``coach.load_stats``, ``query_record`` through ``learner_record`` -- so the
    suite patches no node: like ``tests/conftest.py`` it moves
    ``CAE_DATABASE_URL`` and clears the three caches that hold the old one. On
    the way out the setting and the caches are put back and the file removed,
    so ``cae.db`` is never opened and nothing outlives the block. A pinned model
    survives the cleared settings: ``use_model`` pins it in the environment.
    """
    import os
    import shutil
    import tempfile
    from pathlib import Path

    from cae.db.session import get_engine, get_sessionmaker

    def clear() -> None:
        get_settings.cache_clear()
        get_engine.cache_clear()
        get_sessionmaker.cache_clear()

    previous = os.environ.get("CAE_DATABASE_URL")
    folder = tempfile.mkdtemp(prefix="cae-chat-eval-")
    os.environ["CAE_DATABASE_URL"] = f"sqlite:///{(Path(folder) / 'learner.db').as_posix()}"
    clear()
    try:
        yield
    finally:
        try:
            get_engine().dispose()
        finally:
            if previous is None:
                os.environ.pop("CAE_DATABASE_URL", None)
            else:
                os.environ["CAE_DATABASE_URL"] = previous
            clear()
            shutil.rmtree(folder, ignore_errors=True)


def _graded_rows(
    part: int, payload: dict[str, Any], wrong: Mapping[str, Mapping[str, Any]], where: str
) -> tuple[dict[int, str], list[dict[str, Any]]]:
    """One attempt's responses and item rows, marked as the app marks them.

    Every item not in ``wrong`` is answered with its key. A slip that needs a
    second opinion gets the slip's ``verdict`` (``reject`` unless it says
    otherwise), as the adjudicator would give it; the evidence is the grader's
    own ``error_evidence``. Raises when the data contradicts itself -- a slip
    the marker accepts, a category the part does not have -- since a labelling
    mistake would otherwise shift every figure the questions expect.
    """
    from cae.domain.answers import apply_adjudication, grade_deterministically
    from cae.domain.taxonomy import categories_for_part
    from cae.graphs.rue_evaluator import error_evidence
    from cae.schemas.rue_task import answer_key, parse_exercise

    keys = answer_key(part, parse_exercise(part, payload))
    responses = {key.number: key.answer for key in keys}
    unknown = {int(n) for n in wrong} - set(responses)
    if unknown:
        raise ValueError(f"{where}: no item {sorted(unknown)} in this exercise")
    responses.update({int(number): slip["answer"] for number, slip in wrong.items()})

    rows: list[dict[str, Any]] = []
    for key, result in zip(keys, grade_deterministically(part, keys, responses)):
        slip = wrong.get(str(key.number))
        if result.needs_adjudication:
            apply_adjudication(result, (slip or {}).get("verdict", "reject"))
        if slip is None and not result.is_correct:
            raise ValueError(f"{where}: the key to item {key.number} marks wrong")
        if slip is not None and result.is_correct:
            raise ValueError(f"{where}: '{slip['answer']}' is right for item {key.number}")
        category = None
        if slip is not None:
            allowed = [c.value for c in categories_for_part(part)]
            category = slip.get("category") or (allowed[0] if len(allowed) == 1 else None)
            if category not in allowed:
                raise ValueError(f"{where}: item {key.number} needs a Part {part} category")
        rows.append(
            {
                "item_number": key.number,
                "user_answer": result.user_answer,
                "correct_answer": " / ".join(key.variants),
                "is_correct": result.is_correct,
                "marks_awarded": result.marks,
                "max_marks": result.max_marks,
                "category": category,
                "explanation": key.explanation,
                "adjudicated": result.adjudicated,
                "evidence": error_evidence(payload, key, result),
            }
        )
    return responses, rows


def seed_chat_learner(
    session: Any,
    learner: Mapping[str, Any],
    exercises: Mapping[str, tuple[int, dict[str, Any]]] | None = None,
    now: Any = None,
) -> int:
    """Record ``learner``'s work through the app's own writers; return their id.

    Attempts go through ``repo.record_attempt`` (which also writes the error
    events), mocks through ``create_mock_exam``/``finish_mock_exam``, Writing
    through ``save_writing_task``/``save_submission``/``save_writing_score``.
    Each piece is then dated ``days_ago`` before ``now`` -- relative to today,
    so "the last seven days" means the same thing whenever the suite runs, as
    the sandbox's ``date('now', ...)`` reads it. Pieces on the same day keep the
    order they are listed in.
    """
    from datetime import datetime, timedelta, timezone

    from sqlalchemy import select

    from cae.db import repo
    from cae.db.models import ErrorEvent
    from cae.domain.parts import PART_SPECS
    from cae.domain.scoring import summarise, summarise_writing, writing_total
    from cae.validators import count_words

    exercises = exercises if exercises is not None else chat_exercises()
    scripts = load_dataset("writing_scripts")
    by_script = {case["id"]: case for case in scripts["cases"]}
    now = now or datetime.now(timezone.utc)
    step = 0

    def at(days_ago: float) -> datetime:
        nonlocal step
        step += 1
        return now - timedelta(days=days_ago, hours=1) + timedelta(minutes=step)

    def date_errors(source_type: str, source_id: int, when: datetime) -> None:
        events = session.scalars(
            select(ErrorEvent)
            .where(ErrorEvent.source_type == source_type, ErrorEvent.source_id == source_id)
            .order_by(ErrorEvent.id)
        ).all()
        for offset, event in enumerate(events):
            event.created_at = when + timedelta(seconds=offset)

    user = repo.create_user(session, f"eval-{learner['id']}", "")
    stored: dict[str, int] = {}

    def sit(attempt: Mapping[str, Any], *, mode: str = "practice", mock_id: int | None = None) -> float:
        part, payload = exercises[attempt["exercise"]]
        if attempt["exercise"] not in stored:
            stored[attempt["exercise"]] = repo.save_exercise(
                session, part=part, payload=payload, topic=str(payload.get("topic", ""))
            ).id
        where = f"{learner['id']} {attempt['exercise']}"
        responses, rows = _graded_rows(part, payload, attempt.get("wrong") or {}, where)
        marks = sum(row["marks_awarded"] for row in rows)
        row = repo.record_attempt(
            session,
            user_id=user.id,
            exercise_id=stored[attempt["exercise"]],
            part=part,
            responses={str(k): v for k, v in responses.items()},
            item_rows=rows,
            marks=marks,
            max_marks=float(PART_SPECS[part].max_marks),
            duration_seconds=int(attempt.get("seconds", 0)),
            mode=mode,
            mock_exam_id=mock_id,
        )
        row.submitted_at = at(attempt["days_ago"])
        date_errors("reading", row.id, row.submitted_at)
        return marks

    for attempt in learner.get("attempts", []):
        sit(attempt)

    for mock in learner.get("mocks", []):
        exam = repo.create_mock_exam(session, user.id)
        exam.started_at = at(mock["days_ago"])
        marks = sum(sit(a, mode="mock", mock_id=exam.id) for a in mock["attempts"])
        maximum = float(
            sum(PART_SPECS[exercises[a["exercise"]][0]].max_marks for a in mock["attempts"])
        )
        repo.finish_mock_exam(
            session,
            exam_id=exam.id,
            marks=marks,
            max_marks=maximum,
            scale_score=summarise(marks, maximum).scale_score,
        )
        exam.finished_at = at(mock["days_ago"])

    for piece in learner.get("writing", []):
        case = by_script[piece["script"]]
        task = scripts["tasks"][case["task"]]
        bands = dict(piece.get("bands") or case["expected_bands"])
        annotations = [
            {
                "span": a["span"],
                "category": a["category"],
                "correction": a.get("correction", ""),
                "explanation": a.get("explanation", ""),
                "severity": "minor",
            }
            for a in case.get("expected_annotations", [])
        ]
        to_improve = piece.get("to_improve") or {}
        task_row = repo.save_writing_task(
            session, part=task["part"], genre=task["genre"], payload=dict(task["spec"])
        )
        submission = repo.save_submission(
            session,
            user_id=user.id,
            task_id=task_row.id,
            text=case["script"],
            word_count=count_words(case["script"]),
            duration_seconds=int(piece.get("seconds", 0)),
        )
        submission.submitted_at = at(piece["days_ago"])
        repo.save_writing_score(
            session,
            submission_id=submission.id,
            user_id=user.id,
            bands=bands,
            total=writing_total(bands),
            scale_score=summarise_writing(bands).scale_score,
            feedback={
                **{
                    s: {
                        "band": bands[s],
                        "justification": "",
                        "evidence": "",
                        "to_improve": to_improve.get(s, ""),
                    }
                    for s in SUBSCALES
                },
                "annotations": annotations,
                "strengths": [],
                "improved": {"text": "", "upgrades": []},
                "status": "ok",
                "message": "",
            },
            annotations=annotations,
        )
        date_errors("writing", submission.id, submission.submitted_at)

    session.flush()
    return user.id


@contextmanager
def _without_index(disabled: bool) -> Iterator[None]:
    """With ``disabled``, the reference corpus as a fresh clone has it: absent.

    The ablation of ``--without-index``: the coach is told there is nothing to
    cite, and a lookup returns nothing, so its exam figures come from memory.
    """
    if not disabled:
        yield
        return
    from unittest import mock

    from cae.rag import store

    with mock.patch.object(store, "search", lambda *a, **k: []), mock.patch.object(
        store, "index_exists", lambda *a, **k: False
    ):
        yield


_QUERY_FAILURES = ("SQL error", "Not allowed", "The query took too long", "That statement")


def _sql_errors(tool_outputs: Sequence[str]) -> int:
    return sum(1 for output in tool_outputs if output.startswith(_QUERY_FAILURES))


def score_chat_turn(turn: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    """The deterministic chat metrics for one turn, keyed as the case row keeps them.

    Only what can be counted exactly: which tools the turn called against the
    ones it needed, the rounds it took, its refused queries, and for an exam
    question whether the lookup came back from the right section. Whether the
    reply is right is the judge's (``evals.judge``).
    """
    from cae.graphs.coach_chat import MAX_TOOL_ROUNDS

    expect = dict(turn.get("expect") or {})
    outputs = [str(o) for o in result.get("tool_outputs") or []]
    called = [call["name"] for call in result.get("tool_calls") or []]
    rounds = int(result.get("tool_rounds") or 0)
    scores: dict[str, Any] = {
        "tool_choice": metrics.tool_choice(
            called, turn.get("tools") or [], turn.get("allowed_tools") or []
        ),
        "tool_rounds": rounds,
        "budget_hit": rounds >= MAX_TOOL_ROUNDS,
        "sql_errors": _sql_errors(outputs),
    }
    if expect.get("sections"):
        scores["section_hit"] = metrics.section_hit(result.get("sources") or [], expect["sections"])
    return scores


def _chat_transcript(turn: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    """The turn as a person would judge it: question, whole reply, what it used."""
    return {
        "question": turn["question"],
        "reference": turn.get("reference", ""),
        "reply": result["reply"],
        "citations": [s.get("citation") for s in result.get("sources") or []],
        "passages": [str(s.get("text", "")) for s in result.get("sources") or []],
        "tool_calls": list(result.get("tool_calls") or []),
        "tool_outputs": list(result.get("tool_outputs") or []),
    }


def _ask(
    row: dict[str, Any],
    turn: Mapping[str, Any],
    history: Sequence[tuple[str, str]],
    user_id: int,
) -> dict[str, Any] | None:
    """Put one turn to the real graph and fill ``row`` in place; the result, or None."""
    from cae.graphs import coach_chat

    try:
        result, elapsed = _timed(
            lambda: coach_chat.answer(user_id, CHAT_USERNAME, list(history), turn["question"])
        )
    except Exception as exc:
        status = "unavailable" if type(exc).__name__ in _unavailable_names() else "failed"
        row.update(
            {
                "ok": False,
                status: True,
                "error": f"{type(exc).__name__}: {exc}",
                "question": turn["question"],
            }
        )
        return None
    row.update(
        {
            "ok": True,
            "seconds": elapsed,
            **score_chat_turn(turn, result),
            **_chat_transcript(turn, result),
        }
    )
    return result


def run_chat(
    group: int | None = None,
    kinds: Sequence[str] | None = None,
    case_ids: Sequence[str] | None = None,
    judge: bool = True,
    without_index: bool = False,
) -> dict[str, Any]:
    """Put each synthetic learner's questions to the real chat graph; score the replies.

    ``group`` runs one learner, ``None`` all three; ``kinds`` keeps only those
    kinds of turn. Each learner is recorded into a scratch database
    (``isolated_database``) through the app's own writers, and the graph runs
    unchanged against it: ``load_context``, the ``query_record`` sandbox, the
    tool loop. A dialogue's later turns are given the model's own earlier
    replies as history, as the page gives them. The paid judge
    (``evals.judge``) marks every reply -- it is the only measure of whether a
    reply is right -- unless ``judge`` is False, which leaves only the tool
    metrics; ``without_index`` takes the reference corpus away, as a fresh
    clone has it.
    """
    from cae.db.session import session_scope
    from cae.graphs import coach
    from cae.rag import store

    if group is not None and group not in chat_groups():
        raise ValueError(f"No learner group {group}; the groups are {chat_groups()}.")
    learners = [
        learner
        for learner in load_dataset("chat_learners")["learners"]
        if group is None or learner["group"] == group
    ]
    exercises = chat_exercises()
    judge_model = None
    if judge:
        from evals.judge import build_judge

        judge_model = build_judge()

    cases: list[dict[str, Any]] = []
    context: dict[str, Any] = {}
    with _without_index(without_index):
        index = "disabled" if without_index else ("built" if store.index_exists() else "absent")
        for learner in learners:
            turns = [
                t
                for t in chat_turns(learner)
                if (kinds is None or t["kind"] in kinds)
                and (case_ids is None or t["id"] in case_ids)
            ]
            if not turns:
                continue
            with isolated_database():
                with session_scope() as session:
                    user_id = seed_chat_learner(session, learner, exercises)
                with session_scope() as session:
                    state: dict[str, Any] = {
                        "user_id": user_id,
                        "username": CHAT_USERNAME,
                        **coach.stats_from(session, user_id),
                    }
                state.update(coach.detect_trend(state))
                context[learner["id"]] = {
                    "figures": coach.prompt_figures(state["snapshot"]),
                    "tables": state["tables"],
                }

                histories: dict[str, list[tuple[str, str]]] = {}
                for turn in turns:
                    history = histories.setdefault(turn.get("dialogue", ""), [])
                    row: dict[str, Any] = {
                        "id": turn["id"],
                        "group": learner["group"],
                        "learner": learner["id"],
                        "kind": turn["kind"],
                        **(
                            {"dialogue": turn["dialogue"], "turn": turn["turn"]}
                            if "turn" in turn
                            else {}
                        ),
                    }
                    past = history if turn.get("dialogue") else []
                    with _case_trace("chat", row, {"question": turn["question"]}):
                        result = _ask(row, turn, past, user_id)
                    if turn.get("dialogue"):
                        reply = result["reply"] if result else ""
                        history.extend([("user", turn["question"]), ("assistant", reply)])
                    cases.append(row)

    if judge_model is not None:
        from evals.judge import judge_cases

        judge_cases(cases, context, judge_model)

    return {
        "suite": "chat",
        "group": group,
        "index": index,
        "learners": context,
        "cases": cases,
        "metrics": chat_metrics(cases),
    }


def _chat_scores(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The chat's figures over ``cases``, pooled by counts rather than averaged.

    Only the turns the model answered count; a figure with nothing to score (no
    exam question, no verdict) is None or absent, not 0. The judge's are its
    verdicts' means, on their own scales (0-2, 1-5), and the share of turns that
    got the top mark.
    """
    good = [c for c in cases if c.get("ok")]
    sections = [c["section_hit"] for c in good if "section_hit" in c]
    return {
        "turns": len(good),
        "tool_choice_rate": _rate(sum(c["tool_choice"]["ok"] for c in good), len(good)),
        "missing_tools": sum(len(c["tool_choice"]["missing"]) for c in good),
        "extra_tools": sum(len(c["tool_choice"]["extra"]) for c in good),
        "section_hit_rate": _rate(sum(s["hit"] for s in sections), len(sections)),
        "budget_hits": sum(c["budget_hit"] for c in good),
        "sql_errors": sum(c["sql_errors"] for c in good),
        "mean_tool_rounds": _mean([c["tool_rounds"] for c in good]) if good else None,
        **_judge_scores(good),
    }


def chat_metrics(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The suite's metrics, over every turn and per kind of turn.

    A turn no model was there for (``unavailable``) measures the provider and
    one it answered unusably (``failed``) has nothing to score: both are only
    counted.
    """
    good = [c for c in cases if c.get("ok")]
    kinds = [k for k in CHAT_KINDS if any(c["kind"] == k for c in cases)]
    return {
        "unavailable": sum(1 for c in cases if c.get("unavailable")),
        "failed": sum(1 for c in cases if c.get("failed")),
        **_chat_scores(cases),
        "by_kind": {k: _chat_scores([c for c in cases if c["kind"] == k]) for k in kinds},
        "mean_seconds": _mean([c["seconds"] for c in good]) if good else None,
    }


SUITES: dict[str, Callable[..., dict[str, Any]]] = {
    "generation": run_generation,
    "marking": run_marking,
    "writing_task": run_writing_task,
    "writing_evaluator": run_writing_evaluator,
    "coach": run_coach,
    "chat": run_chat,
}

__all__ = [
    "ARCHETYPES",
    "CHAT_KINDS",
    "SUITES",
    "adjudication_routes",
    "chat_exercises",
    "chat_groups",
    "chat_metrics",
    "chat_turns",
    "coach_case_id",
    "coach_facts",
    "coach_groups",
    "coach_metrics",
    "generation_case_id",
    "generation_groups",
    "generation_metrics",
    "isolated_database",
    "marking_groups",
    "marking_metrics",
    "run_chat",
    "run_coach",
    "run_generation",
    "run_marking",
    "run_writing_evaluator",
    "run_writing_task",
    "score_chat_turn",
    "seed_chat_learner",
    "seeded_learner",
    "writing_evaluator_groups",
    "writing_evaluator_metrics",
    "writing_task_case_id",
    "writing_task_groups",
    "writing_task_metrics",
]
