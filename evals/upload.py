"""Push a finished eval run to LangSmith as an experiment.

The target is a lookup into results already computed, so uploading uses no
quota. Examples are added by ``id`` when missing; existing ones keep their
inputs and outputs, and only gain any metadata or split they lack.

Each experiment row carries the whole case — the material, the feedback or the
reply, not just its scores — and ``trace_url`` when the run was traced, so two
models can be read and scored by hand side by side in the comparison view.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from cae.domain.parts import SUBSCALES

from evals import load_dataset

DATASET_NAMES = {
    "generation": "cae-generation",
    "marking": "cae-marking",
    "writing_task": "cae-writing-task",
    "writing_evaluator": "cae-writing-evaluator",
    "coach": "cae-coach",
    "chat": "cae-chat",
}


def _generation_examples() -> list[dict[str, Any]]:
    from evals.suites import generation_case_id

    seeds = load_dataset("generation_seeds")["seeds"]
    return [
        {
            "inputs": {
                "id": generation_case_id(s["part"], s["group"]),
                "part": s["part"],
                "group": s["group"],
                "topic": s["topic"],
                "focus": "; ".join(s["focus"]),
            },
            "outputs": {"valid": True, "key_self_consistent": True},
            "metadata": {
                "case": generation_case_id(s["part"], s["group"]),
                "part": s["part"],
                "group": s["group"],
            },
            "split": f"group-{s['group']}",
        }
        for s in seeds
    ]


def _marking_examples() -> list[dict[str, Any]]:
    cases = load_dataset("reading_cases")["cases"]
    return [
        {
            "inputs": {"id": c["id"], "part": c["part"], "responses": c["responses"]},
            "outputs": {
                "adjudication": c.get("expected_adjudication") or {},
                "categories": {
                    k: v for k, v in (c.get("expected_category") or {}).items() if v
                },
            },
            "metadata": {"case": c["id"], "part": c["part"], "group": c["group"]},
            "split": f"group-{c['group']}",
        }
        for c in cases
    ]


def _writing_task_examples() -> list[dict[str, Any]]:
    from evals.suites import writing_task_case_id

    seeds = load_dataset("writing_task_seeds")["seeds"]
    return [
        {
            "inputs": {
                "id": writing_task_case_id(s["genre"], s["group"]),
                "part": s["part"],
                "genre": s["genre"],
                "group": s["group"],
                "topic": s["topic"],
            },
            "outputs": {"valid": True, "repairs": 0},
            "metadata": {
                "case": writing_task_case_id(s["genre"], s["group"]),
                "part": s["part"],
                "genre": s["genre"],
                "group": s["group"],
            },
            "split": f"group-{s['group']}",
        }
        for s in seeds
    ]


def _writing_evaluator_examples() -> list[dict[str, Any]]:
    dataset = load_dataset("writing_scripts")
    tasks = dataset["tasks"]
    return [
        {
            "inputs": {
                "id": c["id"],
                "genre": tasks[c["task"]]["genre"],
                "script": c["script"],
                "stresses": c.get("stresses"),
            },
            "outputs": {
                "expected_bands": c.get("expected_bands"),
                "expected_annotations": c.get("expected_annotations"),
            },
            "metadata": {
                "case": c["id"],
                "group": c["group"],
                "genre": tasks[c["task"]]["genre"],
            },
            "split": f"group-{c['group']}",
        }
        for c in dataset["cases"]
    ]


def _coach_examples() -> list[dict[str, Any]]:
    profiles = load_dataset("coach_profiles")["profiles"]
    return [
        {
            "inputs": {"id": p["id"], "group": p["group"], "archetype": p["archetype"]},
            "outputs": {"reference": p.get("reference", ""), "expect": p.get("expect") or {}},
            "metadata": {"case": p["id"], "group": p["group"], "archetype": p["archetype"]},
            "split": f"group-{p['group']}",
        }
        for p in profiles
    ]


def _chat_examples() -> list[dict[str, Any]]:
    from evals.suites import chat_turns

    learners = load_dataset("chat_learners")["learners"]
    return [
        {
            "inputs": {
                "id": t["id"],
                "group": learner["group"],
                "learner": learner["id"],
                "kind": t["kind"],
                "question": t["question"],
            },
            "outputs": {
                "reference": t.get("reference", ""),
                "expect": t.get("expect") or {},
                "tools": t.get("tools") or [],
            },
            "metadata": {"case": t["id"], "group": learner["group"], "kind": t["kind"]},
            "split": f"group-{learner['group']}",
        }
        for learner in learners
        for t in chat_turns(learner)
    ]


EXAMPLE_BUILDERS: dict[str, Callable[[], list[dict[str, Any]]]] = {
    "generation": _generation_examples,
    "marking": _marking_examples,
    "writing_task": _writing_task_examples,
    "writing_evaluator": _writing_evaluator_examples,
    "coach": _coach_examples,
    "chat": _chat_examples,
}


def _valid(outputs: dict, reference_outputs: dict) -> dict:
    return {"key": "valid", "score": float(bool(outputs.get("ok")))}


def _self_consistent(outputs: dict, reference_outputs: dict) -> dict:
    consistency = outputs.get("self_consistency") or {}
    return {"key": "key_self_consistent", "score": float(bool(consistency.get("consistent")))}


def _attempts(outputs: dict, reference_outputs: dict) -> dict:
    return {"key": "attempts", "score": float(outputs.get("attempts", 0))}


def _repairs(outputs: dict, reference_outputs: dict) -> dict:
    """None when no model answered: that measures the provider, not the repairs."""
    value = outputs.get("repairs")
    return {"key": "repairs", "score": None if value is None else float(value)}


def _first_try(outputs: dict, reference_outputs: dict) -> dict:
    if outputs.get("repairs") is None:
        return {"key": "first_try", "score": None}
    return {"key": "first_try", "score": float(bool(outputs.get("ok")) and outputs["repairs"] == 0)}


def _adjudicated(outputs: dict) -> dict | None:
    """The case's adjudication score, or None for a paper with nothing to adjudicate.

    Part 1 is never adjudicated: scoring it zero would read as a failure.
    """
    scores = outputs.get("adjudication") or {}
    return scores if scores.get("items") else None


def _adjudication(outputs: dict, reference_outputs: dict) -> dict:
    scores = _adjudicated(outputs)
    return {"key": "adjudication_accuracy", "score": None if scores is None else float(scores["accuracy"])}


def _adjudication_balanced(outputs: dict, reference_outputs: dict) -> dict:
    """Each expected verdict weighs the same, so rejecting everything does not pass."""
    from evals.metrics import balanced_accuracy

    scores = _adjudicated(outputs)
    value = None if scores is None else balanced_accuracy(scores.get("by_verdict") or {})
    return {"key": "adjudication_balanced", "score": value}


def _over_accepted(outputs: dict, reference_outputs: dict) -> dict:
    """Credit given to a wrong answer."""
    scores = _adjudicated(outputs)
    return {"key": "over_accepted", "score": None if scores is None else float(scores["over_accepted"])}


def _tagged(outputs: dict) -> dict | None:
    """The case's tagging score, or None when nothing was tagged, e.g. a paper the
    model was unavailable for: it measures the provider, not the model."""
    scores = outputs.get("tagging") or {}
    return scores if scores.get("items") else None


def _category_exact(outputs: dict, reference_outputs: dict) -> dict:
    scores = _tagged(outputs)
    return {"key": "category_exact", "score": None if scores is None else float(scores["exact"])}


def _category_family(outputs: dict, reference_outputs: dict) -> dict:
    scores = _tagged(outputs)
    return {"key": "category_family", "score": None if scores is None else float(scores["family"])}


def _band_accuracy(key: str) -> Callable[..., dict]:
    """An evaluator for one subscale's band accuracy, or the total's; None for a
    script with a degraded subscale, whose bands nobody fully awarded."""

    def evaluator(outputs: dict, reference_outputs: dict) -> dict:
        value = (outputs.get("band_accuracy") or {}).get(key)
        return {"key": f"accuracy_{key}", "score": None if value is None else float(value)}

    evaluator.__name__ = f"_accuracy_{key}"
    return evaluator


ANNOTATION_KEYS = {"recall": "recall", "precision": "precision", "category_accuracy": "accuracy_category"}


def _annotation_score(name: str) -> Callable[..., dict]:
    """An evaluator for one of ``annotation_scores``' rates; None where it has none,
    e.g. precision on a script with no annotation, or a failed annotate node."""

    def evaluator(outputs: dict, reference_outputs: dict) -> dict:
        value = (outputs.get("annotation_scores") or {}).get(name)
        return {"key": ANNOTATION_KEYS[name], "score": None if value is None else float(value)}

    evaluator.__name__ = f"_annotation_{name}"
    return evaluator


def _coach_score(key: str, read: Callable[[dict], Any]) -> Callable[..., dict]:
    """An evaluator for one of the judge's verdicts on a coach report; None for a
    report the model did not write or the judge did not mark."""

    def evaluator(outputs: dict, reference_outputs: dict) -> dict:
        value = read(outputs) if outputs.get("ok") else None
        return {"key": key, "score": None if value is None else float(value)}

    evaluator.__name__ = f"_coach_{key}"
    return evaluator


COACH_EVALUATORS = [
    _coach_score("judge_correctness", lambda o: (o.get("judge") or {}).get("correctness")),
    _coach_score("judge_faithfulness", lambda o: (o.get("judge") or {}).get("faithfulness")),
    _coach_score("judge_helpfulness", lambda o: (o.get("judge") or {}).get("helpfulness")),
]


def _chat_score(key: str, read: Callable[[dict], Any]) -> Callable[..., dict]:
    """An evaluator for one chat figure; None for a turn the model did not answer,
    or a figure the turn has nothing to score on (no lookup, no judge)."""

    def evaluator(outputs: dict, reference_outputs: dict) -> dict:
        value = read(outputs) if outputs.get("ok") else None
        return {"key": key, "score": None if value is None else float(value)}

    evaluator.__name__ = f"_chat_{key}"
    return evaluator


CHAT_EVALUATORS = [
    _chat_score("judge_correctness", lambda o: (o.get("judge") or {}).get("correctness")),
    _chat_score("judge_faithfulness", lambda o: (o.get("judge") or {}).get("faithfulness")),
    _chat_score("judge_helpfulness", lambda o: (o.get("judge") or {}).get("helpfulness")),
    _chat_score("tool_choice", lambda o: (o.get("tool_choice") or {}).get("ok")),
    _chat_score("section_hit", lambda o: (o.get("section_hit") or {}).get("hit")),
    _chat_score("sql_errors", lambda o: o.get("sql_errors")),
    _chat_score("tool_rounds", lambda o: o.get("tool_rounds")),
]


EVALUATORS: dict[str, list[Callable[..., dict]]] = {
    "generation": [_valid, _self_consistent, _attempts],
    "marking": [
        _adjudication,
        _adjudication_balanced,
        _over_accepted,
        _category_exact,
        _category_family,
    ],
    "writing_task": [_valid, _first_try, _repairs],
    "writing_evaluator": [
        *(_band_accuracy(key) for key in (*SUBSCALES, "total")),
        *(_annotation_score(name) for name in ANNOTATION_KEYS),
    ],
    "coach": COACH_EVALUATORS,
    "chat": CHAT_EVALUATORS,
}


def _case_id(case: dict[str, Any]) -> str:
    """Cases are keyed by ``id``; a case with only a part falls back to it."""
    if case.get("id"):
        return str(case["id"])
    return f"part{case['part']}" if case.get("part") else ""


def examples_run(examples: Iterable[Any], by_id: dict[str, Any]) -> list[Any]:
    """The dataset examples this run has a row for.

    A run may cover part of the dataset (one seed group, one ``--part``); the
    examples it skipped were not attempted, and must not score as failures.
    """
    return [e for e in examples if (e.inputs or {}).get("id") in by_id]


def ensure_dataset(client, suite: str):
    """Create the suite's dataset if it is missing, and add any new examples."""
    name = DATASET_NAMES[suite]
    if client.has_dataset(dataset_name=name):
        dataset = client.read_dataset(dataset_name=name)
    else:
        dataset = client.create_dataset(
            dataset_name=name,
            description=f"Reference cases for the cae-trainer '{suite}' suite.",
        )

    existing = {
        (example.inputs or {}).get("id"): example
        for example in client.list_examples(dataset_id=dataset.id)
    }
    wanted = EXAMPLE_BUILDERS[suite]()
    missing = [example for example in wanted if example["inputs"]["id"] not in existing]
    if missing:
        client.create_examples(dataset_id=dataset.id, examples=missing)
    stale = stale_labels(wanted, existing)
    if stale:
        client.update_examples(dataset_id=dataset.id, updates=stale)
    return dataset


def stale_labels(wanted: Iterable[dict[str, Any]], existing: dict[str, Any]) -> list[dict[str, Any]]:
    """Updates bringing existing examples' metadata and split up to date.

    Inputs and outputs are left alone, so experiments already run against an
    example keep comparing like with like; only the labels are backfilled.
    """
    updates = []
    for example in wanted:
        current = existing.get(example["inputs"]["id"])
        if current is None:
            continue
        labels = example.get("metadata") or {}
        split = example.get("split")
        metadata = dict(current.metadata or {})
        splits = metadata.pop("dataset_split", None) or []
        if all(metadata.get(k) == v for k, v in labels.items()) and (
            split is None or split in splits
        ):
            continue
        update: dict[str, Any] = {"id": current.id, "metadata": {**metadata, **labels}}
        if split is not None:
            update["split"] = split
        updates.append(update)
    return updates


def experiment_rows(
    suite: str,
    result: dict[str, Any],
    link: Callable[[str], str | None] = lambda trace_id: None,
) -> dict[str, dict[str, Any]]:
    """Each case's outputs as the experiment will show them, keyed by example id.

    ``link`` turns a case's ``trace_id`` into the URL of its trace.
    """
    rows = {_case_id(case): dict(case) for case in result.get("cases") or []}
    for row in rows.values():
        if suite == "writing_evaluator" and result.get("task") and "task" not in row:
            row["task"] = result["task"]
        if row.get("trace_id"):
            url = link(row["trace_id"])
            if url:
                row["trace_url"] = url
    return rows


def _trace_linker(client) -> Callable[[str], str | None]:
    """Build trace URLs locally, after one lookup of the tracing project.

    The run's own ``get_url()`` asks the backend once per run; this asks once.
    """
    from types import SimpleNamespace

    from cae.config import get_settings

    try:
        project_id = client.read_project(project_name=get_settings().langsmith_project).id
    except Exception:
        return lambda trace_id: None

    def link(trace_id: str) -> str | None:
        try:
            run = SimpleNamespace(id=trace_id, session_id=project_id)
            return client._construct_run_url(run=run)
        except Exception:
            return None

    return link


def upload_run(
    *,
    suite: str,
    model: str,
    result: dict[str, Any],
    tiers: Iterable[str] = (),
) -> str:
    """Upload one model's run of one suite as a LangSmith experiment.

    ``tiers`` are the tiers ``model`` was pinned to; the others kept their chain.
    """
    import langsmith as ls

    client = ls.Client()
    dataset = ensure_dataset(client, suite)
    if any(case.get("trace_id") for case in result.get("cases") or []):
        by_id = experiment_rows(suite, result, _trace_linker(client))
    else:
        by_id = experiment_rows(suite, result)

    def target(inputs: dict) -> dict:
        return by_id.get(inputs["id"], {"ok": False, "missing": True})

    pinned = sorted(tiers)
    group = result.get("group")
    prefix = (
        f"{suite}-{model}"
        + (f"-{'+'.join(pinned)}" if pinned else "")
        + (f"-g{group}" if group else "")
    )
    experiment = ls.evaluate(
        target,
        data=examples_run(client.list_examples(dataset_id=dataset.id), by_id),
        evaluators=EVALUATORS[suite],
        experiment_prefix=prefix,
        metadata={
            "model": model,
            "suite": suite,
            "pinned_tiers": ",".join(pinned),
            **({"group": group} if group else {}),
            **(result.get("metrics") or {}),
        },
        client=client,
    )
    return _experiment_url(experiment, client, dataset)


def _experiment_url(experiment: Any, client, dataset) -> str:
    """Best effort: the SDK has moved this around between versions."""
    for attribute in ("experiment_url", "url"):
        value = getattr(experiment, attribute, None)
        if isinstance(value, str):
            return value
    name = getattr(experiment, "experiment_name", None)
    return f"{client.api_url.replace('api.', '')} (experiment {name or 'created'})"


def iter_suites(names: Iterable[str]) -> list[str]:
    return [n for n in names if n in DATASET_NAMES]


__all__ = [
    "DATASET_NAMES",
    "EVALUATORS",
    "ensure_dataset",
    "examples_run",
    "experiment_rows",
    "stale_labels",
    "upload_run",
]
