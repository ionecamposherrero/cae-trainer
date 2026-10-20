"""Run the evaluation suites against the real API.

    python scripts/run_evals.py --suite writing_evaluator
    python scripts/run_evals.py --suite all
    python scripts/run_evals.py --suite writing_evaluator --repeat 3
    python scripts/run_evals.py --suite marking --models gemini-3.5-flash-lite,mistral:ministral-14b-latest
    python scripts/run_evals.py --suite chat --chat-models mistral:ministral-14b-latest,groq:openai/gpt-oss-120b
    python scripts/run_evals.py --suite writing_evaluator --upload
    python scripts/run_evals.py --suite generation --models a,b --tier rue_task
    python scripts/run_evals.py --suite generation --group 2 --models a,b --tier rue_task
    python scripts/run_evals.py --suite marking --models a,b --tier rue_evaluator
    python scripts/run_evals.py --suite marking --group 3 --models a,b
    python scripts/run_evals.py --suite writing_task --group 1 --models a,b --tier writing_task
    python scripts/run_evals.py --suite writing_evaluator --group 1 --models a,b --tier writing_evaluator
    python scripts/run_evals.py --suite coach --group 1
    python scripts/run_evals.py --suite coach --archetype threshold
    python scripts/run_evals.py --suite coach --group 1 --no-judge
    python scripts/run_evals.py --suite coach --group 1 --models a,b --tier coach
    python scripts/run_evals.py --suite chat --group 1
    python scripts/run_evals.py --suite chat --group 1 --kind exam --without-index
    python scripts/run_evals.py --suite chat --group 1 --no-judge

Every run writes ``evals/reports/<timestamp>-<suite>.json`` (``-g<N>`` appended
for one group of the generation, marking, writing_task, writing, coach or chat
dataset, ``--group``; ``-noindex`` before it for a ``--without-index`` run). ``--models`` runs the suite once per model, with the
``fast``, ``smart``, ``rue_task``, ``rue_evaluator``, ``writing_task``,
``writing_evaluator`` and ``coach`` tiers pinned to it, or only the one named by
``--tier`` (the others keep their chain);
``--chat-models`` does the same for the ``chat`` tier and runs only the chat
suite. A pinned tier is a chain of one, so there is no fallback: a sweep measures
that model and nothing else. Every result records which model answered each call.
A suite that calls none of the pinned tiers is skipped (``SUITE_TIERS``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cae import llm  # noqa: E402
from cae.config import TIER_SETTINGS, get_settings  # noqa: E402
from cae.domain.parts import WRITING_PART1_GENRES, WRITING_PART2_GENRES  # noqa: E402
from cae.llm import QuotaExhaustedError  # noqa: E402
from cae.observability import configure_tracing  # noqa: E402
from evals import REPORTS_DIR  # noqa: E402
from evals import suites as suite_module  # noqa: E402

SUITE_NAMES = ("generation", "marking", "writing_task", "writing_evaluator", "coach", "chat")
GROUPED_SUITES = frozenset(
    {"generation", "marking", "writing_task", "writing_evaluator", "coach", "chat"}
)

SUITE_TIERS: dict[str, frozenset[str]] = {
    "generation": frozenset({"rue_task"}),
    "marking": frozenset({"rue_evaluator"}),
    "writing_task": frozenset({"writing_task"}),
    "writing_evaluator": frozenset({"writing_evaluator"}),
    "coach": frozenset({"coach"}),
    "chat": frozenset({"chat"}),
}

PINNABLE_TIERS: tuple[str, ...] = (
    "fast", "smart", "rue_task", "rue_evaluator", "writing_task", "writing_evaluator", "coach"
)

TIER_CHOICES: dict[str, tuple[str, ...]] = {
    "all": PINNABLE_TIERS,
    **{tier: (tier,) for tier in PINNABLE_TIERS},
}


def measures(suite: str, tiers: tuple[str, ...]) -> bool:
    """Whether pinning ``tiers`` changes anything ``suite`` calls."""
    return bool(SUITE_TIERS[suite] & set(tiers))


def use_model(model: str, *, tiers: tuple[str, ...] = PINNABLE_TIERS) -> None:
    """Pin ``tiers`` to one model, with no fallback behind it.

    Both ``get_settings`` and the built clients are cached, so both are cleared.
    """
    for tier in tiers:
        os.environ[f"CAE_{tier.upper()}_MODEL"] = model
    get_settings.cache_clear()
    llm.clear_caches()


def current_models() -> dict[str, str]:
    settings = get_settings()
    return {tier: settings.chain(tier) for tier in TIER_SETTINGS}


def run_suite(name: str, args: argparse.Namespace) -> dict[str, Any]:
    if name == "generation":
        return suite_module.run_generation(
            parts=[args.part] if args.part else None, group=args.group
        )
    if name == "marking":
        return suite_module.run_marking(group=args.group)
    if name == "writing_task":
        return suite_module.run_writing_task(
            genres=[args.genre] if args.genre else None, group=args.group
        )
    if name == "writing_evaluator":
        return suite_module.run_writing_evaluator(
            group=args.group, genres=[args.genre] if args.genre else None, repeat=args.repeat
        )
    if name == "coach":
        return suite_module.run_coach(
            group=args.group,
            archetypes=[args.archetype] if args.archetype else None,
            judge=not args.no_judge,
        )
    if name == "chat":
        return suite_module.run_chat(
            group=args.group,
            kinds=[args.kind] if args.kind else None,
            judge=not args.no_judge,
            without_index=args.without_index,
        )
    raise ValueError(f"Unknown suite {name!r}")


def run_for_model(
    model: str | None,
    suite_names: list[str],
    args: argparse.Namespace,
    *,
    tiers: tuple[str, ...] = PINNABLE_TIERS,
) -> dict[str, Any]:
    """Run the suites with ``model`` pinned to ``tiers``; the rest keep their chain."""
    if model:
        use_model(model, tiers=tiers)
    label = model or "configured"
    print(f"\n=== {label} ({current_models()}) ===")

    results: dict[str, Any] = {}
    for name in suite_names:
        if model and not measures(name, tiers):
            print(
                f"\n-- {name} --\n  skipped: it calls {'/'.join(sorted(SUITE_TIERS[name]))}, "
                f"and this sweep pins only {'/'.join(tiers)}"
            )
            continue
        print(f"\n-- {name} --")
        started = time.perf_counter()
        llm.reset_route_usage()
        try:
            results[name] = run_suite(name, args)
        except QuotaExhaustedError as exc:
            print(f"  QUOTA EXHAUSTED: {exc}")
            results[name] = {"suite": name, "quota_exhausted": True, "error": str(exc)}
            break
        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            results[name] = {"suite": name, "error": f"{type(exc).__name__}: {exc}"}
            continue
        results[name]["routes"] = {
            f"{tier} {route}": calls for (tier, route), calls in llm.route_usage().items()
        }
        print_suite(results[name])
        print(f"  ({time.perf_counter() - started:.1f}s)")
    return {
        "model": label,
        "pinned_tiers": list(tiers) if model else [],
        "models": current_models(),
        "suites": results,
    }


def print_suite(result: dict[str, Any]) -> None:
    if result.get("skipped"):
        print(f"  skipped: {result['skipped']}")
        return
    for key, value in (result.get("metrics") or {}).items():
        if key == "by_genre" and result.get("suite") == "writing_evaluator":
            print_by_genre(value)
        elif key == "by_archetype":
            print_by_archetype(value)
        elif key == "by_kind":
            print_by_kind(value)
        elif key == "repair_reasons":
            print(f"  {key:<28} {'none' if not value else ''}")
            for rule, times in value:
                print(f"    {times:>3}x  {rule}")
        elif isinstance(value, float):
            print(f"  {key:<28} {value:g}")
        else:
            print(f"  {key:<28} {value}")
    routes = result.get("routes") or {}
    if routes:
        answered = ", ".join(f"{route} x{calls}" for route, calls in sorted(routes.items()))
        print(f"  {'answered by':<28} {answered}")
    for case in result.get("cases") or []:
        if case.get("ok") is False:
            print(f"  FAILED {case.get('id', case.get('part'))}: {case.get('error')}")


GENRE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("accuracy_content", "content"),
    ("accuracy_communicative_achievement", "comm"),
    ("accuracy_organisation", "organis"),
    ("accuracy_language", "language"),
    ("accuracy_total", "total"),
    ("recall", "recall"),
    ("precision", "precision"),
    ("accuracy_category", "category"),
)


def print_by_genre(by_genre: dict[str, dict[str, Any]], indent: str = "  ") -> None:
    """One row per genre: band accuracy per subscale and total, then annotation."""
    print(f"{indent}{'by genre':<14}" + "".join(f"{title:>10}" for _, title in GENRE_COLUMNS))
    for genre, scores in by_genre.items():
        cells = "".join(
            f"{'-':>10}" if scores.get(key) is None else f"{scores[key]:>10g}"
            for key, _ in GENRE_COLUMNS
        )
        print(f"{indent}{genre:<14}{cells}")


ARCHETYPE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("profiles", "reports"),
    ("judge_correctness_mean", "correct/2"),
    ("judge_faithfulness_mean", "faithful/2"),
    ("judge_helpfulness", "helpful/5"),
    ("judge_correct_rate", "all right"),
    ("judge_faithful_rate", "all true"),
)


def print_by_archetype(by_archetype: dict[str, dict[str, Any]], indent: str = "  ") -> None:
    """One row per kind of learner: what the judge made of the coach's reports."""
    print(
        f"{indent}{'by archetype':<14}"
        + "".join(f"{title:>10}" for _, title in ARCHETYPE_COLUMNS)
    )
    for archetype, scores in by_archetype.items():
        cells = "".join(
            f"{'-':>10}" if scores.get(key) is None else f"{scores[key]:>10g}"
            for key, _ in ARCHETYPE_COLUMNS
        )
        print(f"{indent}{archetype:<14}{cells}")


KIND_COLUMNS: tuple[tuple[str, str], ...] = (
    ("judge_correctness_mean", "correct/2"),
    ("judge_faithfulness_mean", "faithful/2"),
    ("judge_helpfulness", "helpful/5"),
    ("tool_choice_rate", "tools"),
    ("mean_tool_rounds", "rounds"),
    ("section_hit_rate", "section"),
)


def print_by_kind(by_kind: dict[str, dict[str, Any]], indent: str = "  ") -> None:
    """One row per kind of chat turn: what the chat got right on each."""
    print(f"{indent}{'by kind':<14}" + "".join(f"{title:>10}" for _, title in KIND_COLUMNS))
    for kind, scores in by_kind.items():
        cells = "".join(
            f"{'-':>10}" if scores.get(key) is None else f"{scores[key]:>10g}"
            for key, _ in KIND_COLUMNS
        )
        print(f"{indent}{kind:<14}{cells}")


HEADLINE: dict[str, list[tuple[str, str]]] = {
    "generation": [("rate", "validity")],
    "marking": [
        ("adjudication_accuracy", "adjudication"),
        ("adjudication_balanced", "adj balanced"),
        ("category_exact", "tagging"),
        ("unavailable_papers", "unavailable"),
        ("failed_papers", "failed"),
    ],
    "writing_task": [
        ("rate", "task validity"),
        ("first_try_rate", "first try"),
        ("mean_repairs", "repairs"),
    ],
    "writing_evaluator": [
        ("accuracy_total", "band acc total"),
        ("recall", "annot recall"),
        ("precision", "annot precision"),
        ("accuracy_category", "annot category"),
        ("degraded_scripts", "degraded"),
    ],
    "coach": [
        ("judge_correctness_mean", "coach correct /2"),
        ("judge_faithfulness_mean", "coach faithful /2"),
        ("judge_helpfulness", "coach helpful /5"),
        ("mean_seconds", "s / report"),
    ],
    "chat": [
        ("judge_correctness_mean", "correct /2"),
        ("judge_faithfulness_mean", "faithful /2"),
        ("judge_helpfulness", "helpful /5"),
        ("tool_choice_rate", "tool choice"),
        ("sql_errors", "SQL errors"),
        ("mean_tool_rounds", "tool rounds"),
        ("mean_seconds", "s / turn"),
    ],
}


def print_comparison(runs: list[dict[str, Any]], suite_names: list[str]) -> None:
    if len(runs) < 2:
        return
    columns = [(n, key, title) for n in suite_names for key, title in HEADLINE.get(n, [])]
    print("\n=== comparison ===")
    print(f"{'model':<40}" + "".join(f"{title:>16}" for _, _, title in columns))
    for run in runs:
        cells = []
        for name, key, _ in columns:
            value = (run["suites"].get(name, {}).get("metrics") or {}).get(key)
            cells.append(f"{'-':>16}" if value is None else f"{value:>16g}")
        print(f"{run['model']:<40}" + "".join(cells))
    print(
        "\nLower is better for SQL errors, tool rounds, seconds, repairs, degraded scripts and "
        "unavailable or failed papers; higher for everything else. The writing_evaluator "
        "figures per genre, the coach figures per archetype and the chat figures per kind "
        "of turn are in each model's section above. A '-' means nothing was answered to score."
    )


def write_report(payload: dict[str, Any], suite_label: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = REPORTS_DIR / f"{stamp}-{suite_label}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def upload(runs: list[dict[str, Any]], suite_names: list[str]) -> None:
    """Push each model's results to LangSmith as an experiment. Uses no quota."""
    from evals.upload import upload_run

    for name in suite_names:
        for run in runs:
            result = run["suites"].get(name)
            if not result or result.get("error") or result.get("skipped"):
                continue
            try:
                url = upload_run(
                    suite=name,
                    model=run["model"],
                    result=result,
                    tiers=run.get("pinned_tiers") or (),
                )
            except Exception as exc:
                print(f"  upload failed for {name}/{run['model']}: {type(exc).__name__}: {exc}")
                continue
            print(f"  uploaded {name}/{run['model']}: {url}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--suite", default="all", choices=(*SUITE_NAMES, "all"), help="which suite to run"
    )
    parser.add_argument("--part", type=int, help="with --suite generation, just one part")
    parser.add_argument(
        "--genre",
        choices=(*WRITING_PART1_GENRES, *WRITING_PART2_GENRES),
        help="with --suite writing_task or writing_evaluator, just one genre",
    )
    parser.add_argument(
        "--archetype",
        choices=suite_module.ARCHETYPES,
        help="with --suite coach, just one kind of learner",
    )
    parser.add_argument(
        "--kind",
        choices=suite_module.CHAT_KINDS,
        help="with --suite chat, just one kind of turn",
    )
    parser.add_argument(
        "--group",
        type=int,
        choices=sorted(
            set(suite_module.generation_groups())
            | set(suite_module.marking_groups())
            | set(suite_module.writing_task_groups())
            | set(suite_module.writing_evaluator_groups())
            | set(suite_module.coach_groups())
            | set(suite_module.chat_groups())
        ),
        help="with --suite generation, one group of seeds (one topic per part); with "
        "--suite marking, one group of four papers, one per part; with --suite "
        "writing_task, one group of five tasks, one per genre; with --suite "
        "writing_evaluator, one group of five scripts, one per genre (30 calls); with "
        "--suite coach, one group of five learners, one per archetype (5 calls); with "
        "--suite chat, one learner's fourteen turns. Omitted: every group",
    )
    parser.add_argument(
        "--models",
        default="",
        help="comma-separated models to sweep; fast, smart, rue_task, rue_evaluator, "
        "writing_task, writing_evaluator and coach are pinned to each in turn",
    )
    parser.add_argument(
        "--tier",
        default="all",
        choices=tuple(TIER_CHOICES),
        help="with --models, pin only this tier; the others keep their configured chain",
    )
    parser.add_argument(
        "--chat-models",
        default="",
        help="comma-separated provider:model ids to sweep on the chat suite",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="with --suite writing_evaluator, mark each script N times and report the spread",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="with --suite chat or coach, skip the paid judge (OPENAI_API_KEY, "
        "CAE_JUDGE_MODEL, one call a turn or a report). The chat keeps only its tool "
        "metrics; the coach's reports are kept unscored, to judge later with "
        "judge_coach_report.py",
    )
    parser.add_argument(
        "--without-index",
        action="store_true",
        help="with --suite chat, take the reference corpus away, as a fresh clone has it",
    )
    parser.add_argument(
        "--upload", action="store_true", help="also push the results to LangSmith"
    )
    args = parser.parse_args()

    settings = get_settings()
    if not any(settings.tier_available(tier) for tier in (*PINNABLE_TIERS, "chat")):
        print(
            "No provider key is set. Copy .env.example to .env and add at least "
            "GOOGLE_API_KEY (all the providers used are free)."
        )
        return 1

    configure_tracing()
    suite_names = list(SUITE_NAMES) if args.suite == "all" else [args.suite]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    chat_models = [m.strip() for m in args.chat_models.split(",") if m.strip()]
    if chat_models and "chat" not in suite_names:
        suite_names.append("chat")
    if chat_models and not models:
        suite_names = ["chat"]
    pinned = TIER_CHOICES[args.tier]
    if models:
        suite_names = [
            n for n in suite_names if measures(n, pinned) or (n == "chat" and chat_models)
        ]

    if models or chat_models:
        runs = [run_for_model(model, suite_names, args, tiers=pinned) for model in models]
        runs += [
            run_for_model(model, suite_names, args, tiers=("chat",)) for model in chat_models
        ]
    else:
        runs = [run_for_model(None, suite_names, args)]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suites": suite_names,
        "repeat": args.repeat,
        "group": args.group,
        "runs": runs,
    }
    label = args.suite
    if args.without_index and "chat" in suite_names:
        label += "-noindex"
    if args.group and GROUPED_SUITES & set(suite_names):
        label += f"-g{args.group}"
    path = write_report(payload, label)
    print_comparison(runs, suite_names)
    print(f"\nreport written to {path}")

    if args.upload:
        print("\nuploading to LangSmith...")
        upload(runs, suite_names)

    failed = [
        name
        for run in runs
        for name, result in run["suites"].items()
        if result.get("error") or result.get("quota_exhausted")
    ]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
