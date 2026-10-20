"""Combine the writing_evaluator reports of the script groups into one comparison.

    python scripts/combine_writing_evaluator_reports.py                 # every -g<N> report
    python scripts/combine_writing_evaluator_reports.py a.json b.json   # these reports

The writing_evaluator twin of ``combine_marking_reports.py``, with the same rules. Each
``run_evals.py --suite writing_evaluator --group N`` writes its own report, and a group may
be run more than once. For every model and script the most recent case counts,
so a later report replaces only the models it ran -- except a script the model
did not really mark: one that raised, or one with a degraded subscale (a band 0
no examiner awarded, usually a spent allowance part-way through the group).
That never replaces an earlier result of the same script.

The kept scripts are pooled per model and the suite's metrics recomputed over
all of them with the suite's own function (``suites.writing_evaluator_metrics``),
overall and per genre: band accuracy is a mean over every scored script, and the
annotation rates are pooled over every labelled error, not means of the groups'
figures. A model missing from a group is reported as incomplete rather than
compared on fewer scripts unannounced.

Writes ``evals/reports/<timestamp>-writing_evaluator-combined.json``. No API calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals import REPORTS_DIR  # noqa: E402
from evals.suites import writing_evaluator_groups, writing_evaluator_metrics  # noqa: E402
from scripts.combine_generation_reports import (  # noqa: E402
    ReportError,
    grouped_reports,
    latest_cases,
    load_reports,
)
from scripts.run_evals import print_by_genre  # noqa: E402


def unmarked(case: dict[str, Any]) -> bool:
    """Whether a script went without a real marking: it raised, or a subscale or
    the annotation node degraded."""
    return (
        not case.get("ok")
        or bool(case.get("degraded_subscales"))
        or bool(case.get("annotate_degraded"))
    )


def combine(reports: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entry per model: its latest result of every script, and metrics over them.

    ``reports`` are oldest first, as ``load_reports`` returns them.
    """
    groups, kept, suite_errors, routes = latest_cases(reports, "writing_evaluator", unmarked)

    combined = []
    for model in routes:
        cases: list[dict[str, Any]] = []
        per_group: dict[int, dict[str, Any]] = {}
        for group in groups:
            scripts = kept.get((model, group))
            if not scripts:
                why = suite_errors.get((model, group), "not in that group's sweeps")
                per_group[group] = {"error": why}
                continue
            group_cases = sorted(scripts.values(), key=lambda c: c["id"])
            cases.extend(group_cases)
            per_group[group] = {
                **writing_evaluator_metrics(group_cases),
                "sources": sorted({c["source"] for c in group_cases}),
            }
        combined.append(
            {
                "model": model,
                "complete": all("error" not in per_group[g] for g in groups),
                "groups": per_group,
                "metrics": writing_evaluator_metrics(cases),
                "routes": dict(routes[model]),
                "cases": cases,
            }
        )
    return combined


def _cell(value: Any) -> str:
    return "-" if value is None else f"{value:g}"


def print_combined(combined: list[dict[str, Any]], groups: list[int]) -> None:
    width = max([len(e["model"]) for e in combined] + [5]) + 2
    print("\n=== writing_evaluator, groups combined ===")
    print(
        f"{'model':<{width}}"
        + "".join(f"{'g' + str(g):>5}" for g in groups)
        + f"{'scripts':>9}{'total acc':>11}{'recall':>8}{'precision':>11}{'category':>10}"
        f"{'degraded':>10}{'failed':>8}"
    )

    def rank(entry: dict[str, Any]) -> float:
        return -(entry["metrics"]["accuracy_total"] or 0.0)

    for e in sorted(combined, key=rank):
        m = e["metrics"]
        print(
            f"{e['model']:<{width}}"
            + "".join(f"{'ERR' if 'error' in e['groups'][g] else 'ok':>5}" for g in groups)
            + f"{m['scripts']:>9}{_cell(m['accuracy_total']):>11}{_cell(m['recall']):>8}"
            f"{_cell(m['precision']):>11}{_cell(m['accuracy_category']):>10}"
            f"{m['degraded_scripts'] + m['degraded_annotations']:>10}{m['failed_scripts']:>8}"
            + ("" if e["complete"] else "   INCOMPLETE")
        )

    for e in sorted(combined, key=rank):
        print(f"\n=== {e['model']}, per genre, every group pooled ===")
        print_by_genre(e["metrics"]["by_genre"], indent="")
        for g, metrics in e["groups"].items():
            if "error" in metrics:
                print(f"group {g}: {metrics['error']}")
    print(
        "\nBand accuracy is 1 - |error| / maximum, averaged over the scripts: per "
        "subscale out of 5, and on the total out of 20 (where a band too high and one "
        "too low cancel, as in the learner's mark). For the annotate node, recall is the "
        "labelled errors it found, precision the annotations that were real errors "
        "(labelled or tolerated), category the errors found with an accepted category; "
        "all pooled over the scripts, categories aside for recall and precision. "
        "degraded counts the scripts with a subscale no examiner marked or no "
        "annotation (quota, provider down, an unreadable reply) and failed those that "
        "raised: their figures are left out."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "reports",
        nargs="*",
        type=Path,
        help="writing_evaluator reports, any number per group; default: every one in "
        "evals/reports/ that was run with --group",
    )
    args = parser.parse_args()

    paths = args.reports or grouped_reports(suite="writing_evaluator")
    if not paths:
        print(f"No writing_evaluator report with a script group in {REPORTS_DIR}.")
        return 1
    try:
        reports = load_reports(paths)
    except ReportError as exc:
        print(exc)
        return 1

    groups = sorted({int(r["group"]) for r in reports})
    print("Combining, oldest first (for each model and group the newest counts):")
    for report in reports:
        models = ", ".join(run["model"] for run in report.get("runs", []))
        print(f"  g{report['group']}  {report['source']}  [{models}]")
    missing = sorted(set(writing_evaluator_groups()) - set(groups))
    if missing:
        print(f"  (no report yet for group {', '.join(map(str, missing))})")

    combined = combine(reports)
    print_combined(combined, groups)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suite": "writing_evaluator",
        "groups": groups,
        "sources": [r["source"] for r in reports],
        "models": combined,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = REPORTS_DIR / f"{stamp}-writing_evaluator-combined.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\ncombined report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
