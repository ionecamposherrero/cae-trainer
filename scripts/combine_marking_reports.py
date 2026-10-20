"""Combine the marking reports of the case groups into one comparison.

    python scripts/combine_marking_reports.py                 # every -g<N> report
    python scripts/combine_marking_reports.py a.json b.json   # these reports

The marking twin of ``combine_generation_reports.py``, with the same rules. Each
``run_evals.py --suite marking --group N`` writes its own report, and a group may
be run more than once. For every model and paper the most recent case counts, so
a later report replaces only the models it ran -- except a paper the model was
unavailable for (spent quota, a provider down, no key), which measures the
provider and never replaces an earlier result of the same paper.

The kept papers are pooled per model and the suite's metrics recomputed over all
of them with the suite's own function (``suites.marking_metrics``), so the
combined accuracy is over every paper, not a mean of the groups' figures. A
model missing from a group is reported as incomplete rather than compared on
fewer papers unannounced.

Writes ``evals/reports/<timestamp>-marking-combined.json``. No API calls.
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
from evals.suites import marking_groups, marking_metrics  # noqa: E402
from scripts.combine_generation_reports import (  # noqa: E402
    ReportError,
    grouped_reports,
    latest_cases,
    load_reports,
)

PARTS = (1, 2, 3, 4)


def unavailable(case: dict[str, Any]) -> bool:
    """Whether a paper went unmarked for want of a model, as ``run_marking`` records it."""
    return bool(case.get("unavailable"))


def combine(reports: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entry per model: its latest result of every paper, and metrics over them.

    ``reports`` are oldest first, as ``load_reports`` returns them.
    """
    groups, kept, suite_errors, routes = latest_cases(reports, "marking", unavailable)

    combined = []
    for model in routes:
        cases: list[dict[str, Any]] = []
        per_group: dict[int, dict[str, Any]] = {}
        for group in groups:
            papers = kept.get((model, group))
            if not papers:
                why = suite_errors.get((model, group), "not in that group's sweeps")
                per_group[group] = {"error": why}
                continue
            group_cases = sorted(papers.values(), key=lambda c: (c.get("part", 0), c["id"]))
            cases.extend(group_cases)
            per_group[group] = {
                **marking_metrics(group_cases),
                "sources": sorted({c["source"] for c in group_cases}),
            }
        combined.append(
            {
                "model": model,
                "complete": all("error" not in per_group[g] for g in groups),
                "groups": per_group,
                "metrics": marking_metrics(cases),
                "routes": dict(routes[model]),
                "cases": cases,
            }
        )
    return combined


def _cell(value: Any) -> str:
    return "-" if value is None else f"{value:g}"


def _pair(metrics: dict[str, Any]) -> str:
    """``adjudication/tagging`` for one group, or ERR when it produced nothing."""
    if "error" in metrics:
        return "ERR"
    return f"{_cell(metrics.get('adjudication_accuracy'))}/{_cell(metrics.get('category_exact'))}"


def print_combined(combined: list[dict[str, Any]], groups: list[int]) -> None:
    width = max([len(e["model"]) for e in combined] + [5]) + 2
    print("\n=== marking, groups combined ===")
    print(
        f"{'model':<{width}}"
        + "".join(f"{'g' + str(g) + ' adj/tag':>15}" for g in groups)
        + f"{'adjud':>8}{'balanced':>10}{'tagging':>9}{'family':>8}{'over+':>7}{'over-':>7}"
        f"{'unavail':>9}{'failed':>8}"
    )

    def rank(entry: dict[str, Any]) -> float:
        return -(entry["metrics"]["adjudication_balanced"] or 0.0)

    for e in sorted(combined, key=rank):
        m = e["metrics"]
        print(
            f"{e['model']:<{width}}"
            + "".join(f"{_pair(e['groups'][g]):>15}" for g in groups)
            + f"{_cell(m['adjudication_accuracy']):>8}{_cell(m['adjudication_balanced']):>10}"
            f"{_cell(m['category_exact']):>9}"
            f"{_cell(m['category_family']):>8}{m['over_accepted']:>7}{m['over_rejected']:>7}"
            f"{m['unavailable_papers']:>9}{m['failed_papers']:>8}"
            + ("" if e["complete"] else "   INCOMPLETE")
        )

    print("\n=== per part, pooled over every group (adjudication / tagging) ===")
    print(f"{'model':<{width}}" + "".join(f"{'P' + str(p):>14}" for p in PARTS))
    for e in sorted(combined, key=lambda e: e["model"]):
        adjudication = e["metrics"]["adjudication_by_part"]
        tagging = e["metrics"]["category_exact_by_part"]
        cells = "".join(
            f"{_cell(adjudication.get(p)) + '/' + _cell(tagging.get(p)):>14}" for p in PARTS
        )
        print(f"{e['model']:<{width}}{cells}")

    for e in combined:
        routes = e["metrics"]["part4_adjudication_by_route"]
        if routes:
            shown = ", ".join(f"{route} {_cell(rate)}" for route, rate in routes.items())
            print(f"\n{e['model']}, Part 4 adjudication by route: {shown}")
        for g, metrics in e["groups"].items():
            if "error" in metrics:
                print(f"\n{e['model']}, group {g}: {metrics['error']}")
    print(
        "\nadjud and tagging are over every paper of every group; balanced weighs each "
        "verdict the same, so rejecting everything scores about 0.44 instead of 0.7; "
        "over+ counts wrong "
        "answers given credit, over- right answers denied it; unavail counts the papers "
        "no model answered (quota, provider down), which say nothing about the model; "
        "failed the papers it answered unusably, which do. Neither has figures. Part 1 is never "
        "adjudicated, so its column shows tagging alone."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "reports",
        nargs="*",
        type=Path,
        help="marking reports, any number per group; default: every one in "
        "evals/reports/ that was run with --group",
    )
    args = parser.parse_args()

    paths = args.reports or grouped_reports(suite="marking")
    if not paths:
        print(f"No marking report with a case group in {REPORTS_DIR}.")
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
    missing = sorted(set(marking_groups()) - set(groups))
    if missing:
        print(f"  (no report yet for group {', '.join(map(str, missing))})")

    combined = combine(reports)
    print_combined(combined, groups)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suite": "marking",
        "groups": groups,
        "sources": [r["source"] for r in reports],
        "models": combined,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = REPORTS_DIR / f"{stamp}-marking-combined.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\ncombined report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
