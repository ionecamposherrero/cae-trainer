"""Combine the coach reports of the profile groups into one comparison.

    python scripts/combine_coach_reports.py                 # every -g<N> report
    python scripts/combine_coach_reports.py a.json b.json   # these reports

The coach twin of ``combine_writing_evaluator_reports.py``, with the same rules.
Each ``run_evals.py --suite coach --group N`` writes its own report, and a group
may be run more than once. For every model and learner the most recent case
counts, so a later report replaces only the models it ran -- except a report the
model did not write (no model was there, or its reply could not be read). That
never replaces an earlier result for the same learner. A judged report
(``judge_coach_report.py``) is newer than the one it judged, so its reports,
verdicts included, win.

The kept cases are pooled per model and the suite's metrics -- the judge's
verdicts, the only measure of a report -- recomputed over all of them with the
suite's own function (``suites.coach_metrics``), overall and per archetype, not
as a mean of the groups' figures. A model missing from a group is reported as
incomplete rather than compared on fewer learners unannounced.

Writes ``evals/reports/<timestamp>-coach-combined.json``. No API calls.
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
from evals.suites import coach_groups, coach_metrics  # noqa: E402
from scripts.combine_generation_reports import (  # noqa: E402
    ReportError,
    grouped_reports,
    latest_cases,
    load_reports,
)
from scripts.run_evals import print_by_archetype  # noqa: E402


def unwritten(case: dict[str, Any]) -> bool:
    """Whether the model wrote no usable report for this learner."""
    return not case.get("ok")


def combine(reports: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entry per model: its latest report on every learner, and metrics over them.

    ``reports`` are oldest first, as ``load_reports`` returns them.
    """
    groups, kept, suite_errors, routes = latest_cases(reports, "coach", unwritten)

    combined = []
    for model in routes:
        cases: list[dict[str, Any]] = []
        per_group: dict[int, dict[str, Any]] = {}
        for group in groups:
            learners = kept.get((model, group))
            if not learners:
                why = suite_errors.get((model, group), "not in that group's sweeps")
                per_group[group] = {"error": why}
                continue
            group_cases = sorted(learners.values(), key=lambda c: c["id"])
            cases.extend(group_cases)
            per_group[group] = {
                **coach_metrics(group_cases),
                "sources": sorted({c["source"] for c in group_cases}),
            }
        combined.append(
            {
                "model": model,
                "complete": all("error" not in per_group[g] for g in groups),
                "groups": per_group,
                "metrics": coach_metrics(cases),
                "routes": dict(routes[model]),
                "cases": cases,
            }
        )
    return combined


def _cell(value: Any) -> str:
    return "-" if value is None else f"{value:g}"


def print_combined(combined: list[dict[str, Any]], groups: list[int]) -> None:
    width = max([len(e["model"]) for e in combined] + [5]) + 2
    print("\n=== coach, groups combined ===")
    print(
        f"{'model':<{width}}"
        + "".join(f"{'g' + str(g):>5}" for g in groups)
        + f"{'reports':>9}{'judged':>8}{'correct':>9}{'faithful':>10}{'helpful':>9}"
        f"{'unavail':>9}{'failed':>8}"
    )

    def rank(entry: dict[str, Any]) -> tuple[float, float]:
        m = entry["metrics"]
        return (-(m.get("judge_correctness_mean") or 0.0), -(m.get("judge_faithfulness_mean") or 0.0))

    for e in sorted(combined, key=rank):
        m = e["metrics"]
        print(
            f"{e['model']:<{width}}"
            + "".join(f"{'ERR' if 'error' in e['groups'][g] else 'ok':>5}" for g in groups)
            + f"{m['profiles']:>9}{m.get('judged', 0):>8}"
            f"{_cell(m.get('judge_correctness_mean')):>9}{_cell(m.get('judge_faithfulness_mean')):>10}"
            f"{_cell(m.get('judge_helpfulness')):>9}{m['unavailable']:>9}{m['failed']:>8}"
            + ("" if e["complete"] else "   INCOMPLETE")
        )

    for e in sorted(combined, key=rank):
        print(f"\n=== {e['model']}, per archetype, every group pooled ===")
        print_by_archetype(e["metrics"]["by_archetype"], indent="")
        for g, metrics in e["groups"].items():
            if "error" in metrics:
                print(f"group {g}: {metrics['error']}")
    print(
        "\ncorrect, faithful and helpful are the judge's verdicts, averaged on their own "
        "scales (0-2, 0-2, 1-5; '-' where no report was judged): correct against the "
        "profile's reference review, faithful against the figures and tables the coach "
        "was handed. all right and all true (per archetype) are the shares of reports "
        "with the top mark. unavail counts the reports no model was there for, failed "
        "those it answered unusably: neither is judged."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "reports",
        nargs="*",
        type=Path,
        help="coach reports, any number per group; default: every one in "
        "evals/reports/ that was run with --group",
    )
    args = parser.parse_args()

    paths = args.reports or grouped_reports(suite="coach")
    if not paths:
        print(f"No coach report with a profile group in {REPORTS_DIR}.")
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
    missing = sorted(set(coach_groups()) - set(groups))
    if missing:
        print(f"  (no report yet for group {', '.join(map(str, missing))})")

    combined = combine(reports)
    print_combined(combined, groups)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suite": "coach",
        "groups": groups,
        "sources": [r["source"] for r in reports],
        "models": combined,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = REPORTS_DIR / f"{stamp}-coach-combined.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\ncombined report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
