"""Combine the chat reports of the learner groups into one comparison.

    python scripts/combine_chat_reports.py                 # every -g<N> report
    python scripts/combine_chat_reports.py a.json b.json   # these reports

The chat twin of ``combine_coach_reports.py``, with the same rules. Each
``run_evals.py --suite chat --group N`` writes its own report, and a group may be
run more than once. For every model and turn the most recent case counts, so a
later report replaces only the models it ran -- except a turn the model did not
answer (no model was there, or its reply could not be read). That never replaces
an earlier result for the same turn. A judged report (``judge_chat_report.py``)
is newer than the one it judged, so its turns, verdicts included, win.

The kept turns are pooled per model and the suite's metrics recomputed over all
of them with the suite's own function (``suites.chat_metrics``), overall and per
kind of turn: every rate is pooled over the counts behind it, not a mean of the
groups' figures. A model missing from a group is reported as incomplete rather
than compared on fewer learners unannounced. ``--without-index`` runs are named
``-chat-noindex-g<N>`` and never pooled here.

Writes ``evals/reports/<timestamp>-chat-combined.json``. No API calls.
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
from evals.suites import chat_groups, chat_metrics  # noqa: E402
from scripts.combine_generation_reports import (  # noqa: E402
    ReportError,
    grouped_reports,
    latest_cases,
    load_reports,
)
from scripts.run_evals import print_by_kind  # noqa: E402


def unanswered(case: dict[str, Any]) -> bool:
    """Whether the model gave no usable reply to this turn."""
    return not case.get("ok")


def combine(reports: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entry per model: its latest answer to every turn, and metrics over them.

    ``reports`` are oldest first, as ``load_reports`` returns them.
    """
    groups, kept, suite_errors, routes = latest_cases(reports, "chat", unanswered)

    combined = []
    for model in routes:
        cases: list[dict[str, Any]] = []
        per_group: dict[int, dict[str, Any]] = {}
        for group in groups:
            turns = kept.get((model, group))
            if not turns:
                why = suite_errors.get((model, group), "not in that group's sweeps")
                per_group[group] = {"error": why}
                continue
            group_cases = list(turns.values())
            cases.extend(group_cases)
            per_group[group] = {
                **chat_metrics(group_cases),
                "sources": sorted({c["source"] for c in group_cases}),
            }
        combined.append(
            {
                "model": model,
                "complete": all("error" not in per_group[g] for g in groups),
                "groups": per_group,
                "metrics": chat_metrics(cases),
                "routes": dict(routes[model]),
                "cases": cases,
            }
        )
    return combined


def _cell(value: Any) -> str:
    return "-" if value is None else f"{value:g}"


def print_combined(combined: list[dict[str, Any]], groups: list[int]) -> None:
    width = max([len(e["model"]) for e in combined] + [5]) + 2
    print("\n=== chat, groups combined ===")
    print(
        f"{'model':<{width}}"
        + "".join(f"{'g' + str(g):>5}" for g in groups)
        + f"{'turns':>7}{'correct/2':>11}{'faithful/2':>12}{'helpful/5':>11}{'tools':>7}"
        f"{'rounds':>8}{'section':>9}{'SQL err':>9}{'unavail':>9}{'failed':>8}"
    )

    def rank(entry: dict[str, Any]) -> tuple[float, float]:
        m = entry["metrics"]
        return (
            -(m.get("judge_correctness_mean") or 0.0),
            -(m.get("judge_faithfulness_mean") or 0.0),
        )

    for e in sorted(combined, key=rank):
        m = e["metrics"]
        print(
            f"{e['model']:<{width}}"
            + "".join(f"{'ERR' if 'error' in e['groups'][g] else 'ok':>5}" for g in groups)
            + f"{m['turns']:>7}{_cell(m.get('judge_correctness_mean')):>11}"
            f"{_cell(m.get('judge_faithfulness_mean')):>12}{_cell(m.get('judge_helpfulness')):>11}"
            f"{_cell(m['tool_choice_rate']):>7}{_cell(m['mean_tool_rounds']):>8}"
            f"{_cell(m['section_hit_rate']):>9}{m['sql_errors']:>9}{m['unavailable']:>9}"
            f"{m['failed']:>8}"
            + ("" if e["complete"] else "   INCOMPLETE")
        )

    for e in sorted(combined, key=rank):
        print(f"\n=== {e['model']}, per kind of turn, every group pooled ===")
        print_by_kind(e["metrics"]["by_kind"], indent="")
        for g, metrics in e["groups"].items():
            if "error" in metrics:
                print(f"group {g}: {metrics['error']}")
    print(
        "\ncorrect, faithful and helpful are the judge's verdicts, averaged on their own "
        "scales ('-' where no judge marked a turn); tools the turns that called the "
        "tools they needed and no others; rounds the tool rounds a turn; section the "
        "exam questions whose lookup came back from the right section. All are pooled "
        "over every turn. unavail counts the turns no model was there for, failed those "
        "it answered unusably: their figures are left out."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "reports",
        nargs="*",
        type=Path,
        help="chat reports, any number per group; default: every one in "
        "evals/reports/ that was run with --group",
    )
    args = parser.parse_args()

    paths = args.reports or grouped_reports(suite="chat")
    if not paths:
        print(f"No chat report with a learner group in {REPORTS_DIR}.")
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
    missing = sorted(set(chat_groups()) - set(groups))
    if missing:
        print(f"  (no report yet for group {', '.join(map(str, missing))})")

    combined = combine(reports)
    print_combined(combined, groups)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suite": "chat",
        "groups": groups,
        "sources": [r["source"] for r in reports],
        "models": combined,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = REPORTS_DIR / f"{stamp}-chat-combined.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\ncombined report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
