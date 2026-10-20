"""Have the judge mark a coach report that has already been run.

    python scripts/judge_coach_report.py evals/reports/<stamp>-coach-g1.json

The coach twin of ``judge_chat_report.py``: the same verdicts ``run_evals.py
--suite coach`` gives, on reports already written -- one run with
``--no-judge``, or one to judge again after the judge changed. ``narrate`` is not
called again, so the judge marks exactly the reports that were written, and a
sweep that spent its free allowance can be judged the next day. Every model's
run in the report is judged.

The judge is paid (``OPENAI_API_KEY``, ``CAE_JUDGE_MODEL``; see ``evals/judge.py``):
one call per written report. Writes ``<now>-coach-g<N>-judged.json`` next to the
report, with the metrics recomputed; being newer, its reports are the ones
``combine_coach_reports.py`` keeps.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals import REPORTS_DIR  # noqa: E402
from evals.judge import CoachVerdict, JudgeUnavailableError, build_judge, judge_coach_cases  # noqa: E402
from evals.suites import coach_metrics  # noqa: E402
from scripts.judge_chat_report import judged_label  # noqa: E402


def judge_report(payload: dict[str, Any], judge) -> dict[str, Any]:
    """``payload`` with every coach run's reports judged and its metrics recomputed."""
    for run in payload.get("runs", []):
        result = run.get("suites", {}).get("coach")
        if not result or not result.get("cases"):
            continue
        judge_coach_cases(result["cases"], judge)
        result["metrics"] = coach_metrics(result["cases"])
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("report", type=Path, help="a report written by run_evals.py --suite coach")
    args = parser.parse_args()

    payload = json.loads(args.report.read_text(encoding="utf-8"))
    if not any((run.get("suites") or {}).get("coach") for run in payload.get("runs", [])):
        print(f"{args.report} has no coach results to judge.")
        return 1
    try:
        judge = build_judge(schema=CoachVerdict)
    except JudgeUnavailableError as exc:
        print(exc)
        return 1

    judged = judge_report(payload, judge)
    judged["judged_from"] = args.report.name
    judged["generated_at"] = datetime.now(timezone.utc).isoformat()

    for run in judged["runs"]:
        metrics = (run["suites"].get("coach") or {}).get("metrics") or {}
        if "judged" in metrics:
            print(
                f"{run['model']}: {metrics['judged']} reports judged; correctness "
                f"{metrics['judge_correctness_mean']}/2 (rate of 2s {metrics['judge_correct_rate']}), "
                f"faithfulness {metrics['judge_faithfulness_mean']}/2 (rate of 2s "
                f"{metrics['judge_faithful_rate']}), helpfulness {metrics['judge_helpfulness']}/5"
            )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = REPORTS_DIR / f"{stamp}-{judged_label(args.report)}-judged.json"
    out.write_text(json.dumps(judged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\njudged report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
