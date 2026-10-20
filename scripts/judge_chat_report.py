"""Have the judge mark a chat report that has already been run.

    python scripts/judge_chat_report.py evals/reports/<stamp>-chat-g1.json

The same verdicts ``run_evals.py --suite chat`` gives, on replies already
collected -- one run with ``--no-judge``, or one to judge again after the judge
changed: the chat is not asked again, so the judge marks exactly the replies
the deterministic metrics scored, and a sweep that spent its free allowance can
be judged the next day. Every model's run in the report is judged.

The judge is paid (``OPENAI_API_KEY``, ``CAE_JUDGE_MODEL``; see ``evals/judge.py``):
one call per answered turn. Writes ``<now>-chat-g<N>-judged.json`` next to the
report, with the metrics recomputed; being newer, its turns are the ones
``combine_chat_reports.py`` keeps.
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
from evals.judge import JudgeUnavailableError, build_judge, judge_cases  # noqa: E402
from evals.suites import chat_metrics  # noqa: E402


def judge_report(payload: dict[str, Any], judge) -> dict[str, Any]:
    """``payload`` with every chat run's turns judged and its metrics recomputed."""
    for run in payload.get("runs", []):
        result = run.get("suites", {}).get("chat")
        if not result or not result.get("cases"):
            continue
        judge_cases(result["cases"], result.get("learners") or {}, judge)
        result["metrics"] = chat_metrics(result["cases"])
    return payload


def judged_label(report: Path) -> str:
    """The judged report's name after its timestamp: the original's, so a
    ``-noindex`` run stays out of ``combine_chat_reports.py`` once judged too."""
    label = report.stem.split("-", 1)[1] if "-" in report.stem else "chat"
    return label.removesuffix("-judged")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("report", type=Path, help="a report written by run_evals.py --suite chat")
    args = parser.parse_args()

    payload = json.loads(args.report.read_text(encoding="utf-8"))
    if not any((run.get("suites") or {}).get("chat") for run in payload.get("runs", [])):
        print(f"{args.report} has no chat results to judge.")
        return 1
    try:
        judge = build_judge()
    except JudgeUnavailableError as exc:
        print(exc)
        return 1

    judged = judge_report(payload, judge)
    judged["judged_from"] = args.report.name
    judged["generated_at"] = datetime.now(timezone.utc).isoformat()

    for run in judged["runs"]:
        metrics = (run["suites"].get("chat") or {}).get("metrics") or {}
        if "judged" in metrics:
            print(
                f"{run['model']}: {metrics['judged']} turns judged; correctness "
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
