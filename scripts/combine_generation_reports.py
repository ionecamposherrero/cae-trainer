"""Combine the generation reports of the seed groups into one comparison.

    python scripts/combine_generation_reports.py                 # every -g<N> report
    python scripts/combine_generation_reports.py a.json b.json   # these reports

Each ``run_evals.py --suite generation --group N`` writes its own report, and a
group may be run more than once: on another day, or for a model added later.
For every model and seed the most recent case counts, so a later report replaces
only the models it ran. The exception is a case that failed because no model
could be reached (spent quota, a provider down, no key): that measures the
provider, not the model, so it never replaces an earlier case of the same seed.
A sweep that runs out of quota half-way through a group keeps the seeds it lost
from the earlier run.

The kept cases are pooled per model and the suite's metrics recomputed over all
of them with the suite's own function (``suites.generation_metrics``), so the
combined rate is over every seed attempted, not a mean of the groups' rates. A
model missing from a group is reported as incomplete rather than compared on
fewer exercises unannounced.

Writes ``evals/reports/<timestamp>-generation-combined.json``. No API calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, NamedTuple, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cae.domain.parts import READING_PART_NUMBERS  # noqa: E402
from cae.llm import LLMUnavailableError  # noqa: E402
from evals import REPORTS_DIR  # noqa: E402
from evals.suites import generation_groups, generation_metrics  # noqa: E402


class ReportError(ValueError):
    """The reports given cannot be combined as they are."""


def grouped_reports(reports_dir: Path = REPORTS_DIR, suite: str = "generation") -> list[Path]:
    """Every report of one group of ``suite``, oldest first. Names start with a UTC timestamp."""
    return [
        path
        for path in sorted(reports_dir.glob(f"*-{suite}-g*.json"))
        if json.loads(path.read_text(encoding="utf-8")).get("group")
    ]


def load_reports(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """The reports oldest first, each tagged with the file it came from."""
    reports = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not payload.get("group"):
            raise ReportError(
                f"{path} has no seed group: it was not run with --group, so there "
                "is nothing to combine it with."
            )
        reports.append({"source": Path(path).name, **payload})
    return sorted(reports, key=lambda r: r.get("generated_at") or "")


UNAVAILABLE_ERRORS = tuple(
    cls.__name__ for cls in (LLMUnavailableError, *LLMUnavailableError.__subclasses__())
)


def unavailable(case: dict[str, Any]) -> bool:
    """Whether a case failed for want of a model rather than because of one."""
    return not case.get("ok") and str(case.get("error", "")).startswith(UNAVAILABLE_ERRORS)


class Latest(NamedTuple):
    """Every model's newest case of every group, from reports oldest first."""

    groups: list[int]
    kept: dict[tuple[str, int], dict[str, dict[str, Any]]]
    suite_errors: dict[tuple[str, int], str]
    routes: dict[str, Counter]


def latest_cases(
    reports: Sequence[dict[str, Any]],
    suite: str,
    unavailable: Callable[[dict[str, Any]], bool],
) -> Latest:
    """For each model and case, the newest result, unless it is ``unavailable``.

    A case no model could answer measures the provider, so it never replaces an
    earlier result of the same case. Shared by the generation and marking combines.
    """
    groups = sorted({int(r["group"]) for r in reports})
    kept: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    suite_errors: dict[tuple[str, int], str] = {}
    routes: dict[str, Counter] = {}
    for report in reports:
        group = int(report["group"])
        for run in report.get("runs", []):
            model = run["model"]
            result = run.get("suites", {}).get(suite) or {}
            routes.setdefault(model, Counter()).update(result.get("routes") or {})
            if not result.get("cases"):
                suite_errors[(model, group)] = result.get("error") or "not run"
                continue
            seen = kept.setdefault((model, group), {})
            for case in result["cases"]:
                case_id = case.get("id") or f"part{case['part']}"
                if case_id in seen and unavailable(case):
                    continue
                seen[case_id] = {**case, "source": report.get("source", "")}
    return Latest(groups, kept, suite_errors, routes)


def combine(reports: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entry per model: its latest case of every seed, and metrics over them.

    ``reports`` are oldest first, as :func:`load_reports` returns them.
    """
    groups, kept, suite_errors, routes = latest_cases(reports, "generation", unavailable)

    combined = []
    for model in routes:
        cases: list[dict[str, Any]] = []
        per_group: dict[int, dict[str, Any]] = {}
        for group in groups:
            seeds = kept.get((model, group))
            if not seeds:
                why = suite_errors.get((model, group), "not in that group's sweeps")
                per_group[group] = {"error": why}
                continue
            group_cases = sorted(seeds.values(), key=lambda c: c.get("part", 0))
            cases.extend(group_cases)
            per_group[group] = {
                **generation_metrics(group_cases),
                "unavailable": sum(1 for c in group_cases if unavailable(c)),
                "sources": sorted({c["source"] for c in group_cases}),
            }
        by_part = {}
        for part in READING_PART_NUMBERS:
            tried = [c for c in cases if c.get("part") == part]
            by_part[part] = {"valid": sum(1 for c in tried if c.get("ok")), "tried": len(tried)}
        combined.append(
            {
                "model": model,
                "complete": all("error" not in per_group[g] for g in groups),
                "groups": per_group,
                "metrics": {
                    **generation_metrics(cases),
                    "unavailable": sum(1 for c in cases if unavailable(c)),
                },
                "by_part": by_part,
                "routes": dict(routes[model]),
                "cases": cases,
            }
        )
    return combined


def _cell(value: Any) -> str:
    return "-" if value is None else f"{value:g}"


def print_combined(combined: list[dict[str, Any]], groups: list[int]) -> None:
    width = max([len(e["model"]) for e in combined] + [5]) + 2
    header = (
        f"{'model':<{width}}"
        + "".join(f"{'g' + str(g):>7}" for g in groups)
        + f"{'valid':>9}{'rate':>7}{'unavail':>9}{'attempts':>10}{'secs':>7}{'key ok':>8}"
    )
    print("\n=== generation, groups combined ===")
    print(header)
    for e in sorted(combined, key=lambda e: -e["metrics"]["rate"]):
        m = e["metrics"]
        per_group = "".join(
            f"{'ERR' if 'error' in e['groups'][g] else _cell(e['groups'][g].get('rate')):>7}"
            for g in groups
        )
        print(
            f"{e['model']:<{width}}{per_group}"
            f"{str(m['valid']) + '/' + str(m['generated']):>9}{_cell(m['rate']):>7}"
            f"{m['unavailable']:>9}{_cell(m['mean_attempts']):>10}{_cell(m['mean_seconds']):>7}"
            f"{_cell(m['key_self_consistency_rate']):>8}"
            + ("" if e["complete"] else "   INCOMPLETE")
        )

    print("\n=== valid exercises per part ===")
    print(f"{'model':<{width}}" + "".join(f"{'P' + str(p):>6}" for p in READING_PART_NUMBERS))
    for e in sorted(combined, key=lambda e: e["model"]):
        cells = "".join(
            f"{str(e['by_part'][p]['valid']) + '/' + str(e['by_part'][p]['tried']):>6}"
            for p in READING_PART_NUMBERS
        )
        print(f"{e['model']:<{width}}{cells}")

    for e in combined:
        for g, metrics in e["groups"].items():
            if "error" in metrics:
                print(f"\n{e['model']}, group {g}: {metrics['error']}")
    print(
        "\nrate is valid/attempted over every seed of every group; unavail counts the "
        "failures where no model answered (quota, provider down), which say nothing "
        "about the model's material; attempts, secs and key ok are over the valid "
        "exercises only."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "reports",
        nargs="*",
        type=Path,
        help="generation reports, any number per group; default: every one in "
        "evals/reports/ that was run with --group",
    )
    args = parser.parse_args()

    paths = args.reports or grouped_reports()
    if not paths:
        print(f"No generation report with a seed group in {REPORTS_DIR}.")
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
    missing = sorted(set(generation_groups()) - set(groups))
    if missing:
        print(f"  (no report yet for group {', '.join(map(str, missing))})")

    combined = combine(reports)
    print_combined(combined, groups)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suite": "generation",
        "groups": groups,
        "sources": [r["source"] for r in reports],
        "models": combined,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = REPORTS_DIR / f"{stamp}-generation-combined.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\ncombined report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
