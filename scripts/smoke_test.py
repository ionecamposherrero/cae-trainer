"""End-to-end checks against the real API, without Streamlit.

    python scripts/smoke_test.py --all-parts      # generate and validate all 8 parts
    python scripts/smoke_test.py --part 4         # just one part
    python scripts/smoke_test.py --grade          # generate, then mark right and wrong answers
    python scripts/smoke_test.py --writing        # generate a task and assess a script
    python scripts/smoke_test.py --writing --weak # assess the weak sample script instead

Exercises generated here are stored, so anything that passes is available to the
app afterwards.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cae.config import get_settings  # noqa: E402
from cae.domain.parts import READING_PART_NUMBERS, get_part_spec  # noqa: E402
from cae.graphs.rue_task import GenerationFailed, generate_exercise  # noqa: E402
from cae.graphs.rue_evaluator import grade_attempt  # noqa: E402
from cae.graphs.writing_evaluator import assess_writing  # noqa: E402
from cae.graphs.writing_task import generate_writing_task  # noqa: E402
from cae.schemas.rue_task import answer_key, parse_exercise  # noqa: E402
from cae.validators.rue_task import validate_exercise  # noqa: E402
from evals import load_dataset  # noqa: E402

_WRITING = load_dataset("writing_scripts")
_SCRIPTS = {case["id"]: case["script"] for case in _WRITING["cases"]}

SAMPLE_SCRIPT = _SCRIPTS["museums-strong"]
WEAK_SCRIPT = _SCRIPTS["museums-weak"]


def check_key() -> None:
    settings = get_settings()
    tiers = ("writing_task", "writing_evaluator", "rue_task", "rue_evaluator")
    if not all(settings.tier_available(tier) for tier in tiers):
        raise SystemExit(
            "No key is set for any model in CAE_GENERATION_MODEL / "
            "CAE_MARKING_MODEL (or a graph's own override). Copy "
            ".env.example to .env and add at least GOOGLE_API_KEY (free)."
        )


def generate_one(part: int) -> dict | None:
    spec = get_part_spec(part)
    print(f"\nPart {part} — {spec.name}")
    started = time.time()
    try:
        result = generate_exercise(part)
    except GenerationFailed as exc:
        print(f"  FAILED after the retry budget: {exc}")
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR: {type(exc).__name__}: {exc}")
        return None

    elapsed = time.time() - started
    exercise = parse_exercise(part, result["exercise"])
    issues = validate_exercise(part, exercise)
    status = "VALID" if not issues else "INVALID"
    print(
        f"  {status} in {result['attempts']} attempt(s), {elapsed:.1f}s "
        f"(stored as exercise {result.get('exercise_id')})"
    )
    print(f"  title: {result['exercise'].get('title')}")
    if issues:
        for issue in issues:
            print(f"    - {issue}")
    return result


def run_generation(parts: list[int]) -> int:
    results = {part: generate_one(part) for part in parts}
    good = [p for p, r in results.items() if r is not None]
    print(f"\n{len(good)}/{len(parts)} parts produced a valid exercise: {good}")
    return 0 if len(good) == len(parts) else 1


def run_grading(part: int) -> int:
    print(f"Generating a Part {part} exercise to mark...")
    result = generate_one(part)
    if result is None:
        return 1
    payload = result["exercise"]
    key = answer_key(part, parse_exercise(part, payload))

    print("\n--- Marking a perfect script ---")
    perfect = grade_attempt(part, payload, {k.number: k.answer for k in key})
    print(f"  {perfect['marks']:g} / {perfect['max_marks']:g}")
    if perfect["marks"] != perfect["max_marks"]:
        print("  WARNING: the answer key did not mark itself full marks.")
        for row in perfect["rows"]:
            if not row["is_correct"]:
                print(f"    item {row['item_number']}: {row['reason']}")

    print("\n--- Marking a script with wrong answers ---")
    wrong = {k.number: ("Z" if len(k.answer) == 1 else "definitely not this") for k in key}
    graded = grade_attempt(part, payload, wrong)
    print(f"  {graded['marks']:g} / {graded['max_marks']:g}")
    for row in graded["rows"]:
        print(
            f"    item {row['item_number']}: {row['marks_awarded']:g} marks, "
            f"tagged {row['category']}"
        )
    tagged = sum(1 for row in graded["rows"] if row["category"])
    print(f"\n  {tagged}/{len(graded['rows'])} wrong answers were given a weakness tag.")
    return 0 if perfect["marks"] == perfect["max_marks"] else 1


SAMPLE_TOPIC = _WRITING["tasks"]["museums-essay"]["topic"]


def run_writing(weak: bool = False) -> int:
    print("Generating a Writing Part 1 task...")
    result = generate_writing_task(part=1, topic=SAMPLE_TOPIC)
    genre, task = result["genre"], result["task"]
    print(f"  {task.get('title')} ({genre}) in {result['attempts']} attempt(s)")
    print(f"  {task.get('instruction')}")

    script = WEAK_SCRIPT if weak else SAMPLE_SCRIPT
    print(f"\nAssessing the {'weak' if weak else 'strong'} sample script...")
    started = time.time()
    result = assess_writing(part=1, genre=genre, task=task, script=script)
    print(f"  assessed in {time.time() - started:.1f}s")
    failures = 0
    for subscale, detail in (
        (name, result.get(name, {}))
        for name in ("content", "communicative_achievement", "organisation", "language")
    ):
        if detail.get("failed"):
            failures += 1
            print(f"  {subscale:<28} NOT ASSESSED: {detail.get('justification', '')[:180]}")
        else:
            print(f"  {subscale:<28} band {detail.get('band')}")
    print(f"  total {result.get('total')}/20, estimated scale score {result.get('scale_score')}")
    annotations = result.get("annotations") or []
    located = sum(1 for a in annotations if a.get("found"))
    print(f"  {len(annotations)} annotation(s), {located} located exactly in the script")
    for annotation in annotations[:3]:
        mark = "located" if annotation.get("found") else "NOT FOUND in script"
        print(
            f"    '{annotation['span']}' -> '{annotation.get('correction')}' "
            f"[{annotation['category']}, {mark}]"
        )
    print(f"  strengths noted: {len(result.get('strengths') or [])}")
    print(f"  improved version: {len(result.get('improved', {}).get('text', '').split())} words")
    if failures:
        print(f"\n  {failures} subscale(s) could not be assessed -- see the messages above.")
    return 0 if result.get("status") == "assessed" and not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-parts", action="store_true", help="generate all eight parts")
    parser.add_argument("--part", type=int, choices=READING_PART_NUMBERS, help="one part")
    parser.add_argument("--grade", action="store_true", help="generate then mark answers")
    parser.add_argument("--writing", action="store_true", help="generate and assess writing")
    parser.add_argument(
        "--weak",
        action="store_true",
        help="with --writing, assess the deliberately weak script instead of the strong one",
    )
    args = parser.parse_args()

    check_key()

    if args.writing:
        return run_writing(weak=args.weak)
    if args.grade:
        return run_grading(args.part or 4)
    if args.part:
        return run_generation([args.part])
    if args.all_parts:
        return run_generation(list(READING_PART_NUMBERS))

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
