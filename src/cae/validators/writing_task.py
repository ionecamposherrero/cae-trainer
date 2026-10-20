"""Structural validation of generated Writing tasks.

Like :mod:`cae.validators.rue_task`, messages are written to be fed back to the
model. Only what the real paper guarantees is enforced.
"""

from __future__ import annotations

import re

from cae.domain.parts import WRITING_WORD_RANGE, get_genre
from cae.validators import count_words, normalise

MIN_WORDS, MAX_WORDS = WRITING_WORD_RANGE

MIN_SCENARIO_WORDS = 12
MAX_NOTE_WORDS = 12
MAX_OPINION_WORDS = 40

PART1_NOTES = 3
PART1_OPINIONS = 3
PART2_NOTE_RANGE = (2, 5)

_GENRE_ALIASES: dict[str, tuple[str, ...]] = {
    "essay": ("essay",),
    "letter_email": ("letter", "email", "e-mail"),
    "proposal": ("proposal",),
    "report": ("report",),
    "review": ("review",),
}


def _mentions(haystack: str, *words: str) -> bool:
    text = normalise(haystack)
    return any(re.search(rf"\b{re.escape(w)}\b", text) for w in words)


def _comparison_key(note: str) -> str:
    """Reduce a content point to lowercase alphanumerics, for duplicate detection."""
    return re.sub(r"[^a-z0-9 ]", "", normalise(note)).strip()


def _check_common(task: dict) -> list[str]:
    issues: list[str] = []
    if not str(task.get("title", "")).strip():
        issues.append("The task has no title.")

    scenario = str(task.get("scenario", ""))
    if count_words(scenario) < MIN_SCENARIO_WORDS:
        issues.append(
            f"The scenario is only {count_words(scenario)} words. Describe a concrete "
            f"situation in at least {MIN_SCENARIO_WORDS} words, so the candidate knows "
            "who they are writing for and why."
        )

    instruction = str(task.get("instruction", ""))
    if not instruction.strip():
        issues.append("The instruction is empty: say what the candidate must write.")
    elif not (str(MIN_WORDS) in instruction and str(MAX_WORDS) in instruction):
        issues.append(
            f"The instruction does not state the length. It must ask for "
            f"{MIN_WORDS}-{MAX_WORDS} words, using both numbers."
        )

    if not str(task.get("reader", "")).strip():
        issues.append(
            "The reader is empty. Name who will read the text, for example "
            "'your college principal'."
        )

    notes = [str(n) for n in task.get("notes") or []]
    if any(not n.strip() for n in notes):
        issues.append("One of the content points is empty.")
    for note in notes:
        if count_words(note) > MAX_NOTE_WORDS:
            issues.append(
                f"The content point '{note}' is {count_words(note)} words. Content "
                f"points are short phrases of at most {MAX_NOTE_WORDS} words, not "
                "sentences."
            )
    lowered = [_comparison_key(n) for n in notes if n.strip()]
    if len(set(lowered)) != len(lowered):
        issues.append("Two of the content points are the same. Each must raise a different aspect.")
    return issues


def _check_part1(task: dict) -> list[str]:
    issues: list[str] = []
    notes = [str(n) for n in task.get("notes") or []]
    opinions = [str(o) for o in task.get("opinions") or []]
    instruction = str(task.get("instruction", ""))

    if len(notes) != PART1_NOTES:
        issues.append(
            f"A Part 1 essay needs exactly {PART1_NOTES} content points, but "
            f"{len(notes)} were given. The candidate chooses two of three."
        )
    if len(opinions) != PART1_OPINIONS:
        issues.append(
            f"A Part 1 essay needs exactly {PART1_OPINIONS} quoted opinions, one for "
            f"each content point, but {len(opinions)} were given."
        )
    for opinion in opinions:
        if not opinion.strip():
            issues.append("One of the quoted opinions is empty.")
        elif count_words(opinion) > MAX_OPINION_WORDS:
            issues.append(
                f"The opinion '{opinion[:60]}...' is {count_words(opinion)} words. "
                f"Each is one overheard remark of at most {MAX_OPINION_WORDS} words."
            )
    if not _mentions(instruction, "essay"):
        issues.append("The instruction must say explicitly that the candidate writes an essay.")
    if not _mentions(instruction, "two", "2"):
        issues.append(
            "The instruction must tell the candidate to discuss TWO of the three "
            "points and say which is more important."
        )
    return issues


def _check_part2(task: dict, genre: str) -> list[str]:
    issues: list[str] = []
    notes = [str(n) for n in task.get("notes") or []]
    opinions = task.get("opinions") or []
    low, high = PART2_NOTE_RANGE

    if opinions:
        issues.append(
            f"{len(opinions)} quoted opinions were given. Only Part 1 essays use "
            "opinions; leave the list empty for Part 2."
        )
    if not low <= len(notes) <= high:
        issues.append(
            f"A Part 2 task needs between {low} and {high} content points, but "
            f"{len(notes)} were given."
        )

    aliases = _GENRE_ALIASES.get(genre, (genre.replace("_", " "),))
    where = f"{task.get('scenario', '')} {task.get('instruction', '')}"
    if not _mentions(where, *aliases):
        issues.append(
            f"Neither the scenario nor the instruction names the genre. Say plainly "
            f"that the candidate is writing a {aliases[0]}."
        )
    return issues


def validate_writing_task(part: int, genre: str, task: dict) -> list[str]:
    """Return a list of problems. An empty list means the task is usable."""
    if part not in (1, 2):
        raise ValueError(f"A writing task is Part 1 or Part 2, not {part!r}")
    get_genre(genre)

    issues = _check_common(task)
    issues += _check_part1(task) if part == 1 else _check_part2(task, genre)
    return issues
