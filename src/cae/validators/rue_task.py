"""Structural validation of generated exercises.

Messages are fed back to the model by the ``repair`` node, so each one says what
is wrong and how to fix it.

Passage and paragraph lengths are deliberately not checked: Cambridge publishes
no word count for them, so the prompt states an approximate length and the
validator leaves it to the model.
"""

from __future__ import annotations

import re

from cae.domain.parts import gap_marker, get_part_spec
from cae.schemas.rue_task import AnyExercise
from cae.validators import GAP_MARKER_RE, normalise


def _contains_quote(haystack: str, quote: str) -> bool:
    """Whether ``quote`` appears in ``haystack``, ignoring whitespace and quote style."""
    if not quote.strip():
        return False
    needle = normalise(quote).strip('"\'.,;: ')
    if not needle:
        return False
    hay = normalise(haystack)
    if needle in hay:
        return True
    chunks = [c.strip() for c in re.split(r"\.\.\.|…", needle) if c.strip()]
    if len(chunks) > 1:
        pos = 0
        for chunk in chunks:
            found = hay.find(chunk, pos)
            if found < 0:
                return False
            pos = found + len(chunk)
        return True
    return False


_DERIVATION_PREFIXES = (
    "counter", "under", "inter", "multi", "semi", "anti", "over", "fore", "self",
    "post", "non", "mis", "dis", "out", "sub", "pre", "mid", "re", "un", "in",
    "im", "il", "ir", "de", "ex", "co", "up",
)


def _plausible_derivation(answer: str, stem: str) -> bool:
    """Whether ``answer`` could be a word built from ``stem``.

    Deliberately loose to allow spelling changes (LONG -> length): compares first
    letters, directly or after one derivational prefix.
    """
    answer, stem = answer.lower(), stem.lower()
    if not answer or not stem:
        return False
    if answer[0] == stem[0]:
        return True
    return any(
        answer.startswith(prefix)
        and len(answer) > len(prefix)
        and answer[len(prefix)] == stem[0]
        for prefix in _DERIVATION_PREFIXES
    )


def _shared_run(a: str, b: str, length: int) -> str | None:
    """Return the first run of ``length`` consecutive words shared by a and b."""
    words_a = normalise(a).split()
    set_b = normalise(b)
    for i in range(len(words_a) - length + 1):
        run = " ".join(words_a[i : i + length])
        if run in set_b:
            return run
    return None


def _check_item_numbering(exercise: AnyExercise, expected: int) -> list[str]:
    items = list(getattr(exercise, "items", []))
    issues: list[str] = []
    if len(items) != expected:
        issues.append(
            f"There must be exactly {expected} items, but {len(items)} were produced."
        )
    numbers = [item.number for item in items]
    if sorted(numbers) != list(range(1, expected + 1)):
        issues.append(
            f"Item numbers must be exactly 1 to {expected} with no repeats or gaps; got {numbers}."
        )
    for item in items:
        if not getattr(item, "explanation", "").strip():
            issues.append(f"Item {item.number}: the explanation is empty.")
    return issues


def _check_gaps(text: str, expected: int) -> list[str]:
    issues: list[str] = []
    found = [int(n) for n in GAP_MARKER_RE.findall(text)]
    for n in range(1, expected + 1):
        occurrences = text.count(gap_marker(n))
        if occurrences == 0:
            issues.append(f"The text is missing the gap marker {gap_marker(n)}.")
        elif occurrences > 1:
            issues.append(
                f"The gap marker {gap_marker(n)} appears {occurrences} times; it must appear once."
            )
    for n in set(found) - set(range(1, expected + 1)):
        issues.append(
            f"The text contains {gap_marker(n)}, which is outside the allowed range 1-{expected}."
        )
    if found and found != sorted(found):
        issues.append("The gap markers must appear in ascending numerical order in the text.")
    return issues


def _check_options(item, count: int, letters: tuple[str, ...]) -> list[str]:
    issues: list[str] = []
    options = list(item.options)
    if len(options) != count:
        issues.append(f"Item {item.number}: needs exactly {count} options, got {len(options)}.")
    if any(not str(o).strip() for o in options):
        issues.append(f"Item {item.number}: one of the options is empty.")
    lowered = [normalise(str(o)) for o in options]
    if len(set(lowered)) != len(lowered):
        issues.append(f"Item {item.number}: the options are not all different from each other.")
    if item.answer not in letters[:count]:
        issues.append(
            f"Item {item.number}: the answer '{item.answer}' is not one of {list(letters[:count])}."
        )
    return issues


def _check_letters(objects, attr: str, expected: tuple[str, ...], what: str) -> list[str]:
    letters = [getattr(o, attr) for o in objects]
    if sorted(letters) != sorted(expected):
        return [
            f"The {what} must be labelled exactly {list(expected)} with no repeats; got {letters}."
        ]
    return []


def _validate_part1(ex) -> list[str]:
    spec = get_part_spec(1)
    issues = _check_item_numbering(ex, spec.item_count)
    issues += _check_gaps(ex.text, spec.item_count)
    for item in ex.items:
        issues += _check_options(item, 4, spec.option_letters)
        for option in item.options:
            if len(str(option).split()) > 1:
                issues.append(
                    f"Item {item.number}: option '{option}' must be a single word. For a "
                    "phrasal verb or fixed phrase, gap only one word and leave the rest "
                    "in the text."
                )
    return issues


def _validate_part2(ex) -> list[str]:
    spec = get_part_spec(2)
    issues = _check_item_numbering(ex, spec.item_count)
    issues += _check_gaps(ex.text, spec.item_count)
    for item in ex.items:
        for label, word in [("answer", item.answer), *[("alternative", a) for a in item.accepted]]:
            cleaned = word.strip()
            if not cleaned:
                issues.append(f"Item {item.number}: the {label} is empty.")
            elif len(cleaned.split()) != 1:
                issues.append(
                    f"Item {item.number}: the {label} '{cleaned}' is more than one word. "
                    "Open cloze answers must be exactly ONE word."
                )
            elif not re.fullmatch(r"[A-Za-z']+", cleaned):
                issues.append(
                    f"Item {item.number}: the {label} '{cleaned}' must be a plain word with "
                    "no digits, hyphens or punctuation."
                )
    return issues


def _validate_part3(ex) -> list[str]:
    spec = get_part_spec(3)
    issues = _check_item_numbering(ex, spec.item_count)
    issues += _check_gaps(ex.text, spec.item_count)
    for item in ex.items:
        stem = item.stem.strip()
        answer = item.answer.strip()
        if not re.fullmatch(r"[A-Z]+", stem):
            issues.append(
                f"Item {item.number}: the stem '{item.stem}' must be a single real word "
                "written in capitals with no spaces."
            )
        if len(answer.split()) != 1:
            issues.append(
                f"Item {item.number}: the answer '{answer}' must be exactly one word."
            )
        if normalise(answer) == normalise(stem):
            issues.append(
                f"Item {item.number}: the answer '{answer}' is identical to the stem "
                f"'{stem}'. The gap must require a change of form."
            )
        if answer and stem and not _plausible_derivation(answer, stem):
            issues.append(
                f"Item {item.number}: the answer '{answer}' does not look like a form of "
                f"the stem '{stem}'."
            )
    return issues


def _validate_part4(ex) -> list[str]:
    spec = get_part_spec(4)
    issues = _check_item_numbering(ex, spec.item_count)
    for item in ex.items:
        key = item.key_word.strip()
        if not re.fullmatch(r"[A-Za-z']+", key):
            issues.append(
                f"Item {item.number}: the key word '{item.key_word}' must be a single word."
            )
        if not item.lead_in.strip():
            issues.append(f"Item {item.number}: the lead-in sentence is empty.")
        if not item.sentence_end.strip():
            issues.append(f"Item {item.number}: the second sentence has no text after the gap.")
        if re.search(rf"\b{re.escape(key)}\b", item.lead_in, flags=re.IGNORECASE):
            issues.append(
                f"Item {item.number}: the key word '{key}' appears in the lead-in sentence. "
                "It must not, or the item is trivial."
            )
        for label, candidate in [
            ("answer", item.answer),
            *[("alternative answer", a) for a in item.accepted],
        ]:
            words = candidate.split()
            if not 3 <= len(words) <= 6:
                issues.append(
                    f"Item {item.number}: the {label} '{candidate}' has {len(words)} words. "
                    "It must have between three and six words, including the key word."
                )
            if not re.search(rf"\b{re.escape(key)}\b", candidate, flags=re.IGNORECASE):
                issues.append(
                    f"Item {item.number}: the {label} '{candidate}' does not contain the key "
                    f"word '{key}' unchanged."
                )
    return issues


def _validate_part5(ex) -> list[str]:
    spec = get_part_spec(5)
    issues = _check_item_numbering(ex, spec.item_count)
    if GAP_MARKER_RE.search(ex.text):
        issues.append("The Part 5 passage must not contain any gap markers.")
    for item in ex.items:
        issues += _check_options(item, 4, spec.option_letters)
        if not _contains_quote(ex.text, item.evidence):
            issues.append(
                f"Item {item.number}: the evidence quotation \"{item.evidence}\" does not "
                "appear in the passage. Quote the passage word for word."
            )
    return issues


def _validate_part6(ex) -> list[str]:
    spec = get_part_spec(6)
    issues = _check_item_numbering(ex, spec.item_count)
    issues += _check_letters(ex.texts, "letter", ("A", "B", "C", "D"), "four texts")
    by_letter = {t.letter: t for t in ex.texts}
    for item in ex.items:
        source = by_letter.get(item.answer)
        if source is None:
            issues.append(f"Item {item.number}: answer '{item.answer}' has no matching text.")
        elif not _contains_quote(source.content, item.evidence):
            issues.append(
                f"Item {item.number}: the evidence \"{item.evidence}\" does not appear in "
                f"text {item.answer}. Quote that text word for word."
            )
    return issues


def _validate_part7(ex) -> list[str]:
    spec = get_part_spec(7)
    issues = _check_item_numbering(ex, spec.item_count)
    issues += _check_gaps(ex.text, spec.item_count)
    issues += _check_letters(ex.options, "letter", spec.option_letters, "seven paragraphs")
    answers = [i.answer for i in ex.items]
    if len(set(answers)) != len(answers):
        issues.append(
            f"Each paragraph may fill only one gap, but the answers repeat: {answers}."
        )
    unused = sorted(set(spec.option_letters) - set(answers))
    if len(unused) != 1:
        issues.append(
            f"Exactly one paragraph must be left unused; currently unused: {unused or 'none'}."
        )
    elif ex.distractor_letter != unused[0]:
        issues.append(
            f"distractor_letter is '{ex.distractor_letter}' but the unused paragraph is "
            f"'{unused[0]}'."
        )
    return issues


def _validate_part8(ex) -> list[str]:
    spec = get_part_spec(8)
    issues = _check_item_numbering(ex, spec.item_count)
    issues += _check_letters(ex.sections, "letter", spec.option_letters, "six sections")
    by_letter = {s.letter: s for s in ex.sections}
    for item in ex.items:
        source = by_letter.get(item.answer)
        if source is None:
            issues.append(f"Item {item.number}: answer '{item.answer}' has no matching section.")
            continue
        if not _contains_quote(source.content, item.evidence):
            issues.append(
                f"Item {item.number}: the evidence \"{item.evidence}\" does not appear in "
                f"section {item.answer}. Quote that section word for word."
            )
        lifted = _shared_run(item.prompt, source.content, 5)
        if lifted:
            issues.append(
                f"Item {item.number}: the prompt copies '{lifted}' straight from section "
                f"{item.answer}. Paraphrase it instead."
            )
    used = {i.answer for i in ex.items}
    missing = sorted(set(spec.option_letters) - used)
    if missing:
        issues.append(
            f"Every section must be the answer to at least one question; unused: {missing}."
        )
    return issues


_VALIDATORS = {
    1: _validate_part1,
    2: _validate_part2,
    3: _validate_part3,
    4: _validate_part4,
    5: _validate_part5,
    6: _validate_part6,
    7: _validate_part7,
    8: _validate_part8,
}


def validate_exercise(part: int, exercise: AnyExercise) -> list[str]:
    """Return a list of problems. An empty list means the exercise is usable."""
    try:
        validator = _VALIDATORS[part]
    except KeyError:
        raise ValueError(f"No validator for part {part!r}") from None
    issues = validator(exercise)
    if not str(getattr(exercise, "title", "")).strip():
        issues.append("The exercise has no title.")
    return issues
