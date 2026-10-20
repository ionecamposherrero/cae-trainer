"""Evaluation metrics: pure Python over data the graphs already return.

:func:`key_self_consistency` invokes the grading graph, but on a script that
matches the key, which needs no API calls.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from typing import Any, Iterable, Sequence

from cae.domain.parts import SUBSCALES
from cae.domain.taxonomy import family
from cae.schemas.rue_task import answer_key, parse_exercise

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


BAND_MAX = 5


def band_errors(predicted: dict[str, Any], expected: dict[str, Any]) -> dict[str, int]:
    """Signed error per subscale: positive means the model marked too generously."""
    return {
        subscale: int(predicted.get(subscale, 0)) - int(expected[subscale])
        for subscale in SUBSCALES
        if subscale in expected
    }


def band_accuracy(predicted: dict[str, Any], expected: dict[str, Any]) -> dict[str, float]:
    """How close the bands are, as ``1 - |error| / maximum``.

    One figure per subscale, out of ``BAND_MAX``, and ``total``, out of the
    whole twenty: the total's error is the sum of the signed errors, so a band
    too high on one subscale and too low on another cancel there, as they do
    in the mark the learner sees.
    """
    errors = band_errors(predicted, expected)
    accuracy = {s: round(1 - abs(e) / BAND_MAX, 3) for s, e in errors.items()}
    accuracy["total"] = round(1 - abs(sum(errors.values())) / (BAND_MAX * len(SUBSCALES)), 3)
    return accuracy


def consistency(totals: Sequence[float]) -> dict[str, float]:
    """Spread across repeated markings of the same script."""
    values = [float(t) for t in totals]
    if len(values) < 2:
        return {"runs": len(values), "mean": round(values[0], 3) if values else 0.0,
                "stdev": 0.0, "spread": 0.0}
    return {
        "runs": len(values),
        "mean": round(statistics.fmean(values), 3),
        "stdev": round(statistics.stdev(values), 3),
        "spread": round(max(values) - min(values), 3),
    }


def _intervals(span: str, text: str) -> list[tuple[int, int]]:
    """Every place ``span`` occurs in ``text``, both already normalised."""
    places: list[tuple[int, int]] = []
    start = text.find(span) if span else -1
    while start != -1:
        places.append((start, start + len(span)))
        start = text.find(span, start + 1)
    return places


def _overlap(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> bool:
    return any(x0 < y1 and y0 < x1 for x0, x1 in a for y0, y1 in b)


def _rate(part: int, whole: int) -> float | None:
    return round(part / whole, 3) if whole else None


def annotation_scores(
    annotations: Iterable[dict[str, Any]],
    script: str,
    expected: Sequence[dict[str, Any]],
    tolerated: Sequence[str] = (),
) -> dict[str, Any]:
    """How the annotation node did against the errors a person labelled.

    ``expected`` are the errors it should find, each a ``span`` of the script,
    the ``category`` it belongs to, and optionally ``also``: other categories a
    defensible reading allows. ``tolerated`` are the other real errors -- minor
    or debatable slips -- that it may annotate or leave alone.

    An annotation *hits* an expected error when their spans overlap in the
    script (case, quotes and spacing aside). Categories play no part in recall
    or precision. Then:

    - recall: expected errors hit / expected errors;
    - precision: annotations that are real errors / annotations, where a real
      error is an expected or a tolerated span. One whose span is not in the
      script at all touches nothing, so it counts against precision;
    - category accuracy: of the expected errors hit, those hit with an accepted
      category. A tolerated span has no category, so it is not checked.

    ``by_category`` holds the counts behind recall and precision per category,
    so suites can pool them over scripts. Rates are None when there is nothing
    to divide by: no expected error, no annotation, nothing hit.
    """
    from cae.validators import normalise

    text = normalise(script)
    errors = [
        {
            "span": e["span"],
            "category": e["category"],
            "accepted": {e["category"], *(e.get("also") or [])},
            "at": _intervals(normalise(e["span"]), text),
        }
        for e in expected
    ]
    tolerated_at = [_intervals(normalise(span), text) for span in tolerated]

    by_category: dict[str, dict[str, int]] = {}

    def tally(category: str) -> dict[str, int]:
        return by_category.setdefault(
            category, {"expected": 0, "found_right": 0, "labelled": 0, "labelled_right": 0}
        )

    hits_by_error: list[list[str]] = [[] for _ in errors]
    counted = hits = 0
    false_positives: list[str] = []
    for row in annotations:
        span, category = str(row.get("span") or ""), str(row.get("category") or "")
        at = _intervals(normalise(span), text)
        touched = [i for i, e in enumerate(errors) if _overlap(at, e["at"])]
        counted += 1
        if not touched and any(_overlap(at, t) for t in tolerated_at):
            hits += 1
            continue
        tally(category)["labelled"] += 1
        if not touched:
            false_positives.append(span)
            continue
        hits += 1
        for i in touched:
            hits_by_error[i].append(category)
        if any(category in errors[i]["accepted"] for i in touched):
            tally(category)["labelled_right"] += 1

    found = right = 0
    missed: list[str] = []
    miscategorised: list[dict[str, Any]] = []
    for error, labels in zip(errors, hits_by_error):
        tally(error["category"])["expected"] += 1
        if not labels:
            missed.append(error["span"])
            continue
        found += 1
        if error["accepted"] & set(labels):
            right += 1
            tally(error["category"])["found_right"] += 1
        else:
            miscategorised.append(
                {"span": error["span"], "expected": error["category"], "got": labels}
            )

    return {
        "expected": len(errors),
        "found": found,
        "annotations": counted,
        "hits": hits,
        "category_right": right,
        "recall": _rate(found, len(errors)),
        "precision": _rate(hits, counted),
        "category_accuracy": _rate(right, found),
        "by_category": by_category,
        "missed": missed,
        "false_positives": false_positives,
        "miscategorised": miscategorised,
    }


def pooled_annotation_scores(scores: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Recall, precision and category accuracy over every error of several scripts,
    and per category: pooled counts, not a mean of the scripts' rates."""
    totals: Counter = Counter()
    per_category: dict[str, Counter] = {}
    for score in scores:
        totals.update(
            {k: score[k] for k in ("expected", "found", "annotations", "hits", "category_right")}
        )
        for category, counts in score["by_category"].items():
            per_category.setdefault(category, Counter()).update(counts)
    return {
        "recall": _rate(totals["found"], totals["expected"]),
        "precision": _rate(totals["hits"], totals["annotations"]),
        "category_accuracy": _rate(totals["category_right"], totals["found"]),
        "expected_errors": totals["expected"],
        "annotations": totals["annotations"],
        "by_category": {
            category: {
                "recall": _rate(c["found_right"], c["expected"]),
                "precision": _rate(c["labelled_right"], c["labelled"]),
            }
            for category, c in sorted(per_category.items())
        },
    }


def category_family(category: str) -> str:
    """The taxonomy's family: GRAMMAR, VOCABULARY, TEXT or READING ("" if unknown)."""
    return family(str(category or "").upper())


def category_accuracy(
    predicted: dict[int, str], expected: dict[int, str]
) -> dict[str, Any]:
    """Exact and family-level agreement on the error category of each item."""
    items = sorted(k for k in expected if expected[k])
    if not items:
        return {"items": 0, "exact": 0.0, "family": 0.0, "disagreements": []}
    exact = 0
    family = 0
    disagreements: list[dict[str, Any]] = []
    for number in items:
        want = str(expected[number])
        got = str(predicted.get(number, ""))
        if got == want:
            exact += 1
            family += 1
            continue
        if got and category_family(got) and category_family(got) == category_family(want):
            family += 1
        disagreements.append({"item": number, "expected": want, "got": got or None})
    return {
        "items": len(items),
        "exact": round(exact / len(items), 3),
        "family": round(family / len(items), 3),
        "disagreements": disagreements,
    }


def adjudication_accuracy(
    predicted: dict[int, str], expected: dict[int, str]
) -> dict[str, Any]:
    """Agreement on accept / partial / reject.

    ``over_accepted`` (credit for a wrong answer) and ``over_rejected`` (a
    defensible answer denied) are reported separately. ``by_verdict`` counts,
    for each expected verdict, how many items had it and how many the model
    matched: what :func:`balanced_accuracy` is computed from.
    """
    items = sorted(k for k in expected if expected[k])
    if not items:
        return {
            "items": 0,
            "accuracy": 0.0,
            "over_accepted": 0,
            "over_rejected": 0,
            "by_verdict": {},
            "disagreements": [],
        }

    rank = {"reject": 0, "partial": 1, "accept": 2}
    agreed = 0
    over_accepted = 0
    over_rejected = 0
    by_verdict: dict[str, dict[str, int]] = {}
    disagreements: list[dict[str, Any]] = []
    for number in items:
        want = str(expected[number])
        got = str(predicted.get(number, ""))
        tally = by_verdict.setdefault(want, {"items": 0, "agreed": 0})
        tally["items"] += 1
        if got == want:
            agreed += 1
            tally["agreed"] += 1
            continue
        disagreements.append({"item": number, "expected": want, "got": got or None})
        if got in rank and want in rank:
            if rank[got] > rank[want]:
                over_accepted += 1
            else:
                over_rejected += 1
    return {
        "items": len(items),
        "accuracy": round(agreed / len(items), 3),
        "over_accepted": over_accepted,
        "over_rejected": over_rejected,
        "by_verdict": {v: by_verdict[v] for v in sorted(by_verdict)},
        "disagreements": disagreements,
    }


def pooled_by_verdict(scores: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """The ``by_verdict`` counts of several papers' adjudication scores, summed."""
    pooled: dict[str, dict[str, int]] = {}
    for score in scores:
        for verdict, tally in (score.get("by_verdict") or {}).items():
            total = pooled.setdefault(verdict, {"items": 0, "agreed": 0})
            total["items"] += tally["items"]
            total["agreed"] += tally["agreed"]
    return {v: pooled[v] for v in sorted(pooled)}


def balanced_accuracy(by_verdict: dict[str, dict[str, int]]) -> float | None:
    """The mean, over the verdicts expected at least once, of the share matched.

    Plain accuracy rewards a marker that rejects everything, because most
    near misses should be rejected; here each verdict weighs the same however
    rare it is, so rejecting everything scores 1/2 where two verdicts are
    possible and 1/3 where three are.
    """
    rates = [t["agreed"] / t["items"] for t in by_verdict.values() if t["items"]]
    return round(sum(rates) / len(rates), 3) if rates else None


def validity_rate(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Share of seeds that came out as a valid exercise after the repair loop.

    Takes every case, failures included: the graph raises on a draft it could
    not repair, so ``ok`` is the validator's verdict.
    """
    rows = list(results)
    if not rows:
        return {"generated": 0, "valid": 0, "rate": 0.0}
    valid = sum(1 for r in rows if r.get("ok"))
    return {
        "generated": len(rows),
        "valid": valid,
        "rate": round(valid / len(rows), 3),
    }


def mean_attempts(results: Iterable[dict[str, Any]]) -> float:
    """Average generation attempts, repairs included."""
    values = [int(r.get("attempts", 1)) for r in results]
    return round(statistics.fmean(values), 3) if values else 0.0


def repairs_histogram(results: Iterable[dict[str, Any]]) -> dict[str, int]:
    """How many valid tasks needed 0, 1, 2... repairs, and how many never passed.

    Keys are strings so the report reads the same after a JSON round trip.
    """
    counts: Counter[str] = Counter()
    for r in results:
        if r.get("ok"):
            counts[str(int(r.get("repairs", 0)))] += 1
        else:
            counts["failed"] += 1
    numbered = sorted((k for k in counts if k != "failed"), key=int)
    return {k: counts[k] for k in numbered} | ({"failed": counts["failed"]} if counts["failed"] else {})


def issue_rule(message: str) -> str:
    """A validator message reduced to the rule behind it.

    The messages quote the offending text and count its words, so two drafts
    failing the same rule never say exactly the same thing. Quotes become '…'
    and numbers N, which leaves the rule's own wording to count by.
    """
    return re.sub(r"\d+", "N", re.sub(r"'.*'", "'…'", message)).strip()


def repair_reasons(
    histories: Iterable[Sequence[Sequence[str]]], top: int = 10
) -> list[list[Any]]:
    """The validator rules that sent drafts back, ``[rule, times]``, most frequent first.

    Each history is what ``validate`` said on every pass of one task; every pass
    that found something either triggered a repair or ended the budget.
    """
    counts: Counter[str] = Counter(
        issue_rule(issue) for history in histories for issues in history for issue in issues
    )
    return [[rule, n] for rule, n in counts.most_common(top)]


def key_self_consistency(part: int, payload: dict[str, Any]) -> dict[str, Any]:
    """Mark the exercise with its own answer key. Anything but full marks is a bug."""
    from cae.graphs.rue_evaluator import grade_attempt

    key = answer_key(part, parse_exercise(part, payload))
    graded = grade_attempt(part, payload, {k.number: k.answer for k in key})
    dropped = [
        row["item_number"] for row in graded["rows"] if not row["is_correct"]
    ]
    return {
        "part": part,
        "marks": graded["marks"],
        "max_marks": graded["max_marks"],
        "consistent": graded["marks"] == graded["max_marks"],
        "dropped_items": dropped,
    }


def numbers_in(text: str) -> list[float]:
    """Every numeric literal in a stretch of prose."""
    return [float(m) for m in _NUMBER_RE.findall(text or "")]


_PART_NUMBERS_RE = re.compile(
    r"\bparts?\s+([1-8](?:\s*(?:,\s*(?:and|or)?|and|&|or)\s*[1-8])*)\b", re.IGNORECASE
)

_DASH = "‐‑–—-"

_URL_RE = re.compile(r"https?://[^\s)\]]+|\bwww\.[^\s)\]]+", re.IGNORECASE)

_NUMBER_WORDS = {
    word: index
    for index, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
}
_NUMBER_WORD_RE = re.compile(rf"\b(?:{'|'.join(_NUMBER_WORDS)})\b", re.IGNORECASE)

ROUNDING = 0.5


def _ascii_dashes(text: str) -> str:
    """``text`` with every dash an ASCII hyphen, so "Cross‑text" (U+2011) reads
    as the part name "Cross-text"."""
    return re.sub(f"[{_DASH}]", "-", text or "")


def parts_named(text: str, names: dict[int, str]) -> set[int]:
    """The parts a stretch of prose names: "Part 3", "Parts 5 and 7", or a name.

    Names are matched longest first, so "Cross-text multiple matching" is not
    also read as "Multiple matching", and a dash counts as a space: "Cross‑text"
    (U+2011) is Part 6, and "multiple-choice accuracy" is Part 5's "Multiple
    choice" once "multiple-choice cloze" (Part 1) has been taken out.
    """
    found = {
        int(digit)
        for match in _PART_NUMBERS_RE.finditer(text)
        for digit in re.findall(r"[1-8]", match.group(1))
    }

    def plain(value: str) -> str:
        return " ".join(_ascii_dashes(value).replace("-", " ").lower().split())

    rest = plain(text)
    for part, name in sorted(names.items(), key=lambda item: -len(item[1])):
        if name and plain(name) in rest:
            found.add(int(part))
            rest = rest.replace(plain(name), " ")
    return found


def _names(facts: dict[str, Any]) -> dict[int, str]:
    """Each part's name, keyed by int as it is after a JSON round trip too."""
    return {int(part): stat["name"] for part, stat in (facts.get("parts") or {}).items()}


def category_terms(category: str, extra: Iterable[str] = ()) -> list[str]:
    """The words that name an error category in prose, lower case.

    Its code, the code as words, its label, and each piece of a compound label
    ("Collocations, phrasal verbs and idioms" -> "phrasal verbs"), plus ``extra``.
    """
    from cae.domain.taxonomy import label

    text = label(category)
    pieces = [p for p in re.split(r",\s*|\s+and\s+", text) if p]
    terms = [category, category.replace("_", " "), text, *pieces, *extra]
    return sorted({t.lower() for t in terms if t.strip()})


def subscale_terms(subscale: str, extra: Iterable[str] = ()) -> list[str]:
    """The words that name a Writing subscale in prose, lower case."""
    from cae.domain.parts import SUBSCALE_LABELS

    terms = [subscale.replace("_", " "), SUBSCALE_LABELS.get(subscale, subscale), *extra]
    return sorted({t.lower() for t in terms if t.strip()})


def mentions(
    text: str,
    target: dict[str, Any],
    facts: dict[str, Any],
    terms: dict[str, list[str]] | None = None,
) -> bool:
    """Whether ``text`` names one target: a part, a category or a subscale."""
    terms = terms or {}
    if "part" in target:
        return int(target["part"]) in parts_named(text, _names(facts))
    if "category" in target:
        words = category_terms(target["category"], terms.get(target["category"], ()))
    else:
        words = subscale_terms(target["subscale"], terms.get(target["subscale"], ()))
    lowered = _ascii_dashes(text).lower()
    return any(re.search(rf"\b{re.escape(_ascii_dashes(word))}", lowered) for word in words)


def in_digits(text: str) -> str:
    """``text`` with the numbers zero to twenty written as words turned into
    digits, and thousands separators dropped ("1,020" is 1020).

    The reference corpus spells small numbers out ("between three and six
    words"); grounding compares digits, so evidence goes through this first.
    """
    spelled = _NUMBER_WORD_RE.sub(lambda m: str(_NUMBER_WORDS[m.group(0).lower()]), text or "")
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", spelled)


def stated_numbers(text: str) -> list[float]:
    """Every number a stretch of prose states, in digits or as a word (zero to
    twenty), with thousands separators read through.

    Unlike :func:`figures_in` nothing is left out -- a part's number, "16
    minutes" -- because this asks whether a figure was *said*, not whether it
    was made up: "item 3" is the answer to "which items did I get wrong?".
    """
    return numbers_in(in_digits(_URL_RE.sub(" ", text or "")))


_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def _normalised(text: str) -> str:
    """Lower case, ASCII dashes and quotes, single spaces: how spans are compared."""
    return " ".join(_ascii_dashes(text).translate(_QUOTES).lower().split())


def _said(span: str, text: str) -> bool:
    return _normalised(span) in _normalised(text)


def _alternatives(figure: Any) -> list[float]:
    """An expected figure is one number, or a list of ways to say it (16 or 960)."""
    return [float(f) for f in (figure if isinstance(figure, list) else [figure])]


def _stated(figure: Any, numbers: Sequence[float]) -> bool:
    return any(abs(n - alt) <= ROUNDING for alt in _alternatives(figure) for n in numbers)


_NOTHING_RE = re.compile(
    r"\b(?:haven'?t|have not|hasn'?t|has not|not yet|yet to|never|nothing|none)\b"
    r"|\bno (?:\w+ )?(?:mocks?|essays?|records?|data|results?|attempts?|exams?|"
    r"figures?|work|scores?|marks?|feedback)\b"
    r"|\b(?:don'?t|do not|doesn'?t|does not) (?:have|show|include|hold) any\b"
    r"|\bnot (?:attempted|sat|taken|done|written|tried|any)\b",
    re.IGNORECASE,
)


def answer_correct(
    reply: str, expect: dict[str, Any], facts: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Whether a chat reply says what the question needed, as ``expect`` lists it.

    Not a metric of any suite: the judge marks correctness. It keeps the
    datasets honest: every ``reference`` the judge compares against -- a chat
    turn's or a coach profile's -- must pass it (tests/test_evals_offline.py).

    Each check is one thing a correct reply must contain: every ``figures``
    entry (a number, or a list of ways to say it: ``[16, 960]`` for minutes or
    seconds), within :data:`ROUNDING`; every ``targets`` entry (a part, a
    category or a subscale, as :func:`mentions` reads them, with ``terms`` for
    extra words); every ``spans`` entry word for word; at least one of
    ``any_spans``; and none of ``absent`` -- what the record holds but the
    question excludes, such as the answers of an earlier attempt when it asks
    about the latest. A ``none`` question -- the record has nothing on it -- has
    one check: that the reply says so.

    This is what groundedness cannot see: "I don't have that figure" invents
    nothing and so grounds perfectly, but it is not an answer.
    """
    facts = facts or {}
    if expect.get("none"):
        said = bool(_NOTHING_RE.search(_normalised(reply)))
        return {
            "checks": 1,
            "passed": int(said),
            "correct": said,
            "missing": [] if said else ["that the record has nothing on it"],
        }

    numbers = stated_numbers(reply)
    terms = dict(expect.get("terms") or {})
    missing: list[str] = []
    for figure in expect.get("figures") or []:
        if not _stated(figure, numbers):
            missing.append("figure " + " or ".join(f"{v:g}" for v in _alternatives(figure)))
    for target in expect.get("targets") or []:
        if not mentions(reply, target, facts, terms):
            missing.append(f"target {next(iter(target.values()))}")
    for span in expect.get("spans") or []:
        if not _said(span, reply):
            missing.append(f'"{span}"')
    any_spans = expect.get("any_spans") or []
    if any_spans and not any(_said(span, reply) for span in any_spans):
        missing.append("one of " + ", ".join(f'"{s}"' for s in any_spans))
    for span in expect.get("absent") or []:
        if _said(span, reply):
            missing.append(f'not "{span}"')

    checks = (
        len(expect.get("figures") or [])
        + len(expect.get("targets") or [])
        + len(expect.get("spans") or [])
        + (1 if any_spans else 0)
        + len(expect.get("absent") or [])
    )
    return {
        "checks": checks,
        "passed": checks - len(missing),
        "correct": not missing,
        "missing": missing,
    }


def tool_choice(
    called: Iterable[str], required: Iterable[str], allowed: Iterable[str] = ()
) -> dict[str, Any]:
    """Whether a turn used the tools it needed and no others.

    ``required`` must each be called; ``allowed`` may be, at no penalty (the
    examples of a mistake are in the context, but fetching more is fair). Any
    other call is ``extra``: a round, and a call on the allowance, the turn did
    not need -- the regression where "how am I doing?" starts costing two calls.
    """
    called_set = set(called)
    missing = sorted(set(required) - called_set)
    extra = sorted(called_set - set(required) - set(allowed))
    return {"called": sorted(called_set), "missing": missing, "extra": extra, "ok": not missing and not extra}


def section_hit(sources: Iterable[dict[str, Any]], sections: Sequence[str]) -> dict[str, Any]:
    """Whether ``reference_lookup`` brought back a passage from one of ``sections``.

    Matched on the heading path the chunk is cited by, not on a chunk id: chunk
    ids move whenever the corpus is re-ingested. The text is not searched, since
    "Part 6" appears in the overview too and would make every lookup a hit.
    """
    rows = list(sources)
    wanted = [_normalised(s) for s in sections]
    paths = [_normalised(f"{r.get('section', '')} {r.get('citation', '')}") for r in rows]
    return {
        "retrieved": len(rows),
        "hit": any(w in p for w in wanted for p in paths),
        "citations": [str(r.get("citation", "")) for r in rows][:4],
    }


__all__ = [
    "adjudication_accuracy",
    "annotation_scores",
    "answer_correct",
    "balanced_accuracy",
    "BAND_MAX",
    "band_accuracy",
    "band_errors",
    "category_accuracy",
    "category_family",
    "category_terms",
    "consistency",
    "in_digits",
    "issue_rule",
    "key_self_consistency",
    "mean_attempts",
    "mentions",
    "numbers_in",
    "parts_named",
    "pooled_annotation_scores",
    "pooled_by_verdict",
    "repair_reasons",
    "repairs_histogram",
    "section_hit",
    "stated_numbers",
    "subscale_terms",
    "tool_choice",
    "validity_rate",
]
