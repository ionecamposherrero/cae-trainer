"""Offline tests for the evaluation metrics and datasets."""

from __future__ import annotations

import pytest

from collections import Counter

from cae.domain.answers import ADJUDICATED_PARTS, LETTER_PARTS, grade_deterministically
from cae.domain.parts import PART_SPECS, READING_PART_NUMBERS, SUBSCALES, WRITING_WORD_RANGE
from cae.domain.taxonomy import LANGUAGE_CATEGORIES, ErrorCategory
from cae.graphs.writing_evaluator import count_words
from cae.schemas.rue_task import answer_key, parse_exercise
from cae.validators.rue_task import validate_exercise

from tests import factories

from evals import load_dataset
from evals import metrics


PERFECT = {s: 5 for s in SUBSCALES}
MIDDLING = {s: 3 for s in SUBSCALES}


def test_identical_bands_are_fully_accurate() -> None:
    accuracy = metrics.band_accuracy(PERFECT, PERFECT)
    assert accuracy == {**{s: 1.0 for s in SUBSCALES}, "total": 1.0}


def test_band_error_is_signed_so_generosity_is_visible() -> None:
    """Marking everything a band too high and a band too low are different faults."""
    generous = metrics.band_errors({s: 4 for s in SUBSCALES}, MIDDLING)
    harsh = metrics.band_errors({s: 2 for s in SUBSCALES}, MIDDLING)

    assert set(generous.values()) == {1}
    assert set(harsh.values()) == {-1}
    assert metrics.band_accuracy({s: 4 for s in SUBSCALES}, MIDDLING) == metrics.band_accuracy(
        {s: 2 for s in SUBSCALES}, MIDDLING
    )


def test_band_accuracy_is_out_of_five_per_subscale_and_out_of_twenty_on_the_total() -> None:
    predicted = {"content": 5, "communicative_achievement": 4,
                 "organisation": 3, "language": 1}
    accuracy = metrics.band_accuracy(predicted, {s: 4 for s in SUBSCALES})

    assert accuracy["content"] == 0.8 and accuracy["communicative_achievement"] == 1.0
    assert accuracy["organisation"] == 0.8 and accuracy["language"] == 0.4
    assert accuracy["total"] == 0.85


def test_opposite_band_errors_cancel_on_the_total_as_in_the_learners_mark() -> None:
    predicted = {**MIDDLING, "content": 4, "language": 2}
    accuracy = metrics.band_accuracy(predicted, MIDDLING)
    assert accuracy["content"] == accuracy["language"] == 0.8
    assert accuracy["total"] == 1.0


def test_only_the_subscales_with_a_reference_are_scored() -> None:
    """A half-labelled case must not be penalised for the labels it lacks."""
    errors = metrics.band_errors(PERFECT, {"content": 3})
    assert errors == {"content": 2}


def test_consistency_reports_the_spread_across_repeated_markings() -> None:
    result = metrics.consistency([14, 14, 14])
    assert result["stdev"] == 0.0 and result["spread"] == 0.0

    wobbly = metrics.consistency([12, 14, 16])
    assert wobbly["spread"] == 4.0
    assert wobbly["mean"] == 14.0
    assert wobbly["stdev"] > 0


def test_a_single_run_has_no_spread_to_report() -> None:
    assert metrics.consistency([15])["stdev"] == 0.0
    assert metrics.consistency([])["runs"] == 0


SCRIPT = (
    "Last month my parents was celebrating, and the waiter explained us the menu. "
    "We ate the better fish of our lives, near of the harbour, and it was lovely."
)
EXPECTED = [
    {"span": "my parents was", "category": "GRAMMAR_WORDS"},
    {"span": "explained us the menu", "category": "WORD_PATTERNS"},
    {"span": "the better fish", "category": "SENTENCE_STRUCTURE", "also": ["WORD_FORM"]},
]
TOLERATED = ["near of the harbour"]


def _note(span: str, category: str) -> dict:
    return {"span": span, "category": category}


def test_annotation_scores_count_recall_precision_and_category_accuracy() -> None:
    annotations = [
        _note("parents was", "GRAMMAR_WORDS"),
        _note("Explained us", "VERB_FORMS"),
        _note("near of", "GRAMMAR_WORDS"),
        _note("it was lovely", "WORD_CHOICE"),
        _note("a phrase nobody wrote", "WORD_FORM"),
    ]
    s = metrics.annotation_scores(annotations, SCRIPT, EXPECTED, TOLERATED)

    assert (s["expected"], s["found"], s["annotations"], s["hits"]) == (3, 2, 5, 3)
    assert s["recall"] == 0.667 and s["precision"] == 0.6 and s["category_accuracy"] == 0.5
    assert s["missed"] == ["the better fish"]
    assert s["false_positives"] == ["it was lovely", "a phrase nobody wrote"]
    assert s["miscategorised"] == [
        {"span": "explained us the menu", "expected": "WORD_PATTERNS", "got": ["VERB_FORMS"]}
    ]
    assert s["by_category"]["GRAMMAR_WORDS"] == {
        "expected": 1, "found_right": 1, "labelled": 1, "labelled_right": 1
    }
    assert s["by_category"]["VERB_FORMS"] == {
        "expected": 0, "found_right": 0, "labelled": 1, "labelled_right": 0
    }
    assert s["by_category"]["WORD_PATTERNS"]["found_right"] == 0


def test_an_accepted_alternative_category_is_right() -> None:
    s = metrics.annotation_scores([_note("better fish", "WORD_FORM")], SCRIPT, EXPECTED)
    assert s["category_accuracy"] == 1.0
    assert s["by_category"]["SENTENCE_STRUCTURE"]["found_right"] == 1
    assert s["by_category"]["WORD_FORM"]["labelled_right"] == 1


def test_a_clean_script_with_no_annotation_has_nothing_to_score() -> None:
    s = metrics.annotation_scores([], "A clean script.", [])
    assert s["recall"] is None and s["precision"] is None and s["category_accuracy"] is None


def test_any_annotation_on_a_clean_script_is_a_false_positive() -> None:
    s = metrics.annotation_scores([_note("clean", "WORD_CHOICE")], "A clean script.", [])
    assert s["precision"] == 0.0 and s["recall"] is None


def test_annotation_scores_are_pooled_by_counts_not_averaged() -> None:
    first = metrics.annotation_scores([_note("parents was", "GRAMMAR_WORDS")], SCRIPT, EXPECTED)
    clean = metrics.annotation_scores([_note("clean", "WORD_CHOICE")], "A clean script.", [])
    pooled = metrics.pooled_annotation_scores([first, clean])

    assert pooled["recall"] == 0.333 and pooled["precision"] == 0.5
    assert pooled["category_accuracy"] == 1.0
    assert pooled["by_category"]["GRAMMAR_WORDS"] == {"recall": 1.0, "precision": 1.0}
    assert pooled["by_category"]["WORD_CHOICE"] == {"recall": None, "precision": 0.0}
    assert pooled["by_category"]["WORD_PATTERNS"] == {"recall": 0.0, "precision": None}


FAMILIES = {
    "VERB_FORMS": "GRAMMAR",
    "SENTENCE_STRUCTURE": "GRAMMAR",
    "GRAMMAR_WORDS": "GRAMMAR",
    "WORD_PATTERNS": "VOCABULARY",
    "WORD_CHOICE": "VOCABULARY",
    "FIXED_PHRASES": "VOCABULARY",
    "WORD_FORM": "VOCABULARY",
    "LINKING": "TEXT",
    "REGISTER": "TEXT",
    "PUNCTUATION": "TEXT",
    "READING_FOR_MEANING": "READING",
    "COMPARING_VIEWS": "READING",
    "TEXT_STRUCTURE": "READING",
    "SCANNING": "READING",
}


def test_every_category_of_the_taxonomy_has_an_expected_family() -> None:
    assert set(FAMILIES) == {c.value for c in ErrorCategory}


@pytest.mark.parametrize(("category", "family"), sorted(FAMILIES.items()))
def test_every_category_lands_in_a_family(category: str, family: str) -> None:
    assert metrics.category_family(category) == family


def test_a_near_miss_within_the_family_still_counts_as_a_family_hit() -> None:
    """The dashboard acts on families, so both levels are worth reporting."""
    result = metrics.category_accuracy(
        {1: "SENTENCE_STRUCTURE", 2: "FIXED_PHRASES"},
        {1: "VERB_FORMS", 2: "FIXED_PHRASES"},
    )
    assert result["exact"] == 0.5
    assert result["family"] == 1.0
    assert result["disagreements"] == [
        {"item": 1, "expected": "VERB_FORMS", "got": "SENTENCE_STRUCTURE"}
    ]


def test_two_unknown_codes_are_not_a_family_hit() -> None:
    """An unknown code has no family, and two unknowns must not count as one."""
    result = metrics.category_accuracy({1: "GRAMMAR_MODAL"}, {1: "GRAMMAR_TENSE"})
    assert result["family"] == 0.0


def test_unlabelled_items_are_skipped_rather_than_failed() -> None:
    """An item with no defensible category must not drag the score down."""
    result = metrics.category_accuracy(
        {1: "VERB_FORMS", 2: "LINKING"}, {1: "VERB_FORMS", 2: None}
    )
    assert result["items"] == 1
    assert result["exact"] == 1.0


def test_adjudication_counts_each_expected_verdict_apart() -> None:
    result = metrics.adjudication_accuracy(
        {1: "reject", 2: "reject", 3: "reject", 4: "accept"},
        {1: "reject", 2: "reject", 3: "accept", 4: "accept"},
    )
    assert result["by_verdict"] == {
        "accept": {"items": 2, "agreed": 1},
        "reject": {"items": 2, "agreed": 2},
    }


def test_balanced_accuracy_weighs_every_verdict_the_same() -> None:
    """Rejecting everything matches every reject and nothing else."""
    lazy = {"accept": {"items": 4, "agreed": 0}, "reject": {"items": 12, "agreed": 12}}
    assert metrics.balanced_accuracy(lazy) == 0.5
    three = {**lazy, "partial": {"items": 3, "agreed": 0}}
    assert metrics.balanced_accuracy(three) == 0.333
    assert metrics.balanced_accuracy({}) is None


def test_by_verdict_counts_pool_across_papers() -> None:
    papers = [
        {"by_verdict": {"accept": {"items": 1, "agreed": 1}}},
        {"by_verdict": {"accept": {"items": 2, "agreed": 0}, "reject": {"items": 3, "agreed": 3}}},
        {},
    ]
    assert metrics.pooled_by_verdict(papers) == {
        "accept": {"items": 3, "agreed": 1},
        "reject": {"items": 3, "agreed": 3},
    }


def test_adjudication_separates_the_two_directions_of_error() -> None:
    result = metrics.adjudication_accuracy(
        {1: "accept", 2: "reject", 3: "accept"},
        {1: "accept", 2: "accept", 3: "reject"},
    )
    assert result["accuracy"] == pytest.approx(1 / 3, abs=0.01)
    assert result["over_accepted"] == 1
    assert result["over_rejected"] == 1


def test_a_partial_between_accept_and_reject_ranks_correctly() -> None:
    generous = metrics.adjudication_accuracy({1: "accept"}, {1: "partial"})
    harsh = metrics.adjudication_accuracy({1: "reject"}, {1: "partial"})
    assert generous["over_accepted"] == 1 and generous["over_rejected"] == 0
    assert harsh["over_rejected"] == 1 and harsh["over_accepted"] == 0


def test_validity_rate_counts_failed_generations_against_the_rate() -> None:
    """A seed the repair loop could not fix raised; it must not leave the denominator."""
    result = metrics.validity_rate(
        [{"ok": True}, {"ok": False, "error": "GenerationFailed: ..."}, {"ok": True}]
    )
    assert result == {"generated": 3, "valid": 2, "rate": 0.667}


def test_mean_attempts_measures_the_repair_loop() -> None:
    assert metrics.mean_attempts([{"attempts": 1}, {"attempts": 3}]) == 2.0


@pytest.mark.parametrize("part", READING_PART_NUMBERS)
def test_every_generated_exercise_marks_itself_full_marks(part: int) -> None:
    """The generator, the key and the marker must agree. No API call is made."""
    payload = factories.build(part).model_dump()
    result = metrics.key_self_consistency(part, payload)

    assert result["consistent"], (
        f"Part {part} did not mark its own key full marks; "
        f"items {result['dropped_items']} dropped marks"
    )
    assert result["marks"] == result["max_marks"]


def test_the_writing_dataset_carries_the_reference_bands() -> None:
    cases = {c["id"]: c for c in load_dataset("writing_scripts")["cases"]}

    assert cases["museums-strong"]["expected_bands"] == {s: 5 for s in SUBSCALES}
    assert cases["museums-weak"]["expected_bands"] == {
        "content": 3,
        "communicative_achievement": 2,
        "organisation": 2,
        "language": 2,
    }


def test_every_writing_script_is_within_the_word_range() -> None:
    """Outside it the evaluator does not assess, so the case would measure nothing."""
    low, high = WRITING_WORD_RANGE
    for case in load_dataset("writing_scripts")["cases"]:
        words = count_words(case["script"])
        assert low <= words <= high, f"{case['id']}: {words} words"


def test_every_writing_case_has_expected_bands() -> None:
    """Every group measures the error against the expected bands."""
    for case in load_dataset("writing_scripts")["cases"]:
        assert set(case["expected_bands"]) == set(SUBSCALES), case["id"]


def _writing_cases(group: int | None = None) -> list[dict]:
    cases = load_dataset("writing_scripts")["cases"]
    return [c for c in cases if group is None or c["group"] == group]


def _genre(case: dict) -> str:
    return load_dataset("writing_scripts")["tasks"][case["task"]]["genre"]


ALL_GENRES = ("essay", "letter_email", "proposal", "report", "review")


def test_the_writing_set_has_three_groups_of_five_with_unique_ids() -> None:
    from evals.suites import writing_evaluator_groups

    cases = _writing_cases()
    assert writing_evaluator_groups() == [1, 2, 3]
    assert len({c["id"] for c in cases}) == len(cases)
    assert Counter(c["group"] for c in cases) == {1: 5, 2: 5, 3: 5}


@pytest.mark.parametrize("group", [1, 2, 3])
def test_every_writing_group_has_one_script_of_each_genre(group: int) -> None:
    """So a model run on one group has still marked every genre once."""
    assert sorted(_genre(c) for c in _writing_cases(group)) == sorted(ALL_GENRES)


@pytest.mark.parametrize("genre", ALL_GENRES)
def test_every_genre_spans_the_scale(genre: str) -> None:
    """Three scripts of one level would measure the marker at that level only."""
    totals = [sum(c["expected_bands"].values()) for c in _writing_cases() if _genre(c) == genre]
    assert len(totals) == 3 and len(set(totals)) == 3, totals
    assert max(totals) == 20


def test_the_writing_set_carries_no_diagnosis_or_ordering_labels() -> None:
    """The set is scored on bands and annotations, not weak-subscale or ranking labels."""
    dataset = load_dataset("writing_scripts")
    assert "comparisons" not in dataset
    assert not any("weakest_subscale" in c for c in _writing_cases())


@pytest.mark.parametrize("task_id", sorted(load_dataset("writing_scripts")["tasks"]))
def test_every_fixed_writing_task_is_a_valid_task(task_id: str) -> None:
    from cae.validators.writing_task import validate_writing_task

    task = load_dataset("writing_scripts")["tasks"][task_id]
    assert validate_writing_task(task["part"], task["genre"], task["spec"]) == []


def test_every_writing_case_names_a_task_of_its_part() -> None:
    from cae.domain.parts import WRITING_PART1_GENRES, WRITING_PART2_GENRES

    tasks = load_dataset("writing_scripts")["tasks"]
    for case in _writing_cases():
        task = tasks[case["task"]]
        genres = WRITING_PART1_GENRES if task["part"] == 1 else WRITING_PART2_GENRES
        assert task["genre"] in genres, case["id"]


def test_every_labelled_error_is_one_place_in_its_script() -> None:
    """A span that is absent, or there twice, would match the wrong annotation."""
    from cae.validators import normalise

    for case in _writing_cases():
        text = normalise(case["script"])
        for error in case["expected_annotations"]:
            assert text.count(normalise(error["span"])) == 1, (case["id"], error["span"])
        for span in case["tolerated_annotations"]:
            assert normalise(span) in text, (case["id"], span)


def test_every_labelled_error_has_writing_categories_and_leaves_room_under_the_cap() -> None:
    """The node reports at most twelve errors and chooses which. Eight must-find
    errors leave it room to pick other real ones without losing recall."""
    from cae.domain.taxonomy import WRITING_CATEGORIES

    allowed = {str(c) for c in WRITING_CATEGORIES}
    for case in _writing_cases():
        assert len(case["expected_annotations"]) <= 8, case["id"]
        for error in case["expected_annotations"]:
            assert {error["category"], *error.get("also", [])} <= allowed, error


@pytest.mark.parametrize("genre", ALL_GENRES)
def test_every_genre_has_clean_scripts_and_scripts_with_errors(genre: str) -> None:
    """Per genre, precision needs a clean script to tempt false positives and
    recall needs errors to find."""
    counts = [len(c["expected_annotations"]) for c in _writing_cases() if _genre(c) == genre]
    assert 0 in counts and max(counts) >= 4, counts


def test_no_writing_case_carries_an_unused_tolerance() -> None:
    """A per-case field nothing reads misleads."""
    assert not any("tolerance" in c for c in _writing_cases())


def test_the_reading_cases_are_structurally_valid_exercises() -> None:
    for case in load_dataset("reading_cases")["cases"]:
        issues = validate_exercise(case["part"], parse_exercise(case["part"], case["exercise"]))
        assert issues == [], f"{case['id']}: {issues}"


def test_the_hand_authored_near_misses_all_reach_the_adjudicator() -> None:
    """A near-miss settled offline scores itself, and tests nothing about the model."""
    for case in load_dataset("reading_cases")["cases"]:
        part = case["part"]
        if part not in ADJUDICATED_PARTS:
            continue
        key = answer_key(part, parse_exercise(part, case["exercise"]))
        responses = {int(k): v for k, v in case["responses"].items()}
        results = grade_deterministically(part, key, responses)

        settled = [r.number for r in results if not r.needs_adjudication]
        assert not settled, (
            f"{case['id']}: items {settled} never reach the model, so their "
            "expected verdicts measure the deterministic pass instead"
        )


USE_OF_ENGLISH_PARTS = (1, 2, 3, 4)


def _reading_cases(group: int | None = None) -> list[dict]:
    cases = load_dataset("reading_cases")["cases"]
    return [c for c in cases if group is None or c["group"] == group]


def _marking_groups() -> list[int]:
    from evals.suites import marking_groups

    return marking_groups()


def _first_pass(case: dict):
    part = case["part"]
    key = answer_key(part, parse_exercise(part, case["exercise"]))
    responses = {int(k): v for k, v in case["responses"].items()}
    return {r.number: r for r in grade_deterministically(part, key, responses)}


def test_the_marking_set_has_three_groups_of_four_papers_with_unique_ids() -> None:
    cases = _reading_cases()
    assert _marking_groups() == [1, 2, 3]
    assert len({c["id"] for c in cases}) == len(cases)
    assert Counter(c["group"] for c in cases) == {1: 4, 2: 4, 3: 4}


@pytest.mark.parametrize("group", [1, 2, 3])
def test_every_group_has_one_paper_for_each_use_of_english_part(group: int) -> None:
    """Groups of the same make-up keep one day's figure comparable with another's."""
    assert sorted(c["part"] for c in _reading_cases(group)) == list(USE_OF_ENGLISH_PARTS)


def test_every_language_category_is_expected_at_least_twice_across_the_groups() -> None:
    """One labelled item per category is a coin toss, not a measurement."""
    labelled = Counter(
        category
        for case in _reading_cases()
        for category in case["expected_category"].values()
        if category
    )
    thin = {c.value: labelled[c.value] for c in LANGUAGE_CATEGORIES if labelled[c.value] < 2}
    assert not thin, f"categories with fewer than two labelled items: {thin}"


@pytest.mark.parametrize("part", sorted(ADJUDICATED_PARTS))
def test_every_verdict_a_part_can_give_is_expected_across_the_groups(part: int) -> None:
    """A verdict that is never expected is one the suite cannot catch the model missing."""
    possible = {"accept", "reject"} | ({"partial"} if PART_SPECS[part].marks_per_item >= 2 else set())
    expected = {
        verdict
        for case in _reading_cases()
        if case["part"] == part
        for verdict in case["expected_adjudication"].values()
    }
    assert expected == possible


def test_part4_reaches_the_adjudicator_both_ways_with_every_verdict() -> None:
    """Close to the key starts at a half mark, different from it at zero: a model
    that leaves either alone scores differently, so both must be measured."""
    seen: dict[str, set[str]] = {"close": set(), "different": set()}
    for case in _reading_cases():
        if case["part"] != 4:
            continue
        first = _first_pass(case)
        for number, verdict in case["expected_adjudication"].items():
            route = "close" if first[int(number)].marks > 0 else "different"
            seen[route].add(verdict)
    assert seen == {
        "close": {"accept", "partial", "reject"},
        "different": {"accept", "partial", "reject"},
    }


def test_every_mark_expected_to_be_lost_has_a_category() -> None:
    """A labelled set with holes measures less than it seems to."""
    for case in _reading_cases():
        losing = (
            set(case["responses"])
            if case["part"] in LETTER_PARTS
            else {n for n, v in case["expected_adjudication"].items() if v != "accept"}
        )
        unlabelled = sorted(losing - {n for n, c in case["expected_category"].items() if c})
        assert not unlabelled, f"{case['id']}: items {unlabelled} have no expected category"


def test_a_letter_paper_answers_every_item_with_a_wrong_option() -> None:
    """Part 1 is never adjudicated, so only a wrong letter reaches the tagger."""
    for case in _reading_cases():
        if case["part"] not in LETTER_PARTS:
            continue
        letters = PART_SPECS[case["part"]].option_letters
        for number, result in _first_pass(case).items():
            assert result.user_answer in letters and not result.is_correct, (
                f"{case['id']} item {number}"
            )
            assert "expected_adjudication" not in case


def test_the_set_tests_spelling_both_ways() -> None:
    """The adjudicator must reject a misspelling yet accept a spelling variant."""
    rationale = " ".join(
        text for case in _reading_cases() for text in case["rationale"].values()
    ).lower()
    assert "misspelling" in rationale and "american spelling" in rationale


def test_every_expected_category_is_allowed_for_its_part() -> None:
    """The taxonomy is closed per part; a label outside it can never be produced."""
    from cae.domain.taxonomy import categories_for_part

    for case in load_dataset("reading_cases")["cases"]:
        allowed = {str(c) for c in categories_for_part(case["part"])}
        for number, category in (case.get("expected_category") or {}).items():
            if category:
                assert category in allowed, (
                    f"{case['id']} item {number}: {category} is not available "
                    f"for Part {case['part']}"
                )


def test_the_suite_never_traces_to_langsmith() -> None:
    """The page tests boot tracing, so the session must force it off."""
    from cae.config import get_settings
    from cae.observability import configure_tracing, tracing_enabled

    assert get_settings().has_tracing is False
    assert configure_tracing() is False
    assert tracing_enabled() is False


def test_every_seed_group_pins_one_topic_for_every_part() -> None:
    """An unseeded part would be generated on a random topic and not be comparable;
    a group short of a part would make one day's run cover less than another's."""
    from evals.suites import generation_groups

    seeds = load_dataset("generation_seeds")["seeds"]
    assert len(generation_groups()) >= 2
    for group in generation_groups():
        parts = [s["part"] for s in seeds if s["group"] == group]
        assert sorted(parts) == list(READING_PART_NUMBERS), f"group {group}: {parts}"
    assert all(s["topic"] for s in seeds)
    topics = [s["topic"] for s in seeds]
    assert len(topics) == len(set(topics)), "a topic repeated across groups measures it twice"


def test_every_seed_focus_is_one_the_part_actually_tests() -> None:
    """plan_task draws focus from PART_SPECS; a seed must not ask for anything else."""
    from cae.domain.parts import get_part_spec

    for seed in load_dataset("generation_seeds")["seeds"]:
        focus = get_part_spec(seed["part"]).focus
        assert seed["focus"], seed
        assert set(seed["focus"]) <= set(focus), f"part {seed['part']}: {seed['focus']}"


def test_one_seed_group_runs_one_exercise_per_part(monkeypatch: pytest.MonkeyPatch) -> None:
    from cae.graphs import rue_task
    from evals import suites

    briefs: list[tuple[int, str, str]] = []

    def fake_generate(part, *, topic, focus, persist_result, **kwargs):
        briefs.append((part, topic, focus))
        return {"exercise": factories.build(part).model_dump(), "attempts": 1, "status": "ok"}

    monkeypatch.setattr(rue_task, "generate_exercise", fake_generate)
    result = suites.run_generation(group=2)

    group2 = [s for s in load_dataset("generation_seeds")["seeds"] if s["group"] == 2]
    assert [b[0] for b in briefs] == [s["part"] for s in group2]
    assert {(p, t) for p, t, _ in briefs} == {(s["part"], s["topic"]) for s in group2}
    assert all(f == "; ".join(s["focus"]) for (_, _, f), s in zip(briefs, group2))
    assert result["group"] == 2
    assert [c["id"] for c in result["cases"]] == [f"part{s['part']}-g2" for s in group2]
    assert result["metrics"]["rate"] == 1.0


def _group_report(group: int, runs: dict[str, list[bool] | str]) -> dict:
    """A run_evals payload for one group: per model, ok flags by part, or an error."""
    payload_runs = []
    for model, outcome in runs.items():
        if isinstance(outcome, str):
            suite = {"suite": "generation", "error": outcome}
        else:
            cases = [
                {"id": f"part{p}-g{group}", "part": p, "group": group, "ok": ok}
                | ({"attempts": 1, "seconds": 2.0, "self_consistency": {"consistent": True}} if ok else {})
                for p, ok in zip(READING_PART_NUMBERS, outcome)
            ]
            suite = {"suite": "generation", "group": group, "cases": cases}
        payload_runs.append({"model": model, "suites": {"generation": suite}})
    return {"group": group, "runs": payload_runs}


def test_combining_groups_pools_every_seed_rather_than_averaging_rates() -> None:
    from scripts.combine_generation_reports import combine

    all_ok, one_bad = [True] * 8, [True] * 7 + [False]
    combined = {
        e["model"]: e
        for e in combine(
            [
                _group_report(1, {"a": all_ok, "b": one_bad}),
                _group_report(2, {"a": one_bad, "b": one_bad}),
                _group_report(3, {"a": all_ok, "b": "QuotaExhaustedError: spent"}),
            ]
        )
    }

    a, b = combined["a"], combined["b"]
    assert a["complete"] and a["metrics"]["generated"] == 24
    assert a["metrics"]["valid"] == 23 and a["metrics"]["rate"] == 0.958
    assert a["by_part"][8] == {"valid": 2, "tried": 3}
    assert not b["complete"] and "spent" in b["groups"][3]["error"]
    assert b["metrics"]["generated"] == 16


def test_a_model_absent_from_one_days_sweep_is_incomplete() -> None:
    from scripts.combine_generation_reports import combine

    combined = combine(
        [_group_report(1, {"a": [True] * 8}), _group_report(2, {"b": [True] * 8})]
    )
    assert all(not e["complete"] for e in combined)


def test_a_model_added_later_joins_a_group_without_erasing_the_others() -> None:
    """Re-running group 1 for a new model must keep the models of the first run."""
    from scripts.combine_generation_reports import combine

    first = _group_report(1, {"a": [True] * 8, "b": [False] * 8})
    later = _group_report(1, {"c": [True] * 8})
    combined = {e["model"]: e for e in combine([first, later])}

    assert set(combined) == {"a", "b", "c"}
    assert all(e["complete"] for e in combined.values())
    assert combined["b"]["metrics"]["valid"] == 0


def test_the_newest_result_of_a_model_in_a_group_replaces_the_older_one() -> None:
    from scripts.combine_generation_reports import combine

    older = {**_group_report(1, {"a": [False] * 8}), "source": "old.json"}
    newer = {**_group_report(1, {"a": [True] * 8}), "source": "new.json"}
    (a,) = combine([older, newer])

    assert a["metrics"]["generated"] == 8 and a["metrics"]["valid"] == 8
    assert a["groups"][1]["sources"] == ["new.json"]


def test_a_later_run_that_produced_nothing_does_not_erase_a_finished_group() -> None:
    from scripts.combine_generation_reports import combine

    finished = _group_report(1, {"a": [True] * 8})
    spent = _group_report(1, {"a": "QuotaExhaustedError: spent"})
    (a,) = combine([finished, spent])

    assert a["complete"] and a["metrics"]["valid"] == 8


def _fail_cases(report: dict, model: str, parts: set[int], error: str) -> dict:
    """Turn some of a model's cases into failures, the way run_generation records them."""
    for run in report["runs"]:
        if run["model"] == model:
            for case in run["suites"]["generation"]["cases"]:
                if case["part"] in parts:
                    for key in ("attempts", "seconds", "self_consistency"):
                        case.pop(key, None)
                    case.update({"ok": False, "error": error})
    return report


def test_a_rerun_that_ran_out_of_quota_half_way_keeps_the_seeds_it_lost() -> None:
    """run_generation catches errors per case, so spent quota arrives as failed cases."""
    from scripts.combine_generation_reports import combine

    first = {**_group_report(1, {"a": [True] * 8}), "source": "first.json"}
    rerun = _fail_cases(
        {**_group_report(1, {"a": [False] * 8}), "source": "rerun.json"},
        "a", {6, 7, 8}, "QuotaExhaustedError: The free allowance for 'a' is used up",
    )
    (a,) = combine([first, rerun])

    by_part = {c["part"]: c for c in a["cases"]}
    assert all(not by_part[p]["ok"] and by_part[p]["source"] == "rerun.json" for p in range(1, 6))
    assert all(by_part[p]["ok"] and by_part[p]["source"] == "first.json" for p in (6, 7, 8))
    assert a["metrics"]["generated"] == 8 and a["metrics"]["unavailable"] == 0
    assert a["groups"][1]["sources"] == ["first.json", "rerun.json"]


def test_a_genuine_failure_does_replace_an_older_success() -> None:
    """Newest wins for real verdicts: an invalid draft is a measurement."""
    from scripts.combine_generation_reports import combine

    first = _group_report(1, {"a": [True] * 8})
    rerun = _fail_cases(
        _group_report(1, {"a": [True] * 8}), "a", {4},
        "GenerationFailed: Could not generate a valid Part 4 exercise.",
    )
    (a,) = combine([first, rerun])
    assert a["metrics"]["valid"] == 7


def test_unavailability_with_nothing_older_still_counts_and_is_reported() -> None:
    from scripts.combine_generation_reports import combine

    report = _fail_cases(
        _group_report(1, {"a": [True] * 8}), "a", {2, 3},
        "ModelUnreachableError: The model is busy",
    )
    (a,) = combine([report])
    assert a["metrics"]["valid"] == 6 and a["metrics"]["unavailable"] == 2
    assert a["groups"][1]["unavailable"] == 2


def test_every_availability_error_of_the_router_is_recognised() -> None:
    """The names come from cae.llm, so a new availability error cannot be missed."""
    from cae import llm
    from scripts.combine_generation_reports import UNAVAILABLE_ERRORS

    for cls in (llm.QuotaExhaustedError, llm.ModelUnreachableError, llm.MissingAPIKeyError):
        assert cls.__name__ in UNAVAILABLE_ERRORS


def test_reports_are_combined_oldest_first_whatever_order_they_are_given_in(tmp_path) -> None:
    import json

    from scripts.combine_generation_reports import combine, load_reports

    newer, older = tmp_path / "newer.json", tmp_path / "older.json"
    newer.write_text(
        json.dumps({**_group_report(1, {"a": [True] * 8}), "generated_at": "2026-10-12T09:00:00+00:00"}),
        encoding="utf-8",
    )
    older.write_text(
        json.dumps({**_group_report(1, {"a": [False] * 8}), "generated_at": "2026-10-11T09:00:00+00:00"}),
        encoding="utf-8",
    )

    reports = load_reports([newer, older])
    assert [r["source"] for r in reports] == ["older.json", "newer.json"]
    assert combine(reports)[0]["metrics"]["valid"] == 8


def test_an_ungrouped_report_is_refused(tmp_path) -> None:
    import json

    from scripts.combine_generation_reports import ReportError, load_reports

    old = tmp_path / "old.json"
    old.write_text(json.dumps({"group": None, "runs": []}), encoding="utf-8")
    with pytest.raises(ReportError, match="no seed group"):
        load_reports([old])


def test_every_grouped_report_is_found_and_nothing_else(tmp_path) -> None:
    import json

    from scripts.combine_generation_reports import grouped_reports

    for name, group in [
        ("20261012T090000Z-generation-g1.json", 1),
        ("20261011T090000Z-generation-g1.json", 1),
        ("20261012T100000Z-generation-g2.json", 2),
        ("20261013T090000Z-generation-combined.json", None),
        ("20261010T090000Z-marking.json", None),
    ]:
        (tmp_path / name).write_text(json.dumps({"group": group}), encoding="utf-8")

    assert [p.name for p in grouped_reports(tmp_path)] == [
        "20261011T090000Z-generation-g1.json",
        "20261012T090000Z-generation-g1.json",
        "20261012T100000Z-generation-g2.json",
    ]


def test_an_unknown_seed_group_is_refused() -> None:
    from evals import suites

    with pytest.raises(ValueError, match="No seed group 99"):
        suites.run_generation(group=99)


def test_every_writing_task_seed_group_pins_one_topic_for_every_genre() -> None:
    """Same reasoning as the generation seeds: a group short of a genre would make
    one day's run cover less than another's."""
    from cae.domain.parts import WRITING_PART1_GENRES, WRITING_PART2_GENRES
    from cae.prompts.writing_task import TASK_TOPICS
    from evals.suites import writing_task_groups

    seeds = load_dataset("writing_task_seeds")["seeds"]
    genres = sorted((*WRITING_PART1_GENRES, *WRITING_PART2_GENRES))
    assert len(writing_task_groups()) >= 2
    for group in writing_task_groups():
        assert sorted(s["genre"] for s in seeds if s["group"] == group) == genres, group
    for seed in seeds:
        assert seed["part"] == (1 if seed["genre"] in WRITING_PART1_GENRES else 2), seed
    topics = [s["topic"] for s in seeds]
    assert all(topics) and len(topics) == len(set(topics))
    assert not set(topics) & set(TASK_TOPICS)


def test_the_writing_task_suite_counts_the_repairs_each_task_needed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cae.graphs import writing_task as task_module
    from evals import suites

    drafts = [
        factories.writing_task_spec(1, "essay"),
        factories.writing_task_spec(2, "letter_email"),
        factories.writing_task_spec(2, "proposal", notes=["one"]),
        factories.writing_task_spec(2, "proposal"),
        factories.writing_task_spec(2, "report", notes=["one"]),
        factories.writing_task_spec(2, "report", title=""),
        factories.writing_task_spec(2, "report"),
        *[factories.writing_task_spec(2, "review", notes=["one"])] * 3,
    ]

    def fake_call(schema, system, user, **kwargs):
        assert kwargs["tier"] == "writing_task"
        return drafts.pop(0)

    monkeypatch.setattr(task_module, "structured_call", fake_call)
    monkeypatch.setattr(task_module.get_settings(), "cae_max_generation_attempts", 3)

    result = suites.run_writing_task(group=1)
    assert drafts == [], "every draft was asked for"
    cases = {c["genre"]: c for c in result["cases"]}

    assert [c["id"] for c in result["cases"]] == [
        f"{s['genre']}-g1" for s in load_dataset("writing_task_seeds")["seeds"] if s["group"] == 1
    ]
    assert cases["essay"]["repairs"] == 0 and cases["essay"]["issue_history"] == [[]]
    assert cases["proposal"]["repairs"] == 1 and cases["proposal"]["ok"]
    assert cases["report"]["repairs"] == 2 and cases["report"]["ok"]
    review = cases["review"]
    assert not review["ok"] and review["failed"] and review["repairs"] == 2
    assert len(review["issue_history"]) == 3 and all(review["issue_history"])

    m = result["metrics"]
    assert m["generated"] == 5 and m["valid"] == 4 and m["rate"] == 0.8
    assert m["first_try"] == 2 and m["first_try_rate"] == 0.4
    assert m["total_repairs"] == 5 and m["mean_repairs"] == 1.0
    assert m["repairs_histogram"] == {"0": 2, "1": 1, "2": 1, "failed": 1}
    assert m["by_genre"]["review"] == {"valid": 0, "tried": 1, "mean_repairs": 2.0}
    (rule, times), *_ = m["repair_reasons"]
    assert "content points" in rule and times == 5


def test_a_writing_task_no_model_answered_is_left_out_of_the_rates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cae.graphs import writing_task as task_module
    from cae.llm import QuotaExhaustedError
    from evals import suites

    def spent(*args, **kwargs):
        raise QuotaExhaustedError("The free allowance is used up")

    monkeypatch.setattr(task_module, "structured_call", spent)
    result = suites.run_writing_task(group=2)

    assert all(c["unavailable"] and "repairs" not in c for c in result["cases"])
    m = result["metrics"]
    assert m["unavailable"] == 5 and m["generated"] == 0 and m["mean_repairs"] is None


def test_an_unknown_writing_task_seed_group_is_refused() -> None:
    from evals import suites

    with pytest.raises(ValueError, match="No seed group 99"):
        suites.run_writing_task(group=99)


def _writing_examiner(
    calls: list[str],
    bands: dict[str, dict[str, int]] | None = None,
    fail: str = "",
    annotations: dict[str, list[tuple[str, str]]] | None = None,
):
    """A stub ``structured_call`` for the writing graph.

    It awards each script its expected bands unless ``bands`` overrides some by
    script id, annotates it with ``annotations`` (``(span, category)`` pairs by
    script id; none otherwise), records every call's label in ``calls``, and
    raises on ``fail``.
    """
    from cae.schemas.writing_evaluator import (
        Annotation,
        AnnotationSet,
        ImprovedVersion,
        SubscaleAssessment,
    )

    cases = _writing_cases()
    overrides = bands or {}
    notes = annotations or {}

    def fake_call(schema, system, user, **kwargs):
        assert kwargs["tier"] == "writing_evaluator"
        label = kwargs["label"]
        calls.append(label)
        if label == fail:
            raise RuntimeError("the free allowance is used up")
        case = next((c for c in cases if c["script"].strip() in user), None)
        if schema is SubscaleAssessment:
            subscale = label.rsplit(".", 1)[1]
            band = {**case["expected_bands"], **overrides.get(case["id"], {})}[subscale]
            return SubscaleAssessment(band=band, justification="j", evidence=[], to_improve="t")
        if schema is AnnotationSet:
            return AnnotationSet(
                annotations=[
                    Annotation(
                        span=span, category=category, correction="c", explanation="e",
                        severity="error",
                    )
                    for span, category in notes.get(case["id"], [])
                ],
                strengths=[],
            )
        return ImprovedVersion(text="Improved.", upgrades=[])

    return fake_call


def test_a_group_is_one_script_per_genre_scored_per_genre(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cae.graphs import writing_evaluator
    from evals import suites

    calls: list[str] = []
    examiner = _writing_examiner(calls, {"museums-strong": {"content": 4}})
    monkeypatch.setattr(writing_evaluator, "structured_call", examiner)

    result = suites.run_writing_evaluator(group=1)

    assert len(calls) == 6 * 5
    assert result["group"] == 1
    assert sorted(c["genre"] for c in result["cases"]) == sorted(ALL_GENRES)
    tasks = load_dataset("writing_scripts")["tasks"]
    by_id = {c["id"]: c for c in _writing_cases(1)}
    assert all(c["task"]["spec"] == tasks[by_id[c["id"]]["task"]]["spec"] for c in result["cases"])

    m = result["metrics"]
    assert list(m["by_genre"]) == list(ALL_GENRES)
    essay = m["by_genre"]["essay"]
    assert essay["scripts"] == 1
    assert essay["accuracy_content"] == 0.8 and essay["accuracy_language"] == 1.0
    assert essay["accuracy_total"] == 0.95
    assert all(m["by_genre"][g]["accuracy_total"] == 1.0 for g in ALL_GENRES if g != "essay")
    assert m["accuracy_content"] == 0.96 and m["accuracy_total"] == 0.99
    assert m["scripts"] == 5 and m["degraded_scripts"] == 0


def test_the_annotate_node_is_scored_per_genre_against_the_labelled_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cae.graphs import writing_evaluator
    from evals import suites

    calls: list[str] = []
    annotations = {
        "sports-centre-report": [("This report summarises the responses", "VERB_FORMS")],
        "restaurant-review-language": [
            ("my parents was", "GRAMMAR_WORDS"),
            ("bringed", "WORD_CHOICE"),
            ("fishermans", "WORD_FORM"),
            ("the best part of the night", "WORD_CHOICE"),
        ],
    }
    examiner = _writing_examiner(calls, annotations=annotations)
    monkeypatch.setattr(writing_evaluator, "structured_call", examiner)

    result = suites.run_writing_evaluator(group=3)

    by_genre = result["metrics"]["by_genre"]
    assert (by_genre["review"]["recall"], by_genre["review"]["precision"]) == (0.25, 0.75)
    assert by_genre["review"]["accuracy_category"] == 0.5
    assert by_genre["report"]["recall"] is None and by_genre["report"]["precision"] == 0.0
    assert by_genre["essay"]["recall"] == 0.0 and by_genre["essay"]["precision"] is None
    m = result["metrics"]
    assert (m["recall"], m["precision"], m["accuracy_category"]) == (0.125, 0.6, 0.5)
    assert m["expected_errors"] == 16 and m["annotations"] == 5


def test_a_marker_that_agrees_with_every_label_scores_perfectly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The labels must be reachable: if agreeing with them exactly scores below
    perfect anywhere, the dataset or a metric is wrong, not the model."""
    from cae.graphs import writing_evaluator
    from evals import suites

    calls: list[str] = []
    labels = {
        c["id"]: [(e["span"], e["category"]) for e in c["expected_annotations"]]
        for c in _writing_cases()
    }
    examiner = _writing_examiner(calls, annotations=labels)
    monkeypatch.setattr(writing_evaluator, "structured_call", examiner)

    m = suites.run_writing_evaluator()["metrics"]

    perfect = {
        **{f"accuracy_{s}": 1.0 for s in SUBSCALES},
        "accuracy_total": 1.0, "recall": 1.0, "precision": 1.0, "accuracy_category": 1.0,
    }
    assert m["scripts"] == len(_writing_cases()) and m["degraded_scripts"] == 0
    for genre, scores in {"all": m, **m["by_genre"]}.items():
        assert {k: scores[k] for k in perfect} == perfect, genre
    assert all(
        rates == {"recall": 1.0, "precision": 1.0}
        for rates in m["annotation_by_category"].values()
    )


def test_a_lazy_marker_that_gives_everything_a_three_scores_badly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure the suite exists to catch must not pass any of its checks."""
    from cae.graphs import writing_evaluator
    from evals import suites

    calls: list[str] = []
    flat = {c["id"]: {s: 3 for s in SUBSCALES} for c in _writing_cases()}
    monkeypatch.setattr(
        writing_evaluator, "structured_call", _writing_examiner(calls, bands=flat)
    )

    m = suites.run_writing_evaluator()["metrics"]

    assert m["accuracy_total"] < 0.85
    assert all(scores["accuracy_total"] < 0.9 for scores in m["by_genre"].values())
    assert m["recall"] == 0.0 and m["precision"] is None


def test_a_failed_annotate_node_is_left_out_of_the_annotation_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No annotation because the call failed is not a clean script."""
    from cae.graphs import writing_evaluator
    from evals import suites

    calls: list[str] = []
    examiner = _writing_examiner(calls, fail="writing_evaluator.annotate")
    monkeypatch.setattr(writing_evaluator, "structured_call", examiner)

    result = suites.run_writing_evaluator(group=2)

    assert all(c["annotate_degraded"] for c in result["cases"])
    assert all("annotation_scores" not in c for c in result["cases"])
    m = result["metrics"]
    assert m["degraded_annotations"] == 5 and m["recall"] is None
    assert m["accuracy_total"] == 1.0


def test_a_script_with_a_degraded_subscale_is_left_out_of_the_band_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A band 0 nobody awarded would read as a severe examiner."""
    from cae.graphs import writing_evaluator
    from evals import suites

    calls: list[str] = []
    examiner = _writing_examiner(calls, fail="writing_evaluator.assess.language")
    monkeypatch.setattr(writing_evaluator, "structured_call", examiner)

    result = suites.run_writing_evaluator(group=2)

    assert all(c["degraded_subscales"] == ["language"] for c in result["cases"])
    assert all("band_accuracy" not in c for c in result["cases"])
    m = result["metrics"]
    assert m["scripts"] == 0 and m["degraded_scripts"] == 5
    assert m["accuracy_total"] is None and m["by_genre"]["essay"]["accuracy_total"] is None
    assert m["recall"] == 0.0


def test_only_the_writing_evaluator_by_genre_prints_as_its_table(capsys) -> None:
    """writing_task has a by_genre too, with other figures: forced into the
    evaluator's columns it would print a table of dashes."""
    from scripts.run_evals import print_suite

    task_figures = {"review": {"valid": 0, "tried": 1, "mean_repairs": 2.0}}
    print_suite({"suite": "writing_task", "metrics": {"by_genre": task_figures}})
    printed = capsys.readouterr().out
    assert "mean_repairs" in printed and "precision" not in printed

    scores = {"review": {"accuracy_total": 0.9, "recall": 0.5}}
    print_suite({"suite": "writing_evaluator", "metrics": {"by_genre": scores}})
    row = next(line for line in capsys.readouterr().out.splitlines() if "review" in line)
    assert "0.9" in row and "0.5" in row


def test_one_genre_can_be_run_alone() -> None:
    from evals import suites

    with pytest.MonkeyPatch.context() as patch:
        from cae.graphs import writing_evaluator

        patch.setattr(writing_evaluator, "structured_call", _writing_examiner([]))
        result = suites.run_writing_evaluator(genres=["review"])

    assert [c["genre"] for c in result["cases"]] == ["review"] * 3
    assert list(result["metrics"]["by_genre"]) == ["review"]


def test_an_unknown_writing_group_is_refused() -> None:
    from evals import suites

    with pytest.raises(ValueError, match="No script group 9"):
        suites.run_writing_evaluator(group=9)


def test_a_validator_message_reduces_to_its_rule() -> None:
    """Two drafts failing the same rule quote different text; they must count as one."""
    a = metrics.issue_rule(
        "The content point 'cost of living in the city centre for students' is 13 words. "
        "Content points are short phrases of at most 12 words, not sentences."
    )
    b = metrics.issue_rule(
        "The content point 'it's what we've always done here' is 15 words. "
        "Content points are short phrases of at most 12 words, not sentences."
    )
    assert a == b
    assert "'" + "…" + "'" in a and "13" not in a


def test_repair_reasons_count_every_pass_that_found_something() -> None:
    histories = [
        [["The task has no title."], []],
        [["The task has no title.", "The reader is empty."], ["The task has no title."]],
        [[]],
    ]
    assert metrics.repair_reasons(histories) == [
        ["The task has no title.", 3],
        ["The reader is empty.", 1],
    ]


def test_the_repairs_histogram_keeps_failures_apart() -> None:
    rows = [{"ok": True, "repairs": 0}, {"ok": True, "repairs": 2}, {"ok": False, "repairs": 2}]
    assert metrics.repairs_histogram(rows) == {"0": 1, "2": 1, "failed": 1}
    assert metrics.repairs_histogram([{"ok": True, "repairs": 1}]) == {"1": 1}


CHAT = load_dataset("chat_learners")
CHAT_LEARNERS = CHAT["learners"]


def _chat_turns() -> list[tuple[dict, dict]]:
    from evals.suites import chat_turns

    return [(learner, turn) for learner in CHAT_LEARNERS for turn in chat_turns(learner)]


def test_a_reply_that_says_the_figures_and_names_the_part_is_correct() -> None:
    facts = {"parts": {7: {"name": "Gapped text"}}}
    expect = {"targets": [{"part": 7}], "figures": [58.3]}
    assert metrics.answer_correct("Part 7 is your weakest, at 58.3%.", expect, facts)["correct"]
    assert metrics.answer_correct("Gapped text, at about 58%.", expect, facts)["correct"]
    wrong = metrics.answer_correct("Part 7, at 62.5%.", expect, facts)
    assert not wrong["correct"] and wrong["missing"] == ["figure 58.3"]


def test_an_evasive_reply_is_grounded_but_not_correct() -> None:
    """Why the chat suite needs more than groundedness: saying nothing invents nothing."""
    evasive = "I don't have that figure to hand, but keep practising!"
    expect = {"figures": [5]}

    assert metrics.answer_correct(evasive, expect)["correct"] is False


def test_a_figure_can_be_said_in_minutes_or_seconds_or_in_words() -> None:
    expect = {"figures": [[16, 960]]}
    for reply in ("About 16 minutes.", "960 seconds on average.", "Sixteen minutes."):
        assert metrics.answer_correct(reply, expect)["correct"], reply
    assert not metrics.answer_correct("About 20 minutes.", expect)["correct"]
    assert metrics.stated_numbers("1,020 seconds, three items") == [1020.0, 3.0]


def test_spans_are_matched_whatever_the_quotes_dashes_and_case() -> None:
    expect = {"spans": ["Neither the brothers live"], "any_spans": ["decision-maker", "boss"]}
    reply = "You wrote “neither the brothers  live” – and wrote to the decision‑maker."
    assert metrics.answer_correct(reply, expect)["correct"]
    assert not metrics.answer_correct("You wrote something else.", expect)["correct"]


def test_saying_what_the_question_excludes_makes_a_reply_wrong() -> None:
    """Asked about the latest Part 4, a reply that adds an earlier one's slips is
    wrong however complete it is."""
    expect = {"figures": [3, 6], "spans": ["warned us from touching"], "absent": ["wish I applied"]}
    right = "Items 3 and 6: you wrote 'warned us from touching'."
    padded = right + " Item 3 of an earlier attempt: 'wish I applied'."

    assert metrics.answer_correct(right, expect)["correct"]
    wrong = metrics.answer_correct(padded, expect)
    assert not wrong["correct"] and wrong["missing"] == ['not "wish I applied"']
    assert wrong["checks"] == 4 and wrong["passed"] == 3


def test_a_question_the_record_has_nothing_on_needs_the_reply_to_say_so() -> None:
    expect = {"none": True}
    for honest in (
        "You haven't sat a mock exam yet.",
        "There are no mock exams in your record.",
        "You have not tried Part 6 yet.",
        "You don't have any essays on record.",
    ):
        assert metrics.answer_correct(honest, expect)["correct"], honest
    assert not metrics.answer_correct("Your mock went well: 61 out of 78.", expect)["correct"]


def test_tool_choice_counts_missing_and_extra_calls() -> None:
    assert metrics.tool_choice([], [])["ok"]
    assert metrics.tool_choice(["query_record"], [], ["query_record"])["ok"]
    extra = metrics.tool_choice(["query_record", "reference_lookup"], ["query_record"])
    assert extra["extra"] == ["reference_lookup"] and not extra["ok"]
    missing = metrics.tool_choice([], ["reference_lookup"])
    assert missing["missing"] == ["reference_lookup"] and not missing["ok"]


def test_a_section_hit_is_matched_on_the_heading_not_on_the_text() -> None:
    """The overview mentions every part, so matching text would make every lookup a hit."""
    overview = {
        "section": "C1 Advanced > Reading and Use of English > Overview of the paper",
        "citation": "c1.md — Reading and Use of English > Overview of the paper",
        "text": "Part 6 and Part 7 award two marks per question.",
    }
    part6 = {"section": "C1 Advanced > Reading and Use of English > Part 6: Cross‑text multiple matching"}
    wanted = ["Part 6: Cross-text multiple matching"]

    assert metrics.section_hit([overview], wanted)["hit"] is False
    assert metrics.section_hit([overview, part6], wanted)["hit"] is True
    assert metrics.section_hit([], wanted) == {"retrieved": 0, "hit": False, "citations": []}


def test_every_chat_group_asks_one_question_of_each_kind() -> None:
    from evals.suites import CHAT_KINDS

    assert sorted(learner["group"] for learner in CHAT_LEARNERS) == [1, 2, 3]
    for learner in CHAT_LEARNERS:
        kinds = Counter(q["kind"] for q in learner["questions"])
        assert set(kinds) == set(CHAT_KINDS), learner["id"]
        assert kinds["exam"] == 3 and all(
            n == 1 for kind, n in kinds.items() if kind != "exam"
        ), learner["id"]
        (dialogue,) = [q for q in learner["questions"] if q["kind"] == "dialogue"]
        assert len(dialogue["turns"]) == 3


def test_every_chat_turn_has_a_unique_id_a_reference_and_known_tools() -> None:
    turns = _chat_turns()
    ids = [turn["id"] for _, turn in turns]
    assert len(ids) == len(set(ids))
    for learner, turn in turns:
        assert turn["id"].startswith(f"g{learner['group']}-"), turn["id"]
        assert turn["question"] and turn["reference"] and turn["expect"], turn["id"]
        tools = {*turn.get("tools", []), *turn.get("allowed_tools", [])}
        assert tools <= {"query_record", "reference_lookup"}, turn["id"]


def test_every_reference_answer_passes_its_own_checks() -> None:
    """The reference is what a correct reply says; if it fails, the label is wrong."""
    from cae.db.session import session_scope
    from cae.graphs import coach
    from evals.suites import chat_turns, coach_facts, isolated_database, seed_chat_learner

    for learner in CHAT_LEARNERS:
        with isolated_database():
            with session_scope() as session:
                user_id = seed_chat_learner(session, learner)
                state = coach.stats_from(session, user_id)
        facts = coach_facts(state["snapshot"], state["examples"], {})
        for turn in chat_turns(learner):
            scored = metrics.answer_correct(turn["reference"], turn["expect"], facts)
            assert scored["correct"], (turn["id"], scored["missing"])


def _gold_rows(conn, sql) -> tuple[list[list], str]:
    from cae.db import sandbox

    rows: list[list] = []
    for query in [sql] if isinstance(sql, str) else sql:
        result = sandbox.run_select(conn, query)
        assert isinstance(result, dict), f"{query}: {result}"
        rows.extend(result["rows"])
    return rows, "\n".join(str(row) for row in rows)


def test_every_chat_label_agrees_with_the_learners_record() -> None:
    """Change a learner's work and the expected figures must still be what the
    record says: the gold SQL runs against the very sandbox the model queries,
    and the preloaded kinds against the figures the system message states."""
    from cae.db.session import session_scope
    from cae.graphs import coach
    from cae.graphs.coach_chat import learner_record
    from evals.suites import PRELOADED_KINDS, chat_turns, coach_facts, isolated_database, seed_chat_learner

    for learner in CHAT_LEARNERS:
        with isolated_database():
            with session_scope() as session:
                user_id = seed_chat_learner(session, learner)
                state = coach.stats_from(session, user_id)
            state.update(coach.detect_trend(state))
            figures = coach.prompt_figures(state["snapshot"])
            facts = coach_facts(state["snapshot"], state["examples"], figures)
            conn = learner_record(user_id)
            for turn in chat_turns(learner):
                expect = turn["expect"]
                if turn["kind"] == "exam":
                    continue
                if turn["kind"] in PRELOADED_KINDS:
                    stated = metrics.stated_numbers(
                        " ".join(str(v) for v in figures.values()) + "\n" + state["tables"]["parts"]
                    )
                    assert all(metrics._stated(f, stated) for f in expect["figures"]), turn["id"]
                    for target in expect.get("targets", []):
                        assert target["part"] == facts["weakest"], turn["id"]
                    continue
                if turn["kind"] == "empty" and not turn.get("tools"):
                    assert "not attempted yet" in state["tables"]["parts"]
                rows, text = _gold_rows(conn, turn["gold_sql"])
                if expect.get("none"):
                    assert rows == [], turn["id"]
                    continue
                numbers = metrics.stated_numbers(text)
                for figure in expect.get("figures", []):
                    assert metrics._stated(figure, numbers), (turn["id"], figure, text)
                for span in [*expect.get("spans", []), *expect.get("any_spans", [])]:
                    assert span.lower() in text.lower(), (turn["id"], span)
                for span in expect.get("absent", []):
                    assert span.lower() not in text.lower(), (turn["id"], span)
                    everything = _gold_rows(conn, "SELECT user_answer FROM item_results")[1]
                    assert span.lower() in everything.lower(), (turn["id"], span)
                for target in expect.get("targets", []):
                    if "category" in target:
                        assert rows[0][0] == target["category"], turn["id"]
                    if "subscale" in target:
                        bands = dict(zip(
                            ("content", "communicative_achievement", "organisation", "language"),
                            rows[0][1:],
                        ))
                        lowest = min(bands.values())
                        assert [s for s, b in bands.items() if b == lowest] == [target["subscale"]]


def test_every_exam_fact_is_in_its_section_of_the_corpus() -> None:
    from pathlib import Path

    from cae.rag.ingest import _sections

    corpus = Path(__file__).resolve().parents[1] / "data" / "reference" / "c1_advanced_structure.md"
    sections = _sections(corpus.read_text(encoding="utf-8"))
    for _, turn in _chat_turns():
        if turn["kind"] != "exam":
            continue
        expect = turn["expect"]
        bodies = [
            f"{path}\n{body}"
            for path, body in sections
            if any(s.lower() in path.lower() for s in expect["sections"])
        ]
        assert bodies, (turn["id"], expect["sections"])
        text = "\n".join(bodies)
        numbers = metrics.stated_numbers(text)
        assert all(metrics._stated(f, numbers) for f in expect["figures"]), turn["id"]
        for target in expect.get("targets", []):
            assert metrics.mentions(text, target, {}), (turn["id"], target)


def test_the_frozen_reading_exercises_are_valid() -> None:
    from evals.suites import chat_exercises

    frozen = CHAT["exercises"]
    assert {e["part"] for e in frozen.values()} == {5, 6, 7, 8}
    for exercise_id, (part, payload) in chat_exercises().items():
        assert validate_exercise(part, parse_exercise(part, payload)) == [], exercise_id


def test_a_slip_the_marker_accepts_is_refused_when_seeding() -> None:
    """A labelling mistake must fail loudly, not shift every figure by a mark."""
    from cae.db.session import session_scope
    from evals.suites import isolated_database, seed_chat_learner

    learner = {
        "id": "bad",
        "attempts": [{"exercise": "part1-urban-beekeeping", "days_ago": 1,
                      "wrong": {"1": {"answer": "B", "category": "WORD_PATTERNS"}}}],
    }
    with isolated_database(), pytest.raises(ValueError, match="is right for item 1"):
        with session_scope() as session:
            seed_chat_learner(session, learner)


def test_the_learner_is_recorded_as_the_app_records_one() -> None:
    """Marks from the grader, events dated with their attempt, Writing with its bands."""
    from cae.db.session import session_scope
    from cae.db import repo
    from evals.suites import isolated_database, seed_chat_learner

    learner = next(l for l in CHAT_LEARNERS if l["group"] == 2)
    with isolated_database():
        with session_scope() as session:
            user_id = seed_chat_learner(session, learner)
            export = repo.learner_export(session, user_id)
    assert len(export["attempts"]) == 12 and len(export["mock_exams"]) == 1
    assert {a["mode"] for a in export["attempts"]} == {"practice", "mock"}
    assert export["mock_exams"][0]["status"] == "finished"
    assert len(export["writing_scores"]) == 4
    reading = [e for e in export["error_events"] if e["source_type"] == "reading"]
    assert all('wrote "' in e["evidence"] for e in reading)


def _corpus_search(query: str, *args, **kwargs) -> list[dict]:
    """The real corpus, retrieved by heading: what a perfect lookup returns."""
    from pathlib import Path

    from cae.rag.ingest import split_document

    corpus = Path(__file__).resolve().parents[1] / "data" / "reference" / "c1_advanced_structure.md"
    chunks = split_document(corpus.read_text(encoding="utf-8"), corpus.name)
    return [
        {"text": c.text, "source": c.source, "section": c.section, "citation": c.citation}
        for c in chunks
        if query.lower() in c.section.lower()
    ]


def _stub_faithful_chat(monkeypatch, groups=(1, 2, 3), kinds=None) -> list:
    """A model that calls exactly the tools each turn needs, with the gold query,
    and then answers with the reference: what a perfect chat does. ``groups`` and
    ``kinds`` must select the turns the suite is about to ask, in its order."""
    from langchain_core.messages import AIMessage, HumanMessage

    from cae.graphs import coach_chat
    from cae.rag import store

    turns = iter(
        turn
        for learner, turn in _chat_turns()
        if learner["group"] in groups and (kinds is None or turn["kind"] in kinds)
    )
    seen: list = []
    current: dict = {}

    def fake_chat(messages, **kwargs):
        seen.append(messages)
        if isinstance(messages[-1], HumanMessage):
            current.clear()
            current.update(next(turns))
            calls = []
            if "query_record" in current.get("tools", []):
                calls.append({"name": "query_record", "args": {"sql": current["gold_sql"]}, "id": "q"})
            if "reference_lookup" in current.get("tools", []):
                query = current["expect"]["sections"][0]
                calls.append({"name": "reference_lookup", "args": {"query": query}, "id": "r"})
            if calls:
                return AIMessage(content="", tool_calls=calls)
        return AIMessage(content=current["reference"])

    monkeypatch.setattr(coach_chat, "chat_call", fake_chat)
    monkeypatch.setattr(store, "search", _corpus_search)
    monkeypatch.setattr(store, "index_exists", lambda *a, **k: True)
    monkeypatch.setattr("evals.judge.build_judge", lambda *a, **k: _FakeJudge())
    return seen


def test_a_faithful_chat_scores_perfectly_on_every_turn(monkeypatch) -> None:
    """When it fails, suspect a heuristic or a label, not the model."""
    from evals.suites import run_chat

    _stub_faithful_chat(monkeypatch)
    result = run_chat()

    assert len(result["cases"]) == len(_chat_turns()) == 42
    assert [c["id"] for c in result["cases"] if not c.get("ok")] == []
    m = result["metrics"]
    assert m["tool_choice_rate"] == m["section_hit_rate"] == 1.0
    assert m["missing_tools"] == m["extra_tools"] == m["sql_errors"] == m["budget_hits"] == 0
    assert m["judged"] == 42 and m["judge_correctness_mean"] == 2
    assert result["index"] == "built" and set(result["learners"]) == {"chat-g1", "chat-g2", "chat-g3"}


def test_one_chat_group_asks_its_learners_fourteen_turns(monkeypatch) -> None:
    from evals.suites import CHAT_KINDS, run_chat

    _stub_faithful_chat(monkeypatch, groups=(2,))
    result = run_chat(group=2)

    assert len(result["cases"]) == 14 and result["group"] == 2
    assert {c["learner"] for c in result["cases"]} == {"chat-g2"}
    assert list(result["metrics"]["by_kind"]) == list(CHAT_KINDS)


def test_an_unknown_chat_group_is_refused() -> None:
    from evals.suites import run_chat

    with pytest.raises(ValueError, match="No learner group 9"):
        run_chat(group=9)


def test_a_dialogue_hands_the_model_its_own_earlier_replies(monkeypatch) -> None:
    from langchain_core.messages import AIMessage, HumanMessage

    from evals.suites import chat_turns, run_chat

    learner = next(l for l in CHAT_LEARNERS if l["group"] == 1)
    dialogue = [t for t in chat_turns(learner) if t["kind"] == "dialogue"]
    seen = _stub_faithful_chat(monkeypatch, groups=(1,), kinds=["dialogue"])

    result = run_chat(group=1, kinds=["dialogue"])

    assert [c["turn"] for c in result["cases"]] == [1, 2, 3]
    last_first_call = [m for m in seen if isinstance(m[-1], HumanMessage)][-1]
    history = [m.content for m in last_first_call if isinstance(m, (HumanMessage, AIMessage))]
    assert history == [
        dialogue[0]["question"], dialogue[0]["reference"],
        dialogue[1]["question"], dialogue[1]["reference"],
        dialogue[2]["question"],
    ]


def test_a_turn_no_model_answered_is_counted_not_scored(monkeypatch) -> None:
    from cae.graphs import coach_chat
    from cae.llm import QuotaExhaustedError
    from evals.suites import run_chat

    def spent(*args, **kwargs):
        raise QuotaExhaustedError("mistral:ministral-14b-latest", "chat")

    monkeypatch.setattr(coach_chat, "chat_call", spent)
    monkeypatch.setattr(coach_chat.reference_store, "index_exists", lambda *a, **k: False)
    result = run_chat(group=3, kinds=["snapshot", "exam"], judge=False)

    assert all(c["ok"] is False and c["unavailable"] for c in result["cases"])
    assert result["metrics"]["unavailable"] == 4 and result["metrics"]["turns"] == 0
    assert result["metrics"]["tool_choice_rate"] is None and "judged" not in result["metrics"]


def test_without_the_index_the_lookup_finds_nothing_and_the_report_says_so(monkeypatch) -> None:
    from cae.rag import store
    from evals.suites import run_chat

    _stub_faithful_chat(monkeypatch, groups=(1,), kinds=["exam"])
    result = run_chat(group=1, kinds=["exam"], without_index=True)

    assert result["index"] == "disabled"
    assert all(c["section_hit"]["hit"] is False for c in result["cases"])
    assert all("Nothing in the reference material" in c["tool_outputs"][0] for c in result["cases"])
    assert store.search is _corpus_search


def test_the_chat_suite_never_writes_to_the_apps_database(monkeypatch) -> None:
    """The learners live in a scratch file: nothing reaches cae.db, and the app
    points back at its own database afterwards."""
    from sqlalchemy import func, select

    from cae.config import get_settings
    from cae.db.models import Attempt, User, WritingSubmission
    from cae.db.session import session_scope
    from evals.suites import run_chat

    def counts() -> tuple:
        with session_scope() as session:
            return tuple(
                session.scalar(select(func.count()).select_from(t))
                for t in (User, Attempt, WritingSubmission)
            )

    url = get_settings().database_url
    before = counts()
    _stub_faithful_chat(monkeypatch, groups=(3,))
    run_chat(group=3)
    assert counts() == before and get_settings().database_url == url


class _FakeJudge:
    def __init__(self, correctness: int = 2, fail_on: str = "") -> None:
        self.correctness, self.fail_on, self.calls = correctness, fail_on, []

    def invoke(self, messages, config=None):
        from evals.judge import ChatVerdict

        self.calls.append(messages)
        if self.fail_on and self.fail_on in messages[1][1]:
            raise RuntimeError("judge down")
        return ChatVerdict(
            correctness=self.correctness, faithfulness=2, helpfulness=4,
            unsupported_claims=[], rationale="Fine.",
        )


def test_the_judges_verdict_has_no_optional_field() -> None:
    from evals.judge import ChatVerdict

    schema = ChatVerdict.model_json_schema()
    assert set(schema["required"]) == set(schema["properties"])
    assert "null" not in str(schema)


def test_the_judge_is_no_route_of_the_app() -> None:
    """The app is free-only: an OpenAI model in a tier is refused, by design."""
    from cae.config import RouteConfigError, parse_route
    from evals.judge import JudgeSettings

    with pytest.raises(RouteConfigError):
        parse_route(JudgeSettings.model_fields["cae_judge_model"].default)


def test_judging_a_run_adds_a_verdict_and_agreement_to_every_answered_turn(monkeypatch) -> None:
    from evals.judge import judge_cases
    from evals.suites import chat_metrics, run_chat

    _stub_faithful_chat(monkeypatch, groups=(1,))
    result = run_chat(group=1, judge=False)
    judge = _FakeJudge(fail_on="How did my mock exams go?")
    judge_cases(result["cases"], result["learners"], judge)
    m = chat_metrics(result["cases"])

    assert len(judge.calls) == 14
    unjudged = [c["id"] for c in result["cases"] if "error" in c["judge"]]
    assert unjudged == ["g1-empty"]
    assert m["judged"] == 13 and m["judge_helpfulness"] == 4
    assert m["judge_correctness_mean"] == m["judge_faithfulness_mean"] == 2
    assert m["judge_correct_rate"] == m["judge_faithful_rate"] == 1.0
    judge_cases(result["cases"], result["learners"], _FakeJudge(correctness=1))
    halved = chat_metrics(result["cases"])
    assert halved["judge_correct_rate"] == 0.0 and halved["judge_correctness_mean"] == 1


def test_the_judge_sees_everything_the_tutor_had() -> None:
    """Leaving any of it out marks a reply unsupported for citing what the tutor
    was in fact given, such as "the C1 pass mark of 180"."""
    from evals.judge import judge_evidence

    learner = {"figures": {"overall": 67.0}, "tables": {"parts": "7 | Gapped text | 2 | 58.3%"}}
    earlier = [{"question": "Weakest subscale?", "reply": "Organisation, band 1."}]
    case = {"kind": "dialogue", "tool_outputs": ["{'rows': [[3]]}"], "passages": ["4 questions"]}
    evidence = judge_evidence(case, learner, earlier)

    for expected in ("180 is the C1 pass mark", "Grade A from 200", "out of 20", "overall: 67.0",
                     "58.3%", "Organisation, band 1.", "[[3]]", "4 questions"):
        assert expected in evidence, expected
    assert "58.3%" in judge_evidence({"kind": "exam", "passages": ["x"]}, learner)


def test_the_judge_is_shown_each_dialogue_so_far(monkeypatch) -> None:
    from evals.judge import dialogue_of, judge_cases

    turns = [
        {"id": "g1-dialogue-1", "kind": "dialogue", "ok": True, "question": "Q1", "reply": "R1"},
        {"id": "g1-dialogue-2", "kind": "dialogue", "ok": True, "question": "Q2", "reply": "R2"},
        {"id": "g1-snapshot", "kind": "snapshot", "ok": True, "question": "Q0", "reply": "R0"},
    ]
    judge = _FakeJudge()
    judge_cases(turns, {}, judge)

    assert "Student: Q1\nTutor: R1" in judge.calls[1][1][1]
    assert "EARLIER IN THE CONVERSATION" not in judge.calls[0][1][1]
    assert "EARLIER IN THE CONVERSATION" not in judge.calls[2][1][1]
    assert dialogue_of(turns[1]) == "g1-dialogue" and dialogue_of(turns[2]) == ""


def test_asking_for_the_judge_without_a_key_fails_before_any_chat_call(monkeypatch) -> None:
    from cae.graphs import coach_chat
    from evals.judge import JudgeUnavailableError
    from evals.suites import run_chat

    calls: list = []
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr(coach_chat, "chat_call", lambda *a, **k: calls.append(1))
    with pytest.raises(JudgeUnavailableError):
        run_chat(group=1, judge=True)
    assert calls == []


def test_a_saved_report_is_judged_without_asking_the_chat_again(monkeypatch) -> None:
    from pathlib import Path

    from evals.suites import run_chat
    from scripts.judge_chat_report import judge_report, judged_label

    _stub_faithful_chat(monkeypatch, groups=(2,))
    result = run_chat(group=2, judge=False)
    payload = {"group": 2, "runs": [{"model": "m", "suites": {"chat": result}}]}
    judged = judge_report(payload, _FakeJudge())

    assert judged["runs"][0]["suites"]["chat"]["metrics"]["judged"] == 14
    assert judged_label(Path("20261016T090000Z-chat-g2.json")) == "chat-g2"
    assert judged_label(Path("20261016T090000Z-chat-noindex-g2.json")) == "chat-noindex-g2"


def _chat_report(group: int, runs: dict[str, list[dict]]) -> dict:
    return {
        "group": group,
        "source": f"chat-g{group}.json",
        "runs": [{"model": m, "suites": {"chat": {"cases": cases}}} for m, cases in runs.items()],
    }


def _turn(id_: str, kind: str, correct: bool, ok: bool = True) -> dict:
    if not ok:
        return {"id": id_, "kind": kind, "ok": False, "unavailable": True}
    return {
        "id": id_, "kind": kind, "ok": True, "seconds": 1.0, "tool_rounds": 0,
        "tool_choice": {"called": [], "missing": [], "extra": [], "ok": True},
        "budget_hit": False, "sql_errors": 0,
        "judge": {"correctness": 2 if correct else 0, "faithfulness": 2, "helpfulness": 4},
    }


def test_combining_chat_groups_pools_every_turn_and_keeps_answered_ones() -> None:
    from scripts.combine_chat_reports import combine

    earlier = _chat_report(1, {"a": [_turn("g1-snapshot", "snapshot", True)]})
    spent = _chat_report(1, {"a": [_turn("g1-snapshot", "snapshot", False, ok=False)]})
    second = _chat_report(2, {"a": [_turn("g2-snapshot", "snapshot", False)]})
    (entry,) = combine([earlier, spent, second])

    assert entry["complete"] and entry["metrics"]["turns"] == 2
    assert entry["metrics"]["judge_correctness_mean"] == 1
    assert entry["metrics"]["by_kind"]["snapshot"]["judge_correct_rate"] == 0.5


def test_a_run_without_the_index_is_never_pooled_with_the_measured_chat(tmp_path) -> None:
    import json

    from scripts.combine_generation_reports import grouped_reports

    for name in (
        "20261016T090000Z-chat-g1.json",
        "20261016T100000Z-chat-noindex-g1.json",
        "20261016T110000Z-chat-g1-judged.json",
        "20261016T120000Z-chat-combined.json",
    ):
        (tmp_path / name).write_text(json.dumps({"group": 1}), encoding="utf-8")

    assert [p.name for p in grouped_reports(tmp_path, suite="chat")] == [
        "20261016T090000Z-chat-g1.json",
        "20261016T110000Z-chat-g1-judged.json",
    ]


def test_the_chat_examples_are_keyed_like_the_rows_the_suite_writes() -> None:
    """An example whose id no row carries uploads as 'missing' and scores nothing."""
    from evals.upload import EXAMPLE_BUILDERS

    examples = EXAMPLE_BUILDERS["chat"]()
    assert [e["inputs"]["id"] for e in examples] == [turn["id"] for _, turn in _chat_turns()]
    by_id = {e["inputs"]["id"]: e for e in examples}
    assert by_id["g2-dialogue-3"]["split"] == "group-2"
    assert by_id["g2-dialogue-3"]["metadata"] == {"case": "g2-dialogue-3", "group": 2, "kind": "dialogue"}
    assert by_id["g1-exam-part6"]["outputs"]["tools"] == ["reference_lookup"]


def test_the_chat_evaluators_score_nothing_the_turn_does_not_have() -> None:
    from evals.upload import EVALUATORS

    unanswered = {"ok": False, "unavailable": True}
    assert all(e(unanswered, {})["score"] is None for e in EVALUATORS["chat"])
    record = _turn("g1-item", "item", True)
    scores = {e(record, {})["key"]: e(record, {})["score"] for e in EVALUATORS["chat"]}
    assert scores["judge_correctness"] == 2.0 and scores["tool_choice"] == 1.0
    assert scores["section_hit"] is None


def test_the_chat_by_kind_prints_as_its_table(capsys) -> None:
    from scripts.run_evals import print_suite

    print_suite({"suite": "chat", "metrics": {"by_kind": {"snapshot": {"judge_correctness_mean": 2.0}}}})
    out = capsys.readouterr().out
    assert "by kind" in out and "snapshot" in out and "{" not in out


def test_the_chat_suite_is_registered() -> None:
    from evals.suites import SUITES

    assert "chat" in SUITES


SUITE_GRAPHS = {
    "generation": "rue_task",
    "marking": "rue_evaluator",
    "writing_task": "writing_task",
    "writing_evaluator": "writing_evaluator",
    "coach": "coach",
    "chat": "coach_chat",
}


@pytest.mark.parametrize(("suite", "graph"), sorted(SUITE_GRAPHS.items()))
def test_each_suite_declares_exactly_the_tiers_its_graph_calls(suite: str, graph: str) -> None:
    """A stale map would skip a suite the sweep does change, or run one it does not."""
    import re
    from pathlib import Path

    from scripts.run_evals import SUITE_TIERS

    graphs = Path(__file__).resolve().parents[1] / "src" / "cae" / "graphs"
    source = (graphs / f"{graph}.py").read_text(encoding="utf-8")
    assert SUITE_TIERS[suite] == set(re.findall(r'tier="(\w+)"', source))


def test_every_suite_has_its_tiers_declared() -> None:
    from scripts.run_evals import SUITE_NAMES, SUITE_TIERS

    assert set(SUITE_TIERS) == set(SUITE_NAMES) == set(SUITE_GRAPHS)


def test_pinning_one_tier_runs_only_the_suites_that_call_it() -> None:
    from scripts.run_evals import measures

    assert measures("coach", ("coach",)) and not measures("coach", ("smart",))
    assert not any(measures(s, ("smart",)) for s in ("marking", "generation", "chat"))
    assert measures("marking", ("rue_evaluator",)) and not measures("marking", ("fast",))
    assert not measures("writing_evaluator", ("rue_evaluator",))
    assert measures("writing_evaluator", ("writing_evaluator",))
    assert not measures("writing_evaluator", ("smart", "writing_task"))
    assert measures("generation", ("rue_task",))
    assert not measures("generation", ("fast", "smart"))
    assert measures("writing_task", ("writing_task",))
    assert not measures("writing_task", ("fast", "smart", "rue_task", "rue_evaluator"))
    assert not measures(
        "chat", ("fast", "smart", "rue_task", "rue_evaluator", "writing_task", "writing_evaluator")
    )


def test_use_model_pins_only_the_tiers_it_is_given(free_providers) -> None:
    from cae.config import get_settings
    from scripts.run_evals import use_model

    fast_before = get_settings().cae_fast_model
    use_model("mistral:ministral-14b-latest", tiers=("smart",))

    settings = get_settings()
    assert settings.cae_smart_model == "mistral:ministral-14b-latest"
    assert settings.cae_fast_model == fast_before


def test_every_suite_uploads_to_its_own_dataset_with_evaluators() -> None:
    from evals.upload import DATASET_NAMES, EVALUATORS, EXAMPLE_BUILDERS
    from scripts.run_evals import SUITE_NAMES

    assert set(DATASET_NAMES) == set(EVALUATORS) == set(EXAMPLE_BUILDERS) == set(SUITE_NAMES)
    assert len(set(DATASET_NAMES.values())) == len(DATASET_NAMES)


def test_every_writing_row_carries_the_task_it_was_marked_against() -> None:
    """Content cannot be judged by hand without the question."""
    from evals.upload import experiment_rows

    task = {"genre": "essay", "title": "Museums", "spec": {"prompt": "..."}}
    rows = experiment_rows(
        "writing_evaluator", {"task": task, "cases": [{"id": "museums-strong"}, {"id": "museums-weak"}]}
    )
    assert set(rows) == {"museums-strong", "museums-weak"}
    assert all(row["task"] == task for row in rows.values())


def test_generation_rows_are_keyed_by_part_and_group() -> None:
    from evals.upload import _generation_examples, experiment_rows

    rows = experiment_rows("generation", {"cases": [{"id": "part4-g2", "part": 4}]})
    assert set(rows) == {"part4-g2"}
    assert "part4-g2" in {e["inputs"]["id"] for e in _generation_examples()}


def test_an_experiment_scores_only_the_examples_the_run_attempted() -> None:
    """One group is a third of the dataset; the other two are not failures."""
    from types import SimpleNamespace

    from evals.upload import examples_run

    examples = [SimpleNamespace(inputs={"id": f"part1-g{g}"}) for g in (1, 2, 3)]
    kept = examples_run(examples, {"part1-g2": {"ok": True}})
    assert [e.inputs["id"] for e in kept] == ["part1-g2"]


def test_every_generation_example_is_labelled_with_its_case_and_split() -> None:
    """LangSmith shows metadata and splits, not the keys of the inputs."""
    from evals.upload import _generation_examples

    for example in _generation_examples():
        inputs = example["inputs"]
        assert example["metadata"] == {
            "case": inputs["id"], "part": inputs["part"], "group": inputs["group"]
        }
        assert example["split"] == f"group-{inputs['group']}"


class _FakeLangSmith:
    """Just enough of langsmith.Client for ensure_dataset."""

    def __init__(self, examples):
        from types import SimpleNamespace

        self.dataset = SimpleNamespace(id="ds", name="cae-generation")
        self.examples = examples
        self.created: list = []
        self.updated: list = []

    def has_dataset(self, dataset_name):
        return True

    def read_dataset(self, dataset_name):
        return self.dataset

    def list_examples(self, dataset_id):
        return iter(self.examples)

    def create_examples(self, dataset_id, examples):
        self.created.extend(examples)

    def update_examples(self, dataset_id, updates):
        self.updated.extend(updates)


def test_existing_examples_gain_their_labels_without_their_inputs_changing() -> None:
    """Backfilling must not require deleting the dataset, which takes its experiments."""
    from types import SimpleNamespace

    from evals.upload import _generation_examples, ensure_dataset

    wanted = _generation_examples()
    unlabelled = [
        SimpleNamespace(id=f"uuid-{i}", inputs=e["inputs"], metadata={"dataset_split": ["base"]})
        for i, e in enumerate(wanted)
    ]
    client = _FakeLangSmith(unlabelled)
    ensure_dataset(client, "generation")

    assert client.created == []
    assert len(client.updated) == len(wanted)
    first = client.updated[0]
    assert set(first) == {"id", "metadata", "split"}, "inputs and outputs are left alone"
    assert first["metadata"]["case"] == wanted[0]["inputs"]["id"]
    assert "dataset_split" not in first["metadata"]


def test_labelled_examples_are_not_updated_again() -> None:
    from types import SimpleNamespace

    from evals.upload import _generation_examples, ensure_dataset

    labelled = [
        SimpleNamespace(
            id=f"uuid-{i}",
            inputs=e["inputs"],
            metadata={**e["metadata"], "dataset_split": [e["split"]]},
        )
        for i, e in enumerate(_generation_examples())
    ]
    client = _FakeLangSmith(labelled)
    ensure_dataset(client, "generation")
    assert client.created == [] and client.updated == []


def test_the_marked_items_are_the_ones_the_model_had_a_say_in() -> None:
    from evals.suites import _marked_items

    rows = [
        {"item_number": 1, "user_answer": "a", "correct_answer": "a", "marks_awarded": 1,
         "max_marks": 1, "adjudicated": False, "category": None},
        {"item_number": 2, "user_answer": "on", "correct_answer": "in", "marks_awarded": 1,
         "max_marks": 1, "adjudicated": True, "category": None, "reason": "also idiomatic"},
        {"item_number": 3, "user_answer": "x", "correct_answer": "y", "marks_awarded": 0,
         "max_marks": 1, "adjudicated": False, "category": "FIXED_PHRASES"},
    ]
    items = _marked_items(
        rows, {2: "accept", 3: "reject"}, {2: "reject"}, {3: "FIXED_PHRASES"}
    )
    assert [i["item"] for i in items] == [2, 3]
    assert items[0]["verdict"] == "accept" and items[0]["expected_verdict"] == "reject"
    assert items[0]["reason"] == "also idiomatic"


def _marking_case(part: int, adjudication: dict, tagging: dict, items=()) -> dict:
    return {
        "ok": True, "part": part, "seconds": 1.0, "items": list(items),
        "adjudication": adjudication, "tagging": tagging,
    }


def _score(items: int, missed: int = 0, **extra) -> dict:
    return {
        "items": items,
        "accuracy": round((items - missed) / items, 3) if items else 0.0,
        "exact": round((items - missed) / items, 3) if items else 0.0,
        "family": 1.0 if items else 0.0,
        "over_accepted": 0, "over_rejected": missed,
        "disagreements": [{"item": i} for i in range(missed)],
        **extra,
    }


def test_a_paper_with_nothing_to_adjudicate_is_not_scored_zero() -> None:
    """Part 1 is never adjudicated: averaging its empty score in would sink the mean."""
    from evals.suites import marking_metrics

    result = marking_metrics(
        [
            _marking_case(1, _score(0), _score(8, missed=2)),
            _marking_case(2, _score(8), _score(4)),
        ]
    )
    assert result["adjudication_accuracy"] == 1.0
    assert result["adjudicated_items"] == 8
    assert result["adjudication_by_part"] == {2: 1.0}
    assert result["category_exact_by_part"] == {1: 0.75, 2: 1.0}


def test_part4_adjudication_is_reported_per_route() -> None:
    from evals.suites import marking_metrics

    items = [
        {"route": "close", "verdict": "partial", "expected_verdict": "partial"},
        {"route": "different", "verdict": "reject", "expected_verdict": "accept"},
        {"route": "different", "verdict": "accept", "expected_verdict": "accept"},
    ]
    result = marking_metrics([_marking_case(4, _score(3, missed=1), _score(0), items)])
    assert result["part4_adjudication_by_route"] == {"close": 1.0, "different": 0.5}


def test_the_routes_of_the_two_part4_papers_are_what_their_names_say() -> None:
    from evals.suites import adjudication_routes

    for case in _reading_cases():
        if case["part"] != 4:
            continue
        responses = {int(k): v for k, v in case["responses"].items()}
        routes = set(adjudication_routes(4, case["exercise"], responses).values())
        assert routes == ({"different"} if "far" in case["id"] else {"close"}), case["id"]


def test_an_unadjudicated_paper_uploads_no_adjudication_score() -> None:
    from evals.upload import _adjudication, _over_accepted

    part1 = {"adjudication": {"items": 0, "accuracy": 0.0, "over_accepted": 0}}
    assert _adjudication(part1, {})["score"] is None
    assert _over_accepted(part1, {})["score"] is None
    part2 = {"adjudication": {"items": 8, "accuracy": 0.875, "over_accepted": 1}}
    assert _adjudication(part2, {})["score"] == 0.875


def test_the_balanced_adjudication_uploads_per_paper() -> None:
    from evals.upload import _adjudication_balanced

    part1 = {"adjudication": {"items": 0, "by_verdict": {}}}
    assert _adjudication_balanced(part1, {})["score"] is None
    lazy = {"adjudication": {"items": 4, "by_verdict": {
        "accept": {"items": 1, "agreed": 0}, "reject": {"items": 3, "agreed": 3}}}}
    assert _adjudication_balanced(lazy, {})["score"] == 0.5


def test_an_unavailable_model_leaves_its_papers_out_rather_than_scoring_them(
    monkeypatch,
) -> None:
    """Pinned to one model with no allowance left, nothing is marked: that measures
    the provider, so it is reported apart from what the model got wrong."""
    from cae.graphs import rue_evaluator
    from cae.llm import QuotaExhaustedError
    from evals.suites import run_marking

    def spent(*args, **kwargs):
        raise QuotaExhaustedError("google:gemini-3.5-flash", "rue_evaluator")

    monkeypatch.setattr(rue_evaluator, "structured_call", spent)

    result = run_marking(case_ids=["part1-urban-beekeeping", "part2-night-train"])

    assert all(c["ok"] is False and c["unavailable"] for c in result["cases"])
    assert "QuotaExhaustedError" in result["cases"][0]["error"]
    m = result["metrics"]
    assert m["unavailable_papers"] == 2 and m["papers"] == 0
    assert m["adjudication_accuracy"] is None and m["category_exact"] is None


def test_an_unusable_reply_fails_the_paper_and_counts_against_the_model(monkeypatch) -> None:
    """The model answered, but left items unmarked: its fault, so it is counted --
    and with no fallback there are no figures to put beside it."""
    from cae.graphs import rue_evaluator
    from cae.schemas.rue_evaluator import AdjudicationBatch
    from evals.suites import run_marking

    monkeypatch.setattr(
        rue_evaluator, "structured_call", lambda *a, **k: AdjudicationBatch(decisions=[])
    )

    result = run_marking(case_ids=["part2-night-train"])

    (case,) = result["cases"]
    assert case["ok"] is False and case["failed"] and "unavailable" not in case
    assert "MarkingFailed" in case["error"]
    m = result["metrics"]
    assert m["failed_papers"] == 1 and m["unavailable_papers"] == 0
    assert m["adjudication_accuracy"] is None


def test_a_paper_with_nothing_tagged_uploads_no_tagging_score() -> None:
    from evals.upload import _category_exact, _category_family

    unavailable = {"ok": False, "unavailable": True}
    assert _category_exact(unavailable, {})["score"] is None
    assert _category_family(unavailable, {})["score"] is None
    tagged = {"tagging": {"items": 5, "exact": 0.8, "family": 1.0}}
    assert _category_exact(tagged, {})["score"] == 0.8


def test_one_marking_group_runs_only_its_four_papers(monkeypatch) -> None:
    from cae.graphs import rue_evaluator
    from evals.suites import run_marking

    monkeypatch.setattr(
        rue_evaluator,
        "structured_call",
        lambda schema, system, user, **k: factories.marking_reply(schema, user),
    )
    result = run_marking(group=2)

    assert result["group"] == 2
    assert all(c["ok"] for c in result["cases"])
    assert [c["id"] for c in result["cases"]] == [c["id"] for c in _reading_cases(2)]
    assert all(c["group"] == 2 for c in result["cases"])


def test_rejecting_everything_does_not_pass_the_balanced_adjudication(monkeypatch) -> None:
    """Most near misses in the set should be rejected, so a marker that rejects them
    all scores about 0.7 on plain accuracy. Balanced accuracy is what catches it."""
    from cae.graphs import rue_evaluator
    from cae.schemas.rue_evaluator import AdjudicationBatch
    from evals.suites import run_marking

    def reject_everything(schema, system, user, **kwargs):
        reply = factories.marking_reply(schema, user)
        assert not isinstance(reply, AdjudicationBatch) or all(
            d.verdict == "reject" for d in reply.decisions
        )
        return reply

    monkeypatch.setattr(rue_evaluator, "structured_call", reject_everything)
    m = run_marking(group=1)["metrics"]

    assert m["adjudication_accuracy"] >= 0.6
    assert m["adjudication_balanced_by_part"] == {2: 0.5, 3: 0.5, 4: 0.333}
    assert m["adjudication_balanced"] == 0.444
    assert m["adjudication_by_verdict"] == {"accept": 0.0, "partial": 0.0, "reject": 1.0}


def test_an_unknown_marking_group_is_refused() -> None:
    from evals.suites import run_marking

    with pytest.raises(ValueError, match="No case group 9"):
        run_marking(group=9)


def test_the_runner_offers_group_for_marking_and_names_the_report_after_it() -> None:
    from scripts.run_evals import GROUPED_SUITES

    assert GROUPED_SUITES == {
        "generation", "marking", "writing_task", "writing_evaluator", "coach", "chat"
    }


def test_the_marking_examples_carry_their_group() -> None:
    from evals.upload import EXAMPLE_BUILDERS

    examples = EXAMPLE_BUILDERS["marking"]()
    by_id = {e["inputs"]["id"]: e for e in examples}
    case = _reading_cases(3)[0]
    assert by_id[case["id"]]["metadata"]["group"] == 3
    assert by_id[case["id"]]["split"] == "group-3"


def _marking_report(group: int, runs: dict[str, list[dict] | str], when: str = "") -> dict:
    """A run_evals payload for one marking group: per model, case rows or an error."""
    payload_runs = []
    for model, outcome in runs.items():
        suite = (
            {"suite": "marking", "error": outcome}
            if isinstance(outcome, str)
            else {"suite": "marking", "group": group, "cases": outcome}
        )
        payload_runs.append({"model": model, "suites": {"marking": suite}})
    return {"group": group, "generated_at": when, "runs": payload_runs, "source": f"g{group}{when}"}


def _paper(case_id: str, part: int, agreed: int, items: int = 4) -> dict:
    """A marked paper whose adjudication and tagging agreed on ``agreed`` of ``items``."""
    score = {
        "items": items, "accuracy": agreed / items, "exact": agreed / items, "family": 1.0,
        "over_accepted": 0, "over_rejected": items - agreed,
        "by_verdict": {
            verdict: tally
            for verdict, tally in (
                ("accept", {"items": items - agreed, "agreed": 0}),
                ("reject", {"items": agreed, "agreed": agreed}),
            )
            if tally["items"]
        },
        "disagreements": [{"item": i} for i in range(items - agreed)],
    }
    return {"id": case_id, "part": part, "ok": True, "seconds": 1.0,
            "adjudication": score, "tagging": dict(score), "items": []}


def test_combining_marking_groups_pools_every_paper() -> None:
    from scripts.combine_marking_reports import combine

    reports = [
        _marking_report(1, {"m": [_paper("a", 2, 4), _paper("b", 3, 2)]}),
        _marking_report(2, {"m": [_paper("c", 2, 4), _paper("d", 3, 4)]}),
    ]
    (entry,) = combine(reports)
    assert entry["complete"]
    assert entry["metrics"]["papers"] == 4
    assert entry["metrics"]["adjudication_accuracy"] == 0.875
    assert entry["groups"][1]["adjudication_accuracy"] == 0.75
    assert entry["metrics"]["adjudication_by_part"] == {2: 1.0, 3: 0.75}
    assert entry["metrics"]["adjudication_balanced_by_part"] == {2: 1.0, 3: 0.5}


def test_an_unavailable_paper_never_replaces_an_earlier_result() -> None:
    from scripts.combine_marking_reports import combine

    earlier = _marking_report(1, {"m": [_paper("a", 2, 3)]}, when="2026-10-11")
    spent = _marking_report(
        1, {"m": [{"id": "a", "part": 2, "ok": False, "unavailable": True, "error": "x"}]},
        when="2026-10-12",
    )
    (entry,) = combine([earlier, spent])
    assert entry["metrics"]["papers"] == 1
    assert entry["metrics"]["adjudication_accuracy"] == 0.75


def test_a_model_missing_a_marking_group_is_incomplete() -> None:
    from scripts.combine_marking_reports import combine

    reports = [
        _marking_report(1, {"a": [_paper("x", 2, 4)], "b": [_paper("x", 2, 4)]}),
        _marking_report(2, {"a": [_paper("y", 2, 4)]}),
    ]
    entries = {e["model"]: e for e in combine(reports)}
    assert entries["a"]["complete"] and not entries["b"]["complete"]
    assert "error" in entries["b"]["groups"][2]


def test_only_grouped_marking_reports_are_found(tmp_path) -> None:
    import json

    from scripts.combine_generation_reports import grouped_reports

    for name, group in [
        ("20261012T090000Z-marking-g1.json", 1),
        ("20261012T100000Z-marking-g2.json", 2),
        ("20261012T110000Z-marking.json", None),
        ("20261013T090000Z-marking-combined.json", None),
        ("20261012T090000Z-generation-g1.json", 1),
    ]:
        (tmp_path / name).write_text(json.dumps({"group": group}), encoding="utf-8")

    assert [p.name for p in grouped_reports(tmp_path, suite="marking")] == [
        "20261012T090000Z-marking-g1.json",
        "20261012T100000Z-marking-g2.json",
    ]


def _writing_report(group: int, runs: dict[str, list[dict]], when: str = "") -> dict:
    """A run_evals payload for one writing group: per model, its case rows."""
    return {
        "group": group,
        "generated_at": when,
        "source": f"g{group}{when}",
        "runs": [
            {"model": model, "suites": {"writing_evaluator": {"suite": "writing_evaluator", "group": group, "cases": rows}}}
            for model, rows in runs.items()
        ],
    }


def _script(case_id: str, group: int, genre: str, total_accuracy: float, **extra) -> dict:
    """A marked script row whose bands scored ``total_accuracy`` on the total, and
    whose annotations found one of two labelled errors with one annotation."""
    row = {
        "id": case_id, "group": group, "genre": genre, "ok": True, "seconds": 1.0,
        "degraded_subscales": [], "annotate_degraded": False,
        "band_accuracy": {**{s: 1.0 for s in SUBSCALES}, "total": total_accuracy},
        "annotation_scores": metrics.annotation_scores(
            [{"span": "one", "category": "WORD_FORM"}], "one two", [
                {"span": "one", "category": "WORD_FORM"},
                {"span": "two", "category": "WORD_FORM"},
            ],
        ),
    }
    return {**row, **extra}


def test_combining_writing_groups_pools_every_script_per_genre() -> None:
    from scripts.combine_writing_evaluator_reports import combine

    reports = [
        _writing_report(1, {"m": [_script("museums-strong", 1, "essay", 1.0),
                                  _script("series-review", 1, "review", 0.9)]}),
        _writing_report(3, {"m": [_script("museums-weak", 3, "essay", 0.8)]}),
    ]
    (entry,) = combine(reports)
    assert entry["complete"]
    m = entry["metrics"]
    assert m["scripts"] == 3 and m["accuracy_total"] == 0.9
    assert m["by_genre"]["essay"]["scripts"] == 2
    assert m["by_genre"]["essay"]["accuracy_total"] == 0.9
    assert m["by_genre"]["review"]["accuracy_total"] == 0.9
    assert (m["by_genre"]["essay"]["recall"], m["by_genre"]["essay"]["precision"]) == (0.5, 1.0)
    assert entry["groups"][1]["accuracy_total"] == 0.95


def test_a_degraded_script_never_replaces_an_earlier_result() -> None:
    from scripts.combine_writing_evaluator_reports import combine

    earlier = _writing_report(
        1, {"m": [_script("museums-strong", 1, "essay", 1.0)]}, when="2026-10-11"
    )
    spent = _writing_report(
        1,
        {"m": [_script("museums-strong", 1, "essay", 0.5, degraded_subscales=["language"])]},
        when="2026-10-12",
    )
    (entry,) = combine([earlier, spent])
    assert entry["metrics"]["accuracy_total"] == 1.0
    assert entry["metrics"]["degraded_scripts"] == 0


def test_only_grouped_writing_reports_are_found(tmp_path) -> None:
    import json

    from scripts.combine_generation_reports import grouped_reports

    for name, group in [
        ("20261014T090000Z-writing_evaluator-g1.json", 1),
        ("20261014T090000Z-writing_evaluator.json", None),
        ("20261014T090000Z-writing_task-g1.json", 1),
        ("20261015T090000Z-writing_evaluator-combined.json", None),
    ]:
        (tmp_path / name).write_text(json.dumps({"group": group}), encoding="utf-8")

    assert [p.name for p in grouped_reports(tmp_path, suite="writing_evaluator")] == [
        "20261014T090000Z-writing_evaluator-g1.json"
    ]


def test_the_writing_examples_carry_their_group() -> None:
    from evals.upload import EXAMPLE_BUILDERS

    by_id = {e["inputs"]["id"]: e for e in EXAMPLE_BUILDERS["writing_evaluator"]()}
    case = _writing_cases(2)[0]
    assert by_id[case["id"]]["metadata"]["group"] == 2
    assert by_id[case["id"]]["split"] == "group-2"


def test_a_case_is_not_traced_when_tracing_is_off() -> None:
    from evals.suites import _case_trace

    row = {"id": "case"}
    with _case_trace("marking", row, {}):
        row["ok"] = True
    assert row == {"id": "case", "ok": True}


def test_a_traced_case_records_its_row_and_links_to_the_trace(monkeypatch) -> None:
    """The row is what the experiment shows; the link is how to reach the prompts."""
    import langsmith

    from evals import suites

    recorded: dict = {}

    class FakeRun:
        trace_id = "0001"

        def add_outputs(self, outputs: dict) -> None:
            recorded["outputs"] = outputs

        def get_url(self) -> str:  # pragma: no cover
            raise AssertionError("resolving a URL per case costs a request each")

    class FakeTrace:
        def __init__(self, name: str, **kwargs) -> None:
            recorded.update(name=name, **kwargs)

        def __enter__(self) -> FakeRun:
            return FakeRun()

        def __exit__(self, *exc) -> None:
            return None

    monkeypatch.setattr(suites, "tracing_enabled", lambda: True)
    monkeypatch.setattr(langsmith, "trace", FakeTrace)

    row = {"part": 4}
    with suites._case_trace("generation", row, {"topic": "bees"}):
        row["ok"] = True

    assert recorded["name"] == "eval.generation"
    assert recorded["inputs"] == {"case": "part4", "topic": "bees"}
    assert recorded["metadata"]["eval_case"] == "part4"
    assert "cae_smart_chain" in recorded["metadata"]
    assert recorded["outputs"] == {"part": 4, "ok": True}
    assert row["trace_id"] == "0001"


def test_an_uploaded_row_links_to_its_trace() -> None:
    from evals.upload import experiment_rows

    result = {"cases": [{"id": "traced", "trace_id": "0001"}, {"id": "untraced"}]}
    rows = experiment_rows("marking", result, lambda trace_id: f"https://smith.example/r/{trace_id}")
    assert rows["traced"]["trace_url"] == "https://smith.example/r/0001"
    assert "trace_url" not in rows["untraced"]


COACH_PROFILES = load_dataset("coach_profiles")["profiles"]


def _computed(profile: dict) -> tuple[dict, dict, dict]:
    """The learner as the graph sees it: its facts, its tables, the figures above them."""
    from cae.graphs import coach
    from evals.suites import coach_facts, seeded_learner

    with seeded_learner(profile) as (session, user_id):
        state = {"user_id": user_id, **coach.stats_from(session, user_id)}
    state.update(coach.detect_trend(state))
    figures = coach.prompt_figures(state["snapshot"])
    return coach_facts(state["snapshot"], state["examples"], figures), state["tables"], figures


REPORT = {
    "headline": "Steady progress, with one part to fix.",
    "summary": "You are close to the 180 pass mark.",
    "strengths": ["Part 1 is improving.", "You practise regularly."],
    "common_errors": ["Word formation and spelling.", "Word choice."],
    "trend": "Part 1 is rising; the other parts are flat.",
    "focus_areas": [
        {"title": "Part 3", "why": "Your weakest part.", "action": "One Part 3 a day."},
        {"title": "Spelling", "why": "Your most frequent slip.", "action": "Keep a list."},
        {"title": "Timing", "why": "Whole papers.", "action": "Sit a mock exam."},
    ],
    "next_session": "A Part 3 exercise.",
}


def _stub_coach(monkeypatch, report: dict | None = None) -> "_FakeJudge":
    """``narrate`` writes ``report``; the judge is a fake one, returned to inspect."""
    from cae.graphs import coach
    from cae.schemas.coach import CoachNarrative

    narrative = CoachNarrative(**(report or REPORT))
    monkeypatch.setattr(coach, "structured_call", lambda *a, **k: narrative)
    judge = _FakeJudge()
    monkeypatch.setattr("evals.judge.build_judge", lambda *a, **k: judge)
    return judge


def test_the_coach_set_has_three_groups_of_five_archetypes_with_unique_ids() -> None:
    from evals.suites import ARCHETYPES, coach_case_id, coach_groups

    assert coach_groups() == [1, 2, 3]
    for group in coach_groups():
        kinds = sorted(p["archetype"] for p in COACH_PROFILES if p["group"] == group)
        assert kinds == sorted(ARCHETYPES)
    assert all(p["id"] == coach_case_id(p["archetype"], p["group"]) for p in COACH_PROFILES)


@pytest.mark.parametrize("profile", COACH_PROFILES, ids=lambda p: p["id"])
def test_every_profile_is_work_the_app_could_have_recorded(profile: dict) -> None:
    from cae.domain.parts import WRITING_PART1_GENRES, WRITING_PART2_GENRES
    from cae.domain.taxonomy import WRITING_CATEGORIES, categories_for_part

    for part, marks in profile["attempts"].items():
        assert all(0 <= m <= PART_SPECS[int(part)].max_marks for m in marks), part
    for error in profile["errors"]:
        allowed = categories_for_part(error["part"]) if error["part"] else WRITING_CATEGORIES
        assert error["category"] in {c.value for c in allowed}, error
        assert error["count"] > 0 and error["evidence"]
        if error["part"]:
            assert str(error["part"]) in profile["attempts"], "an error from a part never sat"
        else:
            assert profile["writing"], "a Writing error with no Writing"
    genres = {1: WRITING_PART1_GENRES, 2: WRITING_PART2_GENRES}
    for piece in profile["writing"]:
        assert piece["genre"] in genres[piece["part"]]
        assert set(piece["bands"]) == set(SUBSCALES)
        assert all(0 <= band <= 5 for band in piece["bands"].values())


@pytest.mark.parametrize("profile", COACH_PROFILES, ids=lambda p: p["id"])
def test_every_label_agrees_with_the_snapshot_the_app_computes(profile: dict) -> None:
    """The labels are a teacher's reading of the figures; this keeps the figures
    saying what the labels claim, so neither can drift from the other."""
    from cae.graphs.coach import MIN_ACTIVITY_FOR_REPORT

    facts, tables, figures = _computed(profile)
    expect = profile["expect"]
    parts = {int(p): s for p, s in facts["parts"].items()}
    attempted = {p: s for p, s in parts.items() if s["attempts"]}

    activity = figures["reading_attempts"] + figures["writing_count"]
    assert activity >= MIN_ACTIVITY_FOR_REPORT, "below the threshold: no report to judge"
    for part, direction in expect["trend"].items():
        trend = parts[int(part)]["trend"]
        assert (trend > 0) == (direction == "up") and abs(trend) >= 5, (part, trend)
    assert expect["insufficient"] == all(s["attempts"] < 3 for s in attempted.values())
    accuracies = sorted(s["accuracy"] for s in attempted.values())
    assert len(accuracies) < 2 or accuracies[0] < accuracies[1]
    counts = [row["count"] for row in facts["errors"]]
    assert len(counts) < 2 or counts[0] > counts[1]
    writing = facts["writing"]
    computed = [{"part": facts["weakest"]}, {"category": facts["errors"][0]["category"]}]
    if writing:
        computed.append({"subscale": min(writing, key=writing.get)})
    assert any(t in computed for t in expect["priority"]), expect["priority"]
    for target in expect["priority"]:
        if "part" in target:
            assert target["part"] in attempted
        elif "category" in target:
            assert any(r["category"] == target["category"] for r in facts["errors"])
        else:
            assert target["subscale"] in writing
    assert expect["targets"] == expect["priority"][:1]
    handed = metrics.stated_numbers(
        " ".join(str(v) for v in figures.values()) + "\n" + "\n".join(tables.values())
    )
    handed = [abs(n) for n in handed] + [abs(t - figures["scale"]) for t in (180, 200)]
    for figure in expect["figures"]:
        assert metrics._stated(figure, handed), (profile["id"], figure)


@pytest.mark.parametrize("profile", COACH_PROFILES, ids=lambda p: p["id"])
def test_every_reference_passes_its_own_checks(profile: dict) -> None:
    """The reference is what the judge compares against: if it fails its own
    labels, the label or the reference is wrong, not the coach."""
    facts, _tables, _figures = _computed(profile)
    expect = profile["expect"]
    scored = metrics.answer_correct(
        profile["reference"], {"figures": expect["figures"], "targets": expect["targets"]}, facts
    )
    assert scored["correct"], (profile["id"], scored["missing"])
    names = {int(p): s["name"] for p, s in facts["parts"].items()}
    for part in expect["trend"]:
        assert int(part) in metrics.parts_named(profile["reference"], names), (profile["id"], part)
    assert ("enough data" in profile["reference"] or "too little data" in profile["reference"]) == (
        expect["insufficient"]
    ), profile["id"]


def test_the_coach_verdict_has_no_optional_field_and_the_chats_scales() -> None:
    from evals.judge import ChatVerdict, CoachVerdict

    schema = CoachVerdict.model_json_schema()
    assert set(schema["required"]) == set(schema["properties"])
    assert "null" not in str(schema)
    chat = ChatVerdict.model_json_schema()["properties"]
    for field, spec in schema["properties"].items():
        assert spec.get("enum") == chat[field].get("enum"), field


def test_the_report_is_shown_whole_in_the_pages_order() -> None:
    from evals.judge import report_text

    text = report_text(REPORT)
    titles = ["Headline:", "Summary:", "Strengths:", "Common errors:", "Trend:", "Focus areas", "Next session:"]
    positions = [text.index(t) for t in titles]
    assert positions == sorted(positions)
    for area in REPORT["focus_areas"]:
        assert area["title"] in text and area["why"] in text and area["action"] in text
    assert "- You practise regularly." in text


def test_the_coach_judge_sees_what_narrate_was_handed_and_the_reference() -> None:
    """Leaving any of it out would mark a report unsupported for citing what the
    coach was in fact given."""
    from evals.judge import COACH_JUDGE_SYSTEM, judge_report

    profile = COACH_PROFILES[1]
    facts, tables, figures = _computed(profile)
    case = {
        "id": profile["id"], "ok": True, "narrative": REPORT,
        "figures": figures, "tables": tables, "reference": profile["reference"],
    }
    judge = _FakeJudge()
    judge_report(judge, case)

    (messages,) = judge.calls
    assert messages[0] == ("system", COACH_JUDGE_SYSTEM)
    user = messages[1][1]
    for expected in ("180 is the C1 pass mark", "Grade A from 200", f"overall: {figures['overall']}",
                     tables["parts"].splitlines()[1], profile["reference"], "Focus areas",
                     REPORT["focus_areas"][0]["action"]):
        assert expected in user, expected


def test_only_written_reports_are_judged_and_a_judge_failure_is_kept() -> None:
    from evals.judge import judge_coach_cases
    from evals.suites import coach_metrics

    cases = [
        {"id": "a", "archetype": "strong", "ok": True, "seconds": 1.0, "narrative": REPORT,
         "figures": {}, "tables": {}, "reference": "fine"},
        {"id": "b", "archetype": "strong", "ok": False, "unavailable": True},
        {"id": "c", "archetype": "strong", "ok": True, "seconds": 1.0,
         "narrative": {**REPORT, "headline": "BREAK"}, "figures": {}, "tables": {}, "reference": "x"},
    ]
    judge = _FakeJudge(fail_on="BREAK")
    judge_coach_cases(cases, judge)

    assert len(judge.calls) == 2 and "judge" not in cases[1]
    assert "error" in cases[2]["judge"] and cases[0]["judge"]["correctness"] == 2
    m = coach_metrics(cases)
    assert (m["profiles"], m["judged"], m["unavailable"]) == (2, 1, 1)
    assert m["judge_correctness_mean"] == 2 and m["judge_helpfulness"] == 4


def test_one_coach_group_is_written_and_judged_whole(monkeypatch) -> None:
    from evals.suites import ARCHETYPES, run_coach

    judge = _stub_coach(monkeypatch)
    result = run_coach(group=1)

    assert sorted(c["id"] for c in result["cases"]) == sorted(
        p["id"] for p in COACH_PROFILES if p["group"] == 1
    )
    assert all(c["ok"] and c["narrative"]["headline"] for c in result["cases"])
    m = result["metrics"]
    assert (m["profiles"], m["judged"], m["judge_correctness_mean"]) == (5, 5, 2)
    assert list(m["by_archetype"]) == list(ARCHETYPES)
    references = {p["id"]: p["reference"] for p in COACH_PROFILES}
    for case, messages in zip(result["cases"], judge.calls):
        assert case["reference"] == references[case["id"]]
        assert case["reference"] in messages[1][1] and case["tables"]["parts"] in messages[1][1]


def test_without_the_judge_the_reports_are_kept_unscored(monkeypatch) -> None:
    from evals.suites import run_coach

    judge = _stub_coach(monkeypatch)
    result = run_coach(group=2, judge=False)

    assert judge.calls == [] and all("judge" not in c for c in result["cases"])
    assert result["metrics"]["profiles"] == 5 and "judged" not in result["metrics"]


def test_asking_for_the_judge_without_a_key_fails_before_any_report(monkeypatch) -> None:
    from cae.graphs import coach
    from evals.judge import JudgeUnavailableError
    from evals.suites import run_coach

    calls: list = []
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr(coach, "structured_call", lambda *a, **k: calls.append(1))
    with pytest.raises(JudgeUnavailableError):
        run_coach(group=1, judge=True)
    assert calls == []


def test_one_archetype_can_be_run_alone(monkeypatch) -> None:
    from evals.suites import run_coach

    _stub_coach(monkeypatch)
    result = run_coach(archetypes=["threshold"])
    assert {c["archetype"] for c in result["cases"]} == {"threshold"}
    assert len(result["cases"]) == 3


def test_an_unavailable_model_is_counted_apart_and_judged_nowhere(monkeypatch) -> None:
    from cae.graphs import coach
    from cae.llm import QuotaExhaustedError
    from evals.suites import run_coach

    judge = _stub_coach(monkeypatch)

    def spent(*args, **kwargs):
        raise QuotaExhaustedError("google:gemini-3.6-flash", "coach")

    monkeypatch.setattr(coach, "structured_call", spent)
    result = run_coach(case_ids=["improving-g1", "threshold-g1"])

    assert all(c["ok"] is False and c["unavailable"] for c in result["cases"])
    assert "QuotaExhaustedError" in result["cases"][0]["error"]
    m = result["metrics"]
    assert (m["unavailable"], m["failed"], m["profiles"]) == (2, 0, 0)
    assert "judged" not in m and judge.calls == []


def test_an_unreadable_reply_fails_the_report_and_counts_against_the_model(monkeypatch) -> None:
    from cae.graphs import coach
    from evals.suites import run_coach

    _stub_coach(monkeypatch)

    def garbled(*args, **kwargs):
        raise ValueError("no JSON in the reply")

    monkeypatch.setattr(coach, "structured_call", garbled)
    (case,) = run_coach(case_ids=["strong-g1"])["cases"]
    assert case["ok"] is False and case["failed"] and "unavailable" not in case


def test_an_unknown_coach_group_is_refused() -> None:
    from evals.suites import run_coach

    with pytest.raises(ValueError, match="No profile group 9"):
        run_coach(group=9, judge=False)


def test_the_coach_suite_never_writes_to_the_apps_database(monkeypatch) -> None:
    """The learners live in memory: nothing reaches cae.db, not even the report cache."""
    from sqlalchemy import func, select

    from cae.db.models import Attempt, CoachReport, User
    from cae.db.session import session_scope
    from evals.suites import run_coach

    def counts() -> tuple:
        with session_scope() as session:
            return tuple(session.scalar(select(func.count()).select_from(t)) for t in (User, Attempt, CoachReport))

    _stub_coach(monkeypatch)
    before = counts()
    run_coach(group=1)
    assert counts() == before


def test_below_the_threshold_no_report_is_written_and_no_model_is_called(monkeypatch) -> None:
    from cae.graphs import coach
    from evals.suites import seeded_learner

    calls: list = []
    monkeypatch.setattr(coach, "structured_call", lambda *a, **k: calls.append(1))
    profile = {"id": "one-exercise", "attempts": {"2": [5]}, "errors": [], "writing": []}
    with seeded_learner(profile) as (session, user_id):
        state = {"user_id": user_id, **coach.stats_from(session, user_id)}
    state.update(coach.detect_trend(state))
    assert coach.narrate(state) == {"narrative": {}, "status": "not_enough_data"}
    assert calls == []


def test_the_report_and_the_judge_see_the_same_figures(monkeypatch) -> None:
    """``narrate`` builds its prompt from ``prompt_figures``, which the row keeps
    as the judge's evidence: if they parted, a quoted figure would read as invented."""
    from cae.graphs import coach
    from cae.prompts import coach as coach_prompts
    from evals.suites import run_coach

    seen: dict = {}
    real = coach_prompts.coach_prompt

    def spy(**kwargs):
        seen.update(kwargs)
        return real(**kwargs)

    _stub_coach(monkeypatch)
    monkeypatch.setattr(coach, "coach_prompt", spy)
    (case,) = run_coach(case_ids=["strong-g1"])["cases"]
    assert {k: seen[k] for k in case["figures"]} == case["figures"]
    assert seen["part_table"] == case["tables"]["parts"]


def test_a_saved_coach_report_is_judged_without_writing_it_again(monkeypatch) -> None:
    from pathlib import Path

    from cae.graphs import coach
    from evals.suites import run_coach
    from scripts.judge_chat_report import judged_label
    from scripts.judge_coach_report import judge_report

    _stub_coach(monkeypatch)
    result = run_coach(group=2, judge=False)
    monkeypatch.setattr(coach, "structured_call", lambda *a, **k: pytest.fail("narrate again"))
    payload = {"group": 2, "runs": [{"model": "m", "suites": {"coach": result}}]}
    judged = judge_report(payload, _FakeJudge())

    assert judged["runs"][0]["suites"]["coach"]["metrics"]["judged"] == 5
    assert judged_label(Path("20261017T090000Z-coach-g2.json")) == "coach-g2"


def _coach_report(group: int, runs: dict[str, list[dict]], when: str = "") -> dict:
    """A run_evals payload for one coach group: per model, its case rows."""
    return {
        "group": group,
        "generated_at": when,
        "source": f"g{group}{when}",
        "runs": [
            {"model": model, "suites": {"coach": {"suite": "coach", "group": group, "cases": rows}}}
            for model, rows in runs.items()
        ],
    }


def _learner(case_id: str, group: int, archetype: str, correctness: int | None = 2, **extra) -> dict:
    """A written coach row, judged with ``correctness`` (None: not judged)."""
    row = {"id": case_id, "group": group, "archetype": archetype, "ok": True, "seconds": 1.0}
    if correctness is not None:
        row["judge"] = {"correctness": correctness, "faithfulness": 2, "helpfulness": 4,
                        "unsupported_claims": [], "rationale": "."}
    return {**row, **extra}


def test_combining_coach_groups_pools_the_verdicts_per_archetype() -> None:
    from scripts.combine_coach_reports import combine

    reports = [
        _coach_report(1, {"m": [_learner("improving-g1", 1, "improving", 2),
                                _learner("strong-g1", 1, "strong", 0)]}),
        _coach_report(2, {"m": [_learner("improving-g2", 2, "improving", 1)]}),
    ]
    (entry,) = combine(reports)
    m = entry["metrics"]
    assert (m["profiles"], m["judged"], m["judge_correctness_mean"]) == (3, 3, 1.0)
    assert m["judge_correct_rate"] == 0.333
    assert m["by_archetype"]["improving"]["judge_correctness_mean"] == 1.5
    assert entry["groups"][1]["judge_correctness_mean"] == 1.0


def test_a_judged_report_replaces_the_one_it_judged() -> None:
    from scripts.combine_coach_reports import combine

    unjudged = _coach_report(1, {"m": [_learner("strong-g1", 1, "strong", None)]}, when="2026-10-17T09")
    judged = _coach_report(1, {"m": [_learner("strong-g1", 1, "strong", 2)]}, when="2026-10-17T10")
    (entry,) = combine([unjudged, judged])
    assert entry["metrics"]["judged"] == 1


def test_a_report_the_model_did_not_write_never_replaces_an_earlier_one() -> None:
    from scripts.combine_coach_reports import combine

    earlier = _coach_report(1, {"m": [_learner("strong-g1", 1, "strong", 2)]}, when="2026-10-15")
    spent = _coach_report(
        1, {"m": [{"id": "strong-g1", "group": 1, "archetype": "strong", "ok": False, "unavailable": True}]},
        when="2026-10-16",
    )
    (entry,) = combine([earlier, spent])
    assert entry["metrics"]["judged"] == 1 and entry["metrics"]["unavailable"] == 0


def test_a_model_missing_a_coach_group_is_incomplete() -> None:
    from scripts.combine_coach_reports import combine

    reports = [
        _coach_report(1, {"a": [_learner("strong-g1", 1, "strong")],
                          "b": [_learner("strong-g1", 1, "strong")]}),
        _coach_report(2, {"a": [_learner("strong-g2", 2, "strong")]}),
    ]
    by_model = {e["model"]: e for e in combine(reports)}
    assert by_model["a"]["complete"] and not by_model["b"]["complete"]


def test_only_grouped_coach_reports_are_found(tmp_path) -> None:
    import json

    from scripts.combine_generation_reports import grouped_reports

    for name, group in [
        ("20261015T090000Z-coach-g1.json", 1),
        ("20261015T100000Z-coach-g1-judged.json", 1),
        ("20261015T090000Z-coach.json", None),
        ("20261016T090000Z-coach-combined.json", None),
        ("20261015T090000Z-chat.json", None),
    ]:
        (tmp_path / name).write_text(json.dumps({"group": group}), encoding="utf-8")

    assert [p.name for p in grouped_reports(tmp_path, suite="coach")] == [
        "20261015T090000Z-coach-g1.json",
        "20261015T100000Z-coach-g1-judged.json",
    ]


def test_the_coach_examples_are_keyed_like_the_rows_and_carry_the_reference() -> None:
    from evals.upload import EXAMPLE_BUILDERS

    examples = EXAMPLE_BUILDERS["coach"]()
    assert sorted(e["inputs"]["id"] for e in examples) == sorted(p["id"] for p in COACH_PROFILES)
    by_id = {e["inputs"]["id"]: e for e in examples}
    profile = next(p for p in COACH_PROFILES if p["id"] == "slipping-g3")
    assert by_id["slipping-g3"]["metadata"] == {"case": "slipping-g3", "group": 3, "archetype": "slipping"}
    assert by_id["slipping-g3"]["split"] == "group-3"
    assert by_id["slipping-g3"]["outputs"]["reference"] == profile["reference"]


def test_the_coach_evaluators_are_the_judges_and_score_nothing_unjudged() -> None:
    from evals.upload import EVALUATORS

    keys = {e({"ok": True}, {})["key"] for e in EVALUATORS["coach"]}
    assert keys == {"judge_correctness", "judge_faithfulness", "judge_helpfulness"}
    for row in ({"ok": False, "unavailable": True}, _learner("strong-g1", 1, "strong", None)):
        assert all(e(row, {})["score"] is None for e in EVALUATORS["coach"])
    judged = _learner("strong-g1", 1, "strong", 1)
    scores = {e(judged, {})["key"]: e(judged, {})["score"] for e in EVALUATORS["coach"]}
    assert scores == {"judge_correctness": 1.0, "judge_faithfulness": 2.0, "judge_helpfulness": 4.0}


def test_the_coach_by_archetype_prints_as_its_table(capsys) -> None:
    from scripts.run_evals import print_suite

    print_suite({"suite": "coach", "metrics": {"by_archetype": {"strong": {"judge_correctness_mean": 2}}}})
    out = capsys.readouterr().out
    assert "by archetype" in out and "strong" in out and "correct/2" in out
