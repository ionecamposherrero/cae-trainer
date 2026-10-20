"""Progress charts.

The bug these guard against is invisible in the code: Vega truncates an axis
label past 180px and drops ones that collide, so a bar chart whose categories
are real names loses the very labels that say which bar is which.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from cae.db.repo import PaperCoverage, PartStat
from cae.ui import charts

LONGEST = "Part 4 · Key word transformations"


def stat(part: int, name: str, accuracy: float, attempts: int = 1) -> PartStat:
    return PartStat(
        part=part,
        name=name,
        attempts=attempts,
        marks=accuracy / 10,
        max_marks=10.0,
        accuracy=accuracy,
        recent_accuracy=accuracy,
        trend=0.0,
    )


def all_eight() -> list[PartStat]:
    from cae.domain.parts import PART_SPECS

    return [stat(part, spec.name, 50.0 + part) for part, spec in PART_SPECS.items()]


def y_axis(chart: Any) -> dict[str, Any]:
    """The y-axis of the layer that carries one, from a layered chart."""
    spec = json.loads(chart.to_json())
    layers = spec.get("layer", [spec])
    for layer in layers:
        axis = layer.get("encoding", {}).get("y", {}).get("axis")
        if axis:
            return axis
    raise AssertionError("no layer carries a y axis")


def height(chart: Any) -> int:
    spec = json.loads(chart.to_json())
    return spec["height"]


def test_part_labels_are_never_truncated() -> None:
    """Vega's default labelLimit of 180px would cut the two longest part names."""
    axis = y_axis(charts.accuracy_by_part(all_eight()))
    assert axis["labelLimit"] == 0, "0 means no limit; anything else truncates"


def test_no_bar_loses_its_label_to_collision() -> None:
    axis = y_axis(charts.accuracy_by_part(all_eight()))
    assert axis["labelOverlap"] is False


def test_the_part_number_and_its_name_go_on_separate_lines() -> None:
    """Keeps the longest label narrow without stealing width from the bars."""
    axis = y_axis(charts.accuracy_by_part(all_eight()))
    assert "split" in axis["labelExpr"]
    assert " · " in LONGEST


def test_every_attempted_part_gets_room_for_a_bar() -> None:
    for count in (1, 4, 8):
        chart = charts.accuracy_by_part(all_eight()[:count])
        assert height(chart) >= charts._BAR_ROW_HEIGHT * count


def test_parts_never_attempted_are_left_out() -> None:
    stats = [stat(1, "Multiple-choice cloze", 40.0), stat(2, "Open cloze", 0.0, attempts=0)]
    spec = json.loads(charts.accuracy_by_part(stats).to_json())
    rendered = json.dumps(spec)
    assert "Multiple-choice cloze" in rendered
    assert "Open cloze" not in rendered


def test_no_attempts_at_all_draws_nothing() -> None:
    assert charts.accuracy_by_part([stat(1, "Multiple-choice cloze", 0.0, attempts=0)]) is None


def test_weakness_labels_are_not_truncated_either() -> None:
    rows = [
        {"category": "LEXIS_COLLOCATION", "label": "Collocation and word partnership", "count": 9, "share": 40.0},
        {"category": "GRAMMAR_INVERSION", "label": "Inversion after a negative adverbial", "count": 4, "share": 18.0},
    ]
    axis = y_axis(charts.error_categories(rows))
    assert axis["labelLimit"] == 0
    assert axis["labelOverlap"] is False


def test_no_errors_draws_nothing() -> None:
    assert charts.error_categories([]) is None


@dataclass
class FakeExam:
    id: int
    scale_score: int
    marks: float = 50.0
    max_marks: float = 78.0
    finished_at: datetime = datetime(2026, 6, 1, tzinfo=UTC)


def sittings(*plan: tuple[int, bool, bool, int]) -> tuple[list[FakeExam], dict[int, PaperCoverage]]:
    """``(id, reading, writing, scale_score)`` per exam, ten days apart."""
    exams = [
        FakeExam(
            id=exam_id,
            scale_score=score,
            finished_at=datetime(2026, 6, 1, tzinfo=UTC) + timedelta(days=10 * exam_id),
        )
        for exam_id, _r, _w, score in plan
    ]
    cover = {i: PaperCoverage(reading=r, writing=w) for i, r, w, _s in plan}
    return exams, cover


def colour_encoding(chart: Any) -> dict[str, Any]:
    spec = json.loads(chart.to_json())
    for layer in spec.get("layer", [spec]):
        colour = layer.get("encoding", {}).get("color")
        if colour:
            return colour
    raise AssertionError("no layer carries a colour encoding")


def plotted(chart: Any) -> list[dict[str, Any]]:
    spec = json.loads(chart.to_json())
    return max(spec["datasets"].values(), key=len)


ALL_THREE = ((1, True, True, 168), (2, True, False, 172), (3, False, True, 160))


def test_each_kind_of_sitting_is_its_own_line() -> None:
    chart = charts.mock_exam_history(*sittings(*ALL_THREE))
    assert colour_encoding(chart)["scale"]["domain"] == list(charts.MOCK_SERIES)


def test_a_complete_exam_is_not_averaged_with_a_single_paper() -> None:
    """The three sittings are not comparable, so no point may share a line."""
    chart = charts.mock_exam_history(*sittings(*ALL_THREE))
    rows = plotted(chart)
    by_series: dict[str, list[int]] = {}
    for row in rows:
        by_series.setdefault(row["series"], []).append(row["score"])
    assert by_series == {
        "Complete exam": [168],
        "Reading & UoE only": [172],
        "Writing only": [160],
    }


def test_a_series_colour_follows_the_paper_not_its_rank() -> None:
    """Sitting only Writing must not repaint that line with the first hue."""
    all_three = colour_encoding(charts.mock_exam_history(*sittings(*ALL_THREE)))
    writing_only = colour_encoding(
        charts.mock_exam_history(*sittings((3, False, True, 160), (4, False, True, 171)))
    )
    writing_index = all_three["scale"]["domain"].index("Writing only")
    expected = all_three["scale"]["range"][writing_index]

    assert writing_only["scale"]["domain"] == ["Writing only"]
    assert writing_only["scale"]["range"] == [expected]


def test_identity_is_never_colour_alone() -> None:
    """A colourblind reader needs a second encoding; the legend names it."""
    spec = json.loads(charts.mock_exam_history(*sittings(*ALL_THREE)).to_json())
    layer = next(lyr for lyr in spec["layer"] if "color" in lyr.get("encoding", {}))
    assert layer["encoding"]["shape"]["scale"]["range"] == list(charts._MOCK_SHAPES)
    assert layer["encoding"]["color"]["legend"]["title"]


def test_an_abandoned_sitting_is_left_off_the_chart() -> None:
    """Nothing was marked, so its zero is not a score that belongs on a line."""
    exams, cover = sittings((1, True, True, 168), (2, False, False, 0))
    rows = plotted(charts.mock_exam_history(exams, cover))
    assert [row["score"] for row in rows] == [168]


def test_one_sitting_still_draws_a_point() -> None:
    """A first result must draw a point, not an empty chart."""
    assert charts.mock_exam_history(*sittings((1, True, True, 168))) is not None


def test_no_sittings_draws_nothing() -> None:
    assert charts.mock_exam_history([], {}) is None


def test_without_coverage_everything_counts_as_a_complete_exam() -> None:
    """With no coverage given, every sitting goes on the complete-exam line."""
    exams, _ = sittings(*ALL_THREE)
    rows = plotted(charts.mock_exam_history(exams))
    assert {row["series"] for row in rows} == {"Complete exam"}


def test_the_time_axis_is_a_date_not_a_sequence_number() -> None:
    spec = json.loads(charts.mock_exam_history(*sittings(*ALL_THREE)).to_json())
    layer = next(lyr for lyr in spec["layer"] if "color" in lyr.get("encoding", {}))
    assert layer["encoding"]["x"]["type"] == "temporal"


def test_the_pass_mark_is_drawn_at_180_on_the_cambridge_scale() -> None:
    spec = json.loads(charts.mock_exam_history(*sittings(*ALL_THREE)).to_json())
    rule = next(lyr for lyr in spec["layer"] if lyr["mark"]["type"] == "rule")
    values = spec["datasets"][rule["data"]["name"]]
    assert values == [{"y": 180}]


def test_the_date_axis_reads_as_whole_days_whatever_the_span() -> None:
    """Vega's default format reads "03 PM" for sittings on one day, "Fri 17" for a week."""
    spec = json.loads(charts.mock_exam_history(*sittings(*ALL_THREE)).to_json())
    x = next(lyr for lyr in spec["layer"] if "color" in lyr.get("encoding", {}))["encoding"]["x"]
    assert x["axis"]["format"] == charts.DATE_FORMAT
    assert x["axis"]["tickCount"] == {"interval": "day", "step": 1}
    assert x["scale"]["nice"] == "day"


def writing_entries(count: int) -> list[dict[str, Any]]:
    return [
        {
            "submitted_at": datetime(2026, 6, 1, tzinfo=UTC) + timedelta(days=n),
            "genre": "essay",
            "word_count": 240,
            "content": 3,
            "communicative_achievement": 3,
            "organisation": 4,
            "language": 3,
            "total": 13,
        }
        for n in range(count)
    ]


def test_writing_submissions_are_counted_not_measured() -> None:
    """A quantitative axis ticks at 1.5 and 2.5: submissions that do not exist."""
    spec = json.loads(charts.writing_subscales(writing_entries(3)).to_json())
    assert spec["encoding"]["x"]["type"] == "ordinal"
    assert sorted({row["n"] for row in plotted(charts.writing_subscales(writing_entries(3)))}) == [1, 2, 3]


def test_a_single_writing_submission_draws_no_line() -> None:
    assert charts.writing_subscales(writing_entries(1)) is None
