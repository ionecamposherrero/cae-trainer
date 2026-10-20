"""Progress charts.

Magnitudes use a single blue hue; the four writing subscales use a categorical
palette and are always paired with a table.
"""

from __future__ import annotations

from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from cae.domain.parts import SUBSCALE_LABELS, SUBSCALES
from cae.domain.scoring import percentage

PASS_MARK_PCT = 60.0
DATE_FORMAT = "%-d %b %Y"
DATETIME_FORMAT = "%-d %b %Y, %H:%M"

_LIGHT = {
    "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "muted": "#898781",
    "rule": "#d03b3b",
}
_DARK = {
    "series": ["#3987e5", "#d95926", "#199e70", "#c98500"],
    "grid": "#2c2c2a",
    "axis": "#383835",
    "muted": "#898781",
    "rule": "#d03b3b",
}


def palette() -> dict[str, Any]:
    """The palette stepped for whichever theme the viewer is using."""
    try:
        return _DARK if st.context.theme.type == "dark" else _LIGHT
    except Exception:
        return _LIGHT


def _value_axis(title: str | None, colours: dict[str, Any]) -> alt.Axis:
    return alt.Axis(
        title=title,
        grid=True,
        gridColor=colours["grid"],
        gridWidth=1,
        domainColor=colours["axis"],
        tickColor=colours["axis"],
        labelColor=colours["muted"],
        titleColor=colours["muted"],
    )


def _category_axis(title: str | None, colours: dict[str, Any], **extra: Any) -> alt.Axis:
    """A category axis. ``extra`` overrides any default, for label handling.

    Vega truncates a label past ``labelLimit`` (180px by default) and silently
    drops ones that collide, so any axis whose categories have real names has to
    say otherwise — see :func:`accuracy_by_part`.
    """
    return alt.Axis(
        title=title,
        grid=False,
        domainColor=colours["axis"],
        tickColor=colours["axis"],
        labelColor=colours["muted"],
        titleColor=colours["muted"],
        **extra,
    )


_BAR_ROW_HEIGHT = 46


def _named_category_axis(colours: dict[str, Any], **extra: Any) -> alt.Axis:
    """A category axis whose labels are names that must be readable in full."""
    return _category_axis(
        None,
        colours,
        labelLimit=0,
        labelOverlap=False,
        labelFontSize=12,
        labelPadding=8,
        **extra,
    )


def accuracy_by_part(part_stats: list[Any]) -> alt.Chart | None:
    """Where the marks are being lost, part by part."""
    rows = [
        {
            "part": f"Part {s.part} · {s.name}",
            "accuracy": s.accuracy,
            "attempts": s.attempts,
            "marks": s.marks,
            "max_marks": s.max_marks,
        }
        for s in part_stats
        if s.attempts > 0
    ]
    if not rows:
        return None
    colours = palette()
    frame = pd.DataFrame(rows)
    bars = (
        alt.Chart(frame)
        .mark_bar(color=colours["series"][0], cornerRadiusEnd=4, height=24)
        .encode(
            x=alt.X(
                "accuracy:Q",
                scale=alt.Scale(domain=[0, 100]),
                axis=_value_axis("Accuracy (%)", colours),
            ),
            y=alt.Y(
                "part:N",
                sort="-x",
                axis=_named_category_axis(
                    colours,
                    labelExpr="split(datum.label, ' · ')",
                    labelLineHeight=13,
                ),
            ),
            tooltip=[
                alt.Tooltip("part:N", title="Part"),
                alt.Tooltip("accuracy:Q", title="Accuracy (%)", format=".1f"),
                alt.Tooltip("attempts:Q", title="Exercises done"),
                alt.Tooltip("marks:Q", title="Marks earned"),
                alt.Tooltip("max_marks:Q", title="Marks available"),
            ],
        )
    )
    labels = bars.mark_text(
        align="left", dx=6, color=colours["muted"], fontSize=12
    ).encode(text=alt.Text("accuracy:Q", format=".0f"))
    rule = (
        alt.Chart(pd.DataFrame({"x": [PASS_MARK_PCT]}))
        .mark_rule(color=colours["rule"], strokeDash=[6, 4], strokeWidth=1)
        .encode(x="x:Q")
    )
    return (rule + bars + labels).properties(
        height=max(200, _BAR_ROW_HEIGHT * len(rows))
    )


def error_categories(top_errors: list[dict[str, Any]]) -> alt.Chart | None:
    """The mistakes that come up again and again."""
    if not top_errors:
        return None
    colours = palette()
    frame = pd.DataFrame(top_errors)
    bars = (
        alt.Chart(frame)
        .mark_bar(color=colours["series"][0], cornerRadiusEnd=4, height=24)
        .encode(
            x=alt.X("count:Q", axis=_value_axis("Times", colours)),
            y=alt.Y("label:N", sort="-x", axis=_named_category_axis(colours)),
            tooltip=[
                alt.Tooltip("label:N", title="Weakness"),
                alt.Tooltip("count:Q", title="Times"),
                alt.Tooltip("share:Q", title="Share of all errors (%)", format=".1f"),
            ],
        )
    )
    labels = bars.mark_text(
        align="left", dx=6, color=colours["muted"], fontSize=12
    ).encode(text=alt.Text("count:Q"))
    return (bars + labels).properties(
        height=max(200, _BAR_ROW_HEIGHT * len(frame))
    )


def writing_subscales(writing_history: list[dict[str, Any]]) -> alt.Chart | None:
    """The four official subscales over successive submissions."""
    if len(writing_history) < 2:
        return None
    colours = palette()
    rows = []
    for index, entry in enumerate(writing_history, start=1):
        for subscale in SUBSCALES:
            rows.append(
                {
                    "n": index,
                    "subscale": SUBSCALE_LABELS[subscale],
                    "band": entry[subscale],
                    "genre": entry["genre"],
                    "date": entry["submitted_at"],
                }
            )
    frame = pd.DataFrame(rows)
    order = [SUBSCALE_LABELS[s] for s in SUBSCALES]
    return (
        alt.Chart(frame)
        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=60))
        .encode(
            x=alt.X(
                "n:O",
                axis=_category_axis(
                    "Submission (# in the table below)", colours, labelAngle=0
                ),
            ),
            y=alt.Y(
                "band:Q",
                scale=alt.Scale(domain=[0, 5]),
                axis=_value_axis("Band", colours),
            ),
            color=alt.Color(
                "subscale:N",
                sort=order,
                scale=alt.Scale(domain=order, range=colours["series"]),
                legend=alt.Legend(title="Subscale", labelColor=colours["muted"],
                                  titleColor=colours["muted"]),
            ),
            tooltip=[
                alt.Tooltip("n:O", title="#"),
                alt.Tooltip("date:T", title="Submitted", format=DATE_FORMAT),
                alt.Tooltip("subscale:N", title="Subscale"),
                alt.Tooltip("band:Q", title="Band"),
                alt.Tooltip("genre:N", title="Genre"),
            ],
        )
        .properties(height=280)
    )


def subscale_table(writing_history: list[dict[str, Any]]) -> pd.DataFrame:
    """The subscale figures as a table, readable without relying on colour."""
    rows = []
    for index, entry in enumerate(writing_history, start=1):
        row = {"#": index, "Genre": entry["genre"], "Words": entry["word_count"]}
        row.update({SUBSCALE_LABELS[s]: entry[s] for s in SUBSCALES})
        row["Total /20"] = entry["total"]
        rows.append(row)
    return pd.DataFrame(rows)


MOCK_SERIES: tuple[str, ...] = (
    "Complete exam",
    "Reading & UoE only",
    "Writing only",
)
_MOCK_SHAPES: tuple[str, ...] = ("circle", "square", "triangle")


def mock_exam_history(
    exams: list[Any], coverage: dict[int, Any] | None = None
) -> alt.LayerChart | None:
    """Estimated scale score over time, one line per kind of sitting.

    ``coverage`` maps exam id to :class:`cae.db.repo.PaperCoverage`. Without it
    every exam is treated as a complete one.
    """
    rows = []
    for exam in exams:
        paper = coverage.get(exam.id) if coverage else None
        if paper is not None and not paper:
            continue
        rows.append(
            {
                "series": paper.series if paper is not None else MOCK_SERIES[0],
                "score": exam.scale_score or 0,
                "accuracy": percentage(exam.marks, exam.max_marks),
                "marks": exam.marks,
                "max_marks": exam.max_marks,
                "date": exam.finished_at,
            }
        )
    if not rows:
        return None

    colours = palette()
    frame = pd.DataFrame(rows)
    present = [s for s in MOCK_SERIES if s in set(frame["series"])]
    indices = [MOCK_SERIES.index(s) for s in present]

    lines = (
        alt.Chart(frame)
        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=70, filled=True))
        .encode(
            x=alt.X(
                "date:T",
                scale=alt.Scale(nice="day"),
                axis=alt.Axis(
                    title="Date taken",
                    format=DATE_FORMAT,
                    tickCount={"interval": "day", "step": 1},
                    labelAngle=0,
                    labelOverlap=True,
                    grid=True,
                    gridColor=colours["grid"],
                    domainColor=colours["axis"],
                    tickColor=colours["axis"],
                    labelColor=colours["muted"],
                    titleColor=colours["muted"],
                ),
            ),
            y=alt.Y(
                "score:Q",
                scale=alt.Scale(domain=[142, 210]),
                axis=_value_axis("Estimated scale score", colours),
            ),
            color=alt.Color(
                "series:N",
                sort=present,
                scale=alt.Scale(
                    domain=present, range=[colours["series"][i] for i in indices]
                ),
                legend=alt.Legend(
                    title="Sitting",
                    labelColor=colours["muted"],
                    titleColor=colours["muted"],
                    orient="top",
                    direction="horizontal",
                ),
            ),
            shape=alt.Shape(
                "series:N",
                sort=present,
                scale=alt.Scale(
                    domain=present, range=[_MOCK_SHAPES[i] for i in indices]
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("series:N", title="Sitting"),
                alt.Tooltip("score:Q", title="Scale score"),
                alt.Tooltip("accuracy:Q", title="Accuracy (%)", format=".1f"),
                alt.Tooltip("marks:Q", title="Marks"),
                alt.Tooltip("max_marks:Q", title="Out of"),
                alt.Tooltip("date:T", title="Taken", format=DATETIME_FORMAT),
            ],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"y": [180]}))
        .mark_rule(color=colours["rule"], strokeDash=[6, 4], strokeWidth=1)
        .encode(y="y:Q")
    )
    return (rule + lines).properties(height=300)
