"""Progress: the numbers, and a coach's reading of them."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st  # noqa: E402

from cae.db import repo  # noqa: E402
from cae.db.session import session_scope  # noqa: E402
from cae.domain.parts import SUBSCALE_LABELS, SUBSCALES  # noqa: E402
from cae.domain.scoring import scale_score  # noqa: E402
from cae.domain.taxonomy import advice  # noqa: E402
from cae.services import coaching_report, progress  # noqa: E402
from cae.ui.charts import (  # noqa: E402
    accuracy_by_part,
    error_categories,
    mock_exam_history,
    subscale_table,
    writing_subscales,
)
from cae.ui.feedback import SCALE_CAVEAT  # noqa: E402
from cae.ui.shell import (  # noqa: E402
    api_key_banner,
    page_header,
    page_setup,
    require_login,
    sidebar_account,
)

page_setup("Progress", icon=":material/insights:")
user = require_login()
sidebar_account(user)

page_header(
    "Progress",
    "Everything you have been marked on, and what it says about where the marks "
    "are going.",
    eyebrow="Your record",
)

snapshot = progress(user["id"])

if snapshot.activity_count == 0:
    st.info(
        "Nothing to show yet. Complete an exercise on the **Reading and Use of "
        "English** page or a task on the **Writing** page and your record starts here.",
        icon=":material/insights:",
    )
    st.page_link(
        "pages/1_Reading_and_Use_of_English.py",
        label="Go to Reading and Use of English",
        icon=":material/edit:",
    )
    st.stop()


estimated = scale_score(snapshot.overall_accuracy)
cols = st.columns(4)
cols[0].metric("Exercises marked", snapshot.reading_attempts)
cols[1].metric("Writing tasks marked", snapshot.writing_submissions)
cols[2].metric("Overall accuracy", f"{snapshot.overall_accuracy:g}%")
cols[3].metric("Estimated scale score", estimated, delta=f"{estimated - 180:+d} vs pass")
st.caption(SCALE_CAVEAT)


st.divider()
with st.container(key="cae_panel_coach"):
    header = st.columns([4, 1], vertical_alignment="bottom")
    with header[0]:
        st.markdown("### Your coach's review")
    with header[1]:
        refresh = st.button("Rewrite", width="stretch", icon=":material/refresh:")

    if api_key_banner():
        with st.spinner("Reading your record."):
            report = coaching_report(user["id"], user["username"], refresh=refresh)

        narrative = report.get("narrative") or {}
        if report.get("status") == "not_enough_data":
            st.info(
                "Complete a couple more exercises and your coach will have enough to "
                "work with.",
                icon=":material/hourglass:",
            )
        elif not narrative:
            st.warning("The review could not be written this time. Try **Rewrite**.")
        else:
            st.markdown(f"#### {narrative.get('headline', '')}")
            st.markdown(narrative.get("summary", ""))

            left, right = st.columns(2, gap="large")
            with left:
                st.markdown("**What is working**")
                for item in narrative.get("strengths", []):
                    st.markdown(f"- {item}")
            with right:
                st.markdown("**What keeps costing you marks**")
                for item in narrative.get("common_errors", []):
                    st.markdown(f"- {item}")

            st.markdown("**Direction of travel**")
            st.markdown(narrative.get("trend", ""))

            st.markdown("**Your three priorities**")
            priorities = st.columns(max(1, len(narrative.get("focus_areas", []) or [1])))
            for col, area in zip(priorities, narrative.get("focus_areas", [])):
                with col:
                    with st.container(border=True):
                        st.markdown(f"**{area.get('title', '')}**")
                        st.caption(area.get("why", ""))
                        st.markdown(f"→ {area.get('action', '')}")

            if narrative.get("next_session"):
                st.success(f"**Next session:** {narrative['next_session']}")
            if report.get("cached"):
                st.caption("This review is unchanged since your last piece of marked work.")

        st.page_link(
            "pages/5_Talk_with_your_coach.py",
            label="Ask your coach about this",
            icon=":material/forum:",
        )


st.divider()
st.markdown("### Reading and Use of English")

st.markdown("#### Accuracy")

by_part = accuracy_by_part(snapshot.part_stats)
if by_part is not None:
    st.caption(
        "Your accuracy in each part you have attempted, worst at the bottom. The "
        "dashed line is the 60% pass mark. Hover a bar for how many exercises it "
        "rests on — one exercise is a hint, not a verdict."
    )
    st.altair_chart(by_part, width="stretch")
else:
    st.caption("Complete an exercise and your accuracy per part appears here.")


if snapshot.strengths:
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Strongest parts")
        for stat in snapshot.strengths:
            with st.container(border=True):
                st.markdown(f"**Part {stat.part} · {stat.name}**")
                st.caption(
                    f"{stat.accuracy:g}% over {stat.attempts} exercise(s) · "
                    f"last five {stat.recent_accuracy:g}%"
                )
    with right:
        st.markdown("#### Weakest parts")
        for stat in snapshot.weaknesses:
            with st.container(border=True):
                st.markdown(f"**Part {stat.part} · {stat.name}**")
                direction = (
                    "improving" if stat.trend > 0.5
                    else "slipping" if stat.trend < -0.5
                    else "flat"
                )
                st.caption(
                    f"{stat.accuracy:g}% over {stat.attempts} exercise(s) · "
                    f"{direction} ({stat.trend:+g} points per exercise)"
                )


if snapshot.writing_history:
    st.divider()
    st.markdown("### Writing")
    averages = st.columns(4)
    for col, subscale in zip(averages, SUBSCALES):
        col.metric(
            SUBSCALE_LABELS[subscale],
            f"{snapshot.writing_averages.get(subscale, 0):g} / 5",
        )
    subscale_chart = writing_subscales(snapshot.writing_history)
    if subscale_chart is not None:
        st.altair_chart(subscale_chart, width="stretch")
    st.dataframe(
        subscale_table(snapshot.writing_history), hide_index=True, width="stretch"
    )


errors = error_categories(snapshot.top_errors)
if errors is not None:
    st.divider()
    st.markdown("### Your recurring mistakes")
    st.altair_chart(errors, width="stretch")
    for row in snapshot.top_errors[:5]:
        with st.expander(
            f"{row['label']} — {row['count']} time(s), {row['share']:g}% of your errors"
        ):
            st.markdown(advice(row["category"]))
            with session_scope() as session:
                examples = repo.error_examples(session, user["id"], row["category"])
            if examples:
                st.markdown("**Recent examples of what you wrote:**")
                for example in examples:
                    st.markdown(f"- {example}")


with session_scope() as session:
    exams = repo.list_mock_exams(session, user["id"])
    coverage = repo.mock_exam_coverage(session, [exam.id for exam in exams])

st.divider()
st.markdown("### Mock exams")

if not exams:
    st.info(
        "You have not finished a mock exam yet. Sitting one against the clock is "
        "the only figure here that covers a whole paper, so it is the closest "
        "estimate of where you would land on the day.",
        icon=":material/timer:",
    )
    st.page_link(
        "pages/3_Mock_exam.py", label="Sit a mock exam", icon=":material/timer:"
    )
else:
    history = mock_exam_history(exams, coverage)
    if history is not None:
        st.caption(
            "Estimated scale score over time, with a separate line for each kind of "
            "sitting: a paper on its own is not comparable with both together. The "
            "dashed line is the 180 pass mark."
        )
        st.altair_chart(history, width="stretch")
        st.caption(SCALE_CAVEAT)
    st.dataframe(
        [
            {
                "Taken": exam.finished_at,
                "Paper": coverage[exam.id].label if exam.id in coverage else "—",
                "Marks": f"{exam.marks:g} / {exam.max_marks:g}",
                "Estimated scale score": exam.scale_score,
            }
            for exam in exams
        ],
        hide_index=True,
        width="stretch",
    )
