"""Structured-output schema for the personalised coaching report."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FocusArea(BaseModel):
    title: str = Field(description="The area to work on, in a few words.")
    why: str = Field(
        description=(
            "One or two sentences citing the learner's own numbers: which part, how "
            "many marks, which recurring error."
        )
    )
    action: str = Field(
        description="One concrete thing to do this week, specific enough to start today."
    )


class CoachNarrative(BaseModel):
    headline: str = Field(
        description="One sentence summing up where the learner stands right now."
    )
    summary: str = Field(
        description=(
            "A short paragraph on overall performance: accuracy, how close they are to "
            "the 180 pass mark, and how much evidence there is so far."
        )
    )
    strengths: list[str] = Field(
        description="Two or three genuine strengths, each naming the part or skill and the figure that shows it."
    )
    common_errors: list[str] = Field(
        description=(
            "Two to four recurring mistakes, each naming the error category and what it "
            "looks like in practice."
        )
    )
    trend: str = Field(
        description=(
            "Two or three sentences on the direction of travel: what is improving, what "
            "is flat or slipping. Say plainly if there is not yet enough data."
        )
    )
    focus_areas: list[FocusArea] = Field(
        description="Exactly three priorities, most important first."
    )
    next_session: str = Field(
        description="What to practise next and why, in one or two sentences."
    )
