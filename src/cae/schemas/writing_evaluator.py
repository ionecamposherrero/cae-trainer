"""Structured-output schemas for assessing a Writing script."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from cae.domain.taxonomy import WRITING_CATEGORIES

Band = Literal[0, 1, 2, 3, 4, 5]

WritingCategory = StrEnum("WritingCategory", {c.name: c.value for c in WRITING_CATEGORIES})


class SubscaleAssessment(BaseModel):
    """One of the four official subscales."""

    band: Band = Field(description="The band from 0 to 5 awarded on this subscale.")
    justification: str = Field(
        description=(
            "Two or three sentences addressed to the learner, in English, saying why "
            "this band and not the one above."
        )
    )
    evidence: list[str] = Field(
        description=(
            "Two or three short quotations copied word for word from the candidate's "
            "text that support the band."
        )
    )
    to_improve: str = Field(
        description="One concrete, actionable thing that would earn the next band up."
    )


class Annotation(BaseModel):
    """One error located in the candidate's own words."""

    span: str = Field(
        description=(
            "The exact stretch of the candidate's text that is wrong, copied word for "
            "word so it can be located. Keep it short: a few words."
        )
    )
    category: WritingCategory = Field(description="The category this error belongs to.")
    correction: str = Field(description="The corrected version of that span.")
    explanation: str = Field(
        description="One sentence, addressed to the learner, naming the rule involved."
    )
    severity: Literal["slip", "error", "impedes"] = Field(
        description=(
            "'slip' for a careless one-off, 'error' for a real gap in knowledge, "
            "'impedes' when the reader would struggle to understand."
        )
    )


class AnnotationSet(BaseModel):
    annotations: list[Annotation] = Field(
        description=(
            "Every language error worth showing, in the order they appear. Twelve at most: "
            "prioritise the ones that cost marks."
        )
    )
    strengths: list[str] = Field(
        description="Two or three things the candidate did well, quoted from the text."
    )


class Upgrade(BaseModel):
    original: str = Field(description="A phrase from the candidate's text, word for word.")
    improved: str = Field(description="The stronger C1 version used in the rewrite.")
    why: str = Field(description="One sentence on what the change gains.")


class ImprovedVersion(BaseModel):
    text: str = Field(
        description=(
            "The candidate's own text rewritten at a strong band 5, keeping their ideas, "
            "their structure and their voice. 220-260 words. Plain text, no Markdown."
        )
    )
    upgrades: list[Upgrade] = Field(
        description="Four to six of the most instructive changes made in the rewrite."
    )
