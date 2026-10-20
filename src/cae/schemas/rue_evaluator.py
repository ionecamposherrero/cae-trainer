"""Structured-output schemas for the Reading & Use of English evaluator."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, create_model

from cae.domain.taxonomy import ErrorCategory, categories_for_part


class AdjudicationDecision(BaseModel):
    number: int = Field(description="The item number this decision refers to.")
    verdict: Literal["accept", "partial", "reject"] = Field(
        description=(
            "'accept' if the candidate's answer is fully correct English for this gap, "
            "'partial' if it is on the right track but flawed (Part 4 only), "
            "'reject' if it is wrong."
        )
    )
    reason: str = Field(
        description="One sentence explaining the decision, addressed to the learner."
    )


class AdjudicationBatch(BaseModel):
    decisions: list[AdjudicationDecision] = Field(
        description="One decision per item you were asked about, in the same order."
    )


class ErrorTag(BaseModel):
    number: int = Field(description="The item number this tag refers to.")
    category: ErrorCategory = Field(
        description="The single category that best describes why the candidate went wrong."
    )
    diagnosis: str = Field(
        description=(
            "One or two sentences addressed to the learner: what their answer suggests "
            "they were thinking, and what the right answer depends on."
        )
    )


class ErrorTagBatch(BaseModel):
    tags: list[ErrorTag] = Field(description="One tag per item you were asked about.")


@lru_cache(maxsize=None)
def error_tag_batch(part: int) -> type[ErrorTagBatch]:
    """``ErrorTagBatch`` with ``category`` narrowed to the part's own categories.

    The enum is what the model decodes against, so a category the part does not
    allow cannot be produced at all. Only Parts 1-4 are ever sent: a Reading part
    has one fixed category and is diagnosed without the model. An enum rather
    than a ``Literal`` keeps the ``$defs`` shape ``ErrorCategory`` always had,
    which every provider's structured output was checked against.
    """
    allowed = StrEnum(
        f"Part{part}Category", {c.name: c.value for c in categories_for_part(part)}
    )
    tag = create_model(
        f"Part{part}ErrorTag",
        __base__=ErrorTag,
        category=(allowed, Field(description=ErrorTag.model_fields["category"].description)),
    )
    return create_model(
        f"Part{part}ErrorTagBatch",
        __base__=ErrorTagBatch,
        tags=(list[tag], Field(description=ErrorTagBatch.model_fields["tags"].description)),
    )
