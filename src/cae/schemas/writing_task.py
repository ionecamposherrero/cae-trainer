"""Structured-output schema for generating a Writing task."""

from __future__ import annotations

from pydantic import BaseModel, Field


class WritingTaskSpec(BaseModel):
    """A generated exam task: what the candidate reads before writing."""

    title: str = Field(description="A short label for the task, e.g. 'Museum funding essay'.")
    scenario: str = Field(
        description=(
            "The situation, written as the exam writes it: 'Your class has attended a "
            "panel discussion on...', 'You recently read an article in...'."
        )
    )
    instruction: str = Field(
        description=(
            "What the candidate must write, including the reader and the word count "
            "(220-260 words)."
        )
    )
    notes: list[str] = Field(
        description=(
            "The content points the candidate must cover. For a Part 1 essay these are "
            "the three notes, of which the candidate chooses two."
        )
    )
    opinions: list[str] = Field(
        description=(
            "Short quoted opinions overheard in the discussion, as in a Part 1 essay. "
            "Empty list for Part 2 tasks."
        )
    )
    reader: str = Field(description="Who will read the text, e.g. 'your college principal'.")
