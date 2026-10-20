"""Argument schemas for the coach chat's tools.

The chat replies in prose, so this module holds tool argument schemas rather than
an output model. Every field is required, and none is a ``user_id``: the tools
close over the authenticated learner.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RecordQueryArgs(BaseModel):
    sql: str = Field(
        description=(
            "One SQLite SELECT (a WITH ... SELECT is fine) over the student's record, "
            "using only the tables and columns in the tool description. Aggregate "
            "in SQL with COUNT, AVG and GROUP BY rather than fetching rows to count "
            "them, and add a LIMIT: at most 50 rows come back."
        )
    )


class ReferenceLookupArgs(BaseModel):
    query: str = Field(
        description=(
            "A question about English or about the exam, to look up in the reference "
            "material. Search for the grammar point or exam concept itself, not for "
            "the learner: 'inversion after negative adverbials', not 'my mistakes'."
        )
    )


__all__ = [
    "RecordQueryArgs",
    "ReferenceLookupArgs",
]
