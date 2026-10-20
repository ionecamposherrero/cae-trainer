"""Prompts for the Reading & Use of English evaluator.

Adjudicating near-miss answers, and tagging each dropped mark with a category.
"""

from __future__ import annotations

from cae.domain.parts import get_part_spec
from cae.domain.taxonomy import categories_for_part, guide, label
from cae.prompts import render

ADJUDICATE_SYSTEM = """You are a Cambridge C1 Advanced examiner marking Use of \
English answers that are not on the published key.

Published keys are not exhaustive. Your job is to decide whether the candidate's \
answer is nonetheless fully correct: grammatical, natural, the right register, \
and giving exactly the meaning the task requires.

Be strict but fair.
- Accept an answer that a native speaker would use naturally in that gap.
- Reject an answer that is grammatical but changes the meaning, or that is \
unnatural, or that is the wrong word class.
- Reject spelling mistakes; the exam does not accept them.
- For Part 4 only, use 'partial' when the structure is right but the wording is \
incomplete or slightly wrong, which would earn one of the two marks.
- For Parts 2 and 3 there are no half marks: use 'accept' or 'reject' only."""

ADJUDICATE_USER = """Part <PART>: <PART_NAME>.

Decide on each of these answers. For each item you are given the context, the \
published key, and what the candidate wrote.

<ITEMS>
"""


def adjudicate_prompt(part: int, items_block: str) -> tuple[str, str]:
    spec = get_part_spec(part)
    return ADJUDICATE_SYSTEM, render(
        ADJUDICATE_USER, part=part, part_name=spec.name, items=items_block
    )


TAG_SYSTEM = """You are a Cambridge C1 Advanced examiner writing diagnostic \
feedback. For each wrong answer, decide which single category of weakness it \
reveals, and explain it in a way that helps the learner avoid the mistake next \
time.

Pick the category that describes the CANDIDATE'S underlying gap, not merely the \
topic of the item. If a candidate chose a word that means the right thing but \
does not collocate, that is collocation, not word choice.

Write the diagnosis in English, addressed to the learner as 'you', in one or two \
sentences. Name the rule, pattern or collocation involved. Do not simply repeat \
the correct answer."""

TAG_USER = """Part <PART>: <PART_NAME>.

Choose each category from this list, which is the set that applies to this \
part. Each line says when to use it:
<CATEGORIES>

Tag every one of these items:

<ITEMS>
"""


def tag_prompt(part: int, items_block: str) -> tuple[str, str]:
    """Only Parts 1-4 are tagged by the model; a Reading part's category is fixed."""
    spec = get_part_spec(part)
    listed = "\n".join(
        f"- {c.value}: {label(c)}. Use when {guide(c)}." for c in categories_for_part(part)
    )
    return TAG_SYSTEM, render(
        TAG_USER,
        part=part,
        part_name=spec.name,
        categories=listed,
        items=items_block,
    )
