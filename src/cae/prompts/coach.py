"""Prompt for turning a learner's statistics into coaching."""

from __future__ import annotations

from cae.prompts import render

COACH_SYSTEM = """You are an experienced Cambridge C1 Advanced tutor writing a \
progress review for one of your students.

You are given their actual figures. Work from those figures and nothing else:
- quote real numbers -- parts, accuracy percentages, error counts;
- never invent a statistic, a trend or a mistake that is not in the data;
- if there is too little data to say something, say so plainly rather than \
guessing. Two attempts at a part is not a trend.

Be direct and encouraging in the way a good teacher is: honest about what is \
weak, specific about what to do, and never vague. Avoid generic advice that \
would apply to any student.

A scale score of 180 is the C1 pass mark; 200 is a Grade A. Write in English, \
addressing the student as 'you'."""

COACH_USER = """Student: <USERNAME>
Work completed: <READING_ATTEMPTS> Reading & Use of English exercises, \
<WRITING_COUNT> writing tasks.
Overall accuracy across all Reading & Use of English work: <OVERALL>%
Estimated scale score at that accuracy: <SCALE> (<GRADE>)

Accuracy by part
-----------------------------------------
<PART_TABLE>

The 'trend' column is the change in accuracy per attempt, in percentage points, \
fitted over that part's history. A blank means too few attempts to fit one.

Most frequent error categories
-----------------------------------------
<ERROR_TABLE>

<EXAMPLES_BLOCK>
Writing subscale averages (0-5)
-----------------------------------------
<WRITING_BLOCK>

Write the progress review."""


def coach_prompt(
    *,
    username: str,
    reading_attempts: int,
    writing_count: int,
    overall: float,
    scale: int,
    grade: str,
    part_table: str,
    error_table: str,
    examples_block: str,
    writing_block: str,
) -> tuple[str, str]:
    return COACH_SYSTEM, render(
        COACH_USER,
        username=username,
        reading_attempts=reading_attempts,
        writing_count=writing_count,
        overall=overall,
        scale=scale,
        grade=grade,
        part_table=part_table,
        error_table=error_table,
        examples_block=examples_block,
        writing_block=writing_block,
    )
