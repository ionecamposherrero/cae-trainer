"""Prompts for generating Writing tasks."""

from __future__ import annotations

from cae.domain.parts import WRITING_WORD_RANGE, get_genre
from cae.prompts import render

MIN_WORDS, MAX_WORDS = WRITING_WORD_RANGE


TASK_SYSTEM = """You are a senior item writer for Cambridge English Qualifications, \
writing tasks for the C1 Advanced Writing paper.

A good task:
- gives the candidate a concrete situation, a named reader and a clear purpose;
- supplies content points that require argument, not just description;
- is answerable in 220-260 words by a candidate who knows nothing special about \
the topic;
- is worded in clear, everyday English, so it tests the candidate's writing, \
not their reading;
- is neutral and internationally accessible: no politics, no religion, no real \
named living individuals, no current events.

Write in British English. Use the wording conventions of the real paper."""

TASK_PART1 = """Write one Part 1 task: the compulsory essay.

Topic area: <TOPIC>

Format contract:
- 'scenario' follows the exam formula: the class has attended a talk, panel \
discussion or seminar, and the candidate has made notes.
- 'notes' contains exactly three content points, each two to four words, phrased \
as aspects of the question (for example "cost to the public").
- 'opinions' contains exactly three short quoted remarks overheard in the \
discussion, each one sentence, each clearly attached to one of the three notes \
and each taking a definite position.
- 'instruction' tells the candidate to write an essay for their tutor discussing \
TWO of the three points, explaining which is more important and why, in \
<MIN_WORDS>-<MAX_WORDS> words.
- 'reader' is the candidate's tutor."""

TASK_PART2 = """Write one Part 2 task in this genre: <GENRE_NAME>.

Topic area: <TOPIC>
Expected register: <REGISTER>
Conventions of the genre: <CONVENTIONS>

Format contract:
- 'scenario' establishes a specific, believable situation that makes this genre \
the natural thing to write, and names the reader.
- 'notes' contains three or four content points the candidate must address, \
phrased as the exam phrases them, inside the scenario or the instruction.
- 'opinions' is an empty list: only Part 1 essays use quoted opinions.
- 'instruction' names the genre explicitly, names the reader and gives the word \
count as <MIN_WORDS>-<MAX_WORDS> words.
- 'reader' is the specific person or group who will read the text."""


def task_prompt(part: int, genre: str, topic: str) -> tuple[str, str]:
    if part == 1:
        user = render(TASK_PART1, topic=topic, min_words=MIN_WORDS, max_words=MAX_WORDS)
        return TASK_SYSTEM, user
    spec = get_genre(genre)
    user = render(
        TASK_PART2,
        genre_name=spec.name,
        topic=topic,
        register=spec.register,
        conventions="; ".join(spec.conventions),
        min_words=MIN_WORDS,
        max_words=MAX_WORDS,
    )
    return TASK_SYSTEM, user


TASK_REPAIR = """The task you produced is not usable yet. Here is what is wrong \
with it:

<ISSUES>

Produce the COMPLETE corrected task, in the same format. Fix every problem listed \
and keep everything that was already right: the situation, the reader and any \
content point that was not mentioned should stay as they were, so the task stays \
coherent.

Here is the task you produced, for reference:

<PREVIOUS>
"""


def task_repair_prompt(issues: list[str], previous_json: str) -> str:
    numbered = "\n".join(f"{i}. {issue}" for i, issue in enumerate(issues, start=1))
    return render(TASK_REPAIR, issues=numbered, previous=previous_json)


TASK_TOPICS: tuple[str, ...] = (
    "public transport in growing cities",
    "how museums and galleries should be funded",
    "the value of learning a second language as an adult",
    "working from home and community life",
    "sport facilities in local neighbourhoods",
    "protecting historic buildings versus building new housing",
    "the role of libraries in the digital age",
    "young people and voluntary work",
    "tourism and its effect on small towns",
    "how schools should teach practical life skills",
    "advertising aimed at children",
    "the arts in the school curriculum",
    "food waste in supermarkets and restaurants",
    "green spaces in city centres",
    "the impact of streaming on live entertainment",
    "lifelong learning and career change",
)
