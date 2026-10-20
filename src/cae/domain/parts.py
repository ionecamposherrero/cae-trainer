"""The C1 Advanced exam specification: the single source of truth for the paper.

Reading & Use of English: 8 parts, 56 questions, 78 marks, 90 minutes.
Writing: 2 tasks of 220-260 words, 90 minutes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

AnswerKind = Literal["option_letter", "single_word", "word_form", "phrase", "match_letter"]

GAP_TEMPLATE = "{{{{{n}}}}}"


def gap_marker(n: int) -> str:
    """Return the placeholder used for gap ``n`` (1-indexed): ``{{1}}``."""
    return "{{" + str(n) + "}}"


@dataclass(frozen=True)
class PartSpec:
    number: int
    name: str
    item_count: int
    marks_per_item: int
    answer_kind: AnswerKind
    instructions: str
    text_words: tuple[int, int] | None = None
    option_letters: tuple[str, ...] = ()
    focus: tuple[str, ...] = ()
    suggested_minutes: int = 10

    @property
    def max_marks(self) -> int:
        return self.item_count * self.marks_per_item

    @property
    def uses_gapped_text(self) -> bool:
        return self.number in (1, 2, 3, 7)


PART_SPECS: dict[int, PartSpec] = {
    1: PartSpec(
        number=1,
        name="Multiple-choice cloze",
        item_count=8,
        marks_per_item=1,
        answer_kind="option_letter",
        option_letters=("A", "B", "C", "D"),
        text_words=(150, 180),
        suggested_minutes=8,
        instructions=(
            "Read the text below and decide which answer (A, B, C or D) best fits each gap."
        ),
        focus=(
            "vocabulary: precise word choice between near-synonyms",
            "collocation and fixed phrases",
            "phrasal verbs and dependent prepositions",
            "idiomatic expressions",
        ),
    ),
    2: PartSpec(
        number=2,
        name="Open cloze",
        item_count=8,
        marks_per_item=1,
        answer_kind="single_word",
        text_words=(150, 180),
        suggested_minutes=8,
        instructions=(
            "Read the text below and think of the word which best fits each gap. "
            "Use only ONE word in each gap."
        ),
        focus=(
            "grammatical and function words: auxiliaries, articles, pronouns",
            "prepositions in fixed and dependent structures",
            "linkers and referencing devices",
            "parts of comparative, conditional and inversion structures",
        ),
    ),
    3: PartSpec(
        number=3,
        name="Word formation",
        item_count=8,
        marks_per_item=1,
        answer_kind="word_form",
        text_words=(150, 180),
        suggested_minutes=8,
        instructions=(
            "Read the text below. Use the word given in capitals at the end of some of the "
            "lines to form a word that fits in the gap in the same line."
        ),
        focus=(
            "affixation: prefixes and suffixes",
            "negative and opposite forms",
            "changes of word class (noun/verb/adjective/adverb)",
            "internal spelling changes and compound forms",
        ),
    ),
    4: PartSpec(
        number=4,
        name="Key word transformations",
        item_count=6,
        marks_per_item=2,
        answer_kind="phrase",
        text_words=None,
        suggested_minutes=14,
        instructions=(
            "Complete the second sentence so that it has a similar meaning to the first "
            "sentence, using the word given. Do not change the word given. You must use "
            "between three and six words, including the word given."
        ),
        focus=(
            "passive, causative and reporting structures",
            "conditionals, inversion and emphatic structures",
            "phrasal verbs and fixed expressions",
            "comparatives, modals of speculation and deduction",
        ),
    ),
    5: PartSpec(
        number=5,
        name="Multiple choice",
        item_count=6,
        marks_per_item=2,
        answer_kind="option_letter",
        option_letters=("A", "B", "C", "D"),
        text_words=(700, 800),
        suggested_minutes=16,
        instructions=(
            "Read the text and then answer the questions. Choose the answer (A, B, C or D) "
            "which you think fits best according to the text."
        ),
        focus=(
            "detail, opinion, attitude and tone",
            "purpose, main idea and implication",
            "the meaning of a word or phrase in context",
            "text organisation and reference",
        ),
    ),
    6: PartSpec(
        number=6,
        name="Cross-text multiple matching",
        item_count=4,
        marks_per_item=2,
        answer_kind="match_letter",
        option_letters=("A", "B", "C", "D"),
        text_words=(560, 640),
        suggested_minutes=14,
        instructions=(
            "Read the four texts below. For each question, choose the text (A, B, C or D) "
            "which matches the description."
        ),
        focus=(
            "comparing and contrasting opinions across writers",
            "identifying agreement and disagreement",
            "distinguishing attitude from content",
        ),
    ),
    7: PartSpec(
        number=7,
        name="Gapped text",
        item_count=6,
        marks_per_item=2,
        answer_kind="match_letter",
        option_letters=("A", "B", "C", "D", "E", "F", "G"),
        text_words=(500, 600),
        suggested_minutes=16,
        instructions=(
            "You are going to read a text from which six paragraphs have been removed. "
            "Choose from the paragraphs A-G the one which fits each gap. There is one extra "
            "paragraph which you do not need to use."
        ),
        focus=(
            "text structure and cohesion",
            "reference words and lexical chains across paragraph boundaries",
            "logical and chronological development",
        ),
    ),
    8: PartSpec(
        number=8,
        name="Multiple matching",
        item_count=10,
        marks_per_item=1,
        answer_kind="match_letter",
        option_letters=("A", "B", "C", "D", "E", "F"),
        text_words=(600, 750),
        suggested_minutes=14,
        instructions=(
            "You are going to read a text containing six sections. For each question, "
            "choose the section (A-F) in which the information is mentioned."
        ),
        focus=(
            "scanning for specific information",
            "recognising paraphrase of the prompt in the text",
            "distinguishing similar information across sections",
        ),
    ),
}

READING_PART_NUMBERS: tuple[int, ...] = tuple(sorted(PART_SPECS))
READING_MAX_MARKS: int = sum(spec.max_marks for spec in PART_SPECS.values())
READING_TIME_MINUTES: int = 90


def get_part_spec(part: int) -> PartSpec:
    try:
        return PART_SPECS[part]
    except KeyError:
        raise ValueError(
            f"Unknown Reading & Use of English part: {part!r} (expected 1-8)"
        ) from None


WRITING_WORD_RANGE: tuple[int, int] = (220, 260)
WRITING_TIME_MINUTES: int = 90
WRITING_MAX_MARKS_PER_TASK: int = 20

SUBSCALES: tuple[str, ...] = (
    "content",
    "communicative_achievement",
    "organisation",
    "language",
)

SUBSCALE_LABELS: dict[str, str] = {
    "content": "Content",
    "communicative_achievement": "Communicative Achievement",
    "organisation": "Organisation",
    "language": "Language",
}


@dataclass(frozen=True)
class WritingGenre:
    key: str
    name: str
    register: str
    conventions: tuple[str, ...] = field(default_factory=tuple)


WRITING_GENRES: dict[str, WritingGenre] = {
    "essay": WritingGenre(
        key="essay",
        name="Essay",
        register="formal, neutral-academic",
        conventions=(
            "responds to two of the three points given in the input",
            "argues a consistent position and reaches a clear conclusion",
            "no headings; continuous paragraphs with explicit linking",
        ),
    ),
    "letter_email": WritingGenre(
        key="letter_email",
        name="Letter or email",
        register="formal or semi-formal depending on the reader",
        conventions=(
            "opens and closes appropriately for the stated reader",
            "states the purpose early",
            "register held consistently throughout",
        ),
    ),
    "proposal": WritingGenre(
        key="proposal",
        name="Proposal",
        register="formal, persuasive",
        conventions=(
            "section headings",
            "forward-looking: recommends action for a decision-maker",
            "clear recommendation section at the end",
        ),
    ),
    "report": WritingGenre(
        key="report",
        name="Report",
        register="formal, impersonal",
        conventions=(
            "section headings",
            "backward-looking: presents findings on something that happened",
            "factual tone, conclusions drawn from the findings",
        ),
    ),
    "review": WritingGenre(
        key="review",
        name="Review",
        register="semi-formal, engaging",
        conventions=(
            "describes and evaluates, not just describes",
            "addresses a specific publication's readership",
            "ends with a recommendation to the reader",
        ),
    ),
}

WRITING_PART1_GENRES: tuple[str, ...] = ("essay",)
WRITING_PART2_GENRES: tuple[str, ...] = ("letter_email", "proposal", "report", "review")


def get_genre(key: str) -> WritingGenre:
    try:
        return WRITING_GENRES[key]
    except KeyError:
        raise ValueError(f"Unknown writing genre: {key!r}") from None
