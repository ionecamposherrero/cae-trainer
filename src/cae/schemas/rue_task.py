"""The eight Reading & Use of English exercise formats and their answer key.

Keep these free of ``Optional`` and omitted defaults: Gemini's structured output
handles required, concrete fields far more reliably. Gaps are written as
``{{1}}``, ``{{2}}``, ...
"""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel, Field

ABCD: TypeAlias = Literal["A", "B", "C", "D"]
ATOG: TypeAlias = Literal["A", "B", "C", "D", "E", "F", "G"]
ATOF: TypeAlias = Literal["A", "B", "C", "D", "E", "F"]


class ExerciseBase(BaseModel):
    """Fields every generated exercise carries."""

    title: str = Field(description="A short, engaging title for the passage.")
    topic: str = Field(description="Two or three words naming the subject area.")


class Part1Item(BaseModel):
    number: int = Field(description="Gap number, from 1 to 8.")
    options: list[str] = Field(
        description="Exactly four single-word or short-phrase options, in the order A, B, C, D."
    )
    answer: ABCD = Field(description="The letter of the only correct option.")
    explanation: str = Field(
        description=(
            "One or two sentences: why the key is right and, briefly, why the nearest "
            "distractor is wrong."
        )
    )


class Part1Exercise(ExerciseBase):
    text: str = Field(
        description=(
            "The passage of 150-180 words containing exactly eight gaps written as "
            "{{1}} through {{8}}, in order."
        )
    )
    items: list[Part1Item] = Field(description="Exactly eight items, numbered 1 to 8.")


class Part2Item(BaseModel):
    number: int = Field(description="Gap number, from 1 to 8.")
    answer: str = Field(description="The single word that best fits the gap.")
    accepted: list[str] = Field(
        description=(
            "Other single words that are also fully correct in this gap. Empty list if "
            "the answer is unique."
        )
    )
    explanation: str = Field(
        description="One or two sentences naming the structure the gap tests."
    )


class Part2Exercise(ExerciseBase):
    text: str = Field(
        description=(
            "The passage of 150-180 words containing exactly eight gaps written as "
            "{{1}} through {{8}}, in order."
        )
    )
    items: list[Part2Item] = Field(description="Exactly eight items, numbered 1 to 8.")


class Part3Item(BaseModel):
    number: int = Field(description="Gap number, from 1 to 8.")
    stem: str = Field(
        description="The prompt word shown in capitals, e.g. RELY. Must be a real base form."
    )
    answer: str = Field(
        description="The word formed from the stem that fits the gap, e.g. unreliable."
    )
    accepted: list[str] = Field(
        description="Other forms of the same stem that also fit. Empty list if unique."
    )
    explanation: str = Field(
        description="One or two sentences naming the affix or class change involved."
    )


class Part3Exercise(ExerciseBase):
    text: str = Field(
        description=(
            "The passage of 150-180 words containing exactly eight gaps written as "
            "{{1}} through {{8}}, in order."
        )
    )
    items: list[Part3Item] = Field(description="Exactly eight items, numbered 1 to 8.")


class Part4Item(BaseModel):
    number: int = Field(description="Item number, from 1 to 6.")
    lead_in: str = Field(description="The complete first sentence.")
    key_word: str = Field(
        description="The word in capitals that must appear unchanged in the answer."
    )
    sentence_start: str = Field(
        description="The text of the second sentence before the gap. May be empty."
    )
    sentence_end: str = Field(
        description="The text of the second sentence after the gap, including final punctuation."
    )
    answer: str = Field(
        description=(
            "The words that fill the gap: between three and six words, including the key "
            "word exactly as given."
        )
    )
    accepted: list[str] = Field(
        description="Other three-to-six-word answers that are also correct. Empty list if unique."
    )
    explanation: str = Field(
        description="One or two sentences naming the structure being tested."
    )

    @property
    def full_sentence(self) -> str:
        return f"{self.sentence_start} {self.answer} {self.sentence_end}".strip()


class Part4Exercise(ExerciseBase):
    items: list[Part4Item] = Field(description="Exactly six items, numbered 1 to 6.")


class Part5Item(BaseModel):
    number: int = Field(description="Question number, from 1 to 6.")
    question: str = Field(description="The question stem or sentence opener.")
    options: list[str] = Field(description="Exactly four options, in the order A, B, C, D.")
    answer: ABCD = Field(description="The letter of the only defensible option.")
    evidence: str = Field(
        description="A short verbatim quotation from the passage that proves the key."
    )
    explanation: str = Field(
        description="Why the key follows from that quotation, and why the closest distractor fails."
    )


class Part5Exercise(ExerciseBase):
    text: str = Field(description="A continuous passage of 700-800 words. No gaps.")
    items: list[Part5Item] = Field(
        description="Exactly six items, numbered 1 to 6, following the order of the text."
    )


class Part6Text(BaseModel):
    letter: ABCD = Field(description="The label of this text: A, B, C or D.")
    author: str = Field(description="The writer's name or role, e.g. 'Reviewer B'.")
    content: str = Field(description="This writer's text, about 150 words.")


class Part6Item(BaseModel):
    number: int = Field(description="Question number, from 1 to 4.")
    prompt: str = Field(
        description=(
            "A question of the form 'Which writer shares/differs from X's view on ...?'"
        )
    )
    answer: ABCD = Field(description="The letter of the text that answers the question.")
    evidence: str = Field(description="A short quotation from that text proving the match.")
    explanation: str = Field(
        description="How the two writers' positions compare, and why the others do not match."
    )


class Part6Exercise(ExerciseBase):
    texts: list[Part6Text] = Field(
        description="Exactly four texts labelled A, B, C, D by four different writers on one topic."
    )
    items: list[Part6Item] = Field(description="Exactly four items, numbered 1 to 4.")


class Part7Option(BaseModel):
    letter: ATOG = Field(description="The label of this paragraph, from A to G.")
    paragraph: str = Field(description="A removed paragraph of 40-70 words.")


class Part7Item(BaseModel):
    number: int = Field(description="Gap number, from 1 to 6.")
    answer: ATOG = Field(description="The letter of the paragraph that fills this gap.")
    explanation: str = Field(
        description=(
            "The cohesion evidence: the reference word, pronoun or lexical link that ties "
            "this paragraph to the text before and after the gap."
        )
    )


class Part7Exercise(ExerciseBase):
    text: str = Field(
        description=(
            "The main text of 500-600 words with exactly six gaps written as {{1}} through "
            "{{6}}, each on its own line where a paragraph was removed."
        )
    )
    options: list[Part7Option] = Field(
        description=(
            "Exactly seven paragraphs labelled A to G. Six belong in the gaps; one is a "
            "plausible distractor that fits nowhere."
        )
    )
    items: list[Part7Item] = Field(description="Exactly six items, numbered 1 to 6.")
    distractor_letter: ATOG = Field(
        description="The letter of the paragraph that is not used in any gap."
    )


class Part8Section(BaseModel):
    letter: ATOF = Field(description="The label of this section, from A to F.")
    heading: str = Field(description="A short heading, e.g. a person or place name.")
    content: str = Field(description="This section's text, about 110-125 words.")


class Part8Item(BaseModel):
    number: int = Field(description="Question number, from 1 to 10.")
    prompt: str = Field(
        description=(
            "A statement describing information found in exactly one section, paraphrased "
            "rather than copied from it."
        )
    )
    answer: ATOF = Field(description="The letter of the section containing that information.")
    evidence: str = Field(description="A short quotation from that section proving the match.")
    explanation: str = Field(description="The paraphrase link between the prompt and the text.")


class Part8Exercise(ExerciseBase):
    sections: list[Part8Section] = Field(
        description="Exactly six sections labelled A to F on a shared theme."
    )
    items: list[Part8Item] = Field(
        description=(
            "Exactly ten items, numbered 1 to 10, in a scrambled order relative to the sections."
        )
    )


AnyExercise: TypeAlias = (
    Part1Exercise
    | Part2Exercise
    | Part3Exercise
    | Part4Exercise
    | Part5Exercise
    | Part6Exercise
    | Part7Exercise
    | Part8Exercise
)

EXERCISE_SCHEMAS: dict[int, type[BaseModel]] = {
    1: Part1Exercise,
    2: Part2Exercise,
    3: Part3Exercise,
    4: Part4Exercise,
    5: Part5Exercise,
    6: Part6Exercise,
    7: Part7Exercise,
    8: Part8Exercise,
}


def schema_for(part: int) -> type[BaseModel]:
    try:
        return EXERCISE_SCHEMAS[part]
    except KeyError:
        raise ValueError(f"No exercise schema for part {part!r}") from None


def parse_exercise(part: int, payload: dict) -> AnyExercise:
    """Rebuild a typed exercise from a stored ``payload_json`` blob."""
    return schema_for(part).model_validate(payload)  # type: ignore[return-value]


class ModerationResult(BaseModel):
    """The moderator's verdict on a draft exercise."""

    issues: list[str] = Field(
        description=(
            "One entry per real problem, each naming the item number and what is wrong. "
            "Empty list if the task is sound."
        )
    )
    verdict: Literal["accept", "revise"] = Field(
        description="'accept' if the task is exam-ready, 'revise' otherwise."
    )


class KeyItem(BaseModel):
    """One item's correct answer, normalised across all eight parts."""

    number: int
    answer: str
    accepted: list[str] = Field(default_factory=list)
    explanation: str = ""
    key_word: str = ""
    context: str = ""

    @property
    def variants(self) -> list[str]:
        return [self.answer, *self.accepted]


def _context_for(part: int, item) -> str:
    if part == 4:
        return f"{item.lead_in} [{item.key_word}] {item.sentence_start} ___ {item.sentence_end}"
    if part in (5, 6, 8):
        return getattr(item, "question", None) or getattr(item, "prompt", "")
    return ""


def answer_key(part: int, exercise: AnyExercise) -> list[KeyItem]:
    """Extract a uniform answer key so grading does not branch per part."""
    items = getattr(exercise, "items", [])
    key: list[KeyItem] = []
    for item in items:
        key.append(
            KeyItem(
                number=item.number,
                answer=str(item.answer),
                accepted=[str(a) for a in getattr(item, "accepted", [])],
                explanation=getattr(item, "explanation", ""),
                key_word=str(getattr(item, "key_word", "")),
                context=_context_for(part, item),
            )
        )
    return sorted(key, key=lambda k: k.number)


LETTERS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G")


def options_by_item(exercise: AnyExercise) -> dict[int, dict[str, str]]:
    """The lettered choices per item, for the parts whose items carry them (1 and 5)."""
    by_item: dict[int, dict[str, str]] = {}
    for item in getattr(exercise, "items", []):
        choices = getattr(item, "options", None)
        if choices:
            by_item[item.number] = dict(zip(LETTERS, [str(c) for c in choices]))
    return by_item


def choice_text(part: int, exercise: AnyExercise, number: int, letter: str) -> str:
    """What a letter answer stood for: the option itself in Parts 1 and 5, or the
    writer, paragraph or section it names in Parts 6-8. Empty if it names nothing."""
    letter = (letter or "").strip().upper()
    if not letter:
        return ""
    if part in (1, 5):
        return options_by_item(exercise).get(number, {}).get(letter, "")
    if part == 6:
        named = {t.letter: t.author for t in exercise.texts}
    elif part == 7:
        named = {o.letter: o.paragraph for o in exercise.options}
    elif part == 8:
        named = {s.letter: s.heading for s in exercise.sections}
    else:
        return ""
    return named.get(letter, "")
