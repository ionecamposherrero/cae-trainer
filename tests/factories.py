"""Builders for structurally valid exercises, one per part.

The tests use these two ways round: as-is, to prove a well-formed exercise
passes validation, and deliberately broken, to prove each rule bites.
"""

from __future__ import annotations

from cae.domain.parts import gap_marker
from cae.schemas.rue_task import (
    Part1Exercise,
    Part2Exercise,
    Part3Exercise,
    Part4Exercise,
    Part5Exercise,
    Part6Exercise,
    Part7Exercise,
    Part8Exercise,
)

_VOCAB = (
    "the study of urban green space has revealed how quickly a neglected plot can "
    "recover once planting begins and residents take an interest in its upkeep over "
    "several seasons which researchers describe as a gradual but measurable shift in "
    "local attitudes towards shared land and its long term value for everyone nearby"
).split()


def lorem(n_words: int, offset: int = 0) -> str:
    """Produce exactly ``n_words`` words of filler, broken into sentences."""
    words = [_VOCAB[(i + offset) % len(_VOCAB)] for i in range(n_words)]
    sentences: list[str] = []
    for start in range(0, n_words, 12):
        chunk = words[start : start + 12]
        if not chunk:
            continue
        sentences.append(chunk[0].capitalize() + " " + " ".join(chunk[1:]) + ".")
    return " ".join(sentences)


def gapped_text(total_words: int, gaps: int, offset: int = 0) -> str:
    """Filler of ``total_words`` words with ``gaps`` markers spread through it."""
    body = lorem(total_words, offset).split()
    step = max(2, len(body) // (gaps + 1))
    out: list[str] = []
    placed = 0
    for i, word in enumerate(body):
        out.append(word)
        if placed < gaps and i > 0 and (i + 1) % step == 0:
            placed += 1
            out.append(gap_marker(placed))
    while placed < gaps:
        placed += 1
        out.append(gap_marker(placed))
    return " ".join(out)


def quote_from(text: str, start: int = 3, length: int = 6) -> str:
    """A verbatim slice of ``text``, for the 'evidence' fields."""
    words = [w for w in text.replace("\n", " ").split() if not w.startswith("{{")]
    return " ".join(words[start : start + length])


def part1() -> Part1Exercise:
    return Part1Exercise(
        title="Reclaiming the city",
        topic="urban rewilding",
        text=gapped_text(165, 8),
        items=[
            Part1Item_kwargs(n)
            for n in range(1, 9)
        ],
    )


def Part1Item_kwargs(n: int):  # noqa: N802
    from cae.schemas.rue_task import Part1Item

    letters = ["A", "B", "C", "D"]
    return Part1Item(
        number=n,
        options=[f"option{n}a", f"option{n}b", f"option{n}c", f"option{n}d"],
        answer=letters[n % 4],
        explanation=f"Item {n} tests a fixed collocation.",
    )


def part2() -> Part2Exercise:
    from cae.schemas.rue_task import Part2Item

    return Part2Exercise(
        title="A quiet revolution",
        topic="urban rewilding",
        text=gapped_text(165, 8, offset=5),
        items=[
            Part2Item(
                number=n,
                answer="which",
                accepted=["that"] if n % 2 else [],
                explanation=f"Item {n} tests a relative pronoun.",
            )
            for n in range(1, 9)
        ],
    )


def part3() -> Part3Exercise:
    from cae.schemas.rue_task import Part3Item

    return Part3Exercise(
        title="Growing pains",
        topic="urban rewilding",
        text=gapped_text(165, 8, offset=9),
        items=[
            Part3Item(
                number=n,
                stem="RELY",
                answer="reliable",
                accepted=[],
                explanation=f"Item {n} tests the adjective suffix -able.",
            )
            for n in range(1, 9)
        ],
    )


def part4() -> Part4Exercise:
    from cae.schemas.rue_task import Part4Item

    return Part4Exercise(
        title="Transformations",
        topic="everyday situations",
        items=[
            Part4Item(
                number=n,
                lead_in="I regret not working harder before the test.",
                key_word="WISH",
                sentence_start="I",
                sentence_end="harder before the test.",
                answer="wish I had worked",
                accepted=[],
                explanation="Past regret takes 'wish' plus the past perfect.",
            )
            for n in range(1, 7)
        ],
    )


def part5() -> Part5Exercise:
    from cae.schemas.rue_task import Part5Item

    text = lorem(750)
    letters = ["A", "B", "C", "D"]
    return Part5Exercise(
        title="The long view",
        topic="urban rewilding",
        text=text,
        items=[
            Part5Item(
                number=n,
                question=f"What does the writer suggest in paragraph {n}?",
                options=[f"claim{n}a", f"claim{n}b", f"claim{n}c", f"claim{n}d"],
                answer=letters[n % 4],
                evidence=quote_from(text, start=10 * n),
                explanation=f"Item {n} tests inference.",
            )
            for n in range(1, 7)
        ],
    )


def part6() -> Part6Exercise:
    from cae.schemas.rue_task import Part6Item, Part6Text

    letters = ["A", "B", "C", "D"]
    texts = [
        Part6Text(letter=letter, author=f"Writer {letter}", content=lorem(150, offset=i * 7))
        for i, letter in enumerate(letters)
    ]
    by_letter = {t.letter: t for t in texts}
    return Part6Exercise(
        title="Four views",
        topic="urban rewilding",
        texts=texts,
        items=[
            Part6Item(
                number=n,
                prompt=f"Which writer shares Writer A's view on point {n}?",
                answer=letters[n % 4],
                evidence=quote_from(by_letter[letters[n % 4]].content, start=5 * n),
                explanation=f"Item {n} compares two positions.",
            )
            for n in range(1, 5)
        ],
    )


def part7() -> Part7Exercise:
    from cae.schemas.rue_task import Part7Item, Part7Option

    letters = ["A", "B", "C", "D", "E", "F", "G"]
    options = [
        Part7Option(letter=letter, paragraph=lorem(50, offset=i * 11))
        for i, letter in enumerate(letters)
    ]
    keys = ["C", "A", "F", "B", "G", "E"]
    return Part7Exercise(
        title="Missing pieces",
        topic="urban rewilding",
        text=gapped_text(550, 6, offset=13),
        options=options,
        items=[
            Part7Item(
                number=n,
                answer=keys[n - 1],
                explanation=f"Gap {n} is fixed by the reference word at the start.",
            )
            for n in range(1, 7)
        ],
        distractor_letter="D",
    )


def part8() -> Part8Exercise:
    from cae.schemas.rue_task import Part8Item, Part8Section

    letters = ["A", "B", "C", "D", "E", "F"]
    sections = [
        Part8Section(letter=letter, heading=f"Case {letter}", content=lorem(118, offset=i * 17))
        for i, letter in enumerate(letters)
    ]
    by_letter = {s.letter: s for s in sections}
    keys = [letters[i % 6] for i in range(10)]
    return Part8Exercise(
        title="Six neighbourhoods",
        topic="urban rewilding",
        sections=sections,
        items=[
            Part8Item(
                number=n,
                prompt=f"somebody who eventually abandoned their original plan, point {n}",
                answer=keys[n - 1],
                evidence=quote_from(by_letter[keys[n - 1]].content, start=4 * n),
                explanation=f"Item {n} paraphrases the section.",
            )
            for n in range(1, 11)
        ],
    )


def writing_task(part: int = 1, genre: str = "essay") -> dict:
    """A structurally valid writing task, ready to be broken by a test."""
    if part == 1:
        return {
            "title": "Museum funding essay",
            "scenario": (
                "Your class has attended a panel discussion on how museums and "
                "galleries should be funded in future. You have made the notes below."
            ),
            "instruction": (
                "Write an essay for your tutor discussing two of the points in your "
                "notes. You should explain which method of funding is more effective, "
                "giving reasons in support of your answer. Write 220-260 words."
            ),
            "notes": ["entrance charges", "corporate sponsorship", "public subsidy"],
            "opinions": [
                "Charging at the door keeps out the people who most need to come.",
                "Sponsors bring expertise as well as money.",
                "Public money should go to hospitals before galleries.",
            ],
            "reader": "your tutor",
            "genre": "essay",
            "part": 1,
        }
    named = {
        "letter_email": "letter",
        "proposal": "proposal",
        "report": "report",
        "review": "review",
    }.get(genre, genre.replace("_", " "))
    return {
        "title": f"Sports facilities {named}",
        "scenario": (
            "The council in your town has money available to improve local sports "
            "facilities and has invited residents to put forward suggestions."
        ),
        "instruction": (
            f"Write a {named} for the council explaining which facilities need "
            "improving, why they matter to residents, and what you recommend. "
            "Write 220-260 words."
        ),
        "notes": ["current facilities", "who would benefit", "your recommendation"],
        "opinions": [],
        "reader": "the town council",
        "genre": genre,
        "part": 2,
    }


_SPEC_FIELDS = ("title", "scenario", "instruction", "notes", "opinions", "reader")


def writing_task_spec(part: int = 1, genre: str = "essay", **changes):
    """The same task as a ``WritingTaskSpec``, for stubbing the model."""
    from cae.schemas.writing_task import WritingTaskSpec

    task = {**writing_task(part, genre), **changes}
    return WritingTaskSpec(**{field: task[field] for field in _SPEC_FIELDS})


BUILDERS = {
    1: part1,
    2: part2,
    3: part3,
    4: part4,
    5: part5,
    6: part6,
    7: part7,
    8: part8,
}


def build(part: int):
    return BUILDERS[part]()


def marking_reply(schema, user: str):
    """A complete reply from the grading model: an answer for every item asked about.

    The grader has no fallback, so a stub that leaves an item out fails the
    marking (``MarkingFailed``). Verdicts are "reject" and each tag takes the
    first category the schema allows, so the deterministic marks stand.
    """
    from cae.schemas.rue_evaluator import AdjudicationBatch, ErrorTagBatch

    numbers = [int(line.split()[1]) for line in user.splitlines() if line.startswith("Item ")]
    if issubclass(schema, AdjudicationBatch):
        return schema(
            decisions=[{"number": n, "verdict": "reject", "reason": "Not accepted."} for n in numbers]
        )
    if issubclass(schema, ErrorTagBatch):
        tag = schema.model_fields["tags"].annotation.__args__[0]
        category = next(iter(tag.model_fields["category"].annotation))
        return schema(
            tags=[
                {"number": n, "category": category.value, "diagnosis": "A stubbed diagnosis."}
                for n in numbers
            ]
        )
    raise AssertionError(f"not a grading schema: {schema}")
