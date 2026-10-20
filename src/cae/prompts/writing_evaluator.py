"""Prompts for assessing a Writing script.

Band descriptors follow the Cambridge Writing Assessment Scale at C1; each
subscale is assessed by its own prompt.
"""

from __future__ import annotations

from cae.domain.parts import WRITING_WORD_RANGE, get_genre
from cae.domain.taxonomy import WRITING_CATEGORIES, guide, label
from cae.prompts import render

MIN_WORDS, MAX_WORDS = WRITING_WORD_RANGE


ASSESS_SYSTEM = """You are a trained Cambridge English examiner marking a C1 \
Advanced Writing script. You mark ONE subscale only: <SUBSCALE>. Ignore the \
other three entirely -- do not let strong language pull up a Content band, or \
weak organisation pull down a Language band.

Award a band from 0 to 5 using these descriptors:

<DESCRIPTORS>

Bands 2 and 4 are awarded when the performance shares features of the bands \
above and below. Band 0 is for a performance below band 1.

Mark to the standard of the real exam. Most C1 candidates who pass score 3 on \
most subscales; a 5 is genuinely impressive and rare. Do not inflate.

Every quotation you give as evidence must be copied word for word from the \
candidate's script. Address the learner as 'you'."""

DESCRIPTORS: dict[str, str] = {
    "content": """Band 5: All content is relevant to the task. The target reader is fully informed.
Band 3: Minor irrelevances and/or omissions may be present. The target reader is on the whole informed.
Band 1: Irrelevances and misinterpretation of the task may be present. The target reader is minimally informed.
Band 0: Content is totally irrelevant. The target reader is not informed.

For a Part 1 essay, the candidate must discuss exactly TWO of the three points \
and say which is more important. Discussing three points thinly, or only one, is \
an omission. For a Part 2 task, every content point in the instruction must be \
addressed.""",
    "communicative_achievement": """Band 5: Uses the conventions of the communicative task with sophistication to hold the target reader's attention and communicate straightforward and complex ideas, as appropriate.
Band 3: Uses the conventions of the communicative task effectively to hold the target reader's attention and communicate straightforward and complex ideas, as appropriate.
Band 1: Uses the conventions of the communicative task to hold the target reader's attention and communicate straightforward ideas.
Band 0: Performance below band 1.

Judge register, genre conventions and the effect on the named reader. A report \
without headings, a review that only describes, a proposal that never \
recommends, or an essay that slides into informal speech all lose marks here.""",
    "organisation": """Band 5: The text is organised impressively and coherently, using a wide range of cohesive devices and organisational patterns with complete flexibility.
Band 3: The text is well organised and coherent, using a variety of cohesive devices and organisational patterns to generally good effect.
Band 1: The text is generally well organised and coherent, using a variety of linking words and cohesive devices.
Band 0: Performance below band 1.

Judge paragraphing, the logic of the sequence, and the range and control of \
cohesive devices, including referencing and substitution, not just linkers. \
Repeating the same three linkers throughout caps this subscale at 3.""",
    "language": """Band 5: Uses a range of vocabulary, including less common lexis, effectively and precisely. Uses a wide range of simple and complex grammatical forms with full control, flexibility and sophistication. Errors, if present, are related to less common words and structures, or occur as slips.
Band 3: Uses a range of vocabulary, including less common lexis, appropriately. Uses a range of simple and complex grammatical forms with control and flexibility. Occasional errors may be present but do not impede communication.
Band 1: Uses a range of everyday vocabulary appropriately, with occasional inappropriate use of less common lexis. Uses a range of simple and some complex grammatical forms with a good degree of control. Errors do not impede communication.
Band 0: Performance below band 1.

Judge range first, then control. A script with no errors but only simple \
structures is not a 5. Errors that impede communication cap this subscale at 2.""",
}

ASSESS_USER = """Task the candidate was given
-----------------------------------------
Genre: <GENRE_NAME> (Writing Part <PART>)
Reader: <READER>
Scenario: <SCENARIO>
Instruction: <INSTRUCTION>
Content points: <NOTES>
<OPINIONS_BLOCK>

The candidate's script
-----------------------------------------
<SCRIPT>
"""


def assess_prompt(
    subscale: str,
    *,
    part: int,
    genre: str,
    task: dict,
    script: str,
) -> tuple[str, str]:
    """The prompts for one subscale.

    The script's length is deliberately absent: only a script within the
    required range is assessed, and within it length says nothing about any
    subscale.
    """
    spec = get_genre(genre)
    opinions = task.get("opinions") or []
    opinions_block = ""
    if opinions:
        opinions_block = "Opinions expressed in the discussion:\n" + "\n".join(
            f'  - "{o}"' for o in opinions
        )
    system = render(
        ASSESS_SYSTEM,
        subscale=label_subscale(subscale),
        descriptors=DESCRIPTORS[subscale],
    )
    user = render(
        ASSESS_USER,
        genre_name=spec.name,
        part=part,
        reader=task.get("reader", ""),
        scenario=task.get("scenario", ""),
        instruction=task.get("instruction", ""),
        notes="; ".join(task.get("notes") or []),
        opinions_block=opinions_block,
        script=script,
    )
    return system, user


def label_subscale(subscale: str) -> str:
    return subscale.replace("_", " ").title()


ANNOTATE_SYSTEM = """You are a Cambridge C1 Advanced examiner marking up a \
candidate's script for feedback.

Find the language errors that actually cost marks. For each one:
- copy the wrong stretch of text WORD FOR WORD from the script, exactly as the \
candidate wrote it, including their spelling. Keep the span short: a few words, \
never a whole paragraph. If you cannot copy it exactly, leave it out.
- correct it minimally: change what is wrong and nothing else;
- classify it with one of the categories listed below;
- explain the rule in one sentence, addressed to the learner as 'you'.

Do not flag valid British English as an error. Do not flag stylistic preferences \
as errors: reserve those for the subscale feedback. Report at most twelve errors, \
choosing the most costly ones. Also list two or three genuine strengths, quoted \
from the script."""

ANNOTATE_USER = """Categories to choose from:
<CATEGORIES>

The task was a <GENRE_NAME> for this reader: <READER>. Expected register: <REGISTER>.

The candidate's script
-----------------------------------------
<SCRIPT>
"""


def annotate_prompt(genre: str, script: str) -> tuple[str, str]:
    spec = get_genre(genre)
    categories = "\n".join(
        f"- {c.value}: {label(c)}. Use when {guide(c)}." for c in WRITING_CATEGORIES
    )
    return ANNOTATE_SYSTEM, render(
        ANNOTATE_USER,
        categories=categories,
        genre_name=spec.name,
        reader="the reader named in the task",
        register=spec.register,
        script=script,
    )


IMPROVE_SYSTEM = """You are a C1 Advanced writing tutor. You will rewrite a \
candidate's script so they can see their own text at band 5.

Rules:
- Keep THEIR ideas, THEIR argument, THEIR paragraph structure and THEIR voice. \
This is their essay improved, not your essay.
- Fix the errors, raise the lexis to precise less-common vocabulary, and vary the \
grammar with the complex structures C1 rewards.
- Stay within <MIN_WORDS>-<MAX_WORDS> words.
- Write plain text, as the candidate would on the exam paper: no Markdown, no # \
or * symbols. If the genre has headings, put each on its own line; separate \
paragraphs with a blank line.
- Then pick four to six of the changes that teach the most, and for each give the \
candidate's original phrase word for word, your version, and what the change gains."""

IMPROVE_USER = """Genre: <GENRE_NAME>. Register: <REGISTER>.
Weakest subscale to prioritise: <WEAKEST>.

The candidate's script
-----------------------------------------
<SCRIPT>
"""


def improve_prompt(genre: str, script: str, weakest: str) -> tuple[str, str]:
    spec = get_genre(genre)
    system = render(IMPROVE_SYSTEM, min_words=MIN_WORDS, max_words=MAX_WORDS)
    user = render(
        IMPROVE_USER,
        genre_name=spec.name,
        register=spec.register,
        weakest=label_subscale(weakest),
        script=script,
    )
    return system, user
