"""Prompts for generating Reading & Use of English exercises.

The per-part format contracts repeat the schema's counts in prose on purpose:
it reduces repair rounds.
"""

from __future__ import annotations

from cae.domain.parts import get_part_spec
from cae.prompts import render

SYSTEM = """You are a senior item writer for Cambridge English Qualifications, \
producing authentic practice material for the C1 Advanced (CAE) Reading and Use \
of English paper.

Rules that apply to everything you write:
- Target level is a typical C1 exam, not C2. Write passages in the vocabulary a \
quality newspaper uses for a general readership: mostly common, current words, \
with less frequent ones only where the topic needs them. Avoid rare, literary, \
archaic or specialist words, and idioms a C1 learner would seldom meet. The \
difficulty should come from what each item tests, not from the words around it.
- Use British English spelling and punctuation throughout.
- Exactly one answer must be defensible for every item. If a plausible second \
answer exists, change the item until it does not.
- Distractors must be genuinely tempting: same word class, same register, \
plausible in isolation, wrong only because of the context.
- Write neutral, internationally accessible topics: science, culture, work, \
urban life, nature, technology, history of everyday things. No politics, no \
religion, no real named living individuals, no claims about current events.
- Never reproduce text from published Cambridge papers. Everything must be original.
- Write in continuous prose of the register a quality newspaper or magazine \
would use. Do not address the candidate.
- Explanations are written for the learner: name the rule or the collocation, \
do not merely restate the answer."""


_COMMON_TASK = """Write one complete, exam-ready Part <PART> task: <PART_NAME>.

Topic: <TOPIC>
Level: standard C1, as in a typical exam session
Testing focus for this task: <FOCUS>
<WEAKNESS_HINT>
"""

WEAKNESS_HINT = """
The learner has been repeatedly losing marks on: <WEAKNESSES>. Where the part \
naturally allows it, aim several items at those areas. Do not distort the task \
to force them in."""


_PART_RULES: dict[int, str] = {
    1: """Format contract:
- A passage of 150-180 words with exactly eight gaps.
- Write the gaps as {{1}}, {{2}}, ... {{8}}, in ascending order in the text.
- The first sentence must contain no gap, so the topic is established first.
- Each item has exactly four options in the order A, B, C, D: each option is \
ONE single word, all the same part of speech. To test a phrasal verb or fixed \
phrase, gap only one of its words and leave the rest in the text \
("...{{3}} up with a solution", options "came", "brought", "put", "took").
- Test lexis: collocation, fixed phrases, phrasal verbs, dependent prepositions, \
idiom, and shades of meaning between near-synonyms. Do not test pure grammar.
- Keys and distractors should be words a C1 candidate already knows; the \
challenge is choosing the one that fits the context, not recognising a rare word.
- Spread the keys: do not let one letter dominate.

Worked micro-example of the intended flavour (do not reuse it):
    text: "...the discovery has cast fresh {{1}} on how the birds navigate."
    options: ["light", "glow", "shine", "beam"], answer "A"
    explanation: "'Cast light on' is a fixed collocation meaning 'clarify'. \
'Glow', 'shine' and 'beam' all relate to light but none collocates with 'cast'."
""",
    2: """Format contract:
- A passage of 150-180 words with exactly eight gaps.
- Write the gaps as {{1}}, {{2}}, ... {{8}}, in ascending order in the text.
- The first sentence must contain no gap.
- Every answer is exactly ONE word, and it must be a grammatical or function \
word: auxiliary, article, pronoun, preposition, quantifier, linker, relative, \
or part of a fixed structure. Never a content word that could only be guessed.
- Test C1 structures: inversion, cleft sentences, comparatives, concession, \
relative clauses with prepositions, and parts of fixed expressions.
- If more than one single word genuinely fits, list the alternatives in \
'accepted'; if none does, leave 'accepted' empty.

Worked micro-example of the intended flavour (do not reuse it):
    text: "Little {{1}} the organisers realise how much interest there would be."
    answer: "did", accepted: []
    explanation: "After the negative adverbial 'little' at the start of a clause, \
the subject and auxiliary invert, so the auxiliary 'did' is required."
""",
    3: """Format contract:
- A passage of 150-180 words with exactly eight gaps.
- Write the gaps as {{1}}, {{2}}, ... {{8}}, in ascending order in the text.
- The first sentence must contain no gap.
- Each item supplies a stem word in CAPITALS. The answer is a different word \
built from that stem and it must be exactly one word.
- The stem must be the BASE form of its word family: RELY, LONG, ACCESS, SUCCEED, \
DECIDE. Never give a stem that is already the derived form -- not SUCCESSFUL, not \
RELIABLE, not DECISION -- because then the answer is just that word with a prefix \
bolted on, which is far too easy for C1.
- Cover a range: prefixes and suffixes, negative forms, changes of word class, \
plural or participial forms, and at least two items that require an internal \
spelling change (for example LONG to LENGTH, STRONG to STRENGTH).
- At least two answers should be negative or opposite forms. Even in those, the \
answer must differ from the stem by more than the prefix: SUCCEED to \
'unsuccessful' is right, SUCCESSFUL to 'unsuccessful' is not.

Worked micro-example of the intended flavour (do not reuse it):
    stem: "RELY", answer: "unreliable"
    explanation: "A negative adjective is needed. 'Rely' becomes the adjective \
'reliable', and the prefix 'un-' gives the negative sense the sentence requires."
""",
    4: """Format contract:
- Exactly six independent items. There is no passage.
- Each item has: a complete lead-in sentence; a KEY WORD in capitals; the second \
sentence split into 'sentence_start' and 'sentence_end' with the gap between them.
- The answer fills that gap and must be between three and six words INCLUDING the \
key word, which must appear exactly as given, with no change of form.
- The key word must NOT appear anywhere in the lead-in sentence.
- The two sentences must mean the same thing, with no added or lost information.
- Cover a spread of C1 structures across the six items: passive or causative, \
a conditional or hypothetical form, inversion or emphasis, a modal of \
speculation, a phrasal verb or fixed expression, and a reporting structure.
- List any other three-to-six-word wording that is fully correct in 'accepted'.

Worked micro-example of the intended flavour (do not reuse it):
    lead_in: "I regret not working harder before the exam."
    key_word: "WISH"
    sentence_start: "I", sentence_end: "harder before the exam."
    answer: "wish I had worked"
    explanation: "Regret about the past is expressed with 'wish' plus the past \
perfect, so 'wish I had worked' is required."
""",
    5: """Format contract:
- A single continuous passage of 700-800 words. NO gaps and no gap markers.
- Write it as a magazine feature, a book extract or a long-form article with a \
clear authorial voice, opinions and shifts of attitude, not a neutral report.
- Exactly six questions, in the same order as the information appears in the text.
- Each question has four options A, B, C, D of similar length and grammar.
- Across the six items test a spread of: a specific detail, the writer's attitude \
or opinion, the purpose of a paragraph, an implication that is not stated \
directly, the meaning of a word or phrase in context, and a reference or \
organisational point.
- 'evidence' must be a short quotation copied WORD FOR WORD from the passage \
that proves the key. Never paraphrase in the evidence field.
- Distractors must each be wrong for a nameable reason: true but not the answer \
to the question asked, a partial truth, a plausible inference the text does not \
support, or a statement about the wrong part of the text.
""",
    6: """Format contract:
- Four short texts of about 150 words each, labelled A, B, C, D, by four \
different writers on ONE topic. Give each writer a role, for example \
"Reviewer A" or "Historian C".
- Each writer must take a clear position on the same two or three sub-issues, \
so that they can be compared. Some writers agree with each other on one \
sub-issue and disagree on another. Do not make them all differ on everything.
- Exactly four questions in the form "Which writer ... ?", each asking who shares \
or differs from another named writer's view on a specified sub-issue.
- 'evidence' must be a short quotation copied WORD FOR WORD from the text you \
name as the answer.
""",
    7: """Format contract:
- A main text of 500-600 words from which six paragraphs have been removed.
- Mark each removal with {{1}} through {{6}} on its own line, in ascending order. \
The text must open with at least one full paragraph before {{1}} and close with \
at least one full paragraph after {{6}}.
- Supply exactly seven paragraphs labelled A to G, each 25-100 words. Six of them \
fill the gaps; one is a distractor that fits nowhere.
- Shuffle the labels: the correct sequence must not run alphabetically.
- Every correct paragraph must be provable from cohesion, not from general topic \
match: put a pronoun, a determiner such as "this shift", a time reference or a \
lexical chain in the paragraph that links backwards, and something that links \
forwards to the text after the gap.
- The distractor must be on-topic and superficially plausible, but must contradict \
or duplicate information, or break a reference chain, so it cannot fit any gap.
- Each 'explanation' must name the specific link words that prove the placement.
- Set 'distractor_letter' to the label of the unused paragraph.
""",
    8: """Format contract:
- One text on a shared theme divided into exactly six sections labelled A to F, \
each about 110-125 words with a short heading (a person, a place, a case).
- The sections must be genuinely comparable: each covers a similar set of aspects, \
so that similar-sounding information appears in more than one section.
- Exactly ten questions. Each states information found in exactly ONE section.
- PARAPHRASE every question. A question must never copy five or more consecutive \
words from its section; use synonyms and a different grammatical structure.
- Every section must be the answer to at least one question, and at least three \
sections must answer two or more.
- Scramble the order: the ten answers must not run A, A, B, B, C, C ...
- 'evidence' must be a short quotation copied WORD FOR WORD from the section you \
name as the answer.
""",
}


def generation_prompt(
    part: int,
    *,
    topic: str,
    focus: str,
    weaknesses: list[str] | None = None,
) -> tuple[str, str]:
    """Return ``(system, user)`` for generating a Part ``part`` exercise."""
    spec = get_part_spec(part)
    hint = ""
    if weaknesses:
        hint = render(WEAKNESS_HINT, weaknesses=", ".join(weaknesses))
    user = render(
        _COMMON_TASK,
        part=part,
        part_name=spec.name,
        topic=topic,
        focus=focus,
        weakness_hint=hint,
    )
    return SYSTEM, user + "\n" + _PART_RULES[part]


REPAIR = """The task you produced is not usable yet. Here is what is wrong with it:

<ISSUES>

Produce the COMPLETE corrected task, in the same format. Fix every problem \
listed. Keep everything that was already correct: the topic, the passage and the \
items that were not mentioned should stay as they were, so that the learner sees \
a coherent task. Do not shorten or rewrite the passage unless a problem \
listed requires it.

Here is the task you produced, for reference:

<PREVIOUS>
"""


def repair_prompt(issues: list[str], previous_json: str) -> str:
    numbered = "\n".join(f"{i}. {issue}" for i, issue in enumerate(issues, start=1))
    return render(REPAIR, issues=numbered, previous=previous_json)


CRITIQUE_SYSTEM = """You are a Cambridge English item moderator. Your job is to \
reject items that would not survive pretesting. You are sceptical and specific.

Check, for every item:
1. Is the key genuinely provable from the text as written?
2. Could a well-prepared C1 candidate defend a second option? If so the item fails.
3. Is any distractor so obviously wrong that the item gives itself away?
4. Does the stated evidence quotation actually appear in the text and support the key?
5. For matching tasks: is the information unique to the section or writer named?

Report only real, item-level problems. If the task is sound, return an empty \
list of issues -- do not invent faults to seem useful. Never rewrite the task; \
just say what is wrong, naming the item number, so it can be fixed."""

CRITIQUE_USER = """Moderate this C1 Advanced Part <PART> task (<PART_NAME>).

<TASK_JSON>
"""


def critique_prompt(part: int, task_json: str) -> tuple[str, str]:
    spec = get_part_spec(part)
    return CRITIQUE_SYSTEM, render(
        CRITIQUE_USER, part=part, part_name=spec.name, task_json=task_json
    )


TOPICS: tuple[str, ...] = (
    "urban rewilding and city green space",
    "the science of sleep",
    "the economics of second-hand clothing",
    "how museums choose what to display",
    "deep-sea exploration",
    "the revival of regional languages",
    "food preservation through history",
    "the psychology of queuing",
    "wind and tidal energy",
    "the design of public libraries",
    "long-distance migration in insects",
    "the rise of the four-day week",
    "cartography and how maps mislead",
    "amateur astronomy",
    "the restoration of historic buildings",
    "why some films become cult classics",
    "the chemistry of coffee",
    "volunteering and civic life",
    "acoustics in concert halls",
    "the history of the bicycle",
    "artificial light and the night sky",
    "seed banks and crop diversity",
    "the etiquette of open-plan offices",
    "handwriting in a digital age",
    "urban beekeeping",
    "the business of live music",
    "weather forecasting before satellites",
    "how languages borrow words",
    "the ethics of wildlife photography",
    "commuting and how it shapes cities",
)
