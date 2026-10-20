"""A paid judge for the chat and coach suites: one OpenAI model, eval-only, outside ``cae.llm``.

The app runs on free tiers only, and ``cae.config`` refuses any route a free
provider does not serve, so an OpenAI model cannot sit in a tier -- and should
not, since a learner's chat must never depend on a paid key. The judge only
reads replies the suite has already collected, so it lives here, with its own
key and its own single model.

There is no fallback chain, on purpose and for the reason embeddings have none:
scores from two judges are not comparable, so a judge that changed model halfway
through a run would mix two scales. A judge that cannot answer leaves the turn
unjudged instead.

It is the chat suite's only measure of whether a reply is right
(``correctness``), true to what the tutor had (``faithfulness``) and of use to
the learner (``helpfulness``). The deterministic metrics beside it count only
what can be counted exactly -- tools, rounds, refused queries, sections.

The coach suite has nothing beside it: the coach's report is all prose. The
same judge marks each report on the same three scales (``CoachVerdict``),
against a reference review and the very figures and tables ``narrate`` was
handed.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from cae.config import PROJECT_ROOT
from cae.prompts import render

MAX_EVIDENCE_CHARS = 12_000


class JudgeSettings(BaseSettings):
    """The judge's key and model, read from the same ``.env`` as the app's."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    openai_api_key: str = ""
    cae_judge_model: str = "gpt-6.1-sol"
    cae_judge_temperature: str = ""


class JudgeUnavailableError(RuntimeError):
    """No key for the judge, or the judge's package is not installed."""


class ChatVerdict(BaseModel):
    """The judge's verdict on one reply. No optional fields: strict schemas refuse them."""

    correctness: Literal[0, 1, 2] = Field(
        description=(
            "2: the reply says everything the reference answer requires, in any wording "
            "(rounding a figure to a whole number is fine). 1: part of it. 0: none of it, "
            "or something that contradicts it."
        )
    )
    faithfulness: Literal[0, 1, 2] = Field(
        description=(
            "2: every claim about the student or the exam is supported by the evidence. "
            "1: a minor detail is not. 0: an invented figure, fact or quotation."
        )
    )
    helpfulness: Literal[1, 2, 3, 4, 5] = Field(
        description=(
            "As a tutor's answer: 5 answers exactly what was asked, briefly, and gives "
            "the student something concrete to do; 3 answers it but vaguely or at "
            "length; 1 does not answer it."
        )
    )
    unsupported_claims: list[str] = Field(
        description="Each claim the evidence does not support, quoted briefly. Empty if none."
    )
    rationale: str = Field(description="Why, in two sentences at most.")


class CoachVerdict(BaseModel):
    """The judge's verdict on one progress review, on ``ChatVerdict``'s scales so the
    two suites read alike. No optional fields: strict schemas refuse them."""

    correctness: Literal[0, 1, 2] = Field(
        description=(
            "2: the review gives the same picture as the reference -- where the student "
            "stands, what is rising or slipping (or that there is too little data to "
            "say), and it puts first what the reference puts first. 1: part of it. "
            "0: none of it, or something that contradicts it."
        )
    )
    faithfulness: Literal[0, 1, 2] = Field(
        description=(
            "2: every claim about the student or the exam is supported by the evidence. "
            "1: a minor detail is not. 0: an invented figure, fact, trend or quotation."
        )
    )
    helpfulness: Literal[1, 2, 3, 4, 5] = Field(
        description=(
            "As a tutor's review: 5 is specific to this student, honest about what is "
            "weak, and gives them something concrete to start today; 3 is accurate but "
            "vague, generic or long; 1 would not help this student."
        )
    )
    unsupported_claims: list[str] = Field(
        description="Each claim the evidence does not support, quoted briefly. Empty if none."
    )
    rationale: str = Field(description="Why, in two sentences at most.")


JUDGE_SYSTEM = """You check the replies of an AI tutor who helps one student prepare \
for the Cambridge C1 Advanced exam.

For each reply you are given the student's question, a reference answer, the \
evidence the tutor had, and the reply.

The reference answer is one good reply, not a checklist. A reply is correct when \
it gives what the question asks for, as the reference does; leaving out a \
supporting detail the reference adds (how many exercises a figure covers, a \
percentage beside a count) does not make it less correct.

The evidence is all the tutor could know: the facts its instructions state, the \
student's record as it was shown to the tutor, the rows its own database queries \
returned, the passages about the exam it retrieved, and, in a conversation, what \
was said before. A figure or fact in the reply that is not in the evidence is \
unsupported, however plausible. Advice and explanations of English are not \
claims about the student or the exam: judge them as a tutor's, not as facts \
needing evidence.

The tutor is told never to show the student SQL, table names or its sources: do \
not penalise it for leaving them out. Judge only the reply, against the \
reference answer and the evidence."""


def tutor_facts() -> str:
    """What the tutor's instructions and the app's tables state, so a reply
    citing them is not marked unsupported: the grade boundaries, the maximum
    each mark is out of, and the app's pages it is told to send the student to."""
    from cae.domain.parts import PART_SPECS, READING_MAX_MARKS, WRITING_MAX_MARKS_PER_TASK
    from cae.domain.scoring import GRADE_THRESHOLDS, SCALE_MAX, grade_for

    grades = ", ".join(
        f"{grade_for(t)[0]} from {t}" for t in GRADE_THRESHOLDS if t != SCALE_MAX
    )
    parts = ", ".join(f"Part {p} out of {s.max_marks}" for p, s in PART_SPECS.items())
    return (
        f"The Cambridge English Scale runs to {SCALE_MAX}; 180 is the C1 pass mark "
        f"({grades}). Reading and Use of English is out of {READING_MAX_MARKS} ({parts}). "
        f"A Writing task is out of {WRITING_MAX_MARKS_PER_TASK}: four subscales, "
        "each banded 0 to 5. For practice, the tutor sends the student to the app's "
        "pages: the Reading and Use of English page (any of Parts 1 to 8), the "
        "Writing page (an essay or a Part 2 task) and the Mock exam page (a full "
        "timed paper)."
    )


JUDGE_USER = """QUESTION
<QUESTION>

REFERENCE ANSWER
<REFERENCE>

EVIDENCE
<EVIDENCE>

THE TUTOR'S REPLY
<REPLY>"""


COACH_JUDGE_SYSTEM = """You check the progress reviews an AI tutor writes for one \
student preparing for the Cambridge C1 Advanced exam.

For each review you are given a reference review, the evidence the tutor had, \
and the review.

The reference is one good review, not a checklist. A review is correct when it \
gives the same picture of the student as the reference: where they stand, what is \
rising or slipping -- or, with too little data, that it is too early to say -- and \
what should come first. Wording, order and extra true detail do not matter; \
leaving out a supporting figure the reference adds does not make it less correct. \
When the reference names more than one acceptable first priority, any of them is \
right.

The evidence is all the tutor could know: the facts its instructions state and \
the student's figures and tables exactly as they were shown to it. A figure, \
trend or fact in the review that is not in the evidence is unsupported, however \
plausible -- an estimate of marks to be gained, a share of the exam, a link. \
Advice, targets the review sets the student and explanations of English are not \
claims about the student or the exam: judge them as a tutor's, not as facts \
needing evidence.

Judge only the review, against the reference and the evidence."""

COACH_JUDGE_USER = """REFERENCE REVIEW
<REFERENCE>

EVIDENCE
<EVIDENCE>

THE TUTOR'S REVIEW
<REVIEW>"""


def build_judge(settings: JudgeSettings | None = None, schema: type[BaseModel] | None = None):
    """The judge model with ``schema`` (``ChatVerdict`` by default; ``CoachVerdict``
    for the coach) bound as strict structured output."""
    settings = settings or JudgeSettings()
    if not settings.openai_api_key.strip():
        raise JudgeUnavailableError(
            "The judge needs OPENAI_API_KEY in .env (it is paid, and used by the evals only)."
        )
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover
        raise JudgeUnavailableError(
            "The judge needs langchain-openai: pip install -e .[evals]"
        ) from exc

    extra: dict[str, Any] = {}
    if settings.cae_judge_temperature.strip():
        extra["temperature"] = float(settings.cae_judge_temperature)
    model = ChatOpenAI(
        model=settings.cae_judge_model,
        api_key=settings.openai_api_key,
        max_retries=2,
        timeout=120,
        **extra,
    )
    return model.with_structured_output(
        schema or ChatVerdict, method="json_schema", strict=True
    )


def judge_evidence(
    case: Mapping[str, Any],
    learner: Mapping[str, Any],
    earlier: Sequence[Mapping[str, Any]] = (),
) -> str:
    """Everything the tutor had for this turn, as the judge reads it.

    The facts its instructions state, the learner's figures and tables as the
    system message showed them, what was said before in a dialogue
    (``earlier``), the rows its queries returned and the passages it retrieved.
    Leaving any of it out would mark a reply unsupported for citing what the
    tutor was in fact given. Cut at :data:`MAX_EVIDENCE_CHARS`.
    """
    figures = learner.get("figures") or {}
    sections = [
        ("WHAT THE TUTOR'S INSTRUCTIONS STATE", tutor_facts()),
        ("THE STUDENT'S RECORD, AS SHOWN TO THE TUTOR", "\n\n".join(
            ["\n".join(f"{k}: {v}" for k, v in figures.items()),
             *(t for t in (learner.get("tables") or {}).values() if t)]
        )),
        ("EARLIER IN THE CONVERSATION", "\n\n".join(
            f"Student: {turn.get('question', '')}\nTutor: {turn.get('reply', '')}"
            for turn in earlier
        )),
        ("ROWS ITS QUERIES RETURNED", "\n\n".join(case.get("tool_outputs") or [])),
        ("PASSAGES IT RETRIEVED", "\n\n".join(case.get("passages") or [])),
    ]
    text = "\n\n".join(f"== {title} ==\n{body}" for title, body in sections if body.strip())
    return text if len(text) <= MAX_EVIDENCE_CHARS else text[:MAX_EVIDENCE_CHARS] + "\n[...cut]"


def judge_turn(
    judge,
    case: Mapping[str, Any],
    learner: Mapping[str, Any],
    earlier: Sequence[Mapping[str, Any]] = (),
) -> ChatVerdict:
    """One verdict on one answered turn."""
    user = render(
        JUDGE_USER,
        question=case.get("question", ""),
        reference=case.get("reference", "") or "(none given)",
        evidence=judge_evidence(case, learner, earlier),
        reply=case.get("reply", ""),
    )
    return judge.invoke(
        [("system", JUDGE_SYSTEM), ("user", user)],
        config={"run_name": "eval.judge", "metadata": {"eval_case": case.get("id")}},
    )


def dialogue_of(case: Mapping[str, Any]) -> str:
    """The conversation a turn belongs to; ``""`` for a single question. A row
    without a ``dialogue`` field is read off its id: ``g1-dialogue-2``."""
    if case.get("kind") != "dialogue":
        return ""
    return str(case.get("dialogue") or str(case.get("id", "")).rsplit("-", 1)[0])


def judge_cases(
    cases: Sequence[dict[str, Any]],
    learners: Mapping[str, Mapping[str, Any]],
    judge=None,
) -> None:
    """Add the judge's verdict to every answered turn, in place.

    ``learners`` is the suite result's ``learners``: what each learner's tutor
    was handed. A turn the judge could not mark keeps ``{"error": ...}`` and is
    left out of the judge's figures; a missing key raises before any call.
    """
    settings = JudgeSettings()
    judge = judge or build_judge(settings)
    said: dict[str, list[Mapping[str, Any]]] = {}
    for case in cases:
        dialogue = dialogue_of(case)
        earlier = said.setdefault(dialogue, []) if dialogue else []
        if case.get("ok"):
            try:
                verdict = judge_turn(
                    judge, case, learners.get(case.get("learner", ""), {}), list(earlier)
                )
            except Exception as exc:
                case["judge"] = {"error": f"{type(exc).__name__}: {exc}"}
            else:
                case["judge"] = {**verdict.model_dump(), "model": settings.cae_judge_model}
        if dialogue:
            earlier.append(case)


REPORT_FIELDS: tuple[tuple[str, str], ...] = (
    ("headline", "Headline"),
    ("summary", "Summary"),
    ("strengths", "Strengths"),
    ("common_errors", "Common errors"),
    ("trend", "Trend"),
    ("focus_areas", "Focus areas, most important first"),
    ("next_session", "Next session"),
)


def report_text(narrative: Mapping[str, Any]) -> str:
    """The coach's report as the judge reads it: each field under its title, lists
    as bullets, each focus area as its title, why and action. Formatting only."""
    blocks: list[str] = []
    for key, title in REPORT_FIELDS:
        value = narrative.get(key)
        if key == "focus_areas":
            body = "\n".join(
                f"{n}. {area.get('title', '')}\n   Why: {area.get('why', '')}\n"
                f"   Action: {area.get('action', '')}"
                for n, area in enumerate(value or [], 1)
            )
        elif isinstance(value, list):
            body = "\n".join(f"- {item}" for item in value)
        else:
            body = str(value or "")
        blocks.append(f"{title}:\n{body}")
    return "\n\n".join(blocks)


def judge_report(judge, case: Mapping[str, Any]) -> CoachVerdict:
    """One verdict on one written report.

    The evidence is what ``narrate`` was handed -- the figures above the tables
    and the tables, stored in the case row -- beside what the tutor's
    instructions state, through the chat's own :func:`judge_evidence`.
    """
    learner = {"figures": case.get("figures") or {}, "tables": case.get("tables") or {}}
    user = render(
        COACH_JUDGE_USER,
        reference=case.get("reference", "") or "(none given)",
        evidence=judge_evidence({}, learner),
        review=report_text(case.get("narrative") or {}),
    )
    return judge.invoke(
        [("system", COACH_JUDGE_SYSTEM), ("user", user)],
        config={"run_name": "eval.judge", "metadata": {"eval_case": case.get("id")}},
    )


def judge_coach_cases(cases: Sequence[dict[str, Any]], judge=None) -> None:
    """Add the judge's verdict to every written report, in place.

    A report the model did not write is not judged; one the judge could not
    mark keeps ``{"error": ...}`` and is left out of the judge's figures; a
    missing key raises before any call.
    """
    settings = JudgeSettings()
    judge = judge or build_judge(settings, CoachVerdict)
    for case in cases:
        if not case.get("ok"):
            continue
        try:
            verdict = judge_report(judge, case)
        except Exception as exc:
            case["judge"] = {"error": f"{type(exc).__name__}: {exc}"}
        else:
            case["judge"] = {**verdict.model_dump(), "model": settings.cae_judge_model}


__all__ = [
    "COACH_JUDGE_SYSTEM",
    "ChatVerdict",
    "CoachVerdict",
    "JUDGE_SYSTEM",
    "JudgeSettings",
    "JudgeUnavailableError",
    "build_judge",
    "dialogue_of",
    "judge_cases",
    "judge_coach_cases",
    "judge_evidence",
    "judge_report",
    "judge_turn",
    "report_text",
    "tutor_facts",
]
