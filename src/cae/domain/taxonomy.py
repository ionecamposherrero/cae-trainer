"""The closed error taxonomy every mistake is tagged with, so counts aggregate.

Language categories are chapters of a grammar or vocabulary book: something the
learner can go and study. Use of English (Parts 1-4) shares one set of them, and
Writing adds register and punctuation. Each Reading part (5-8) has a single,
fixed category -- the task itself -- so there is nothing for the model to choose.
"""

from __future__ import annotations

from enum import StrEnum


class ErrorCategory(StrEnum):
    VERB_FORMS = "VERB_FORMS"
    SENTENCE_STRUCTURE = "SENTENCE_STRUCTURE"
    GRAMMAR_WORDS = "GRAMMAR_WORDS"

    WORD_PATTERNS = "WORD_PATTERNS"
    WORD_CHOICE = "WORD_CHOICE"
    FIXED_PHRASES = "FIXED_PHRASES"
    WORD_FORM = "WORD_FORM"

    LINKING = "LINKING"
    REGISTER = "REGISTER"
    PUNCTUATION = "PUNCTUATION"

    READING_FOR_MEANING = "READING_FOR_MEANING"
    COMPARING_VIEWS = "COMPARING_VIEWS"
    TEXT_STRUCTURE = "TEXT_STRUCTURE"
    SCANNING = "SCANNING"


class Family(StrEnum):
    """Four coarse groups, for summaries that a dozen categories would clutter."""

    GRAMMAR = "GRAMMAR"
    VOCABULARY = "VOCABULARY"
    TEXT = "TEXT"
    READING = "READING"


CATEGORY_FAMILY: dict[ErrorCategory, Family] = {
    ErrorCategory.VERB_FORMS: Family.GRAMMAR,
    ErrorCategory.SENTENCE_STRUCTURE: Family.GRAMMAR,
    ErrorCategory.GRAMMAR_WORDS: Family.GRAMMAR,
    ErrorCategory.WORD_PATTERNS: Family.VOCABULARY,
    ErrorCategory.WORD_CHOICE: Family.VOCABULARY,
    ErrorCategory.FIXED_PHRASES: Family.VOCABULARY,
    ErrorCategory.WORD_FORM: Family.VOCABULARY,
    ErrorCategory.LINKING: Family.TEXT,
    ErrorCategory.REGISTER: Family.TEXT,
    ErrorCategory.PUNCTUATION: Family.TEXT,
    ErrorCategory.READING_FOR_MEANING: Family.READING,
    ErrorCategory.COMPARING_VIEWS: Family.READING,
    ErrorCategory.TEXT_STRUCTURE: Family.READING,
    ErrorCategory.SCANNING: Family.READING,
}

CATEGORY_LABELS: dict[ErrorCategory, str] = {
    ErrorCategory.VERB_FORMS: "Verb forms",
    ErrorCategory.SENTENCE_STRUCTURE: "Sentence structure",
    ErrorCategory.GRAMMAR_WORDS: "Grammar words and agreement",
    ErrorCategory.WORD_PATTERNS: "Word patterns",
    ErrorCategory.WORD_CHOICE: "Word choice",
    ErrorCategory.FIXED_PHRASES: "Collocations, phrasal verbs and idioms",
    ErrorCategory.WORD_FORM: "Word formation and spelling",
    ErrorCategory.LINKING: "Linking and reference",
    ErrorCategory.REGISTER: "Register and tone",
    ErrorCategory.PUNCTUATION: "Punctuation",
    ErrorCategory.READING_FOR_MEANING: "Detail, opinion and implication",
    ErrorCategory.COMPARING_VIEWS: "Comparing writers' opinions",
    ErrorCategory.TEXT_STRUCTURE: "Text structure and cohesion",
    ErrorCategory.SCANNING: "Finding paraphrased information",
}

CATEGORY_GUIDE: dict[ErrorCategory, str] = {
    ErrorCategory.VERB_FORMS: (
        "the verb phrase is wrong: tense or aspect, including the auxiliary or "
        "participle that builds it (has been, have gone); a modal (including "
        "perfect modals); the verbs of a conditional or of wish/if only; a "
        "passive or causative; passive reporting (it is said that, she is "
        "thought to)"
    ),
    ErrorCategory.SENTENCE_STRUCTURE: (
        "the shape of the sentence is wrong: word order (including indirect "
        "questions); inversion after a negative or limiting adverbial (Never "
        "have I, Only then did) or in a conditional (Had I known, Should you "
        "need); cleft sentences; relative clauses, including the choice of "
        "relative pronoun (who, which, whose); comparative structures (than, "
        "as ... as), though an idiom built on one (as good as new) is "
        "FIXED_PHRASES"
    ),
    ErrorCategory.GRAMMAR_WORDS: (
        "a small grammar word is wrong: article, determiner, personal or "
        "reflexive pronoun, an auxiliary in a short structure (so do I, "
        "question tags), a quantifier that does not suit its noun (much/many, "
        "each of the), a double negative, a preposition chosen by its meaning "
        "(at night, by car, for or since a time), or agreement within the "
        "clause (its/their, is/are, a plural noun after many)"
    ),
    ErrorCategory.WORD_PATTERNS: (
        "what a verb, noun or adjective requires after it is wrong: a dependent "
        "preposition chosen by the word before it (depend on, attributable to) "
        "or -ing / to-infinitive / that-clause, including after a reporting "
        "verb (suggest doing, persuade someone to do). A phrasal verb's "
        "particle is FIXED_PHRASES"
    ),
    ErrorCategory.WORD_CHOICE: (
        "a real word that fits the grammar but means the wrong thing: "
        "near-synonyms, precision, strength (even of a quantifier: all for "
        "most), connotation, false friends, or two derivatives with different "
        "meanings (economic/economical). A wrong linker is LINKING, an -ing/-ed "
        "mix-up is WORD_FORM, and a wrong phrasal verb, collocation or idiom is "
        "FIXED_PHRASES, even when the word chosen is a real one"
    ),
    ErrorCategory.FIXED_PHRASES: (
        "words that go together and cannot be swapped: collocation (make a "
        "decision, heavy rain), phrasal verb, idiom or fixed expression, even "
        "when the wrong word or particle makes a real phrase with another "
        "meaning (give up for give in). A wrong collocating word is this "
        "category even if a preposition is wrong too; a wrong dependent "
        "preposition alone is WORD_PATTERNS"
    ),
    ErrorCategory.WORD_FORM: (
        "the right word in the wrong form: the wrong word class, -ing for -ed "
        "or the reverse on an adjective (a bored/boring audience), a prefix or "
        "suffix that does not exist on that word, or a misspelling. A wrong "
        "form inside the verb phrase (have went) is VERB_FORMS"
    ),
    ErrorCategory.LINKING: (
        "how clauses and sentences connect: linkers and concessive words "
        "(despite, in spite of, whereas, rather than, whatever, no matter how) "
        "and what may follow them (a noun or -ing after despite, never a "
        "clause), including a linker that is grammatical but does the wrong "
        "job (adds where the sentence concedes); words referring back across "
        "sentences (this, such); substitution"
    ),
    ErrorCategory.REGISTER: "the formality or tone does not suit the reader or the genre",
    ErrorCategory.PUNCTUATION: "commas, full stops, apostrophes and the like",
}

CATEGORY_ADVICE: dict[ErrorCategory, str] = {
    ErrorCategory.VERB_FORMS: "Review perfect aspect, modals of deduction about the past ('must have been'), mixed conditionals and passive reporting ('it is thought that...').",
    ErrorCategory.SENTENCE_STRUCTURE: "Practise inversion after negative adverbials, cleft sentences ('It was X that...'), relatives with prepositions and comparative patterns.",
    ErrorCategory.GRAMMAR_WORDS: "Focus on articles with abstract nouns, pronouns that must agree with their noun, and auxiliaries in short structures.",
    ErrorCategory.WORD_PATTERNS: "Record verbs, nouns and adjectives with what follows them: 'object to doing', 'attributable to', 'insist on'.",
    ErrorCategory.WORD_CHOICE: "Compare near-synonyms in a good learner's dictionary before choosing one.",
    ErrorCategory.FIXED_PHRASES: "Record new vocabulary in whole chunks rather than single words, and phrasal verbs grouped by particle.",
    ErrorCategory.WORD_FORM: "Build word families and note the spelling shifts across them ('pronounce', 'pronunciation').",
    ErrorCategory.LINKING: "Check what each linker does (add, contrast, concede) and use referencing ('this shift', 'such concerns') instead of repeating nouns.",
    ErrorCategory.REGISTER: "Match formality to genre: proposals and reports are impersonal, reviews are engaging.",
    ErrorCategory.PUNCTUATION: "Check comma use with subordinate clauses and avoid comma splices.",
    ErrorCategory.READING_FOR_MEANING: "Find the line that proves every word of your option; reject options that repeat the text's words but say something else.",
    ErrorCategory.COMPARING_VIEWS: "Underline each writer's verdict on the point, not just the point: two writers can discuss the same thing and disagree.",
    ErrorCategory.TEXT_STRUCTURE: "Read across the gap in both directions and follow 'this', 'they' or 'such' back to what they refer to.",
    ErrorCategory.SCANNING: "Rephrase the statement in your own words, then scan for the idea: the text rarely repeats the prompt's wording.",
}

LANGUAGE_CATEGORIES: tuple[ErrorCategory, ...] = (
    ErrorCategory.VERB_FORMS,
    ErrorCategory.SENTENCE_STRUCTURE,
    ErrorCategory.GRAMMAR_WORDS,
    ErrorCategory.WORD_PATTERNS,
    ErrorCategory.WORD_CHOICE,
    ErrorCategory.FIXED_PHRASES,
    ErrorCategory.WORD_FORM,
    ErrorCategory.LINKING,
)

WRITING_CATEGORIES: tuple[ErrorCategory, ...] = (
    *LANGUAGE_CATEGORIES,
    ErrorCategory.REGISTER,
    ErrorCategory.PUNCTUATION,
)

PART_CATEGORIES: dict[int, tuple[ErrorCategory, ...]] = {
    1: LANGUAGE_CATEGORIES,
    2: LANGUAGE_CATEGORIES,
    3: LANGUAGE_CATEGORIES,
    4: LANGUAGE_CATEGORIES,
    5: (ErrorCategory.READING_FOR_MEANING,),
    6: (ErrorCategory.COMPARING_VIEWS,),
    7: (ErrorCategory.TEXT_STRUCTURE,),
    8: (ErrorCategory.SCANNING,),
}


def label(category: ErrorCategory | str) -> str:
    """Human-readable label, tolerant of raw strings coming out of the DB."""
    try:
        return CATEGORY_LABELS[ErrorCategory(category)]
    except (ValueError, KeyError):
        return str(category).replace("_", " ").title()


def advice(category: ErrorCategory | str) -> str:
    try:
        return CATEGORY_ADVICE[ErrorCategory(category)]
    except (ValueError, KeyError):
        return "Review this area with targeted practice."


def guide(category: ErrorCategory | str) -> str:
    """When to use a language category; empty for Reading, which is never chosen."""
    try:
        return CATEGORY_GUIDE.get(ErrorCategory(category), "")
    except ValueError:
        return ""


def family(category: ErrorCategory | str) -> str:
    """The category's family; an unknown code is ``""``."""
    try:
        return CATEGORY_FAMILY[ErrorCategory(category)].value
    except (ValueError, KeyError):
        return ""


def categories_for_part(part: int) -> tuple[ErrorCategory, ...]:
    return PART_CATEGORIES.get(part, tuple(ErrorCategory))
