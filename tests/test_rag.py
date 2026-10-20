"""Chunking and retrieval, offline.

Retrieval runs against a real Chroma collection with deterministic fake embeddings.
"""

from __future__ import annotations

import math
import zlib
from pathlib import Path

import pytest

from cae.rag import ingest
from cae.rag import store as reference_store

HANDBOOK = """# Reading and Use of English

The paper has eight parts and lasts ninety minutes.

## Part 2: Open cloze

Eight gaps, each requiring exactly one word. The answers are grammatical or
function words: auxiliaries, articles, pronouns, prepositions and parts of fixed
structures.

## Inversion

After a negative adverbial at the start of a clause, the subject and the
auxiliary change places. Little did the organisers realise how much interest
there would be. The same happens after seldom, rarely, never and no sooner.
"""

GRAMMAR = """# Conditionals

Mixed conditionals combine a past condition with a present result, or the
reverse. If I had studied medicine, I would be a doctor now.
"""


class BagOfWords:
    """Deterministic hashing bag-of-words embedder with the LangChain interface.

    Uses ``crc32`` because the built-in ``hash`` is salted per process.
    """

    DIMENSIONS = 1024

    def _vector(self, text: str) -> list[float]:
        counts = [0.0] * self.DIMENSIONS
        for word in text.lower().split():
            cleaned = word.strip(".,;:'\"()")
            if not cleaned:
                continue
            counts[zlib.crc32(cleaned.encode()) % self.DIMENSIONS] += 1.0
        norm = math.sqrt(sum(c * c for c in counts)) or 1.0
        return [c / norm for c in counts]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    directory = tmp_path / "reference"
    directory.mkdir()
    (directory / "handbook.md").write_text(HANDBOOK, encoding="utf-8")
    (directory / "grammar.md").write_text(GRAMMAR, encoding="utf-8")
    (directory / "README.md").write_text("# Reference corpus\nPut documents here.", encoding="utf-8")
    return directory


@pytest.fixture
def index(tmp_path: Path, corpus: Path, monkeypatch: pytest.MonkeyPatch):
    """A real Chroma collection, in a scratch directory, with fake embeddings."""
    monkeypatch.setattr(reference_store, "INDEX_DIR", tmp_path / "chroma")
    reference_store._cached_store.cache_clear()
    embeddings = BagOfWords()
    ingest.build_index(corpus, embeddings=embeddings)
    yield embeddings
    reference_store._cached_store.cache_clear()


def test_a_chunk_keeps_the_heading_it_came_from() -> None:
    """Provenance is the point: an uncited chunk can only be asserted, not checked."""
    chunks = ingest.split_document(HANDBOOK, source="handbook.md")

    sections = {c.section for c in chunks}
    assert any("Inversion" in s for s in sections)
    assert any("Part 2: Open cloze" in s for s in sections)

    inversion = next(c for c in chunks if c.section.endswith("Inversion"))
    assert "negative adverbial" in inversion.text
    assert inversion.citation.startswith("handbook.md — ")


def test_a_chunk_is_cited_by_its_heading_path_not_just_its_heading() -> None:
    """Repeated headings such as "THE TASK" are told apart by their path."""
    document = """# Paper 1: Reading and Use of English

## Part 2

### THE TASK

Eight gaps, each requiring exactly one word.

## Part 4

### THE TASK

Six questions, each with a key word and a gap of three to six words.
"""
    chunks = ingest.split_document(document, source="handbook.md")
    by_text = {c.text[:20]: c for c in chunks}

    part2 = next(c for c in chunks if "Eight gaps" in c.text)
    part4 = next(c for c in chunks if "Six questions" in c.text)

    assert part2.section != part4.section
    assert part2.section.endswith("Part 2 > THE TASK")
    assert part4.section.endswith("Part 4 > THE TASK")
    assert by_text


def test_a_deeper_heading_does_not_swallow_its_siblings() -> None:
    """A level-3 heading must not stay in the path of the next level-2 section."""
    document = """# Guide

## Writing

### Organisation

Paragraphing and cohesive devices.

## Listening

Four parts, thirty questions.
"""
    chunks = ingest.split_document(document, source="g.md")
    listening = next(c for c in chunks if "thirty questions" in c.text)
    assert "Organisation" not in listening.section
    assert listening.section.endswith("Listening")


def test_the_citation_path_is_capped_so_it_stays_readable() -> None:
    document = "\n\n".join(
        [f"{'#' * level} Level {level}" for level in range(1, 7)] + ["Body text here."]
    )
    chunk = ingest.split_document(document, source="deep.md")[0]
    assert chunk.section.count(ingest.PATH_SEPARATOR) == ingest.MAX_PATH_DEPTH - 1
    assert chunk.section.endswith("Level 6")


def test_the_heading_path_is_part_of_what_gets_embedded() -> None:
    """A heading kept only in metadata would be invisible to the retriever."""
    chunk = ingest.split_document("## Part 2: Open cloze\n\nEight gaps.", source="h.md")[0]

    assert chunk.embedded.startswith("Part 2: Open cloze")
    assert "Eight gaps." in chunk.embedded
    assert chunk.text == "Eight gaps."


def test_an_unheaded_chunk_is_embedded_as_itself() -> None:
    chunk = ingest.split_document("Just prose.", source="n.txt")[0]
    assert chunk.embedded == "Just prose."


def test_a_document_without_headings_is_still_chunked() -> None:
    chunks = ingest.split_document("Just a paragraph of prose.", source="notes.txt")
    assert len(chunks) == 1
    assert chunks[0].section == ""
    assert chunks[0].citation == "notes.txt"


def test_long_sections_are_split_with_overlap() -> None:
    """The overlap is what stops a rule that straddles a boundary being lost."""
    body = " ".join(f"word{i}" for i in range(ingest.CHUNK_WORDS * 2))
    chunks = ingest.split_document(f"# Long\n\n{body}", source="long.md")

    assert len(chunks) > 1
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    assert first_words[-ingest.CHUNK_OVERLAP:] == second_words[:ingest.CHUNK_OVERLAP]


def test_the_readme_is_not_indexed_as_reference_material(corpus: Path) -> None:
    """Otherwise the coach cites the instructions at a learner asking about grammar."""
    sources = {chunk.source for chunk in ingest.load_corpus(corpus)}
    assert sources == {"handbook.md", "grammar.md"}


def test_a_missing_corpus_directory_is_empty_not_an_error(tmp_path: Path) -> None:
    assert ingest.load_corpus(tmp_path / "nope") == []


def test_a_lookup_returns_the_relevant_passage_with_its_citation(index) -> None:
    hits = reference_store.search(
        "negative adverbial inversion auxiliary", k=2, embeddings=index
    )

    assert hits, "the index returned nothing"
    assert "Inversion" in hits[0]["section"]
    assert hits[0]["source"] == "handbook.md"
    assert hits[0]["citation"] == "handbook.md — Reading and Use of English > Inversion"
    assert "subject" in hits[0]["text"]


def test_a_rebuild_does_not_leave_the_old_chunks_behind(
    index, corpus: Path
) -> None:
    """A stale chunk is a citation pointing at text no longer in the source."""
    before = reference_store.count(embeddings=index)

    (corpus / "grammar.md").unlink()
    ingest.build_index(corpus, embeddings=index)

    assert reference_store.count(embeddings=index) < before
    sources = {
        hit["source"]
        for hit in reference_store.search("conditionals", k=5, embeddings=index)
    }
    assert "grammar.md" not in sources


def test_search_survives_a_missing_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh clone has no corpus; the chat must degrade, not crash."""
    monkeypatch.setattr(reference_store, "INDEX_DIR", tmp_path / "empty")
    reference_store._cached_store.cache_clear()
    monkeypatch.setattr(
        reference_store, "get_embeddings", lambda: (_ for _ in ()).throw(RuntimeError("no key"))
    )

    assert reference_store.search("anything") == []
    assert reference_store.index_exists() is False
    reference_store._cached_store.cache_clear()


def test_an_empty_query_costs_nothing(index) -> None:
    assert reference_store.search("   ", embeddings=index) == []


class OtherModel(BagOfWords):
    """Same interface, different model: its vectors mean something else."""


def test_an_index_built_with_another_model_counts_as_missing(index) -> None:
    """Querying another model's vectors returns confident nonsense, so it must not run."""
    assert reference_store.index_exists(embeddings=index)
    assert reference_store.indexed_model(embeddings=index) == "custom:BagOfWords"

    other = OtherModel()
    assert reference_store.index_exists(embeddings=other) is False
    assert reference_store.search("negative adverbial inversion", embeddings=other) == []


def test_an_index_without_a_recorded_model_counts_as_missing(
    tmp_path: Path, corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An index that does not record its embedding model cannot be trusted."""
    monkeypatch.setattr(reference_store, "INDEX_DIR", tmp_path / "old")
    embeddings = BagOfWords()
    store = reference_store._open_store(embeddings)
    store.add_texts(texts=["Inversion after negative adverbials."])

    assert reference_store.count(embeddings=embeddings) == 1
    assert reference_store.index_exists(embeddings=embeddings) is False
