"""Split the reference documents into citable chunks and build the index.

Everything except :func:`build_index` is pure Python with no API calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from cae.config import REFERENCE_DIR

CHUNK_WORDS = 220
CHUNK_OVERLAP = 40

SUPPORTED_SUFFIXES = (".md", ".txt")

EXCLUDED_STEMS = frozenset({"readme"})

_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*\S)\s*$")


@dataclass(frozen=True)
class Chunk:
    """One retrievable passage, and enough provenance to cite it."""

    text: str
    source: str
    section: str

    @property
    def citation(self) -> str:
        return f"{self.source} — {self.section}" if self.section else self.source

    @property
    def embedded(self) -> str:
        """The indexed text: heading path, then body.

        The path is embedded so queries can match section names the body omits.
        """
        return f"{self.section}\n\n{self.text}" if self.section else self.text


PATH_SEPARATOR = " > "

MAX_PATH_DEPTH = 3


def _sections(text: str) -> list[tuple[str, str]]:
    """Split a document at Markdown headings into ``(heading path, body)`` pairs.

    A document with no headings comes back as one unnamed section.
    """
    sections: list[tuple[str, list[str]]] = []
    stack: list[tuple[int, str]] = []
    current_lines: list[str] = []

    def path() -> str:
        titles = [title for _, title in stack][-MAX_PATH_DEPTH:]
        return PATH_SEPARATOR.join(titles)

    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match is None:
            current_lines.append(line)
            continue
        if current_lines and "".join(current_lines).strip():
            sections.append((path(), current_lines))
        level = len(match.group(1))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, match.group(2).strip()))
        current_lines = []

    if current_lines and "".join(current_lines).strip():
        sections.append((path(), current_lines))

    return [(heading, "\n".join(lines).strip()) for heading, lines in sections]


def _windows(words: list[str]) -> list[list[str]]:
    """Overlapping windows of ``CHUNK_WORDS`` words."""
    if len(words) <= CHUNK_WORDS:
        return [words]
    step = CHUNK_WORDS - CHUNK_OVERLAP
    windows = [words[start : start + CHUNK_WORDS] for start in range(0, len(words), step)]
    if len(windows) > 1 and len(windows[-1]) <= CHUNK_OVERLAP:
        windows.pop()
    return windows


def split_document(text: str, source: str) -> list[Chunk]:
    """Chunk one document, keeping each chunk's heading with it."""
    chunks: list[Chunk] = []
    for section, body in _sections(text):
        words = body.split()
        if not words:
            continue
        for window in _windows(words):
            chunks.append(Chunk(text=" ".join(window), source=source, section=section))
    return chunks


def load_corpus(directory: Path | None = None) -> list[Chunk]:
    """Read and chunk every supported document in the reference directory."""
    root = directory or REFERENCE_DIR
    if not root.exists():
        return []
    chunks: list[Chunk] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES or not path.is_file():
            continue
        if path.stem.lower() in EXCLUDED_STEMS:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        chunks.extend(split_document(text, source=path.relative_to(root).as_posix()))
    return chunks


def estimated_tokens(chunks: list[Chunk]) -> int:
    """Rough token count (four characters per token), for pricing an ingest."""
    return sum(len(chunk.text) for chunk in chunks) // 4


def build_index(directory: Path | None = None, *, embeddings=None) -> dict[str, object]:
    """Embed the corpus and persist it, rebuilding from scratch. Returns a report."""
    from cae.rag.store import reset_collection

    chunks = load_corpus(directory)
    if not chunks:
        return {"chunks": 0, "documents": 0, "tokens": 0}

    store = reset_collection(embeddings=embeddings)
    store.add_texts(
        texts=[chunk.embedded for chunk in chunks],
        metadatas=[
            {"source": chunk.source, "section": chunk.section, "citation": chunk.citation}
            for chunk in chunks
        ],
    )
    return {
        "chunks": len(chunks),
        "documents": len({chunk.source for chunk in chunks}),
        "tokens": estimated_tokens(chunks),
    }


__all__ = [
    "CHUNK_OVERLAP",
    "CHUNK_WORDS",
    "Chunk",
    "build_index",
    "estimated_tokens",
    "load_corpus",
    "split_document",
]
