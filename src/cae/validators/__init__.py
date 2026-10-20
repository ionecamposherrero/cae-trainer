"""Structural validation of generated Reading exercises and Writing tasks.

Holds the text helpers shared by :mod:`cae.validators.rue_task` and
:mod:`cae.validators.writing_task`.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_SMART_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})

GAP_MARKER_RE = re.compile(r"\{\{\s*(\d+)\s*\}\}")


def count_words(text: str) -> int:
    """Count the words in a passage. Gap placeholders are not words."""
    return len(_WORD_RE.findall(GAP_MARKER_RE.sub(" ", text)))


def normalise(text: str) -> str:
    """Lowercase, straighten smart quotes and collapse whitespace; keep punctuation."""
    text = text.translate(_SMART_QUOTES).lower()
    return re.sub(r"\s+", " ", text).strip()
