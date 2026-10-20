"""Prompt templates.

Placeholders are ``<TOKEN>`` rather than ``str.format`` fields, because exercise
texts contain literal ``{{1}}`` gap markers.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"<([A-Z_]+)>")


def render(template: str, **values: object) -> str:
    """Substitute ``<TOKEN>`` placeholders, leaving unknown tokens untouched."""

    def replace(match: re.Match[str]) -> str:
        key = match.group(1).lower()
        return str(values[key]) if key in values else match.group(0)

    return _TOKEN_RE.sub(replace, template)
