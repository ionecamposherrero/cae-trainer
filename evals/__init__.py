"""Evaluation harness.

* :mod:`evals.metrics` is pure Python, unit tested in ``tests/test_evals_offline.py``.
* :mod:`evals.suites` defines the cases and what is expected of them.
* ``scripts/run_evals.py`` runs them against the real API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATASETS_DIR = Path(__file__).resolve().parent / "datasets"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def load_dataset(name: str) -> dict[str, Any]:
    """Load ``datasets/<name>.json``."""
    path = DATASETS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"No eval dataset at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = ["DATASETS_DIR", "REPORTS_DIR", "load_dataset"]
