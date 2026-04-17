"""Evaluator protocol: everything that scores predictions against ground truth."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from ..runspec import EvalResult


class Evaluator(Protocol):
    def score(
        self,
        predictions: list[dict[str, Any]],
        ground_truth: dict[str, dict[str, Any]],
    ) -> EvalResult: ...


def load_ground_truth(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load ground_truth.jsonl keyed by id.

    Each line: {"id": "...", "fields": {...}} — 'fields' is the expected structured value.
    We tolerate the variant where the whole line *is* the expected structure keyed by top-level 'id'.
    """
    out: dict[str, dict[str, Any]] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        doc_id = obj["id"]
        if "fields" in obj:
            out[doc_id] = obj["fields"]
        else:
            rest = {k: v for k, v in obj.items() if k != "id"}
            out[doc_id] = rest
    return out
