"""Unified-diff rendering for `autoptim inspect`."""
from __future__ import annotations

import difflib


def unified(a: str, b: str, a_name: str = "parent", b_name: str = "candidate") -> str:
    return "".join(
        difflib.unified_diff(
            a.splitlines(keepends=True),
            b.splitlines(keepends=True),
            fromfile=a_name,
            tofile=b_name,
            n=3,
        )
    )
