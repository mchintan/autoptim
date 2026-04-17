"""Loader for a user-supplied eval.py.

The module must expose:

    def score(predictions: list[dict], ground_truth: dict[str, dict]) -> EvalResult

Where EvalResult is the dataclass in autoptim.runspec.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from ..runspec import EvalResult


class CustomEvaluator:
    def __init__(self, eval_py_path: str | Path):
        self.path = Path(eval_py_path).expanduser().resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"custom eval.py not found: {self.path}")
        spec = importlib.util.spec_from_file_location(
            f"autoptim_user_eval_{self.path.stem}", self.path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load eval module from {self.path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        if not hasattr(mod, "score"):
            raise AttributeError(
                f"{self.path} must define `def score(predictions, ground_truth) -> EvalResult`"
            )
        self._score = mod.score

    def score(
        self,
        predictions: list[dict[str, Any]],
        ground_truth: dict[str, dict[str, Any]],
    ) -> EvalResult:
        result = self._score(predictions, ground_truth)
        if not isinstance(result, EvalResult):
            raise TypeError(
                "custom eval.score() must return autoptim.runspec.EvalResult; "
                f"got {type(result).__name__}"
            )
        return result
