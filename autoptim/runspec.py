"""Dataclasses describing a run's on-disk state."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


StrategyTag = Literal[
    "seed",
    "prompt_mutation",
    "pipeline_restructuring",
    "model_param_swap",
    "error_pattern_fix",
    "representation",
    "verification",
]


ALL_STRATEGIES: tuple[StrategyTag, ...] = (
    "prompt_mutation",
    "pipeline_restructuring",
    "model_param_swap",
    "error_pattern_fix",
    "representation",
    "verification",
)


@dataclass
class FieldScore:
    name: str
    expected: Any
    predicted: Any
    match: bool
    score: float


@dataclass
class PerDocScore:
    id: str
    score: float
    fields: list[FieldScore]
    error: str | None = None


@dataclass
class Failure:
    id: str
    summary: str
    expected: Any
    predicted: Any


@dataclass
class EvalResult:
    metric_name: str
    overall: float
    target: float
    per_doc: list[PerDocScore]
    failures: list[Failure]
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "overall": self.overall,
            "target": self.target,
            "per_doc": [
                {
                    "id": d.id,
                    "score": d.score,
                    "error": d.error,
                    "fields": [
                        {
                            "name": f.name,
                            "expected": f.expected,
                            "predicted": f.predicted,
                            "match": f.match,
                            "score": f.score,
                        }
                        for f in d.fields
                    ],
                }
                for d in self.per_doc
            ],
            "failures": [
                {
                    "id": f.id,
                    "summary": f.summary,
                    "expected": f.expected,
                    "predicted": f.predicted,
                }
                for f in self.failures
            ],
            "extra": self.extra,
        }


@dataclass
class IterationRecord:
    """One line in history.jsonl — enough for dashboards and resume."""

    iter: int
    ts: str
    parent: int | str
    strategy_tag: str
    hypothesis: str
    score: float | None
    delta: float | None
    kept: bool
    new_best: bool
    frontier_tokens_in: int
    frontier_tokens_out: int
    frontier_usd: float
    worker_seconds: float
    ollama_calls: int
    predicted_delta: float | None
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class RunState:
    """In-memory mirror of a run directory."""

    run_id: str
    run_dir: str
    task_name: str
    iters: list[IterationRecord] = field(default_factory=list)
    best_iter: int | None = None
    best_score: float | None = None
    total_frontier_usd: float = 0.0
    started_ts: str = ""

    def last_iter(self) -> int:
        return self.iters[-1].iter if self.iters else -1
