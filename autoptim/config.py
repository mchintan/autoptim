"""Pydantic models for task.yaml and runtime config."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class FieldSpec(BaseModel):
    match: Literal["exact", "fuzzy", "numeric", "date_iso", "list_of_objects", "contains"]
    weight: float = 1.0
    tol: float | None = None
    threshold: float | None = None
    key: str | None = None
    fields: dict[str, "FieldSpec"] | None = None

    model_config = {"extra": "forbid"}


class MetricSpec(BaseModel):
    type: Literal["schema_match", "custom"]
    schema_: dict[str, FieldSpec] | None = Field(default=None, alias="schema")
    aggregate: Literal["weighted_f1", "mean_score", "min_score"] = "weighted_f1"
    target: float = 0.95
    eval_py: str | None = None

    model_config = {"extra": "forbid", "populate_by_name": True}

    @field_validator("target")
    @classmethod
    def _target_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("metric.target must be in [0, 1]")
        return v


class Budgets(BaseModel):
    max_iters: int = 50
    wall_clock_h: float = 8.0
    frontier_usd: float = 10.0
    per_iter_timeout_s: int = 600

    model_config = {"extra": "forbid"}


class MetaConfig(BaseModel):
    provider: Literal["anthropic", "openai", "openrouter", "gemini"] = "gemini"
    model: str = "gemini-3-pro"
    max_history: int = 5
    # Some reasoning models (e.g. claude-opus-4-7) reject `temperature`. Leave unset
    # by default and only pass through when the user explicitly configures it.
    temperature: float | None = None

    model_config = {"extra": "forbid"}


class WorkerConfig(BaseModel):
    ollama_host: str = "http://localhost:11434"
    default_model: str = "gemma4:latest"
    memory_mb: int = 4096

    model_config = {"extra": "forbid"}


class TaskConfig(BaseModel):
    """A fully resolved task spec (paths absolute)."""

    name: str
    inputs_dir: str
    ground_truth: str
    seed_process: str
    metric: MetricSpec
    budgets: Budgets = Budgets()
    meta: MetaConfig = MetaConfig()
    worker: WorkerConfig = WorkerConfig()
    task_root: str  # the directory task.yaml was loaded from (for relative resolution)

    model_config = {"extra": "forbid"}


def load_task(path: str | Path) -> TaskConfig:
    """Load and validate a task.yaml, resolving relative paths against the yaml's directory."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"task file not found: {p}")
    raw = yaml.safe_load(p.read_text()) or {}
    root = p.parent

    def _resolve(key: str) -> None:
        val = raw.get(key)
        if val and not Path(val).is_absolute():
            raw[key] = str((root / val).resolve())

    for k in ("inputs_dir", "ground_truth", "seed_process"):
        _resolve(k)

    if raw.get("metric", {}).get("eval_py"):
        ev = raw["metric"]["eval_py"]
        if not Path(ev).is_absolute():
            raw["metric"]["eval_py"] = str((root / ev).resolve())

    raw["task_root"] = str(root)
    return TaskConfig.model_validate(raw)
