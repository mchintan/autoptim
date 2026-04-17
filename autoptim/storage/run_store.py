"""Atomic, append-only IO for run artifacts under runs/<run_id>/."""
from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from ..runspec import IterationRecord, RunState


def new_run_id(task_name: str) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return f"{task_name}-{ts}"


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _atomic_write_json(path: Path, obj: Any) -> None:
    _atomic_write_text(path, json.dumps(obj, indent=2, default=str))


class RunStore:
    """All mutation funnels through this object so layout stays consistent."""

    def __init__(self, runs_root: str | Path, run_id: str):
        self.root = Path(runs_root).expanduser().resolve()
        self.run_id = run_id
        self.run_dir = self.root / run_id

    @classmethod
    def create(
        cls,
        runs_root: str | Path,
        run_id: str,
        frozen_task_yaml: dict[str, Any],
    ) -> "RunStore":
        store = cls(runs_root, run_id)
        store.run_dir.mkdir(parents=True, exist_ok=True)
        (store.run_dir / "history.jsonl").touch(exist_ok=True)
        _atomic_write_text(
            store.run_dir / "run.yaml",
            yaml.safe_dump(frozen_task_yaml, sort_keys=False),
        )
        return store

    def iter_dir(self, iter_n: int) -> Path:
        return self.run_dir / f"iter_{iter_n:03d}"

    def write_iter_process(self, iter_n: int, process_py: str) -> None:
        d = self.iter_dir(iter_n)
        d.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(d / "process.py", process_py)

    def write_iter_hypothesis(self, iter_n: int, hypothesis: str, strategy_tag: str, parent: int | str) -> None:
        d = self.iter_dir(iter_n)
        d.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(d / "hypothesis.md", hypothesis.strip() + "\n")
        _atomic_write_text(d / "strategy_tag.txt", strategy_tag)
        _atomic_write_text(d / "parent.txt", str(parent))

    def write_iter_eval(self, iter_n: int, eval_json: dict[str, Any]) -> None:
        _atomic_write_json(self.iter_dir(iter_n) / "eval.json", eval_json)

    def write_iter_proposal(self, iter_n: int, proposal: dict[str, Any]) -> None:
        """Full meta-agent proposal (hypothesis, strategy, predicted delta, failure modes, tokens)."""
        _atomic_write_json(self.iter_dir(iter_n) / "proposal.json", proposal)

    def write_iter_timing(self, iter_n: int, timing: dict[str, Any]) -> None:
        _atomic_write_json(self.iter_dir(iter_n) / "timing.json", timing)

    def write_iter_decision(self, iter_n: int, decision: dict[str, Any]) -> None:
        _atomic_write_json(self.iter_dir(iter_n) / "decision.json", decision)

    def write_iter_stdio(self, iter_n: int, stdout: str, stderr: str) -> None:
        d = self.iter_dir(iter_n)
        d.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(d / "worker_stdout.log", stdout)
        _atomic_write_text(d / "worker_stderr.log", stderr)

    def append_history(self, rec: IterationRecord) -> None:
        line = json.dumps(rec.to_json(), default=str)
        with open(self.run_dir / "history.jsonl", "a") as f:
            f.write(line + "\n")

    def update_best(self, iter_n: int, score: float) -> None:
        _atomic_write_json(
            self.run_dir / "best.json",
            {"iter": iter_n, "score": score, "path": str(self.iter_dir(iter_n))},
        )

    def read_best(self) -> dict[str, Any] | None:
        p = self.run_dir / "best.json"
        if not p.exists():
            return None
        return json.loads(p.read_text())

    def read_history(self) -> list[dict[str, Any]]:
        p = self.run_dir / "history.jsonl"
        if not p.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in p.read_text().splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def read_process(self, iter_n: int) -> str:
        return (self.iter_dir(iter_n) / "process.py").read_text()

    def read_eval(self, iter_n: int) -> dict[str, Any]:
        return json.loads((self.iter_dir(iter_n) / "eval.json").read_text())

    def read_hypothesis(self, iter_n: int) -> str:
        p = self.iter_dir(iter_n) / "hypothesis.md"
        return p.read_text() if p.exists() else ""

    def write_final(self, final: dict[str, Any]) -> None:
        _atomic_write_json(self.run_dir / "final.json", final)


def list_runs(runs_root: str | Path) -> list[str]:
    root = Path(runs_root).expanduser()
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def find_latest_for_task(runs_root: str | Path, task_name: str) -> str | None:
    ids = [r for r in list_runs(runs_root) if r.startswith(task_name + "-")]
    return ids[-1] if ids else None
