import textwrap
from pathlib import Path

import pytest

from autoptim.config import load_task


def _write_task(tmp_path: Path, body: str) -> Path:
    (tmp_path / "task.yaml").write_text(textwrap.dedent(body))
    (tmp_path / "seed_process.py").write_text("def run(inputs, ctx): return []\n")
    (tmp_path / "inputs").mkdir()
    (tmp_path / "ground_truth.jsonl").write_text("")
    return tmp_path / "task.yaml"


def test_resolves_relative_paths(tmp_path: Path):
    p = _write_task(tmp_path, """
        name: demo
        inputs_dir: ./inputs
        ground_truth: ./ground_truth.jsonl
        seed_process: ./seed_process.py
        metric:
          type: schema_match
          schema:
            x: {match: exact, weight: 1}
          aggregate: mean_score
          target: 0.9
    """)
    task = load_task(p)
    assert Path(task.inputs_dir).is_absolute()
    assert task.inputs_dir.endswith("/inputs")
    assert task.metric.schema_["x"].match == "exact"


def test_rejects_bad_target(tmp_path: Path):
    p = _write_task(tmp_path, """
        name: demo
        inputs_dir: ./inputs
        ground_truth: ./ground_truth.jsonl
        seed_process: ./seed_process.py
        metric:
          type: schema_match
          schema:
            x: {match: exact, weight: 1}
          aggregate: mean_score
          target: 1.5
    """)
    with pytest.raises(Exception):
        load_task(p)
