"""Custom evaluator for maze_solver.

Each prediction is a `path`: a list of [x, y] cells from S to G. For each maze:

- Validate: path starts at S, ends at G, every step is 4-adjacent to its
  predecessor, every cell is open (non-wall) and in-bounds, no repeats.
- Score: optimal_len / path_len  (1.0 = optimal, 0.0 = invalid or missing).

The score is continuous so the meta-agent sees smooth progress (e.g. DFS
returning a 42-step path through a maze with an 18-step optimum scores 0.43,
not 0).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autoptim.runspec import EvalResult, Failure, FieldScore, PerDocScore

INPUTS_DIR = Path(__file__).parent / "inputs"


def _load_maze(qid: str) -> dict[str, Any] | None:
    p = INPUTS_DIR / f"{qid}.json"
    if not p.exists():
        return None
    with open(p, "r") as f:
        return json.load(f)


def _validate(maze: dict[str, Any], path: Any) -> tuple[bool, str]:
    if not isinstance(path, list) or not path:
        return False, "empty or non-list path"
    grid = maze["grid"]
    width = int(maze["width"])
    height = int(maze["height"])
    start = (int(maze["start"][0]), int(maze["start"][1]))
    goal = (int(maze["goal"][0]), int(maze["goal"][1]))
    try:
        coords = [(int(step[0]), int(step[1])) for step in path]
    except (TypeError, ValueError, IndexError):
        return False, "path cells must be [x, y] integer pairs"
    if coords[0] != start:
        return False, f"path must start at {list(start)}, got {list(coords[0])}"
    if coords[-1] != goal:
        return False, f"path must end at {list(goal)}, got {list(coords[-1])}"
    seen: set[tuple[int, int]] = set()
    for i, (x, y) in enumerate(coords):
        if not (0 <= x < width and 0 <= y < height):
            return False, f"step {i} {[x, y]} out of bounds"
        cell = grid[y][x]
        if cell == "#":
            return False, f"step {i} {[x, y]} is a wall"
        if (x, y) in seen:
            return False, f"step {i} revisits cell {[x, y]}"
        seen.add((x, y))
        if i > 0:
            px, py = coords[i - 1]
            if abs(x - px) + abs(y - py) != 1:
                return False, f"step {i - 1}->{i} not 4-adjacent"
    return True, ""


def score(
    predictions: list[dict[str, Any]],
    ground_truth: dict[str, dict[str, Any]],
) -> EvalResult:
    pred_by_id = {
        p["id"]: (p.get("prediction") or {}) for p in predictions if "id" in p
    }

    per_doc: list[PerDocScore] = []
    for qid, expected in ground_truth.items():
        optimal_len = expected.get("optimal_len")
        maze = _load_maze(qid)
        pred = pred_by_id.get(qid)
        path = (pred or {}).get("path") if isinstance(pred, dict) else None

        if maze is None or not isinstance(optimal_len, int):
            per_doc.append(
                PerDocScore(
                    id=qid,
                    score=0.0,
                    fields=[],
                    error="malformed ground truth or missing maze file",
                )
            )
            continue

        ok, err = _validate(maze, path)
        if not ok:
            per_doc.append(
                PerDocScore(
                    id=qid,
                    score=0.0,
                    fields=[
                        FieldScore(
                            name="path_len",
                            expected=optimal_len,
                            predicted=len(path) if isinstance(path, list) else None,
                            match=False,
                            score=0.0,
                        )
                    ],
                    error=err,
                )
            )
            continue

        path_len = len(path)
        item_score = optimal_len / path_len if path_len > 0 else 0.0
        per_doc.append(
            PerDocScore(
                id=qid,
                score=item_score,
                fields=[
                    FieldScore(
                        name="path_len",
                        expected=optimal_len,
                        predicted=path_len,
                        match=item_score >= 0.999,
                        score=item_score,
                    )
                ],
            )
        )

    overall = sum(d.score for d in per_doc) / len(per_doc) if per_doc else 0.0

    failures: list[Failure] = []
    worst = sorted(per_doc, key=lambda d: d.score)[:3]
    for d in worst:
        if d.score >= 0.999:
            continue
        f = d.fields[0] if d.fields else None
        summary = d.error or (
            f"path_len={f.predicted if f else '?'} vs optimal={f.expected if f else '?'}"
        )
        failures.append(
            Failure(
                id=d.id,
                summary=summary,
                expected={"optimal_len": f.expected if f else None},
                predicted={"path_len": f.predicted if f else None},
            )
        )

    return EvalResult(
        metric_name="maze_path_optimality",
        overall=overall,
        target=0.98,
        per_doc=per_doc,
        failures=failures,
    )
