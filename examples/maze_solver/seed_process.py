"""Seed process.py for the maze_solver task.

Finds a path from S to G in each maze with plain iterative DFS using a fixed
neighbor order (N, E, S, W) and marking cells visited on push. On perfect
mazes this returns the unique (optimal) path. On braided mazes it often
commits to an early branch and returns a path that's visibly longer than
the shortest — which is where the meta-agent's headroom is.

Experiment space:
- Swap DFS for BFS → guaranteed optimal on unit-weight grids.
- Use A* with a Manhattan heuristic → same result, fewer nodes expanded.
- Tie-breaking, bidirectional search, JPS, iterative deepening, etc.
"""
from __future__ import annotations

import json
from typing import Any


def _load_maze(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _solve(
    grid: list[str], start: tuple[int, int], goal: tuple[int, int]
) -> list[list[int]] | None:
    width = len(grid[0])
    height = len(grid)
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    stack: list[tuple[int, int]] = [start]
    dirs = ((0, -1), (1, 0), (0, 1), (-1, 0))  # N, E, S, W
    while stack:
        x, y = stack.pop()
        if (x, y) == goal:
            path: list[list[int]] = []
            cur: tuple[int, int] | None = (x, y)
            while cur is not None:
                path.append([cur[0], cur[1]])
                cur = parent[cur]
            path.reverse()
            return path
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < width
                and 0 <= ny < height
                and grid[ny][nx] != "#"
                and (nx, ny) not in parent
            ):
                parent[(nx, ny)] = (x, y)
                stack.append((nx, ny))
    return None


def run(inputs: list[dict], ctx: dict) -> list[dict]:
    log = ctx["log"]
    predictions: list[dict] = []
    for item in inputs:
        try:
            m = _load_maze(item["path"])
            grid = m["grid"]
            start = (int(m["start"][0]), int(m["start"][1]))
            goal = (int(m["goal"][0]), int(m["goal"][1]))
            path = _solve(grid, start, goal)
            predictions.append({"id": item["id"], "prediction": {"path": path}})
        except Exception as e:
            log(f"{item['id']}: error {type(e).__name__}: {e}")
            predictions.append({"id": item["id"], "prediction": None})

    for p in predictions[:2]:
        path = (p.get("prediction") or {}).get("path")
        log(f"{p['id']}: path_len={len(path) if path else None}")
    return predictions
