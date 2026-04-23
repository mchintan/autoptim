"""Generate the maze_solver test set.

Produces 20 mazes of mixed size. Most are lightly *braided* (a fraction of
interior walls knocked out after carving a perfect maze) so that multiple
start-to-goal paths exist — which means DFS/greedy solvers score strictly
worse than BFS/A*, giving the meta-agent headroom. A couple of pure perfect
mazes are mixed in so even weak solvers get some credit.

    python examples/maze_solver/build_dataset.py

Deterministic: fixed per-maze seed, so the checked-in dataset is reproducible.
"""
from __future__ import annotations

import json
import random
from collections import deque
from pathlib import Path

HERE = Path(__file__).parent

# (width, height, braid_fraction, seed). width and height must be odd so
# recursive-backtracker can place chambers on odd indices.
MAZE_SPECS: list[tuple[int, int, float, int]] = [
    (15, 15, 0.15, 1),
    (15, 15, 0.20, 2),
    (15, 15, 0.10, 3),
    (15, 15, 0.00, 4),   # perfect maze
    (17, 17, 0.15, 5),
    (17, 17, 0.20, 6),
    (17, 17, 0.10, 7),
    (19, 19, 0.15, 8),
    (19, 19, 0.20, 9),
    (19, 19, 0.15, 10),
    (21, 21, 0.15, 11),
    (21, 21, 0.20, 12),
    (21, 21, 0.10, 13),
    (21, 21, 0.00, 14),  # perfect maze
    (23, 23, 0.15, 15),
    (25, 25, 0.20, 16),
    (25, 25, 0.15, 17),
    (27, 27, 0.15, 18),
    (29, 29, 0.20, 19),
    (31, 31, 0.15, 20),
]


def _carve_perfect(width: int, height: int, rng: random.Random) -> list[list[str]]:
    assert width % 2 == 1 and height % 2 == 1
    grid = [["#"] * width for _ in range(height)]
    grid[1][1] = "."
    stack = [(1, 1)]
    while stack:
        x, y = stack[-1]
        candidates = []
        for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            nx, ny = x + dx, y + dy
            if 1 <= nx < width - 1 and 1 <= ny < height - 1 and grid[ny][nx] == "#":
                candidates.append((nx, ny, dx, dy))
        if not candidates:
            stack.pop()
            continue
        nx, ny, dx, dy = rng.choice(candidates)
        grid[ny][nx] = "."
        grid[y + dy // 2][x + dx // 2] = "."
        stack.append((nx, ny))
    return grid


def _braid(grid: list[list[str]], frac: float, rng: random.Random) -> None:
    height = len(grid)
    width = len(grid[0])
    candidates = []
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if grid[y][x] != "#":
                continue
            if grid[y][x - 1] == "." and grid[y][x + 1] == ".":
                candidates.append((x, y))
            elif grid[y - 1][x] == "." and grid[y + 1][x] == ".":
                candidates.append((x, y))
    rng.shuffle(candidates)
    for x, y in candidates[: int(len(candidates) * frac)]:
        grid[y][x] = "."


def _bfs_optimal_len(
    grid: list[list[str]], start: tuple[int, int], goal: tuple[int, int]
) -> int | None:
    height = len(grid)
    width = len(grid[0])
    q = deque([(start, 1)])
    seen = {start}
    while q:
        (x, y), d = q.popleft()
        if (x, y) == goal:
            return d
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < width
                and 0 <= ny < height
                and grid[ny][nx] != "#"
                and (nx, ny) not in seen
            ):
                seen.add((nx, ny))
                q.append(((nx, ny), d + 1))
    return None


def _build(width: int, height: int, braid_frac: float, seed: int) -> tuple[list[str], int]:
    rng = random.Random(seed)
    grid = _carve_perfect(width, height, rng)
    if braid_frac > 0:
        _braid(grid, braid_frac, rng)
    sx, sy = 1, 1
    gx, gy = width - 2, height - 2
    optimal = _bfs_optimal_len(grid, (sx, sy), (gx, gy))
    assert optimal is not None, f"unsolvable maze at seed={seed}"
    grid[sy][sx] = "S"
    grid[gy][gx] = "G"
    rows = ["".join(row) for row in grid]
    return rows, optimal


def main() -> None:
    inputs_dir = HERE / "inputs"
    inputs_dir.mkdir(exist_ok=True)
    for f in inputs_dir.glob("maze_*.json"):
        f.unlink()

    gt_lines: list[str] = []
    for i, (w, h, braid, seed) in enumerate(MAZE_SPECS, start=1):
        rows, optimal = _build(w, h, braid, seed)
        mid = f"maze_{i:04d}"
        (inputs_dir / f"{mid}.json").write_text(
            json.dumps(
                {
                    "id": mid,
                    "width": w,
                    "height": h,
                    "start": [1, 1],
                    "goal": [w - 2, h - 2],
                    "grid": rows,
                },
                indent=2,
            )
            + "\n"
        )
        gt_lines.append(json.dumps({"id": mid, "fields": {"optimal_len": optimal}}))

    (HERE / "ground_truth.jsonl").write_text("\n".join(gt_lines) + "\n")
    print(f"Wrote {len(MAZE_SPECS)} mazes to {inputs_dir}")
    print(f"Wrote ground truth to {HERE / 'ground_truth.jsonl'}")


if __name__ == "__main__":
    main()
