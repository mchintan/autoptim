# maze_solver

**Drop a walker at `S`, find the way out to `G`.** The meta-agent rewrites the pathfinder in `process.py` each iteration. No LLM inside the worker — `process.py` is pure Python solving 20 mazes in milliseconds.

This is a policy/algorithm optimization example in the same family as `quadratic_nn`, but the thing being optimized is a classical search routine rather than a training run. The trajectory reads almost like a textbook: DFS → BFS → A*.

## What it looks like

Here's `maze_0001` (15×15, lightly braided). Left is the raw maze; middle is the seed DFS solver's path (65 steps); right is the optimal BFS path (37 steps). You can see the DFS walker commit to a long detour through the upper-right chamber before eventually finding the goal, while BFS takes the straight shot down the left column.

```
              raw                         DFS  (65 steps)                  BFS  (37 steps)
###############              ###############                   ###############
#S....#.......#              #S....#..·····#                   #S····#.......#
#.###.###.###.#              #·###.###·###·#                   #.###·###.###.#
#...#.........#              #·..#....···.·#                   #...#·········#
#.#.##..###.#.#              #·#.##..###·#·#                   #.#.##..###.#·#
#...#...#...#.#              #·..#...#···#·#                   #...#...#...#·#
#.#.#.###.###.#              #·#.#.###·###·#                   #.#.#.###.###·#
#.#.#.#...#.#.#              #·#.#.#···#.#·#                   #.#.#.#...#.#·#
#.###.#.###.#.#              #·###.#·###.#·#                   #.###.#.###.#·#
#.....#.#.....#              #·....#·#·····#                   #.....#.#·····#
#.#####.#.#####              #·#####·#·#####                   #.#####.#·#####
#.......#.#...#              #·.·····#·#···#                   #.......#·#···#
#...#####.#.#.#              #·.·#####·#·#·#                   #...#####·#·#·#
#...#.......#G#              #···#....···#G#                   #...#....···#G#
###############              ###############                   ###############
```

The gap gets more dramatic on larger braided mazes — on `maze_0015` (23×23) the seed's DFS produces a 99-step path against an optimum of 49. That's the score 0.495 that the overall average is averaging in, and it's the score BFS closes to 1.000.

## The task

Each input is a maze:

```json
{
  "id": "maze_0001",
  "width": 15, "height": 15,
  "start": [1, 1], "goal": [13, 13],
  "grid": ["###############",
           "#S.#.........#",
           ...]
}
```

`process.py` returns `{"path": [[x, y], ...]}` — the sequence of 4-adjacent open cells from `S` to `G`. The evaluator validates the path and scores:

```
score_per_maze = optimal_len / your_path_len     (invalid path → 0)
overall        = mean across 20 mazes
```

So 1.0 means shortest path on every maze. The dataset is 20 mazes mixing pure (single-solution) mazes and lightly braided mazes with multiple paths — braiding is what makes DFS measurably worse than BFS.

## What gets optimized

Seed is iterative **DFS with a fixed N/E/S/W neighbor order**, returning the first path it reaches via parent pointers. That's optimal on perfect mazes and often badly suboptimal on braided ones. Seed scores ~0.77.

Expected arc:

1. **Swap DFS → BFS** — biggest single win; guaranteed optimal on unit-weight grids. Scores jump to 1.00.
2. **A\* with Manhattan heuristic** — same score, fewer nodes expanded (if you care about speed).
3. **Bidirectional search, JPS, tie-breaking, iterative deepening** — diminishing-returns refinements.

Target is **0.98**; any correct BFS implementation clears it.

## Setup

```bash
pip install -e .
export GEMINI_API_KEY=...
```

No worker-tier LLM needed (`worker.no_endpoint: true`).

## Build the dataset (only to regenerate)

```bash
python examples/maze_solver/build_dataset.py
```

Deterministic per-maze seeds — the checked-in dataset is what you'll get.

## Run

```bash
autoptim run examples/maze_solver/task.yaml --dashboard
```

Each iteration takes well under a second on CPU, so the full run finishes in a minute or two. `autoptim inspect <run_id> <iter>` shows the diff vs. parent plus the per-maze `path_len vs optimal` breakdown — the mazes where you're strictly suboptimal are exactly the braided ones.

## Why it's a good template

Clone this example when your problem is **"I have an algorithm with a programmatic correctness metric and I want the harness to discover a better algorithm, not a better prompt."** Swap the dataset builder, swap the scorer, and the rest is the same shape.
