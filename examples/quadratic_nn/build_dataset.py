"""Generate the quadratic_nn test set.

Run once to produce ./inputs/q_NNNN.json and ./ground_truth.jsonl with 100
real-rooted quadratics whose coefficients and roots are drawn from a bounded
distribution the NN can plausibly learn.

    python examples/quadratic_nn/build_dataset.py

Deterministic: fixed seed, so the checked-in dataset is reproducible.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

N_PROBLEMS = 100
SEED = 0
HERE = Path(__file__).parent


def main() -> None:
    rng = random.Random(SEED)
    inputs_dir = HERE / "inputs"
    inputs_dir.mkdir(exist_ok=True)
    for f in inputs_dir.glob("q_*.json"):
        f.unlink()

    gt_lines: list[str] = []
    for i in range(1, N_PROBLEMS + 1):
        # Sample two real roots; that guarantees discriminant >= 0 and keeps
        # the coefficient magnitudes in a learnable range.
        r1 = round(rng.uniform(-5.0, 5.0), 3)
        r2 = round(rng.uniform(-5.0, 5.0), 3)
        a = round(rng.uniform(0.5, 2.0), 3)
        # (x - r1)(x - r2) * a  =>  a*x^2 - a*(r1+r2)*x + a*r1*r2
        b = round(-a * (r1 + r2), 4)
        c = round(a * r1 * r2, 4)

        qid = f"q_{i:04d}"
        (inputs_dir / f"{qid}.json").write_text(
            json.dumps({"a": a, "b": b, "c": c}, indent=2) + "\n"
        )
        sorted_roots = sorted([r1, r2])
        gt_lines.append(
            json.dumps({"id": qid, "fields": {"roots": sorted_roots}})
        )

    (HERE / "ground_truth.jsonl").write_text("\n".join(gt_lines) + "\n")
    print(f"Wrote {N_PROBLEMS} problems to {inputs_dir}")
    print(f"Wrote ground truth to {HERE / 'ground_truth.jsonl'}")


if __name__ == "__main__":
    main()
