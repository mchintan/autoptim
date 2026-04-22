"""Seed process.py for the quadratic_nn task.

Trains a tiny neural net to predict the two real roots of ax^2 + bx + c = 0
from the coefficients (a, b, c), then evaluates it on the held-out test set
passed in via `inputs`.

Deliberately mediocre baseline so the meta-agent has room to improve:
- 1 hidden layer of 8 units, ReLU
- 200 synthetic training samples
- 20 training steps
- SGD at a conservative LR
- Raw (a, b, c) features — no discriminant, no normalization

The meta-agent's experiment space is huge here: width/depth, activation,
optimizer, LR schedule, training steps, feature engineering (the single
biggest win is feeding b^2 - 4ac as a feature), loss function, and so on.
"""
from __future__ import annotations

import json
import random
from typing import Any


def _load_input(path: str) -> dict[str, float]:
    with open(path, "r") as f:
        return json.load(f)


def _solve_analytic(a: float, b: float, c: float) -> tuple[float, float]:
    """Used ONLY to generate synthetic training labels."""
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        disc = 0.0
    s = disc ** 0.5
    r1 = (-b - s) / (2.0 * a)
    r2 = (-b + s) / (2.0 * a)
    return (min(r1, r2), max(r1, r2))


def _make_training_data(n: int, seed: int = 0) -> tuple[list[list[float]], list[list[float]]]:
    rng = random.Random(seed)
    xs: list[list[float]] = []
    ys: list[list[float]] = []
    while len(xs) < n:
        r1 = rng.uniform(-5.0, 5.0)
        r2 = rng.uniform(-5.0, 5.0)
        a = rng.uniform(0.5, 2.0)
        b = -a * (r1 + r2)
        c = a * r1 * r2
        roots = sorted([r1, r2])
        xs.append([a, b, c])
        ys.append(roots)
    return xs, ys


def run(inputs: list[dict], ctx: dict) -> list[dict]:
    log = ctx["log"]

    # Deterministic seed so "did this iter actually change anything?" is signal, not noise
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(1)

    # --- training data ---
    train_x_raw, train_y_raw = _make_training_data(200, seed=0)
    train_x = torch.tensor(train_x_raw, dtype=torch.float32)
    train_y = torch.tensor(train_y_raw, dtype=torch.float32)

    # --- model ---
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 2),
    )
    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.001)

    # --- training ---
    model.train()
    for step in range(20):
        pred = model(train_x)
        loss = loss_fn(pred, train_y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0:
            log(f"train step {step}: loss={loss.item():.4f}")

    # --- predict on the held-out test set ---
    model.eval()
    predictions: list[dict] = []
    with torch.no_grad():
        for item in inputs:
            try:
                prob = _load_input(item["path"])
                x = torch.tensor(
                    [[prob["a"], prob["b"], prob["c"]]], dtype=torch.float32
                )
                out = model(x).squeeze(0).tolist()
                roots = sorted(float(v) for v in out)
                predictions.append(
                    {"id": item["id"], "prediction": {"roots": roots}}
                )
            except Exception as e:
                log(f"{item['id']}: error {type(e).__name__}: {e}")
                predictions.append({"id": item["id"], "prediction": None})

    # Log a few examples so the meta-agent can see what's actually happening
    for p in predictions[:3]:
        log(f"sample prediction: {p}")
    return predictions
