# quadratic_nn

**Train a tiny neural network to solve quadratic equations.** The twist: the neural network's architecture, optimizer, features, and training schedule are rewritten by the autoptim meta-agent each iteration. The loop isn't optimizing a prompt — it's optimizing a training run.

This is the purest demonstration of autoptim's premise: continuously improve a verifiable process. Here the process is the code in `process.py` that trains a model end-to-end. No LLM is involved inside `process.py` at all.

## What gets optimized

The seed is deliberately underpowered:

- 1 hidden layer of 8 units, ReLU
- 200 synthetic training samples
- 20 SGD steps at LR 0.001
- Raw `(a, b, c)` features (no discriminant, no normalization)

Expected seed score: ~0.10–0.30. Expected trajectory as the meta-agent iterates:

1. **Feature engineering** (biggest single win): feed `b² - 4ac` (the discriminant) as a feature. The closed-form solution becomes almost trivial from there.
2. **More training data + steps**.
3. **Adam optimizer with a reasonable LR** (1e-3 → 1e-2).
4. **Wider / deeper network**.
5. **Normalize inputs** so the optimizer converges cleanly.

A mature `process.py` scores >0.90.

## Setup

```bash
pip install -e ".[examples]"      # pulls in torch + numpy
export GEMINI_API_KEY=...          # or set another meta-agent provider in task.yaml
```

No worker-tier LLM needed — the `worker.no_endpoint: true` flag tells autoptim there's no chat endpoint to preflight.

## Build the dataset (only needed if you want to regenerate)

```bash
python examples/quadratic_nn/build_dataset.py
```

Produces 100 problems in `inputs/q_NNNN.json` and roots in `ground_truth.jsonl`. Deterministic (fixed seed), so the checked-in dataset is the one you get.

## Run

```bash
autoptim run examples/quadratic_nn/task.yaml --dashboard
```

Watch the score trajectory climb. Each iteration, the meta-agent:

- Sees the current `process.py` (the training code)
- Sees the per-problem score breakdown
- Sees which problems are furthest from the correct roots
- Proposes a new `process.py` with a stated rationale

The `autoptim inspect <run_id> <iter>` command shows the exact diff vs. the parent, including which hyperparameters / features / architecture changed.

## Why it's a good template

If you have an ML problem where:
- The metric is programmatic (numeric error, test pass rate, held-out accuracy),
- You can write a `process.py` that trains and evaluates a model end-to-end in under a few minutes, and
- You suspect there's headroom you haven't found yet,

…clone this example. The only things you need to change are the dataset, the seed architecture, and the evaluator.
