# autoptim

A harness that continuously improves a verifiable process.

You give it three things: a set of inputs, a ground truth, and a metric that scores a proposed output against that ground truth. You also give it a seed `process.py` — the thing being optimized. The harness then runs a simple improvement loop:

1. **Run** the current `process.py` against all inputs.
2. **Score** the outputs against the ground truth.
3. **Propose** a new `process.py` — a frontier model reads the current code, the score, the per-item breakdown, and the history of prior attempts, then rewrites the file with a stated hypothesis.
4. **Keep** the new version if it beat the previous best; otherwise discard and go back to the previous best.
5. **Repeat** until the target score is reached or a budget cap (iterations, wall-clock, or dollars) is hit.

The work inside `process.py` is done by a **local Ollama model** — cheap enough to call thousands of times per iteration. The **frontier meta-agent** (Gemini / Claude / OpenAI / OpenRouter) is called once per iteration to propose the next experiment. That split is what makes the loop affordable to run for many iterations.

## Quick start

```bash
pip install -e .
ollama pull gemma4:latest        # whichever Gemma tag you already have works — just match it in task.yaml
export GEMINI_API_KEY=...        # or ANTHROPIC_API_KEY / OPENAI_API_KEY / OPENROUTER_API_KEY
autoptim run examples/invoice_extraction/task.yaml
```

The shipped example uses `gemma4:latest` as the local worker and `gemini-3.1-flash-lite-preview` as the meta-agent. Edit the `worker.default_model` and `meta.model` entries in `task.yaml` to switch either one.

Inspect a run:

```bash
autoptim list                       # all runs
autoptim inspect <run_id>           # trajectory + best score
autoptim inspect <run_id> <iter>    # diff + hypothesis + eval breakdown
autoptim tail <run_id>              # live follow
```

## Components

- **Orchestrator** — drives the loop; enforces budgets; writes every iteration to `runs/<run_id>/iter_NNN/`.
- **Worker** — runs each candidate `process.py` inside a subprocess sandbox (timeout + memory cap, env scrubbed).
- **Evaluator** — `schema_match` for declarative field-level scoring, or a user-supplied `eval.py` for custom metrics.
- **Meta-agent** — frontier model that proposes the next `process.py` via a structured tool call. Output is Pydantic-validated before it hits disk.
- **Strategy scheduler** — tracks which axes of change have been tried (prompt tweaks, pipeline shape, model/params, error-driven fixes, input representation, verification passes) and forces the agent to an unused axis when it stalls or hill-climbs.

See [CONTRACT.md](CONTRACT.md) for the `process.py` interface.
