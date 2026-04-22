# autoptim

A harness that continuously improves a verifiable process.

You give it three things: a set of inputs, a ground truth, and a metric that scores a proposed output against that ground truth. You also give it a seed `process.py` — the thing being optimized. The harness then runs a simple improvement loop:

1. **Run** the current `process.py` against all inputs.
2. **Score** the outputs against the ground truth.
3. **Propose** a new `process.py` — a frontier model reads the current code, the score, the per-item breakdown, and the history of prior attempts, then rewrites the file with a stated hypothesis.
4. **Keep** the new version if it beat the previous best; otherwise discard and go back to the previous best.
5. **Repeat** until the target score is reached or a budget cap (iterations, wall-clock, or dollars) is hit.

The work inside `process.py` is done by a **worker** LLM — cheap enough to call thousands of times per iteration. The **frontier meta-agent** (Gemini / OpenAI / OpenRouter) is called once per iteration to propose the next experiment. That split is what makes the loop affordable to run for many iterations.

The worker is any **OpenAI-compatible chat endpoint** — LM Studio locally, or a cloud provider (Groq / Together / Fireworks / OpenRouter / vLLM). One code path, one contract.

## Quick start

**Local (LM Studio, free):**

```bash
pip install -e .
# Open LM Studio → load a model (e.g. google/gemma-4-e4b) → click "Start Server"
export GEMINI_API_KEY=...
autoptim run examples/invoice_extraction/task.yaml
```

**Cloud (Groq free tier, no local model needed):**

```bash
pip install -e .
export GEMINI_API_KEY=...
export GROQ_API_KEY=gsk_...      # from https://console.groq.com/keys
autoptim run examples/invoice_extraction/task_groq.yaml
```

To point at a different provider, edit the task file: `worker.base_url` (the `/v1` endpoint), `worker.default_model`, and optionally `worker.api_key_env` (the env-var name — omit for auth-less local servers).

## What it looks like

### Per-iteration output (main terminal during `autoptim run`)

Every iteration prints the meta-agent's reasoning before any code runs, then the colorized diff, then the per-doc scores and a pass/fail summary:

```
──────────────── iter 3  (parent = iter 1) ─────────────────
╭───────── iter 3 proposal  ·  from iter 1 ──────────╮
│ strategy       prompt_mutation                     │
│ predicted Δ    +0.040                              │
│ frontier cost  4,412 in / 918 out  (cumulative     │
│                $0.06)                              │
│ forced axis    prompt_mutation (scheduler)         │
│                                                    │
│ Rationale (why this experiment):                   │
│ Three docs still fail on date (04/02/2024 parsed   │
│ as Feb 4 not Apr 2). Parent prompt doesn't tell    │
│ the model which format its input uses. I'll add    │
│ an explicit "output dates as YYYY-MM-DD" rule and  │
│ two few-shot examples showing the conversion.      │
│                                                    │
│ Hypothesis: Constraining output to ISO-8601 dates  │
│ will fix inv_002/007/009.                          │
│                                                    │
│ Expected failure modes:                            │
│   • model may still emit local-format dates        │
│   • few-shot bias may hurt other fields            │
╰────────────────────────────────────────────────────╯
╭─────────────── diff vs parent ─────────────────────╮
│ --- iter_001/process.py                            │
│ +++ iter_003/process.py                            │
│ @@ -24,7 +24,15 @@                                 │
│      prompt = (                                    │
│ -        "Extract invoice fields as JSON..."       │
│ +        "Extract invoice fields as JSON.\n"       │
│ +        "Dates: always output YYYY-MM-DD.\n"      │
│ +        "Example: '04/02/2024 (EU format)' →      │
│ +         '2024-02-04'.\n"                         │
│ +        "Example: 'Apr 2, 2024' → '2024-04-02'.\n"│
│          f"INVOICE:\n{text}\n\n"                   │
│      )                                             │
╰────────────────────────────────────────────────────╯
 iter 3 — per-doc scores
┏━━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃ doc     ┃ score ┃ error ┃
┡━━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ inv_001 │ 1.000 │       │
│ inv_002 │ 1.000 │       │
│ inv_003 │ 1.000 │       │
│ inv_004 │ 0.833 │       │
│ inv_005 │ 1.000 │       │
│ inv_006 │ 0.968 │       │
│ inv_007 │ 1.000 │       │
│ inv_008 │ 0.833 │       │
│ inv_009 │ 1.000 │       │
│ inv_010 │ 1.000 │       │
└─────────┴───────┴───────┘
╭──────────────── iter 3 result ─────────────────────╮
│ overall = 0.963 / target 0.980   Δ=+0.050          │
│ (predicted +0.040)   NEW BEST                      │
│ worker 58.1s · frontier 4,412 in / 918 out ·       │
│ cumulative $0.06                                   │
│                                                    │
│ worst failures:                                    │
│ ✗ inv_004: vendor: expected='Northwind Traders     │
│   GmbH' got='Northwind Traders GmbH, Musterstraße  │
│   15, 10115 Berlin'                                │
│ ✗ inv_008: currency: expected='USD' got=None       │
╰────────────────────────────────────────────────────╯
```

### Live dashboard (`autoptim tail <run_id>` or `autoptim run --dashboard`)

A Lightning-style terminal dashboard refreshing every 2 seconds — score sparkline, budget meters, last 8 iterations, and a "what's happening right now" panel:

```
╭────────────────────────────────────────────────────────────╮
│ invoice_extraction-20260419-140322   target 0.980   ● running │
╰────────────────────────────────────────────────────────────╯
╭─ trajectory ───────────────────────────────────────────────╮
│ scores over 7 iterations  (▁=0.00 █=1.00, target 0.980):   │
│ ▆▇▆▇█▇█                                                     │
│ best: iter 6 = 0.974   ✓ near target                        │
╰────────────────────────────────────────────────────────────╯
╭─ budgets ──────────────────────────────────────────────────╮
│ iters     ████████░░░░░░░░░░░░░░░░░░░░░░░░░░   7/30        │
│ wall      ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   8m04s / 4.0h│
│ frontier  █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   $0.12 / $5  │
╰────────────────────────────────────────────────────────────╯
╭─ recent iterations ────────────────────────────────────────╮
│ iter │ parent │ strategy              │ score │   Δ   │ ✓  │
│ ─────┼────────┼───────────────────────┼───────┼───────┼────│
│   0  │  seed  │ seed                  │ 0.913 │   —   │ ★  │
│   1  │   0    │ prompt_mutation       │ 0.928 │ +0.015│ ★  │
│   2  │   1    │ error_pattern_fix     │ 0.920 │ -0.008│    │
│   3  │   1    │ prompt_mutation       │ 0.963 │ +0.035│ ★  │
│   4  │   3    │ pipeline_restructur.. │ 0.941 │ -0.022│    │
│   5  │   3    │ verification          │ 0.961 │ -0.002│    │
│   6  │   3    │ representation        │ 0.974 │ +0.011│ ★  │
╰────────────────────────────────────────────────────────────╯
╭─ now ──────────────────────────────────────────────────────╮
│ ● running worker  iter 7  strategy=verification            │
│ rationale: Add a second pass that re-reads the source      │
│ for inv_004 and inv_008 to strip address lines from the    │
│ vendor field and confirm currency.                         │
│ scheduler: Strategy usage so far: prompt_mutation=2,       │
│ pipeline_restructuring=1, model_param_swap=0...            │
╰────────────────────────────────────────────────────────────╯
```

### Postmortem (`autoptim inspect <run_id> <iter>`)

Saves everything to disk so you can reconstruct any decision after the fact:

```
$ autoptim inspect invoice_extraction-20260419-140322 3
invoice_extraction-20260419-140322 / iter 3
  parent: 1
  strategy: prompt_mutation

rationale:
  Three docs still fail on date (04/02/2024 parsed as Feb 4 not Apr 2)...

hypothesis:
  Constraining output to ISO-8601 dates will fix inv_002/007/009.

expected failure modes:
  • model may still emit local-format dates
  • few-shot bias may hurt other fields

predicted Δ: +0.04  ·  tokens: 4,412 in / 918 out

[... unified diff vs parent ...]
[... per-doc eval table ...]
```

## Components

- **Orchestrator** — drives the loop; enforces budgets; writes every iteration to `runs/<run_id>/iter_NNN/`.
- **Worker** — runs each candidate `process.py` inside a subprocess sandbox (timeout + memory cap, env scrubbed). Talks to an OpenAI-compatible chat endpoint.
- **Evaluator** — `schema_match` for declarative field-level scoring, or a user-supplied `eval.py` for custom metrics.
- **Meta-agent** — frontier model that proposes the next `process.py` via a structured tool call (rationale + hypothesis + strategy tag + predicted delta + expected failure modes). Output is Pydantic-validated before it hits disk.
- **Strategy scheduler** — tracks which axes of change have been tried (prompt tweaks, pipeline shape, model/params, error-driven fixes, input representation, verification passes) and forces the agent to an unused axis when it stalls or hill-climbs.

See [CONTRACT.md](CONTRACT.md) for the `process.py` interface.

## Examples

Four shipped examples spanning the main use-case shapes. Pick whichever matches your problem:

### [`invoice_extraction/`](examples/invoice_extraction/) — LLM pipeline, declarative metric

Extract structured fields (`invoice_no`, `date`, `vendor`, `total`, `currency`) from messy text invoices. Scored with `schema_match` — per-field exact/fuzzy/numeric/date matchers with weighted aggregation. Ships 10 hand-authored invoices across multiple currencies and date formats.

```bash
autoptim run examples/invoice_extraction/task.yaml             # LM Studio
autoptim run examples/invoice_extraction/task_groq.yaml        # Groq cloud
```

Clone this template when your problem is **"extract structured JSON from unstructured text, score against a field schema."**

### [`support_triage/`](examples/support_triage/) — LLM pipeline, declarative metric, different domain

Classify and extract from customer support tickets: `category`, `priority`, `sentiment`, `customer_id` (nullable), `requested_action`. Same `schema_match` machinery as `invoice_extraction` but a different domain, so it's the concrete template for adapting the harness to your own ticketing / inquiry workflow. Ships 12 hand-authored tickets balanced across enum values and `customer_id` presence.

```bash
autoptim run examples/support_triage/task.yaml                 # LM Studio
autoptim run examples/support_triage/task_groq.yaml            # Groq cloud
```

Clone this template when your problem is **"extract a typed-field schema from unstructured text with enum vocabularies."**

### [`regex_synth/`](examples/regex_synth/) — LLM pipeline, executable metric

Worker LLM writes a Python regex given positive and negative string examples. The evaluator **compiles the regex and tests every example** — no LLM-as-judge. 10 hand-curated tasks covering US zip codes, emails, hex colors, phone numbers, IPv4 addresses, SemVer, UUID, 24h times, integers without leading zeros, and ISO dates.

```bash
autoptim run examples/regex_synth/task.yaml                    # LM Studio
autoptim run examples/regex_synth/task_groq.yaml               # Groq cloud
```

Expected trajectory: seeds land around 0.55–0.70 (common failures: missing anchors, out-of-range IPv4 octets). Mature patterns hit >0.95 in ~10–15 iterations.

Clone this template when your problem is **"LLM writes a small artifact (regex, SQL, glob, config snippet) that you can execute to score correctness."**

### [`quadratic_nn/`](examples/quadratic_nn/) — ML-training optimization, no LLM worker

The meta-agent rewrites a **PyTorch training loop** each iteration. `process.py` trains a small neural network on the fly, then predicts roots of held-out quadratics. No worker-tier LLM calls at all — only the frontier meta-agent spends tokens.

```bash
pip install -e ".[examples]"                                   # adds torch
autoptim run examples/quadratic_nn/task.yaml --dashboard
```

Seed is deliberately underpowered (1 hidden layer of 8 units, 20 SGD steps, raw coefficients as features) and scores 0.23. A mature implementation (discriminant feature + wider net + Adam + more steps) reaches 0.98. Each iteration takes under a second on CPU, so runs finish quickly.

Clone this template when your problem is **"I have a training loop with a programmatic metric and I suspect there's headroom I haven't found yet."**

## Commands

```
autoptim run <task.yaml>              # start a run (add --dashboard to auto-open a second window)
autoptim run <task.yaml> --resume     # resume the latest run for this task
autoptim tail <run_id>                # live Lightning-style dashboard
autoptim list                         # all runs under ./runs
autoptim inspect <run_id>             # trajectory summary
autoptim inspect <run_id> <iter>      # rationale + diff + per-doc breakdown
autoptim replay <run_id> <iter>       # re-execute that iter's process.py (debug)
autoptim models --provider gemini     # list models available to your Gemini key
```

## Real screenshots

The ASCII snippets above are accurate mockups of the real panels; to capture native screenshots, run `autoptim run --dashboard examples/invoice_extraction/task.yaml` in a terminal that supports true color, screenshot the two windows, and drop the PNGs into `docs/img/`.
