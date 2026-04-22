# regex_synth

**Synthesize a regex from positive and negative example strings.** The worker LLM outputs one pattern per task; the evaluator actually **compiles** it and tests each example. No LLM-as-judge — the metric is "does this regex accept everything it should and reject everything it shouldn't?"

This is the simplest example of an **executable metric**: the output of `process.py` is a single string, and the evaluator runs it. If the model hallucinates a pattern that looks plausible but matches "192.168.1.1 " with a trailing space, the score surfaces that instantly. No room for the LLM being confidently wrong without the loop noticing.

## Tasks

10 hand-curated problems covering common real-world regex needs:

| ID | Task |
|----|------|
| `t_01_zip` | US 5-digit zip codes |
| `t_02_email` | simple email addresses |
| `t_03_hex_color` | `#rgb` or `#rrggbb` CSS hex |
| `t_04_phone_us` | `(xxx) xxx-xxxx` phone |
| `t_05_ipv4` | IPv4 octets 0–255, no leading zeros |
| `t_06_semver` | SemVer with optional pre-release |
| `t_07_uuid` | UUID 8-4-4-4-12 hex |
| `t_08_time24` | `HH:MM` 24-hour clock |
| `t_09_positive_int_no_leading_zero` | non-negative int, no leading zeros |
| `t_10_iso_date` | ISO-8601 dates in 1900–2099 |

Each task ships with 4–7 positives and 7–9 negatives. The scoring is `(correct_positives_matched + correct_negatives_rejected) / total_examples`, then averaged across tasks.

## Run

```bash
# Local via LM Studio
autoptim run examples/regex_synth/task.yaml --dashboard

# Cloud via Groq (no local model needed)
export GROQ_API_KEY=...
autoptim run examples/regex_synth/task_groq.yaml --dashboard
```

## What to expect

- **Seed** (single-shot, no CoT, no format guidance): typically lands around 0.55–0.70. Common failures: missing anchors (so leading/trailing whitespace slips through), accepting 6-digit numbers for zip codes, missing the 0-255 constraint in IPv4.
- **Mature process** (~10–15 iterations): should reach >0.95. Expected strategy progression:
  1. `prompt_mutation`: explicitly request `fullmatch`-compatible anchored patterns.
  2. `error_pattern_fix`: read per-task failures and target the most common (usually IPv4 octet bounds and SemVer pre-release edge cases).
  3. `pipeline_restructuring`: per-task specialized prompts with few-shot examples, or a verification pass that tests the regex against the examples before returning.

## Why this is a good template

Clone this example if your problem shape is: **"I want an LLM to write a small artifact (a SQL query, a regex, a glob, an AST-matcher rule) that I can then execute to score correctness."** The shape is:

- `inputs/<id>.json`: task definition + test cases
- `seed_process.py`: ask the LLM to produce the artifact
- `eval.py`: execute the artifact against the test cases; aggregate

Swap `_score_task` for your own executor (SQL against SQLite, glob against a file tree, whatever) and you're done.
