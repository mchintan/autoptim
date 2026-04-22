# support_triage

**Classify and extract structured fields from customer support tickets.** Same *shape* as `invoice_extraction` (schema-match scoring on structured extraction from unstructured text) but a completely different *domain*, so it serves as a concrete template for anyone who wants to adapt the harness to their own ticketing / inquiry / complaint workflow.

## What each ticket becomes

```json
{
  "category":         "billing | technical | account | feature_request | feedback | complaint",
  "priority":         "low | medium | high | urgent",
  "sentiment":        "negative | neutral | positive",
  "customer_id":      "string or null",
  "requested_action": "free text — one-line summary of what the customer wants"
}
```

Scored with `schema_match`:

| Field | Matcher | Weight | Notes |
|---|---|---|---|
| `category` | `exact` | 2 | enum; heavily weighted — mis-routing is costly |
| `priority` | `exact` | 2 | enum; drives on-call escalation |
| `sentiment` | `exact` | 1 | enum; used for satisfaction tracking |
| `customer_id` | `exact` | 1 | string or `null`; compares case-insensitively |
| `requested_action` | `fuzzy` | 1 | free text; 0.5 threshold is lenient on paraphrasing |

## Dataset

12 hand-authored tickets spanning:

- **Categories**: billing (3), technical (3), account (3), feature_request (2), feedback (1)
- **Priorities**: urgent (3), high (1), medium (3), low (5)
- **Sentiments**: negative (4), neutral (6), positive (2)
- **customer_id presence**: 7 tickets mention one; 5 don't
- **Tone variation**: SEV1 on-call page, angry cancellation threat, terse beginner question, polite feature request, short thank-you note

The intent is that no single failure mode dominates — the meta-agent needs to get both the enum-matching and the "when is there really no customer ID?" signal right.

## Run

```bash
# Local via LM Studio
autoptim run examples/support_triage/task.yaml --dashboard

# Cloud via Groq (no local model needed)
export GROQ_API_KEY=...
autoptim run examples/support_triage/task_groq.yaml --dashboard
```

## What to expect

- **Seed** (single-shot prompt with no enum vocabulary): typically ~0.55–0.70. Common failures: inventing new category values ("login_issue" instead of `account`), over-weighting politeness as `positive` sentiment when the body is clearly negative, guessing at customer IDs that aren't there.
- **Mature process** (~10–15 iterations): should reach >0.90. Expected strategy progression:
  1. `prompt_mutation`: pin the enum vocabulary and require `null` (not empty string or "N/A") when customer ID is absent.
  2. `error_pattern_fix`: address the most common misclassification (usually urgent-vs-high or feature_request-vs-account).
  3. `verification`: add a second pass that re-reads the ticket and corrects the `priority` field based on explicit urgency signals.

## Clone this template when...

...your problem is **"I have messy unstructured text and I want to extract a fixed schema of typed fields, then score per-field against a ground truth."** Swap:

- `inputs/` → your own text corpus (one file per item)
- `ground_truth.jsonl` → your labels
- `task.yaml` → update the `metric.schema` to your field names, matchers, and weights

Nothing else changes. No `eval.py` needed — `schema_match` handles all of it.
