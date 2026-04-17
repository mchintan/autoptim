# Experiment Strategy Catalog

You will pick exactly one `strategy_tag` per iteration. It must be one of:

- `prompt_mutation`
- `pipeline_restructuring`
- `model_param_swap`
- `error_pattern_fix`
- `representation`
- `verification`

## prompt_mutation

Tweak how the LLM is asked to produce the output. Keep the pipeline shape fixed.

- Add chain-of-thought ("think step by step inside <scratch>, then emit JSON")
- Add few-shot examples drawn from a held-out subset
- Tighten the output JSON schema and show a canonical empty example
- Change the role / persona ("You are a meticulous accounts-payable clerk…")
- Add negative examples of common failure modes
- Add an explicit "if the value is not clearly present, return null" rule
- Switch from free-form to structured-output mode (`format="json"`)

## pipeline_restructuring

Change the shape of the process. Not just the prompt — the flow.

- Single-pass → extract-then-verify (two LLM calls: extract, then self-audit against the doc)
- Per-field specialist calls (one focused prompt per field rather than one omnibus prompt)
- Chunk-and-merge for long documents (split by page / token window, extract per chunk, merge)
- OCR pre-pass when the PDF text layer is poor
- Two-model ensemble with a deterministic disagreement resolver
- Retrieval-first: find the single most relevant page / paragraph, then extract

## model_param_swap

Change the model or its sampling parameters. Only use Ollama models you've verified are installed.

- Swap to a larger / stronger installed model (check `ctx["ollama_host"]/api/tags` at run start)
- Swap to a smaller model and compensate with better prompting (faster iteration)
- Lower temperature for determinism, raise it for diversity + self-consistency voting
- Raise `num_ctx` when documents are long
- Adjust `top_p`

## error_pattern_fix

Study the latest `eval.failures` and the per-doc breakdown. Cluster the dominant failure, then write a targeted fix.

- E.g. "vendor name is often an address line" → add explicit vendor-vs-address rule + example
- E.g. "totals off by tax" → prompt for `subtotal`, `tax`, `total` separately and reconcile
- E.g. "dates in mixed formats" → normalize via an explicit format-conversion step

## representation

Change how the input is presented to the LLM.

- PDF → plain text (pypdf) vs. PDF → markdown (preserving headings) vs. layout-preserving text
- Feed page images instead of text (requires a vision-capable model — verify first)
- Pre-extract tables with a deterministic parser, feed as markdown tables
- Feed the full document vs. a summary + relevant excerpts

## verification

Add a self-checking step that catches errors the first pass made.

- After extraction, ask the model to locate each field in the source and strike those that can't be supported
- Cross-field consistency checks (e.g. `line_items_total ≈ subtotal`)
- Confidence gating: ask the model to rate each field, re-extract low-confidence ones with a different approach

## Rules

- Do **not** stack multiple axes in one iteration. One clean change per experiment or you can't learn from the result.
- If the strategy scheduler directs you to a specific axis, you MUST use that axis this iteration.
- Your `hypothesis` should be a falsifiable sentence: "Swapping to qwen2.5:14b will improve total-field accuracy because…"
- Your `predicted_delta` should be realistic. Overclaiming reduces trust.
