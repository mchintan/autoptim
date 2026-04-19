# Worker Contract (v2, frozen)

Every `process.py` you produce MUST conform to this contract. The harness imports the file and calls `run(...)` in an isolated subprocess.

## Required entrypoint

```python
def run(inputs: list[dict], ctx: dict) -> list[dict]:
    ...
```

## Inputs

`inputs` is a list of dicts, one per document/task-item:

```
{
  "id":   str,   # stable identifier, matches ground_truth.jsonl key
  "path": str,   # absolute path to the input file (may be PDF, image, text, ...)
  "mime": str    # best-effort MIME type, e.g. "application/pdf", "text/plain"
}
```

`ctx` is provided by the harness:

```
{
  "model_hint":   str,   # default model id from task.yaml (e.g. "google/gemma-4-e4b")
  "base_url":     str,   # OpenAI-compatible endpoint, e.g. "http://127.0.0.1:1234/v1"
  "api_key":      str,   # bearer token — use in Authorization header only, never log
  "scratch_dir":  str,   # absolute path, freely writable by the worker
  "log":          callable(str) -> None,   # captured to worker_stdout.log
  "per_iter_timeout_s": int,                # remaining wall-clock budget for this call
}
```

Call the worker via: `POST {base_url}/chat/completions` with `Authorization: Bearer {api_key}`.
Standard OpenAI chat-completions payload works (`{model, messages, temperature, max_tokens}`). The `openai` Python SDK works too.

This endpoint covers LM Studio (local), llama.cpp server (local), Groq / Together / Fireworks / OpenRouter / vLLM (cloud). They're all the same shape.

## Outputs

Return a list of dicts, one per input, in any order:

```
{
  "id":         str,    # MUST match an input id
  "prediction": <JSON>  # task-defined. Structure is judged by the evaluator.
}
```

Every input id must be represented exactly once. Missing ids count as empty predictions.

## Rules

1. **Determinism**: results depend only on `inputs` and `ctx`. No hidden state, no global mutation that outlives the call.
2. **Network**: the ONLY allowed outbound destination is `ctx["base_url"]`. Do not log, persist, or echo `ctx["api_key"]`.
3. **Filesystem**: you may read `inputs[i]["path"]` and write anywhere inside `ctx["scratch_dir"]`. Do not write elsewhere.
4. **Model choice**: use `ctx["model_hint"]` by default. If you switch models, log your choice via `ctx["log"]`. Only use models you have confirmed are available at `ctx["base_url"]`.
5. **Time**: must return within `ctx["per_iter_timeout_s"]`. If you parallelize, do so with `concurrent.futures`. No background threads that outlive `run`.
6. **Failure handling**: per-document errors should be caught, logged via `ctx["log"]`, and returned as `{"id": <id>, "prediction": null}`. Do NOT raise — a raise fails the whole iteration.
7. **No shell execution**: do not call `os.system`, `subprocess`, or similar. Python-only processing.
8. **No eval/exec** of LLM outputs.

## Encouraged patterns

- Use `httpx.post` or the `openai` Python SDK with `base_url=ctx["base_url"]` and `api_key=ctx["api_key"]`.
- Parse LLM JSON defensively: strip markdown fences, retry on parse failure with a tighter prompt.
- Log one line per document for traceability — but never log `ctx["api_key"]`.
- Keep the file self-contained — no imports from other files in the run directory.
