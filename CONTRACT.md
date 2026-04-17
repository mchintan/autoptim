# Worker Contract (v1, frozen)

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

`ctx` is a dict provided by the harness. Shape depends on the configured worker backend:

**Always present:**
```
{
  "backend":       str,          # "ollama" | "openai_compat"
  "model_hint":    str,          # default model id from task.yaml
  "scratch_dir":   str,          # absolute path, writable by the worker
  "log":           callable(str) -> None,   # captured to worker_stdout.log
  "per_iter_timeout_s": int,     # remaining wall-clock budget for this call
}
```

**When `backend == "ollama"` (local, free):**
```
  "ollama_host": str,            # e.g. "http://localhost:11434"
```
Call Ollama's native endpoint: `POST {ollama_host}/api/chat`.

**When `backend == "openai_compat"` (cloud: Groq, Together, Fireworks, OpenRouter, Ollama /v1):**
```
  "base_url": str,               # e.g. "https://api.groq.com/openai/v1"
  "api_key":  str,               # bearer token — keep in headers only, do NOT log
```
Call: `POST {base_url}/chat/completions` with `Authorization: Bearer {api_key}`.
The standard OpenAI chat-completions payload works. Use the `openai` Python SDK if you prefer.

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
2. **Network**: the ONLY allowed outbound destination is the configured worker endpoint:
   - backend=`ollama`        → `ctx["ollama_host"]` (localhost)
   - backend=`openai_compat` → `ctx["base_url"]` (the cloud provider in task.yaml)
   No other HTTP calls. Do not log, persist, or echo `ctx["api_key"]`.
3. **Filesystem**: you may read `inputs[i]["path"]` and write anywhere inside `ctx["scratch_dir"]`. Do not write elsewhere.
4. **Model choice**: use `ctx["model_hint"]` by default. You may switch to another Ollama model you have verified is locally available; if you do, `log(...)` which model and why.
5. **Time**: must return within `ctx["per_iter_timeout_s"]`. If you parallelize, do so with `concurrent.futures`. No background threads that outlive `run`.
6. **Failure handling**: per-document errors should be caught, logged via `ctx["log"]`, and returned as `{"id": <id>, "prediction": null}`. Do NOT raise — a raise fails the whole iteration.
7. **No shell execution**: do not call `os.system`, `subprocess`, or similar. Python-only processing.
8. **No eval/exec** of LLM outputs.

## Encouraged patterns

- Branch once on `ctx["backend"]` and call the right endpoint. Ollama uses `/api/chat` with `{model, messages, stream:false, options}`. OpenAI-compat uses `/chat/completions` with `{model, messages, temperature, max_tokens}` + `Authorization: Bearer {api_key}`.
- Parse LLM JSON defensively: strip markdown fences, retry on parse failure with a tighter prompt.
- Log one line per document so debugging is tractable — but never log `ctx["api_key"]`.
- Keep the file self-contained — no imports from other files in the run directory.
