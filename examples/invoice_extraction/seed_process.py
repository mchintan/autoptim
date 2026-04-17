"""Seed process.py for invoice extraction.

Deliberately mediocre. One naive prompt, one worker call per doc, no retries,
no verification. Establishes a baseline the meta-agent can improve on.

Supports two worker backends via `ctx`:

- `ctx["backend"] == "ollama"`        → POST to Ollama's native /api/chat
- `ctx["backend"] == "openai_compat"` → any OpenAI-shaped endpoint (Groq, Together,
                                        Fireworks, OpenRouter, Ollama's /v1 compat).
                                        Uses `ctx["base_url"]` + `ctx["api_key"]`.
"""
from __future__ import annotations

import json
from typing import Any

import httpx


def _read_text(path: str) -> str:
    with open(path, "r", errors="replace") as f:
        return f.read()


def _chat(ctx: dict, messages: list[dict], *, temperature: float = 0.2, timeout_s: int = 120) -> str:
    backend = ctx.get("backend", "ollama")
    model = ctx["model_hint"]

    if backend == "openai_compat":
        base_url = ctx["base_url"].rstrip("/")
        api_key = ctx["api_key"]
        r = httpx.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
            },
            timeout=timeout_s,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    # Default: Ollama native API
    host = ctx["ollama_host"].rstrip("/")
    r = httpx.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=min(timeout_s, 120),
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def _parse_json_loose(text: str) -> dict[str, Any]:
    text = text.strip()
    # Strip common markdown fencing
    if text.startswith("```"):
        text = text.strip("`")
        # After stripping backticks we may have a leading "json\n"
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        pass
    i = text.find("{")
    if i < 0:
        return {}
    depth = 0
    for j in range(i, len(text)):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[i : j + 1])
                    return obj if isinstance(obj, dict) else {}
                except json.JSONDecodeError:
                    return {}
    return {}


def run(inputs: list[dict], ctx: dict) -> list[dict]:
    log = ctx["log"]
    timeout = int(ctx.get("per_iter_timeout_s", 300))

    predictions: list[dict] = []
    for item in inputs:
        try:
            text = _read_text(item["path"])
            prompt = (
                "Extract invoice fields as JSON. Output ONLY JSON, no prose.\n\n"
                f"INVOICE:\n{text}\n\n"
                "Fields to extract: invoice_no, date, vendor, total, currency."
            )
            raw = _chat(
                ctx,
                [{"role": "user", "content": prompt}],
                temperature=0.2,
                timeout_s=timeout,
            )
            parsed = _parse_json_loose(raw)
            predictions.append({"id": item["id"], "prediction": parsed})
            # Preview of raw output so the meta-agent can see what the worker actually returned
            # (this ends up in worker_stdout.log, which the orchestrator tails into the next
            # iteration's prompt).
            preview = raw.replace("\n", " ")[:150]
            log(f"{item['id']}: ok ({len(raw)} chars) keys={sorted(parsed)} preview={preview!r}")
        except Exception as e:
            log(f"{item['id']}: error {type(e).__name__}: {e}")
            predictions.append({"id": item["id"], "prediction": None})
    return predictions
