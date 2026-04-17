"""Seed process.py for invoice extraction.

Deliberately mediocre. One naive prompt, one model call, text-only input,
no retries, no verification. The point is to establish a baseline that the
meta-agent can then improve on.
"""
from __future__ import annotations

import json
from typing import Any

import httpx


def _read_text(path: str) -> str:
    with open(path, "r", errors="replace") as f:
        return f.read()


def _ollama_chat(host: str, model: str, prompt: str, timeout_s: int) -> str:
    r = httpx.post(
        f"{host.rstrip('/')}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.2},
        },
        timeout=min(timeout_s, 120),
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def _parse_json_loose(text: str) -> dict[str, Any]:
    # Try straight parse, then locate first balanced {...}
    text = text.strip()
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
    model = ctx["model_hint"]
    host = ctx["ollama_host"]
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
            raw = _ollama_chat(host, model, prompt, timeout)
            parsed = _parse_json_loose(raw)
            predictions.append({"id": item["id"], "prediction": parsed})
            # Log a preview of what Ollama actually returned — this is what the meta-agent
            # sees in the worker log tail, and it's the fastest way to spot "Ollama is
            # returning prose", "Ollama is returning empty", or "Ollama is returning
            # something the parser can't salvage" without reading the raw run artifacts.
            preview = raw.replace("\n", " ")[:150]
            log(f"{item['id']}: ok ({len(raw)} chars) keys={sorted(parsed)} preview={preview!r}")
        except Exception as e:
            log(f"{item['id']}: error {type(e).__name__}: {e}")
            predictions.append({"id": item["id"], "prediction": None})
    return predictions
