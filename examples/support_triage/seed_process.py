"""Seed process.py for support_triage.

Naive one-shot extraction: give the LLM the raw ticket text and ask for
JSON with five fields. No enum vocabulary hints, no few-shot examples,
no explicit null handling for customer_id. Plenty of room for the
meta-agent to improve.
"""
from __future__ import annotations

import json
from typing import Any

import httpx


def _read_text(path: str) -> str:
    with open(path, "r", errors="replace") as f:
        return f.read()


def _chat(ctx: dict, messages: list[dict], *, temperature: float = 0.2, timeout_s: int = 120) -> str:
    base_url = ctx["base_url"].rstrip("/")
    api_key = ctx.get("api_key") or "not-needed"
    model = ctx["model_hint"]
    r = httpx.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={"model": model, "messages": messages, "temperature": temperature},
        timeout=timeout_s,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _parse_json_loose(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
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
    timeout = int(ctx.get("per_iter_timeout_s", 180))

    predictions: list[dict] = []
    for item in inputs:
        try:
            text = _read_text(item["path"])
            prompt = (
                "Classify and extract from this customer support ticket. "
                "Output ONLY JSON with these keys: category, priority, "
                "sentiment, customer_id, requested_action.\n\n"
                f"TICKET:\n{text}"
            )
            raw = _chat(
                ctx,
                [{"role": "user", "content": prompt}],
                temperature=0.2,
                timeout_s=timeout,
            )
            parsed = _parse_json_loose(raw)
            predictions.append({"id": item["id"], "prediction": parsed})
            preview = raw.replace("\n", " ")[:150]
            log(f"{item['id']}: ok ({len(raw)} chars) keys={sorted(parsed)} preview={preview!r}")
        except Exception as e:
            log(f"{item['id']}: error {type(e).__name__}: {e}")
            predictions.append({"id": item["id"], "prediction": None})
    return predictions
