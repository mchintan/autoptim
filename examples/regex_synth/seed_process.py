"""Seed process.py for regex_synth.

For each task, ask the worker LLM to write a Python regex that matches all
`positives` and none of the `negatives`. Return the raw regex string.

Deliberately mediocre:
- One shot, no CoT, no examples of the output format.
- No anchoring hint (re.fullmatch vs re.search).
- No robustness to "here is a regex: `...`" prose wrapping.
"""
from __future__ import annotations

import json
from typing import Any

import httpx


def _load_task(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


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


def run(inputs: list[dict], ctx: dict) -> list[dict]:
    log = ctx["log"]
    timeout = int(ctx.get("per_iter_timeout_s", 180))

    predictions: list[dict] = []
    for item in inputs:
        try:
            task = _load_task(item["path"])
            prompt = (
                "Write a Python regular expression.\n\n"
                f"TASK: {task.get('task', '')}\n\n"
                f"POSITIVES (must match): {task['positives']}\n"
                f"NEGATIVES (must NOT match): {task['negatives']}\n\n"
                "Output only the regex pattern. No code fences, no explanation."
            )
            raw = _chat(
                ctx,
                [{"role": "user", "content": prompt}],
                temperature=0.1,
                timeout_s=timeout,
            )
            pattern = raw.strip()
            predictions.append(
                {"id": item["id"], "prediction": {"pattern": pattern}}
            )
            log(f"{item['id']}: pattern={pattern!r}")
        except Exception as e:
            log(f"{item['id']}: error {type(e).__name__}: {e}")
            predictions.append({"id": item["id"], "prediction": None})
    return predictions
