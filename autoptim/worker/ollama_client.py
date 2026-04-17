"""Small helper for process.py to talk to a local Ollama.

Intentionally minimal: `chat()` for JSON/text generation, `list_models()` for
introspection, and `available()` for health checks.
"""
from __future__ import annotations

import json
import re
from typing import Any

import httpx

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434", timeout_s: float = 180.0):
        self.host = host.rstrip("/")
        self.timeout_s = timeout_s

    def available(self) -> bool:
        try:
            r = httpx.get(f"{self.host}/api/tags", timeout=5.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def list_models(self) -> list[str]:
        r = httpx.get(f"{self.host}/api/tags", timeout=10.0)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        format: str | None = None,  # e.g. "json" to coerce JSON output
        temperature: float = 0.2,
        num_ctx: int | None = None,
        top_p: float | None = None,
        timeout_s: float | None = None,
    ) -> str:
        """Return the assistant message content as a string."""
        options: dict[str, Any] = {"temperature": temperature}
        if num_ctx is not None:
            options["num_ctx"] = num_ctx
        if top_p is not None:
            options["top_p"] = top_p
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if format:
            payload["format"] = format
        r = httpx.post(
            f"{self.host}/api/chat",
            json=payload,
            timeout=timeout_s or self.timeout_s,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]

    def chat_json(
        self,
        model: str,
        messages: list[dict[str, str]],
        **kw: Any,
    ) -> dict[str, Any] | list[Any]:
        """Chat that insists on JSON; tolerates markdown fences and trailing prose."""
        kw.setdefault("format", "json")
        raw = self.chat(model, messages, **kw)
        return parse_json_loose(raw)


def parse_json_loose(text: str) -> dict[str, Any] | list[Any]:
    """Parse JSON forgivingly — strips markdown fences and locates the first {...} or [...]."""
    if not text:
        raise ValueError("empty LLM response")
    text = text.strip()
    m = _JSON_FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find first balanced JSON object or array
    for opener, closer in (("{", "}"), ("[", "]")):
        i = text.find(opener)
        if i < 0:
            continue
        depth = 0
        for j in range(i, len(text)):
            c = text[j]
            if c == opener:
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    snippet = text[i : j + 1]
                    try:
                        return json.loads(snippet)
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"could not parse JSON from LLM output: {text[:200]!r}")
