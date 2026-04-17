"""Thin adapter over Anthropic / OpenAI / OpenRouter so the meta-agent is provider-agnostic.

Every provider call must: (a) accept system + user + one tool schema, (b) return a
parsed dict from the tool call, and (c) report tokens_in / tokens_out.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class LLMResponse:
    tool_args: dict[str, Any]
    tokens_in: int
    tokens_out: int
    raw_text: str | None = None  # non-tool assistant text if any (debug)


class Provider(Protocol):
    def call_tool(
        self,
        *,
        system: str,
        user: str,
        tool_name: str,
        tool_description: str,
        tool_schema: dict[str, Any],
        model: str,
        temperature: float | None,
        max_tokens: int,
    ) -> LLMResponse: ...


class AnthropicProvider:
    def __init__(self, api_key: str):
        import anthropic  # lazy import so OpenAI-only users don't need it

        self._client = anthropic.Anthropic(api_key=api_key)

    def call_tool(
        self,
        *,
        system: str,
        user: str,
        tool_name: str,
        tool_description: str,
        tool_schema: dict[str, Any],
        model: str,
        temperature: float | None,
        max_tokens: int,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": model,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "tools": [
                {
                    "name": tool_name,
                    "description": tool_description,
                    "input_schema": tool_schema,
                }
            ],
            "tool_choice": {"type": "tool", "name": tool_name},
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        resp = self._client.messages.create(**kwargs)
        tool_args: dict[str, Any] | None = None
        raw_text_parts: list[str] = []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "tool_use" and getattr(block, "name", None) == tool_name:
                tool_args = dict(block.input)  # type: ignore[arg-type]
            elif btype == "text":
                raw_text_parts.append(block.text)  # type: ignore[attr-defined]
        if tool_args is None:
            raise RuntimeError(
                f"anthropic response contained no tool_use for {tool_name!r}: {resp.content!r}"
            )
        usage = resp.usage
        return LLMResponse(
            tool_args=tool_args,
            tokens_in=getattr(usage, "input_tokens", 0) or 0,
            tokens_out=getattr(usage, "output_tokens", 0) or 0,
            raw_text="\n".join(raw_text_parts) if raw_text_parts else None,
        )


class OpenAIProvider:
    def __init__(self, api_key: str, base_url: str | None = None):
        import openai

        self._client = openai.OpenAI(api_key=api_key, base_url=base_url) if base_url else openai.OpenAI(api_key=api_key)

    def call_tool(
        self,
        *,
        system: str,
        user: str,
        tool_name: str,
        tool_description: str,
        tool_schema: dict[str, Any],
        model: str,
        temperature: float | None,
        max_tokens: int,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_description,
                        "parameters": tool_schema,
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": tool_name}},
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        tool_args: dict[str, Any] | None = None
        if msg.tool_calls:
            call = msg.tool_calls[0]
            if call.function.name == tool_name:
                tool_args = json.loads(call.function.arguments)
        if tool_args is None:
            raise RuntimeError(
                f"openai response contained no function call for {tool_name!r}: {msg!r}"
            )
        usage = resp.usage
        return LLMResponse(
            tool_args=tool_args,
            tokens_in=getattr(usage, "prompt_tokens", 0) or 0,
            tokens_out=getattr(usage, "completion_tokens", 0) or 0,
            raw_text=msg.content,
        )


class OpenRouterProvider(OpenAIProvider):
    def __init__(self, api_key: str):
        super().__init__(api_key=api_key, base_url="https://openrouter.ai/api/v1")


def make_provider(provider: str, api_key: str) -> Provider:
    if provider == "anthropic":
        return AnthropicProvider(api_key)
    if provider == "openai":
        return OpenAIProvider(api_key)
    if provider == "openrouter":
        return OpenRouterProvider(api_key)
    raise ValueError(f"unknown provider: {provider!r}")
