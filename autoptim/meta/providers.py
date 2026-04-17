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


class GeminiProvider:
    """Google Gemini via the native google-genai SDK.

    The OpenAI-compatible endpoint at generativelanguage.googleapis.com is
    available, but it rejects requests with "Multiple authentication credentials
    received" when the Authorization header overlaps with ambient Google auth
    in the environment. The native SDK sidesteps the problem entirely.

    API key env var: `GEMINI_API_KEY` (preferred) or `GOOGLE_API_KEY` (fallback).
    """

    def __init__(self, api_key: str):
        from google import genai

        self._genai = genai
        self._client = genai.Client(api_key=api_key)

    def list_models(self) -> list[str]:
        """Best-effort model listing — used by the CLI preflight."""
        try:
            names: list[str] = []
            for m in self._client.models.list():
                n = getattr(m, "name", "") or ""
                # Google returns "models/<id>" — strip the prefix for display/config
                names.append(n.split("/", 1)[1] if n.startswith("models/") else n)
            return [n for n in names if n]
        except Exception:
            return []

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
        from google.genai import types

        # Gemini's function declarations accept an OpenAPI-shaped schema. Strip fields
        # the subset doesn't support to avoid silent weirdness.
        clean_schema = _strip_for_gemini(tool_schema)

        tool = types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=tool_name,
                    description=tool_description,
                    parameters=clean_schema,  # type: ignore[arg-type]
                )
            ]
        )

        cfg_kwargs: dict[str, Any] = {
            "system_instruction": system,
            "tools": [tool],
            "tool_config": types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=[tool_name],
                )
            ),
            "max_output_tokens": max_tokens,
        }
        if temperature is not None:
            cfg_kwargs["temperature"] = temperature

        resp = self._client.models.generate_content(
            model=model,
            contents=user,
            config=types.GenerateContentConfig(**cfg_kwargs),
        )

        tool_args: dict[str, Any] | None = None
        raw_text_parts: list[str] = []
        for cand in resp.candidates or []:
            content = getattr(cand, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", None) or []:
                fc = getattr(part, "function_call", None)
                if fc and getattr(fc, "name", None) == tool_name:
                    tool_args = dict(fc.args or {})
                elif getattr(part, "text", None):
                    raw_text_parts.append(part.text)

        if tool_args is None:
            preview = "\n".join(raw_text_parts)[:500]
            raise RuntimeError(
                f"gemini response contained no function_call for {tool_name!r}. "
                f"Text preview: {preview!r}"
            )

        usage = getattr(resp, "usage_metadata", None)
        return LLMResponse(
            tool_args=tool_args,
            tokens_in=getattr(usage, "prompt_token_count", 0) or 0,
            tokens_out=getattr(usage, "candidates_token_count", 0) or 0,
            raw_text="\n".join(raw_text_parts) if raw_text_parts else None,
        )


_GEMINI_UNSUPPORTED_KEYS = {"additionalProperties", "$schema", "$defs", "definitions"}


def _strip_for_gemini(schema: Any) -> Any:
    """Recursively drop JSON-Schema keywords that Gemini's function-calling subset rejects."""
    if isinstance(schema, dict):
        return {
            k: _strip_for_gemini(v)
            for k, v in schema.items()
            if k not in _GEMINI_UNSUPPORTED_KEYS
        }
    if isinstance(schema, list):
        return [_strip_for_gemini(v) for v in schema]
    return schema


def make_provider(provider: str, api_key: str) -> Provider:
    if provider == "anthropic":
        return AnthropicProvider(api_key)
    if provider == "openai":
        return OpenAIProvider(api_key)
    if provider == "openrouter":
        return OpenRouterProvider(api_key)
    if provider == "gemini":
        return GeminiProvider(api_key)
    raise ValueError(f"unknown provider: {provider!r}")
