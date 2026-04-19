"""Regression: providers must omit `temperature` when it's None.

Some reasoning models reject `temperature` in the request body entirely — not
merely ignoring it but returning HTTP 400.
"""
from __future__ import annotations

import types
from unittest.mock import MagicMock

from autoptim.meta.providers import OpenAIProvider


def _make_openai_fake():
    call = types.SimpleNamespace(
        function=types.SimpleNamespace(
            name="propose_iteration",
            arguments='{"hypothesis": "x"}',
        )
    )
    choice = types.SimpleNamespace(
        message=types.SimpleNamespace(content=None, tool_calls=[call])
    )
    resp = types.SimpleNamespace(
        choices=[choice],
        usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )
    fake = MagicMock()
    fake.chat.completions.create.return_value = resp
    return fake


def test_openai_omits_temperature_when_none():
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider._client = _make_openai_fake()

    provider.call_tool(
        system="sys",
        user="usr",
        tool_name="propose_iteration",
        tool_description="",
        tool_schema={"type": "object"},
        model="o1-preview",
        temperature=None,
        max_tokens=1000,
    )
    kwargs = provider._client.chat.completions.create.call_args.kwargs
    assert "temperature" not in kwargs


def test_openai_includes_temperature_when_set():
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider._client = _make_openai_fake()

    provider.call_tool(
        system="sys",
        user="usr",
        tool_name="propose_iteration",
        tool_description="",
        tool_schema={"type": "object"},
        model="gpt-4.1",
        temperature=0.3,
        max_tokens=1000,
    )
    kwargs = provider._client.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0.3


def _make_gemini_fake(tool_args):
    """Fake google-genai client returning a single function_call part."""
    function_call = types.SimpleNamespace(name="propose_iteration", args=tool_args)
    part = types.SimpleNamespace(function_call=function_call, text=None)
    candidate = types.SimpleNamespace(
        content=types.SimpleNamespace(parts=[part])
    )
    resp = types.SimpleNamespace(
        candidates=[candidate],
        usage_metadata=types.SimpleNamespace(
            prompt_token_count=42, candidates_token_count=17
        ),
    )
    client = MagicMock()
    client.models.generate_content.return_value = resp
    return client


def test_gemini_call_forces_function_and_parses_response():
    from autoptim.meta.providers import GeminiProvider

    provider = GeminiProvider.__new__(GeminiProvider)
    provider._client = _make_gemini_fake({"hypothesis": "x"})

    result = provider.call_tool(
        system="sys",
        user="usr",
        tool_name="propose_iteration",
        tool_description="",
        tool_schema={"type": "object", "properties": {}, "additionalProperties": False},
        model="gemini-3-pro",
        temperature=None,
        max_tokens=1000,
    )
    assert result.tool_args == {"hypothesis": "x"}
    assert result.tokens_in == 42
    assert result.tokens_out == 17

    kwargs = provider._client.models.generate_content.call_args.kwargs
    assert kwargs["model"] == "gemini-3-pro"
    # tool_config should force a function call on the named tool
    cfg = kwargs["config"]
    tc = cfg.tool_config.function_calling_config
    assert tc.mode == "ANY"
    assert tc.allowed_function_names == ["propose_iteration"]


def test_gemini_omits_temperature_when_none():
    from autoptim.meta.providers import GeminiProvider

    provider = GeminiProvider.__new__(GeminiProvider)
    provider._client = _make_gemini_fake({"hypothesis": "x"})

    provider.call_tool(
        system="sys",
        user="usr",
        tool_name="propose_iteration",
        tool_description="",
        tool_schema={"type": "object"},
        model="gemini-3-pro",
        temperature=None,
        max_tokens=1000,
    )
    cfg = provider._client.models.generate_content.call_args.kwargs["config"]
    # temperature was never set — Pydantic's .model_fields_set excludes unset fields
    assert "temperature" not in cfg.model_fields_set


def test_gemini_strips_unsupported_schema_keys():
    from autoptim.meta.providers import _strip_for_gemini

    stripped = _strip_for_gemini({
        "type": "object",
        "additionalProperties": False,
        "$schema": "http://example",
        "properties": {
            "x": {"type": "string", "additionalProperties": False},
        },
    })
    assert "additionalProperties" not in stripped
    assert "$schema" not in stripped
    assert "additionalProperties" not in stripped["properties"]["x"]


def test_make_provider_rejects_unknown():
    import pytest

    from autoptim.meta.providers import make_provider

    with pytest.raises(ValueError):
        make_provider("some-retired-vendor", "fake-key")
