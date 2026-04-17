"""Regression: providers must omit `temperature` when it's None.

Some reasoning models (e.g. claude-opus-4-7) reject `temperature` in the request
body entirely — not merely ignoring it but returning HTTP 400.
"""
from __future__ import annotations

import types
from unittest.mock import MagicMock

from autoptim.meta.providers import AnthropicProvider, OpenAIProvider


def _make_anthropic_fake(return_tool_args):
    # Construct a fake response object shaped like the anthropic SDK returns
    tool_block = types.SimpleNamespace(
        type="tool_use",
        name="propose_iteration",
        input=return_tool_args,
    )
    resp = types.SimpleNamespace(
        content=[tool_block],
        usage=types.SimpleNamespace(input_tokens=10, output_tokens=5),
    )
    fake_client = MagicMock()
    fake_client.messages.create.return_value = resp
    return fake_client


def test_anthropic_omits_temperature_when_none():
    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider._client = _make_anthropic_fake({"hypothesis": "x"})

    result = provider.call_tool(
        system="sys",
        user="usr",
        tool_name="propose_iteration",
        tool_description="",
        tool_schema={"type": "object"},
        model="claude-opus-4-7",
        temperature=None,
        max_tokens=1000,
    )
    assert result.tool_args == {"hypothesis": "x"}
    kwargs = provider._client.messages.create.call_args.kwargs
    assert "temperature" not in kwargs


def test_anthropic_includes_temperature_when_set():
    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider._client = _make_anthropic_fake({"hypothesis": "x"})

    provider.call_tool(
        system="sys",
        user="usr",
        tool_name="propose_iteration",
        tool_description="",
        tool_schema={"type": "object"},
        model="claude-sonnet-4-6",
        temperature=0.5,
        max_tokens=1000,
    )
    kwargs = provider._client.messages.create.call_args.kwargs
    assert kwargs["temperature"] == 0.5


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


def test_make_provider_gemini_uses_google_base_url():
    from autoptim.meta.providers import GeminiProvider, make_provider

    provider = make_provider("gemini", "fake-key")
    assert isinstance(provider, GeminiProvider)
    # The OpenAI SDK stores base_url on the client. It should point at Google's endpoint.
    assert "generativelanguage.googleapis.com" in str(provider._client.base_url)


def test_gemini_call_passes_tool_choice_and_parses_response():
    from autoptim.meta.providers import GeminiProvider

    provider = GeminiProvider.__new__(GeminiProvider)
    provider._client = _make_openai_fake()
    result = provider.call_tool(
        system="sys",
        user="usr",
        tool_name="propose_iteration",
        tool_description="",
        tool_schema={"type": "object"},
        model="gemini-3-pro",
        temperature=None,
        max_tokens=1000,
    )
    assert result.tool_args == {"hypothesis": "x"}
    kwargs = provider._client.chat.completions.create.call_args.kwargs
    assert kwargs["tool_choice"] == {"type": "function", "function": {"name": "propose_iteration"}}
    assert kwargs["model"] == "gemini-3-pro"
