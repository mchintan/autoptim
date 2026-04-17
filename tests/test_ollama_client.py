"""Tests for OllamaClient's smoke_test — the thing that catches a crashed llama runner
before we start burning frontier-agent tokens against a broken baseline.
"""
from __future__ import annotations

import httpx

from autoptim.worker.ollama_client import OllamaClient


def _client_with_transport(transport: httpx.MockTransport) -> OllamaClient:
    client = OllamaClient("http://localhost:11434")
    # We patch the httpx module-level `post`/`get` used by the client.
    return client


def test_smoke_test_ok(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "ok"}},
        )

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: httpx.Client(transport=transport).post(*a, **kw))

    ok, detail = OllamaClient("http://localhost:11434").smoke_test("qwen2.5:7b")
    assert ok is True
    assert "ok" in detail


def test_smoke_test_catches_500(monkeypatch):
    """A crashed llama runner returns HTTP 500. smoke_test must surface the body so the
    user knows what's broken instead of blindly proceeding to burn frontier tokens."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            text="llama runner process has terminated: %!w(<nil>)",
        )

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: httpx.Client(transport=transport).post(*a, **kw))

    ok, detail = OllamaClient("http://localhost:11434").smoke_test("qwen2.5:7b")
    assert ok is False
    assert "500" in detail
    assert "llama runner" in detail


def test_smoke_test_catches_connection_refused(monkeypatch):
    def raise_conn(*a, **kw):
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(httpx, "post", raise_conn)

    ok, detail = OllamaClient("http://localhost:11434").smoke_test("qwen2.5:7b")
    assert ok is False
    assert "ConnectError" in detail or "refused" in detail.lower()
