"""Tests for the openai_compat (cloud) worker path.

Covers:
- WorkerConfig accepts the new fields.
- Orchestrator threads api_key + base_url into ctx for openai_compat.
- seed_process._chat routes to /chat/completions with Bearer auth when backend=openai_compat.
"""
from __future__ import annotations

import importlib.util
import types
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

from autoptim.config import WorkerConfig


def test_worker_config_defaults_to_ollama():
    cfg = WorkerConfig()
    assert cfg.backend == "ollama"
    assert cfg.base_url is None
    assert cfg.api_key_env is None


def test_worker_config_openai_compat():
    cfg = WorkerConfig(
        backend="openai_compat",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        default_model="llama-3.1-8b-instant",
    )
    assert cfg.base_url == "https://api.groq.com/openai/v1"
    assert cfg.api_key_env == "GROQ_API_KEY"


def _load_seed() -> types.ModuleType:
    path = Path(__file__).resolve().parents[1] / "examples" / "invoice_extraction" / "seed_process.py"
    spec = importlib.util.spec_from_file_location("seed_process_under_test", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_seed_chat_routes_to_ollama_by_default(monkeypatch):
    seed = _load_seed()
    captured: dict = {}

    def fake_post(url, *, json, timeout, **_):
        captured["url"] = url
        captured["json"] = json
        return _FakeResp({"message": {"content": "hello"}})

    monkeypatch.setattr(httpx, "post", fake_post)
    out = seed._chat(
        {"backend": "ollama", "ollama_host": "http://localhost:11434", "model_hint": "llama3.2:1b"},
        [{"role": "user", "content": "hi"}],
    )
    assert out == "hello"
    assert captured["url"].endswith("/api/chat")
    assert captured["json"]["model"] == "llama3.2:1b"
    assert captured["json"]["stream"] is False


def test_seed_chat_routes_to_openai_compat_with_bearer(monkeypatch):
    seed = _load_seed()
    captured: dict = {}

    def fake_post(url, *, headers, json, timeout, **_):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return _FakeResp({"choices": [{"message": {"content": "cloud-hello"}}]})

    monkeypatch.setattr(httpx, "post", fake_post)
    out = seed._chat(
        {
            "backend": "openai_compat",
            "base_url": "https://api.groq.com/openai/v1",
            "api_key": "gsk_test_abc",
            "model_hint": "llama-3.1-8b-instant",
        },
        [{"role": "user", "content": "hi"}],
    )
    assert out == "cloud-hello"
    assert captured["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer gsk_test_abc"
    assert captured["json"]["model"] == "llama-3.1-8b-instant"
    # OpenAI-compat payload shouldn't have Ollama's "stream"/"options" keys
    assert "stream" not in captured["json"]
    assert "options" not in captured["json"]


def test_orchestrator_threads_cloud_creds_into_ctx(tmp_path, monkeypatch):
    """Verify the orchestrator passes backend/base_url/api_key through to the sandbox ctx."""
    from autoptim.config import load_task
    from autoptim.orchestrator import Orchestrator
    from autoptim.worker.sandbox import SandboxResult

    # Build a minimal cloud-flavored task
    root = tmp_path / "task"
    root.mkdir()
    (root / "inputs").mkdir()
    (root / "inputs" / "a.txt").write_text("foo")
    (root / "ground_truth.jsonl").write_text('{"id": "a", "fields": {"v": "x"}}\n')
    (root / "seed_process.py").write_text(
        "def run(inputs, ctx):\n    return [{'id': i['id'], 'prediction': {'v': 'x'}} for i in inputs]\n"
    )
    (root / "task.yaml").write_text(
        "name: demo\n"
        "inputs_dir: ./inputs\n"
        "ground_truth: ./ground_truth.jsonl\n"
        "seed_process: ./seed_process.py\n"
        "metric:\n  type: schema_match\n  schema:\n    v: {match: exact, weight: 1}\n"
        "  aggregate: mean_score\n  target: 0.99\n"
        "budgets:\n  max_iters: 1\n  wall_clock_h: 1\n  frontier_usd: 10\n  per_iter_timeout_s: 30\n"
        "worker:\n  backend: openai_compat\n  base_url: https://api.groq.com/openai/v1\n"
        "  api_key_env: GROQ_API_KEY\n  default_model: llama-3.1-8b-instant\n"
        "meta:\n  provider: anthropic\n  model: claude-opus-4-7\n"
    )
    task = load_task(root / "task.yaml")

    from autoptim.worker import ollama_client as oc
    monkeypatch.setattr(oc.OllamaClient, "available", lambda self: True)
    monkeypatch.setattr(oc.OllamaClient, "list_models", lambda self: [])

    from autoptim.meta import agent as agent_mod
    monkeypatch.setattr(agent_mod, "make_provider", lambda *a, **kw: MagicMock())

    captured_ctx: dict = {}
    from autoptim import orchestrator as orch_mod

    def fake_sandbox(*, process_py_path, inputs, ctx, out_dir, timeout_s, memory_mb):
        captured_ctx.update(ctx)
        return SandboxResult(
            predictions=[{"id": "a", "prediction": {"v": "x"}}],
            stdout="",
            stderr="",
            elapsed_s=0.01,
            error=None,
            returncode=0,
            timed_out=False,
        )
    monkeypatch.setattr(orch_mod, "run_process_py", fake_sandbox)

    orch = Orchestrator(task, api_key="fake", runs_root=tmp_path / "runs", worker_api_key="gsk_test")
    orch.run()

    assert captured_ctx["backend"] == "openai_compat"
    assert captured_ctx["base_url"] == "https://api.groq.com/openai/v1"
    assert captured_ctx["api_key"] == "gsk_test"
    assert "ollama_host" in captured_ctx  # still present for backwards compat
