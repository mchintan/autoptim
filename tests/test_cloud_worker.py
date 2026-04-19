"""Tests for the OpenAI-compatible worker path.

Covers:
- WorkerConfig accepts the new (single-backend) fields.
- Orchestrator threads api_key + base_url into ctx.
- seed_process._chat routes to /chat/completions with Bearer auth.
"""
from __future__ import annotations

import importlib.util
import types
from pathlib import Path
from unittest.mock import MagicMock

import httpx

from autoptim.config import WorkerConfig


def test_worker_config_defaults_to_lm_studio():
    cfg = WorkerConfig()
    assert cfg.base_url.startswith("http://")
    assert "1234" in cfg.base_url  # LM Studio default port
    assert cfg.api_key_env is None


def test_worker_config_accepts_cloud_provider():
    cfg = WorkerConfig(
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        default_model="llama-3.1-8b-instant",
    )
    assert cfg.api_key_env == "GROQ_API_KEY"
    assert cfg.default_model == "llama-3.1-8b-instant"


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


def test_seed_chat_hits_chat_completions_with_bearer(monkeypatch):
    seed = _load_seed()
    captured: dict = {}

    def fake_post(url, *, headers, json, timeout, **_):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return _FakeResp({"choices": [{"message": {"content": "hello"}}]})

    monkeypatch.setattr(httpx, "post", fake_post)
    out = seed._chat(
        {
            "base_url": "http://127.0.0.1:1234/v1",
            "api_key": "sk-test",
            "model_hint": "google/gemma-4-e4b",
        },
        [{"role": "user", "content": "hi"}],
    )
    assert out == "hello"
    assert captured["url"] == "http://127.0.0.1:1234/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["json"]["model"] == "google/gemma-4-e4b"


def test_seed_chat_uses_dummy_key_when_none_provided(monkeypatch):
    """LM Studio / llama.cpp ignore auth; the seed should still send a non-empty bearer."""
    seed = _load_seed()
    captured: dict = {}

    def fake_post(url, *, headers, json, timeout, **_):
        captured["headers"] = headers
        return _FakeResp({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(httpx, "post", fake_post)
    seed._chat(
        {"base_url": "http://127.0.0.1:1234/v1", "model_hint": "m"},  # no api_key
        [{"role": "user", "content": "hi"}],
    )
    # Bearer must be present and non-empty even when no key supplied
    auth = captured["headers"]["Authorization"]
    assert auth.startswith("Bearer ") and len(auth) > len("Bearer ")


def test_orchestrator_threads_worker_creds_into_ctx(tmp_path, monkeypatch):
    from autoptim.config import load_task
    from autoptim.orchestrator import Orchestrator
    from autoptim.worker.sandbox import SandboxResult

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
        "worker:\n  base_url: https://api.groq.com/openai/v1\n"
        "  api_key_env: GROQ_API_KEY\n  default_model: llama-3.1-8b-instant\n"
        "meta:\n  provider: openai\n  model: gpt-4.1\n"
    )
    task = load_task(root / "task.yaml")

    # Skip real worker-model listing
    monkeypatch.setattr(
        "autoptim.orchestrator.Orchestrator._probe_worker_models",
        lambda self: ["llama-3.1-8b-instant"],
    )

    from autoptim.meta import agent as agent_mod
    monkeypatch.setattr(agent_mod, "make_provider", lambda *a, **kw: MagicMock())

    captured_ctx: dict = {}
    from autoptim import orchestrator as orch_mod

    def fake_sandbox(*, process_py_path, inputs, ctx, out_dir, timeout_s, memory_mb):
        captured_ctx.update(ctx)
        return SandboxResult(
            predictions=[{"id": "a", "prediction": {"v": "x"}}],
            stdout="", stderr="", elapsed_s=0.01,
            error=None, returncode=0, timed_out=False,
        )
    monkeypatch.setattr(orch_mod, "run_process_py", fake_sandbox)

    orch = Orchestrator(task, api_key="fake", runs_root=tmp_path / "runs", worker_api_key="gsk_test")
    orch.run()

    assert captured_ctx["base_url"] == "https://api.groq.com/openai/v1"
    assert captured_ctx["api_key"] == "gsk_test"
    assert captured_ctx["model_hint"] == "llama-3.1-8b-instant"
