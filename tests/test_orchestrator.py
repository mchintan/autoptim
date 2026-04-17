"""Integration-ish test: runs the full orchestrator with a stub meta-agent and a stub worker.

We don't hit Ollama or a real frontier model. We monkey-patch the worker sandbox and
the MetaAgent.propose() to return canned proposals, and verify score monotonicity,
budget enforcement, and run-store artifacts.
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest

from autoptim.config import load_task
from autoptim.meta.agent import MetaCallResult, ProposedIteration
from autoptim.meta.strategy_scheduler import Directive
from autoptim.orchestrator import Orchestrator
from autoptim.worker.sandbox import SandboxResult


@pytest.fixture
def example_task(tmp_path: Path) -> Path:
    root = tmp_path / "task"
    root.mkdir()
    (root / "inputs").mkdir()
    for i in range(3):
        (root / "inputs" / f"doc_{i}.txt").write_text(f"doc {i} body")
    (root / "ground_truth.jsonl").write_text(
        "\n".join(
            [
                '{"id": "doc_0", "fields": {"v": "zero"}}',
                '{"id": "doc_1", "fields": {"v": "one"}}',
                '{"id": "doc_2", "fields": {"v": "two"}}',
            ]
        )
    )
    (root / "seed_process.py").write_text(
        "def run(inputs, ctx):\n"
        "    return [{'id': i['id'], 'prediction': {'v': 'wrong'}} for i in inputs]\n"
    )
    (root / "task.yaml").write_text(
        textwrap.dedent(
            """
            name: demo
            inputs_dir: ./inputs
            ground_truth: ./ground_truth.jsonl
            seed_process: ./seed_process.py
            metric:
              type: schema_match
              schema:
                v: {match: exact, weight: 1}
              aggregate: mean_score
              target: 0.99
            budgets:
              max_iters: 4
              wall_clock_h: 1
              frontier_usd: 10
              per_iter_timeout_s: 30
            meta:
              provider: anthropic
              model: claude-opus-4-7
            worker:
              ollama_host: http://localhost:99999
              default_model: none
            """
        )
    )
    return root / "task.yaml"


def test_orchestrator_monotonic_with_stub(example_task: Path, tmp_path: Path, monkeypatch):
    task = load_task(example_task)

    # Fake the Ollama probe so constructor doesn't care about it
    from autoptim.worker import ollama_client as oc

    monkeypatch.setattr(oc.OllamaClient, "available", lambda self: True)
    monkeypatch.setattr(oc.OllamaClient, "list_models", lambda self: ["stub:1"])

    # Don't touch real providers — stub out the provider factory
    from autoptim.meta import agent as agent_mod

    class _StubProvider:
        def call_tool(self, **kw):
            raise AssertionError("should not be called when propose is stubbed")

    monkeypatch.setattr(agent_mod, "make_provider", lambda *a, **kw: _StubProvider())

    orch = Orchestrator(task, api_key="fake", runs_root=tmp_path / "runs")

    # Stub the sandbox to return canned predictions keyed by iteration
    iter_to_preds: dict[int, list[dict[str, Any]]] = {
        # seed: all wrong
        0: [{"id": f"doc_{i}", "prediction": {"v": "wrong"}} for i in range(3)],
        # iter 1: 2/3 right
        1: [
            {"id": "doc_0", "prediction": {"v": "zero"}},
            {"id": "doc_1", "prediction": {"v": "one"}},
            {"id": "doc_2", "prediction": {"v": "wrong"}},
        ],
        # iter 2: worse than iter 1
        2: [{"id": f"doc_{i}", "prediction": {"v": "wrong"}} for i in range(3)],
        # iter 3: perfect -> should hit target
        3: [
            {"id": "doc_0", "prediction": {"v": "zero"}},
            {"id": "doc_1", "prediction": {"v": "one"}},
            {"id": "doc_2", "prediction": {"v": "two"}},
        ],
    }

    call_count = {"n": 0}

    def fake_run_process_py(**kwargs):
        n = call_count["n"]
        call_count["n"] += 1
        return SandboxResult(
            predictions=iter_to_preds[n],
            stdout="",
            stderr="",
            elapsed_s=0.01,
            error=None,
            returncode=0,
            timed_out=False,
        )

    from autoptim import orchestrator as orch_mod

    monkeypatch.setattr(orch_mod, "run_process_py", fake_run_process_py)

    # Stub meta-agent propose to return a trivially different process each call
    def fake_propose(self, *, iter_n, **kw):
        return MetaCallResult(
            proposal=ProposedIteration(
                rationale=f"canned stub at iter {iter_n}: the parent eval shows 3/3 docs failing exact match; try returning canonical values.",
                hypothesis=f"iteration {iter_n}: canned mutation",
                strategy_tag="prompt_mutation",
                process_py=f"def run(inputs, ctx):  # iter {iter_n}\n    return []\n",
                predicted_delta=0.1,
                expected_failure_modes=["no-op"],
            ),
            tokens_in=100,
            tokens_out=50,
            usd=0.0,
            directive=Directive(message="", forced_strategy=None),
        )

    monkeypatch.setattr(agent_mod.MetaAgent, "propose", fake_propose)

    reason = orch.run()
    # Should halt on "target" when iter 3 scores 1.0
    assert reason.code == "target", reason
    assert call_count["n"] == 4  # seed + 3 meta-proposed iterations

    hist_path = Path(orch.runs_root) / next(Path(orch.runs_root).iterdir()).name / "history.jsonl"
    lines = hist_path.read_text().strip().splitlines()
    assert len(lines) == 4


def test_budget_cap_halts_run(example_task: Path, tmp_path: Path, monkeypatch):
    task = load_task(example_task)

    from autoptim.worker import ollama_client as oc
    monkeypatch.setattr(oc.OllamaClient, "available", lambda self: True)
    monkeypatch.setattr(oc.OllamaClient, "list_models", lambda self: ["stub:1"])

    from autoptim.meta import agent as agent_mod

    class _StubProvider:
        def call_tool(self, **kw):
            raise AssertionError("should not be called")
    monkeypatch.setattr(agent_mod, "make_provider", lambda *a, **kw: _StubProvider())

    orch = Orchestrator(task, api_key="fake", runs_root=tmp_path / "runs")

    # Starve the cost cap
    orch.cost.cap_usd = 0.0001
    orch.cost.spent_usd = 0.0

    from autoptim import orchestrator as orch_mod

    def fake_sandbox(**kw):
        return SandboxResult(
            predictions=[{"id": f"doc_{i}", "prediction": {"v": "wrong"}} for i in range(3)],
            stdout="",
            stderr="",
            elapsed_s=0.01,
            error=None,
            returncode=0,
            timed_out=False,
        )
    monkeypatch.setattr(orch_mod, "run_process_py", fake_sandbox)

    reason = orch.run()
    assert reason.code == "usd_cap"
