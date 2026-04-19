"""The main loop: execute → eval → meta → record. Budget-aware."""
from __future__ import annotations

import datetime as dt
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from .config import TaskConfig
from .evaluator.base import Evaluator, load_ground_truth
from .evaluator.custom import CustomEvaluator
from .evaluator.schema_match import SchemaMatchEvaluator
from .meta.agent import BudgetExceeded, MetaAgent, MetaCallResult, MetaValidationError
from .runspec import IterationRecord, RunState
from .storage.diff import unified as diff_unified
from .storage.run_store import RunStore, new_run_id, find_latest_for_task
from .util.cost import CostTracker
from .util.logging import console, logger
from .worker.sandbox import run_process_py

log = logger("autoptim.orchestrator")


@dataclass
class HaltReason:
    code: str  # "target" | "max_iters" | "wall_clock" | "usd_cap" | "user" | "error"
    detail: str


class Orchestrator:
    def __init__(
        self,
        task: TaskConfig,
        api_key: str,
        runs_root: str | Path = "runs",
        worker_api_key: str = "not-needed",
        run_id: str | None = None,
    ):
        self.task = task
        self.api_key = api_key
        self.worker_api_key = worker_api_key
        self.runs_root = Path(runs_root).expanduser().resolve()
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self._preset_run_id = run_id

        self.evaluator: Evaluator = self._build_evaluator()
        self.ground_truth = load_ground_truth(task.ground_truth)
        self.inputs = self._discover_inputs()
        if not self.inputs:
            raise RuntimeError(f"no inputs found under {task.inputs_dir}")
        if not self.ground_truth:
            raise RuntimeError(f"no ground truth entries in {task.ground_truth}")

        self.worker_models = self._probe_worker_models()

        self.meta = MetaAgent(
            provider=task.meta.provider,
            api_key=api_key,
            model=task.meta.model,
            task_name=task.name,
            metric_name=f"{task.metric.type}/{task.metric.aggregate}",
            target=task.metric.target,
            worker_models=self.worker_models,
            max_history=task.meta.max_history,
            temperature=task.meta.temperature,
        )
        self.cost = CostTracker(
            cap_usd=task.budgets.frontier_usd,
            provider=task.meta.provider,
            model=task.meta.model,
        )

    # ---- setup helpers ----

    def _build_evaluator(self) -> Evaluator:
        if self.task.metric.type == "schema_match":
            return SchemaMatchEvaluator(self.task.metric)
        if self.task.metric.type == "custom":
            if not self.task.metric.eval_py:
                raise ValueError("metric.type=custom requires metric.eval_py path")
            return CustomEvaluator(self.task.metric.eval_py)
        raise ValueError(f"unknown metric.type {self.task.metric.type!r}")

    def _discover_inputs(self) -> list[dict[str, Any]]:
        root = Path(self.task.inputs_dir).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"inputs_dir missing: {root}")
        out = []
        for p in sorted(root.iterdir()):
            if not p.is_file() or p.name.startswith("."):
                continue
            mime, _ = mimetypes.guess_type(p.name)
            out.append({"id": p.stem, "path": str(p), "mime": mime or "application/octet-stream"})
        # Filter to only inputs that have ground truth
        gt_ids = set(self.ground_truth)
        out = [i for i in out if i["id"] in gt_ids]
        return out

    def _probe_worker_models(self) -> list[str]:
        """Best-effort listing of models at the OpenAI-compat endpoint.

        Many providers expose /v1/models. If not, return just the configured default —
        the meta-agent uses this to know what's safe to swap to.
        """
        import openai

        try:
            client = openai.OpenAI(
                api_key=self.worker_api_key or "not-needed",
                base_url=self.task.worker.base_url,
            )
            return [m.id for m in client.models.list().data]
        except Exception as e:
            log.warning("worker /v1/models failed (%s); the meta-agent will only see the configured default", e)
            return [self.task.worker.default_model]

    # ---- run lifecycle ----

    def start_new_run(self) -> tuple[RunStore, RunState]:
        run_id = self._preset_run_id or new_run_id(self.task.name)
        frozen = self._frozen_task_yaml()
        store = RunStore.create(self.runs_root, run_id, frozen)
        state = RunState(
            run_id=run_id,
            run_dir=str(store.run_dir),
            task_name=self.task.name,
            started_ts=_now_iso(),
        )
        return store, state

    def resume_latest(self) -> tuple[RunStore, RunState] | None:
        existing = find_latest_for_task(self.runs_root, self.task.name)
        if not existing:
            return None
        store = RunStore(self.runs_root, existing)
        hist_raw = store.read_history()
        iters: list[IterationRecord] = []
        for h in hist_raw:
            iters.append(
                IterationRecord(
                    iter=h["iter"],
                    ts=h.get("ts", ""),
                    parent=h.get("parent", "seed"),
                    strategy_tag=h.get("strategy_tag", "seed"),
                    hypothesis=h.get("hypothesis", ""),
                    score=h.get("score"),
                    delta=h.get("delta"),
                    kept=bool(h.get("kept", False)),
                    new_best=bool(h.get("new_best", False)),
                    frontier_tokens_in=h.get("frontier_tokens_in", 0),
                    frontier_tokens_out=h.get("frontier_tokens_out", 0),
                    frontier_usd=h.get("frontier_usd", 0.0),
                    worker_seconds=h.get("worker_seconds", 0.0),
                    worker_calls=h.get("worker_calls", h.get("ollama_calls", 0)),
                    predicted_delta=h.get("predicted_delta"),
                    error=h.get("error"),
                )
            )
        best = store.read_best()
        state = RunState(
            run_id=existing,
            run_dir=str(store.run_dir),
            task_name=self.task.name,
            iters=iters,
            best_iter=best.get("iter") if best else None,
            best_score=best.get("score") if best else None,
            total_frontier_usd=sum(i.frontier_usd for i in iters),
            started_ts=iters[0].ts if iters else _now_iso(),
        )
        self.cost.spent_usd = state.total_frontier_usd
        return store, state

    def _frozen_task_yaml(self) -> dict[str, Any]:
        return {
            "name": self.task.name,
            "inputs_dir": self.task.inputs_dir,
            "ground_truth": self.task.ground_truth,
            "seed_process": self.task.seed_process,
            "metric": {
                "type": self.task.metric.type,
                "schema": {
                    k: v.model_dump(exclude_none=True)
                    for k, v in (self.task.metric.schema_ or {}).items()
                },
                "aggregate": self.task.metric.aggregate,
                "target": self.task.metric.target,
                "eval_py": self.task.metric.eval_py,
            },
            "budgets": self.task.budgets.model_dump(),
            "meta": self.task.meta.model_dump(),
            "worker": self.task.worker.model_dump(),
            "frozen_at": _now_iso(),
        }

    # ---- the loop ----

    def run(
        self,
        *,
        resume: bool = False,
        on_iteration: "callable | None" = None,
    ) -> HaltReason:
        resumed = self.resume_latest() if resume else None
        store, state = resumed if resumed else self.start_new_run()
        log.info("run %s started (task=%s)", state.run_id, self.task.name)

        wall_start = time.time() - _wall_used_so_far(state)
        reason: HaltReason | None = None

        # Seed iteration if empty
        if not state.iters:
            console().rule("[bold]iter 0  ·  seed baseline[/bold]")
            console().print(
                "[dim]Running the user-provided seed_process.py to establish a baseline score. "
                "No meta-agent call happens for the seed.[/dim]"
            )
            reason = self._run_iteration(
                store, state,
                iter_n=0,
                parent_iter="seed",
                strategy_tag="seed",
                hypothesis="Baseline: run the user-provided seed_process.py as-is.",
                process_py=Path(self.task.seed_process).read_text(),
                predicted_delta=None,
                wall_start=wall_start,
            )
            if on_iteration:
                on_iteration(state)
            if reason:
                store.write_final(self._final_json(state, reason))
                return reason
            # Circuit breaker: if the seed produced no usable output at all,
            # refuse to proceed. The meta-agent can't design useful experiments
            # against a completely broken baseline.
            degenerate = self._diagnose_seed(store)
            if degenerate is not None:
                store.write_final(self._final_json(state, degenerate))
                return degenerate

        # Main loop
        while True:
            reason = self._check_halt_pre(state, wall_start)
            if reason:
                break

            iter_n = state.last_iter() + 1
            parent = state.best_iter if state.best_iter is not None else 0
            parent_process_py = store.read_process(parent)
            parent_eval = store.read_eval(parent)
            # Previous iteration's log (not parent's — we want what happened most recently).
            last_iter_n = state.iters[-1].iter if state.iters else parent
            worker_log_tail = _read_log_tail(
                store.iter_dir(last_iter_n) / "worker_stdout.log", 30
            )

            console().rule(f"[bold cyan]iter {iter_n}[/bold cyan]  (parent = iter {parent})")

            try:
                with console().status(
                    f"[cyan]Asking {self.task.meta.model} to design iter {iter_n} "
                    f"(${self.cost.remaining():.2f} of frontier budget left)...[/cyan]",
                    spinner="dots",
                ):
                    meta_result = self.meta.propose(
                        iter_n=iter_n,
                        max_iters=self.task.budgets.max_iters,
                        wall_used_h=(time.time() - wall_start) / 3600.0,
                        wall_cap_h=self.task.budgets.wall_clock_h,
                        cost_tracker=self.cost,
                        parent_process_py=parent_process_py,
                        parent_score=state.best_score,
                        eval_json=parent_eval,
                        history=state.iters,
                        worker_log_tail=worker_log_tail,
                        best=(
                            {
                                "iter": state.best_iter,
                                "score": state.best_score,
                                "hypothesis": state.iters[state.best_iter].hypothesis
                                if state.best_iter is not None and state.best_iter < len(state.iters)
                                else "",
                            }
                            if state.best_iter is not None
                            else None
                        ),
                    )
            except BudgetExceeded as e:
                reason = HaltReason("usd_cap", str(e))
                break
            except MetaValidationError as e:
                log.error("meta-agent invalid output: %s", e)
                reason = HaltReason("error", f"meta-agent failed validation: {e}")
                break

            self._render_proposal(iter_n, parent, parent_process_py, meta_result)
            store.write_iter_proposal(iter_n, _proposal_to_json(meta_result))

            reason = self._run_iteration(
                store, state,
                iter_n=iter_n,
                parent_iter=parent,
                strategy_tag=meta_result.proposal.strategy_tag,
                hypothesis=meta_result.proposal.hypothesis,
                process_py=meta_result.proposal.process_py,
                predicted_delta=meta_result.proposal.predicted_delta,
                wall_start=wall_start,
                meta_tokens_in=meta_result.tokens_in,
                meta_tokens_out=meta_result.tokens_out,
            )
            if on_iteration:
                on_iteration(state)
            if reason:
                break

        final = reason or HaltReason("error", "loop exited without reason")
        store.write_final(self._final_json(state, final))
        log.info("run %s halted: %s (%s)", state.run_id, final.code, final.detail)
        return final

    def _run_iteration(
        self,
        store: RunStore,
        state: RunState,
        *,
        iter_n: int,
        parent_iter: int | str,
        strategy_tag: str,
        hypothesis: str,
        process_py: str,
        predicted_delta: float | None,
        wall_start: float,
        meta_tokens_in: int = 0,
        meta_tokens_out: int = 0,
    ) -> HaltReason | None:
        log.info(
            "iter %d (strategy=%s, parent=%s): %s",
            iter_n, strategy_tag, parent_iter,
            (hypothesis or "").splitlines()[0][:100],
        )
        store.write_iter_process(iter_n, process_py)
        store.write_iter_hypothesis(iter_n, hypothesis, strategy_tag, parent_iter)

        iter_dir = store.iter_dir(iter_n)
        t_iter = time.time()

        with console().status(
            f"[cyan]Running worker (iter {iter_n}) on {len(self.inputs)} inputs "
            f"(timeout {self.task.budgets.per_iter_timeout_s}s)...[/cyan]",
            spinner="dots",
        ):
            worker_ctx: dict[str, Any] = {
                "model_hint": self.task.worker.default_model,
                "base_url": self.task.worker.base_url,
                "api_key": self.worker_api_key,
            }
            sandbox_result = run_process_py(
                process_py_path=iter_dir / "process.py",
                inputs=self.inputs,
                ctx=worker_ctx,
                out_dir=iter_dir,
                timeout_s=self.task.budgets.per_iter_timeout_s,
                memory_mb=self.task.worker.memory_mb,
            )
        store.write_iter_stdio(iter_n, sandbox_result.stdout, sandbox_result.stderr)

        if sandbox_result.error:
            # Score unknown; record and move on
            console().print(
                Panel.fit(
                    f"[red]Worker failed:[/red] {sandbox_result.error}\n"
                    + (f"\n[dim]stderr (last 20 lines):\n{_tail(sandbox_result.stderr, 20)}[/dim]" if sandbox_result.stderr else ""),
                    title=f"iter {iter_n} — worker error",
                    border_style="red",
                )
            )
            rec = IterationRecord(
                iter=iter_n,
                ts=_now_iso(),
                parent=parent_iter,
                strategy_tag=strategy_tag,
                hypothesis=hypothesis,
                score=None,
                delta=None,
                kept=False,
                new_best=False,
                frontier_tokens_in=meta_tokens_in,
                frontier_tokens_out=meta_tokens_out,
                frontier_usd=self.cost.spent_usd - (state.total_frontier_usd),
                worker_seconds=sandbox_result.elapsed_s,
                worker_calls=0,
                predicted_delta=predicted_delta,
                error=sandbox_result.error,
            )
            state.iters.append(rec)
            state.total_frontier_usd = self.cost.spent_usd
            store.append_history(rec)
            store.write_iter_decision(
                iter_n,
                {"keep": False, "reason": f"worker error: {sandbox_result.error}"},
            )
            store.write_iter_timing(
                iter_n,
                {
                    "wall_clock_s": time.time() - t_iter,
                    "worker_s": sandbox_result.elapsed_s,
                    "frontier_tokens_in": meta_tokens_in,
                    "frontier_tokens_out": meta_tokens_out,
                },
            )
            return None  # non-fatal; keep going

        result = self.evaluator.score(sandbox_result.predictions, self.ground_truth)
        store.write_iter_eval(iter_n, result.to_json())

        prev_best = state.best_score
        new_best = (prev_best is None) or (result.overall > prev_best + 1e-9)
        delta = None if prev_best is None else result.overall - prev_best

        if new_best:
            state.best_iter = iter_n
            state.best_score = result.overall
            store.update_best(iter_n, result.overall)

        rec = IterationRecord(
            iter=iter_n,
            ts=_now_iso(),
            parent=parent_iter,
            strategy_tag=strategy_tag,
            hypothesis=hypothesis,
            score=result.overall,
            delta=delta,
            kept=new_best,
            new_best=new_best,
            frontier_tokens_in=meta_tokens_in,
            frontier_tokens_out=meta_tokens_out,
            frontier_usd=self.cost.spent_usd - state.total_frontier_usd,
            worker_seconds=sandbox_result.elapsed_s,
            worker_calls=0,  # counted by worker if it wants; not measured in v1
            predicted_delta=predicted_delta,
            error=None,
        )
        state.iters.append(rec)
        state.total_frontier_usd = self.cost.spent_usd
        store.append_history(rec)
        store.write_iter_decision(
            iter_n,
            {
                "keep": new_best,
                "reason": "new best" if new_best else "did not improve best score",
                "score": result.overall,
                "delta": delta,
                "predicted_delta": predicted_delta,
            },
        )
        store.write_iter_timing(
            iter_n,
            {
                "wall_clock_s": time.time() - t_iter,
                "worker_s": sandbox_result.elapsed_s,
                "frontier_tokens_in": meta_tokens_in,
                "frontier_tokens_out": meta_tokens_out,
                "frontier_usd_cumulative": self.cost.spent_usd,
            },
        )

        self._render_eval(
            iter_n=iter_n,
            eval_result=result,
            delta=delta,
            new_best=new_best,
            predicted_delta=predicted_delta,
            best_score=state.best_score,
            worker_s=sandbox_result.elapsed_s,
            meta_tokens_in=meta_tokens_in,
            meta_tokens_out=meta_tokens_out,
            cumulative_usd=self.cost.spent_usd,
        )

        if result.overall >= self.task.metric.target:
            return HaltReason("target", f"hit target {self.task.metric.target} at iter {iter_n}")
        return None

    def _check_halt_pre(self, state: RunState, wall_start: float) -> HaltReason | None:
        if state.last_iter() + 1 >= self.task.budgets.max_iters:
            return HaltReason("max_iters", f"reached {self.task.budgets.max_iters} iterations")
        wall_used_h = (time.time() - wall_start) / 3600.0
        if wall_used_h >= self.task.budgets.wall_clock_h:
            return HaltReason("wall_clock", f"wall-clock cap {self.task.budgets.wall_clock_h}h hit")
        if self.cost.spent_usd >= self.task.budgets.frontier_usd:
            return HaltReason("usd_cap", f"frontier spend ${self.cost.spent_usd:.2f} >= cap")
        return None

    def _final_json(self, state: RunState, reason: HaltReason) -> dict[str, Any]:
        return {
            "run_id": state.run_id,
            "task_name": state.task_name,
            "halt_code": reason.code,
            "halt_detail": reason.detail,
            "iterations": len(state.iters),
            "best_iter": state.best_iter,
            "best_score": state.best_score,
            "target": self.task.metric.target,
            "frontier_usd_spent": self.cost.spent_usd,
            "ended_ts": _now_iso(),
        }


    # ---- diagnostics ----

    def _diagnose_seed(self, store: RunStore) -> "HaltReason | None":
        """If iter 0 shows every prediction empty/null, halt with actionable diagnostics.

        This protects the frontier budget: we've seen runs burn $1+ of meta-agent
        calls while every worker call was returning HTTP 500 because Ollama's
        llama runner had crashed. No point designing experiments when the
        infrastructure beneath them is down.
        """
        try:
            eval_json = store.read_eval(0)
        except FileNotFoundError:
            return None
        per_doc = eval_json.get("per_doc") or []
        if not per_doc:
            return None
        all_null_or_empty = all(
            d.get("score", 0.0) == 0.0
            and (d.get("error") or all(f.get("predicted") in (None, "", {}) for f in d.get("fields", [])))
            for d in per_doc
        )
        if not all_null_or_empty:
            return None

        # Gather the evidence
        tail = _read_log_tail(store.run_dir / "iter_000" / "worker_stdout.log", 15)
        err_tail = _read_log_tail(store.run_dir / "iter_000" / "worker_stderr.log", 10)

        body = (
            "[red bold]Refusing to proceed: the seed iteration produced no usable output.[/red bold]\n\n"
            "Every document came back with either an error or all-null fields. There is no signal "
            "for the meta-agent to improve on — it would just burn frontier tokens against a broken baseline.\n\n"
            "[bold]Worker stdout (last 15 lines):[/bold]\n"
            f"[dim]{tail or '(empty)'}[/dim]\n"
        )
        if err_tail:
            body += f"\n[bold]Worker stderr (last 10 lines):[/bold]\n[dim]{err_tail}[/dim]\n"
        body += (
            "\n[bold]Likely causes + fixes:[/bold]\n"
            f"  • Worker endpoint [cyan]{self.task.worker.base_url}[/cyan] is unreachable or the "
            f"loaded model does not match [cyan]{self.task.worker.default_model}[/cyan].\n"
            "  • If using LM Studio: open the app, click 'Start Server' on the chosen model, "
            "confirm it responds to a manual chat in the UI.\n"
            "  • If using a cloud provider: verify the API key is active and the model id is valid.\n"
            f"  • seed_process.py may have a bug unrelated to the endpoint → "
            f"[bold]autoptim replay {store.run_id} 0[/bold] to reproduce with fresh logs\n\n"
            "Once fixed, re-run. [dim](Full logs live at runs/<run_id>/iter_000/.)[/dim]"
        )
        console().print(Panel(body, title="seed diagnosis", border_style="red"))
        return HaltReason("degenerate_seed", "seed iteration produced no usable predictions; see diagnostic panel")

    # ---- rendering helpers ----

    def _render_proposal(
        self,
        iter_n: int,
        parent_iter: int | str,
        parent_process_py: str,
        meta_result: MetaCallResult,
    ) -> None:
        p = meta_result.proposal
        usd_this_call = self.cost.spent_usd - (sum(i.frontier_usd for i in []) or 0)  # not meaningful across calls; use incremental from cost record below
        # Header table
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold cyan")
        table.add_column()
        table.add_row("strategy", p.strategy_tag)
        table.add_row("predicted Δ", f"{p.predicted_delta:+.3f}")
        table.add_row("frontier cost", f"{meta_result.tokens_in:,} in / {meta_result.tokens_out:,} out  (cumulative ${self.cost.spent_usd:.2f})")
        if meta_result.directive.forced_strategy:
            table.add_row("forced axis", f"[yellow]{meta_result.directive.forced_strategy}[/yellow] (scheduler)")
        failure_text = "\n".join(f"  • {m}" for m in p.expected_failure_modes) or "  (none given)"
        body = (
            f"[bold magenta]Rationale (why this experiment):[/bold magenta]\n{p.rationale}\n\n"
            f"[bold]Hypothesis (what it will prove):[/bold]\n{p.hypothesis}\n\n"
            f"[bold]Expected failure modes:[/bold]\n{failure_text}\n\n"
            f"[dim]Scheduler directive:[/dim] [dim]{meta_result.directive.message}[/dim]"
        )
        console().print(
            Panel.fit(
                _rich_concat(table, body),
                title=f"iter {iter_n} proposal  ·  from iter {parent_iter}",
                border_style="cyan",
            )
        )
        # Diff panel
        diff = diff_unified(
            parent_process_py,
            p.process_py,
            a_name=f"iter_{parent_iter}/process.py" if isinstance(parent_iter, int) else f"{parent_iter}/process.py",
            b_name=f"iter_{iter_n:03d}/process.py",
        )
        if diff:
            trimmed = _trim_diff(diff, max_lines=80)
            console().print(
                Panel(
                    Syntax(trimmed, "diff", theme="ansi_dark", line_numbers=False, word_wrap=True),
                    title="diff vs parent",
                    border_style="dim",
                )
            )
        else:
            console().print("[dim](no textual change vs parent — meta-agent returned identical code)[/dim]")

    def _render_eval(
        self,
        *,
        iter_n: int,
        eval_result: Any,
        delta: float | None,
        new_best: bool,
        predicted_delta: float | None,
        best_score: float | None,
        worker_s: float,
        meta_tokens_in: int,
        meta_tokens_out: int,
        cumulative_usd: float,
    ) -> None:
        # Per-doc table
        t = Table(title=f"iter {iter_n} — per-doc scores", show_lines=False)
        t.add_column("doc", style="cyan")
        t.add_column("score", justify="right")
        t.add_column("error", style="red")
        for d in eval_result.per_doc:
            t.add_row(d.id, f"{d.score:.3f}", d.error or "")
        console().print(t)

        # Summary line
        target = eval_result.target
        delta_str = f"{delta:+.3f}" if delta is not None else "—"
        pdelta_str = f"{predicted_delta:+.3f}" if predicted_delta is not None else "—"
        verdict = "[green bold]NEW BEST[/green bold]" if new_best else "[yellow]kept parent[/yellow]"
        summary = (
            f"overall = [bold]{eval_result.overall:.3f}[/bold] / target {target:.3f}   "
            f"Δ={delta_str}  (predicted {pdelta_str})   {verdict}\n"
            f"worker {worker_s:.1f}s  ·  frontier {meta_tokens_in:,} in / {meta_tokens_out:,} out  "
            f"·  cumulative ${cumulative_usd:.2f}"
        )
        if eval_result.failures:
            failures = "\n".join(
                f"[red]✗[/red] [cyan]{f.id}[/cyan]: {f.summary}"
                for f in eval_result.failures[:3]
            )
            summary += "\n\n[bold]worst failures:[/bold]\n" + failures
        console().print(
            Panel.fit(summary, title=f"iter {iter_n} result", border_style=("green" if new_best else "yellow"))
        )


def _proposal_to_json(meta_result: MetaCallResult) -> dict[str, Any]:
    p = meta_result.proposal
    return {
        "rationale": p.rationale,
        "hypothesis": p.hypothesis,
        "strategy_tag": p.strategy_tag,
        "predicted_delta": p.predicted_delta,
        "expected_failure_modes": p.expected_failure_modes,
        "tokens_in": meta_result.tokens_in,
        "tokens_out": meta_result.tokens_out,
        "scheduler": {
            "message": meta_result.directive.message,
            "forced_strategy": meta_result.directive.forced_strategy,
        },
    }


def _rich_concat(*parts: Any) -> "Any":
    from rich.console import Group
    return Group(*parts)


def _trim_diff(diff: str, max_lines: int = 80) -> str:
    lines = diff.splitlines()
    if len(lines) <= max_lines:
        return diff
    keep_head = max_lines // 2
    keep_tail = max_lines - keep_head
    elided = len(lines) - max_lines
    return "\n".join(
        lines[:keep_head]
        + [f"... [{elided} diff lines elided; see iter_NNN/process.py on disk] ..."]
        + lines[-keep_tail:]
    )


def _tail(s: str, n: int) -> str:
    return "\n".join(s.splitlines()[-n:])


def _read_log_tail(path: "Path", n: int) -> str:
    try:
        return _tail(path.read_text(), n)
    except (FileNotFoundError, OSError):
        return ""


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _wall_used_so_far(state: RunState) -> float:
    """Total wall-seconds recorded in history so far — used for resumes."""
    # Approximate: ordering by ts first vs. last iteration's recorded ts
    if len(state.iters) < 2:
        return 0.0
    try:
        first = dt.datetime.fromisoformat(state.iters[0].ts)
        last = dt.datetime.fromisoformat(state.iters[-1].ts)
        return max(0.0, (last - first).total_seconds())
    except ValueError:
        return 0.0
