"""CLI entrypoint: run, inspect, resume, replay, tail."""
from __future__ import annotations

import json
import os
import stat
import sys
import time
from pathlib import Path
from typing import Any, Optional

import typer
from rich.prompt import Prompt
from rich.table import Table

from .config import load_task
from .orchestrator import Orchestrator
from .storage.diff import unified as diff_unified
from .storage.run_store import RunStore, list_runs
from .util.logging import configure, console
from .worker.ollama_client import OllamaClient

app = typer.Typer(add_completion=False, help="autoptim — agentic loop for verifiable processes.")

RUNS_ROOT_ENV = "AUTOPTIM_RUNS"
CRED_PATH = Path("~/.autoptim/credentials").expanduser()


def _runs_root() -> Path:
    return Path(os.environ.get(RUNS_ROOT_ENV, "runs")).expanduser().resolve()


_PROVIDER_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    # Gemini accepts either GEMINI_API_KEY (preferred) or GOOGLE_API_KEY (Google's default name).
    "gemini": "GEMINI_API_KEY",
}

_PROVIDER_ENV_FALLBACK = {
    "gemini": "GOOGLE_API_KEY",
}


def _resolve_api_key(provider: str) -> str:
    env_var = _PROVIDER_ENV.get(provider)
    if not env_var:
        raise typer.Exit(f"unknown provider: {provider}")
    val = os.environ.get(env_var, "").strip()
    if val:
        return val
    fallback = _PROVIDER_ENV_FALLBACK.get(provider)
    if fallback:
        fv = os.environ.get(fallback, "").strip()
        if fv:
            return fv
    # Check persisted creds
    if CRED_PATH.exists():
        try:
            creds = json.loads(CRED_PATH.read_text())
            if creds.get(provider):
                return creds[provider]
        except json.JSONDecodeError:
            pass
    # Prompt
    console().print(
        f"[yellow]{env_var} not set. Enter API key for provider `{provider}` (will be stored 0600 at {CRED_PATH}):[/yellow]"
    )
    key = Prompt.ask("API key", password=True).strip()
    if not key:
        raise typer.Exit("no key provided")
    persist = Prompt.ask("Persist to ~/.autoptim/credentials?", choices=["y", "n"], default="y")
    if persist == "y":
        CRED_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing: dict[str, Any] = {}
        if CRED_PATH.exists():
            try:
                existing = json.loads(CRED_PATH.read_text())
            except json.JSONDecodeError:
                existing = {}
        existing[provider] = key
        CRED_PATH.write_text(json.dumps(existing, indent=2))
        os.chmod(CRED_PATH, stat.S_IRUSR | stat.S_IWUSR)
    return key


def _resolve_worker_api_key(api_key_env: str) -> str:
    """Read an OpenAI-compat worker's API key from env / credentials / prompt.

    Parallel to `_resolve_api_key` but for the worker tier. Keys are persisted to
    `~/.autoptim/credentials` under the env-var name (e.g. "GROQ_API_KEY") so
    subsequent runs pick them up automatically.
    """
    val = os.environ.get(api_key_env, "").strip()
    if val:
        return val
    if CRED_PATH.exists():
        try:
            creds = json.loads(CRED_PATH.read_text())
            if creds.get(api_key_env):
                return creds[api_key_env]
        except json.JSONDecodeError:
            pass
    console().print(
        f"[yellow]{api_key_env} not set. Enter API key for the worker (will be stored 0600 at {CRED_PATH}):[/yellow]"
    )
    key = Prompt.ask("API key", password=True).strip()
    if not key:
        raise typer.Exit("no key provided")
    persist = Prompt.ask("Persist to ~/.autoptim/credentials?", choices=["y", "n"], default="y")
    if persist == "y":
        CRED_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing: dict[str, Any] = {}
        if CRED_PATH.exists():
            try:
                existing = json.loads(CRED_PATH.read_text())
            except json.JSONDecodeError:
                existing = {}
        existing[api_key_env] = key
        CRED_PATH.write_text(json.dumps(existing, indent=2))
        os.chmod(CRED_PATH, stat.S_IRUSR | stat.S_IWUSR)
    return key


def _preflight_cloud_worker(base_url: str, model: str, api_key: str) -> None:
    """Smoke-test a cloud OpenAI-compat worker endpoint before starting the loop."""
    import openai

    try:
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with only: ok"}],
            max_tokens=8,
            temperature=0.0,
        )
        reply = (resp.choices[0].message.content or "").strip()[:80]
        console().print(
            f"[green]Cloud worker OK[/green] · [cyan]{model}[/cyan] @ {base_url} replied: {reply!r}"
        )
    except Exception as e:
        console().print(
            f"[red]Cloud worker rejected smoke test for [cyan]{model}[/cyan] @ {base_url}:[/red] "
            f"{type(e).__name__}: {e!s}\n\n"
            f"Check: (a) model id is valid on that provider, (b) API key still active, "
            f"(c) base_url is correct (must end with /v1 for OpenAI-compat endpoints)."
        )
        raise typer.Exit(2)


def _preflight_worker(task: Any) -> str | None:
    """Preflight the worker tier. Returns the resolved API key if cloud, else None."""
    if task.worker.backend == "openai_compat":
        if not task.worker.base_url or not task.worker.api_key_env:
            console().print(
                "[red]worker.backend=openai_compat requires both base_url and api_key_env in task.yaml[/red]"
            )
            raise typer.Exit(2)
        key = _resolve_worker_api_key(task.worker.api_key_env)
        _preflight_cloud_worker(task.worker.base_url, task.worker.default_model, key)
        return key
    # Native Ollama
    _preflight_ollama(task.worker.ollama_host, task.worker.default_model)
    return None


def _preflight_gemini(api_key: str, wanted_model: str) -> None:
    """List installed Gemini models and warn if `wanted_model` is not exposed by the endpoint."""
    try:
        from .meta.providers import GeminiProvider

        provider = GeminiProvider(api_key)
        models = provider.list_models()
    except Exception as e:
        console().print(f"[yellow]Could not list Gemini models ({e!s}). Proceeding anyway.[/yellow]")
        return
    if not models:
        console().print("[yellow]Gemini model list came back empty. Proceeding — the endpoint may still accept the requested model.[/yellow]")
        return
    gemini_only = [m for m in models if m.startswith("gemini")]
    # Match exact or with common "-latest"/"-preview" suffixes
    wanted_prefix = wanted_model.rsplit("-preview", 1)[0].rsplit("-latest", 1)[0]
    exact = wanted_model in gemini_only
    similar = [m for m in gemini_only if m.startswith(wanted_prefix)]
    console().print(
        f"[green]Gemini OK[/green] · {len(gemini_only)} models available, requesting [cyan]{wanted_model}[/cyan]"
    )
    if not exact:
        console().print(
            f"[yellow]Note:[/yellow] [cyan]{wanted_model}[/cyan] is not in the account's listed models."
        )
        if similar:
            console().print(f"  Similar available: {', '.join(similar[:6])}")
        else:
            likely = [m for m in gemini_only if "pro" in m or "flash" in m][:8]
            console().print(f"  Some available Gemini models: {', '.join(likely)}")
        console().print(
            "  If the request fails with 404/INVALID_ARGUMENT, set the right id in task.yaml → meta.model."
        )


def _preflight_ollama(host: str, worker_model: str) -> None:
    client = OllamaClient(host)
    if not client.available():
        console().print(
            f"[red]Ollama not reachable at {host}.[/red] Start it with `ollama serve` "
            f"and pull a model (e.g. `ollama pull {worker_model}`) before running."
        )
        raise typer.Exit(2)
    models = client.list_models()
    if not models:
        console().print(
            f"[yellow]Ollama is running but has no models installed. Pull one first, e.g. `ollama pull {worker_model}`.[/yellow]"
        )
        raise typer.Exit(2)
    if worker_model not in models:
        console().print(
            f"[red]Worker model [cyan]{worker_model}[/cyan] is not in `ollama list`.[/red] "
            f"Installed: {', '.join(models[:6])}{'…' if len(models) > 6 else ''}\n"
            f"Run: [bold]ollama pull {worker_model}[/bold]"
        )
        raise typer.Exit(2)

    # Actual chat to verify the runner is healthy, not just the server. Catches
    # crashed llama-runner / corrupt model / OOM cases that /api/tags misses.
    ok, detail = client.smoke_test(worker_model)
    if not ok:
        console().print(
            f"[red]Ollama rejected a test chat with [cyan]{worker_model}[/cyan]:[/red] {detail}\n\n"
            f"Common fixes:\n"
            f"  • [bold]ollama serve[/bold] is running but the llama runner may have crashed — "
            f"restart it and try [bold]ollama run {worker_model} \"hi\"[/bold] to confirm it replies.\n"
            f"  • Re-pull the model: [bold]ollama rm {worker_model} && ollama pull {worker_model}[/bold]\n"
            f"  • Check [bold]ollama ps[/bold] and system memory — large models may OOM silently.\n\n"
            f"The harness refuses to proceed because every worker call would fail and the frontier "
            f"meta-agent would burn tokens against a broken baseline."
        )
        raise typer.Exit(2)
    console().print(
        f"[green]Ollama OK[/green] · {len(models)} models installed · [cyan]{worker_model}[/cyan] replied in smoke test"
    )


@app.command()
def run(
    task_file: Path = typer.Argument(..., help="Path to task.yaml"),
    resume: bool = typer.Option(False, "--resume", help="Resume latest run for this task"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Start or resume an optimization run."""
    configure(log_level)
    task = load_task(task_file)
    worker_api_key = _preflight_worker(task)
    key = _resolve_api_key(task.meta.provider)
    if task.meta.provider == "gemini":
        _preflight_gemini(key, task.meta.model)
    orch = Orchestrator(
        task,
        api_key=key,
        runs_root=_runs_root(),
        worker_api_key=worker_api_key,
    )
    reason = orch.run(resume=resume)
    console().print(
        f"\n[bold]Halted[/bold]: {reason.code} — {reason.detail}\n"
        f"Best score: [cyan]{orch.cost.spent_usd:.2f}$ spent[/cyan]"
    )


@app.command()
def models(
    provider: str = typer.Option("gemini", "--provider", help="Provider to query (only gemini is queryable today)"),
) -> None:
    """List the models the configured provider reports as available on your account."""
    configure("WARNING")
    if provider != "gemini":
        console().print(f"[yellow]Listing is only implemented for gemini. Got {provider!r}.[/yellow]")
        raise typer.Exit(1)
    key = _resolve_api_key("gemini")
    from .meta.providers import GeminiProvider

    gp = GeminiProvider(key)
    models_list = gp.list_models()
    if not models_list:
        console().print("[yellow]No models returned.[/yellow]")
        return
    table = Table(title=f"{provider} models")
    table.add_column("id", style="cyan")
    for m in sorted(models_list):
        table.add_row(m)
    console().print(table)


@app.command("list")
def list_cmd() -> None:
    """List all runs under the runs root."""
    configure("WARNING")
    ids = list_runs(_runs_root())
    if not ids:
        console().print(f"no runs under {_runs_root()}")
        return
    table = Table(title=f"runs under {_runs_root()}")
    table.add_column("run_id")
    table.add_column("iters", justify="right")
    table.add_column("best", justify="right")
    table.add_column("halted")
    for rid in ids:
        store = RunStore(_runs_root(), rid)
        hist = store.read_history()
        best = store.read_best()
        final = {}
        final_path = store.run_dir / "final.json"
        if final_path.exists():
            final = json.loads(final_path.read_text())
        table.add_row(
            rid,
            str(len(hist)),
            f"{best['score']:.3f}" if best else "—",
            final.get("halt_code", "running"),
        )
    console().print(table)


@app.command()
def inspect(
    run_id: str = typer.Argument(..., help="Run id (see `autoptim list`)"),
    iter_n: Optional[int] = typer.Argument(None, help="Optional iteration number"),
) -> None:
    """Inspect a run (summary) or a single iteration (hypothesis + diff + eval breakdown)."""
    configure("WARNING")
    store = RunStore(_runs_root(), run_id)
    if not store.run_dir.exists():
        raise typer.Exit(f"no such run: {run_id}")

    if iter_n is None:
        _inspect_run(store)
    else:
        _inspect_iter(store, iter_n)


def _inspect_run(store: RunStore) -> None:
    hist = store.read_history()
    best = store.read_best()
    final_path = store.run_dir / "final.json"
    final = json.loads(final_path.read_text()) if final_path.exists() else None

    console().print(f"[bold]{store.run_id}[/bold]")
    console().print(f"  dir: {store.run_dir}")
    console().print(f"  iterations: {len(hist)}")
    if best:
        console().print(f"  best: iter {best['iter']} score {best['score']:.3f}")
    if final:
        console().print(f"  halted: {final['halt_code']} — {final['halt_detail']}")
        console().print(f"  frontier spend: ${final.get('frontier_usd_spent', 0):.2f}")

    table = Table(title="trajectory")
    for col in ("iter", "parent", "strategy", "score", "Δ", "kept", "hypothesis"):
        table.add_column(col)
    for h in hist:
        table.add_row(
            str(h["iter"]),
            str(h["parent"]),
            str(h.get("strategy_tag", "—")),
            f"{h['score']:.3f}" if h.get("score") is not None else "—",
            f"{h['delta']:+.3f}" if h.get("delta") is not None else "—",
            "✓" if h.get("kept") else "",
            (h.get("hypothesis") or "").splitlines()[0][:80],
        )
    console().print(table)


def _inspect_iter(store: RunStore, iter_n: int) -> None:
    d = store.iter_dir(iter_n)
    if not d.exists():
        raise typer.Exit(f"no such iteration: {iter_n}")
    hyp = store.read_hypothesis(iter_n)
    parent_txt = (d / "parent.txt").read_text() if (d / "parent.txt").exists() else "?"
    strategy = (d / "strategy_tag.txt").read_text() if (d / "strategy_tag.txt").exists() else "?"

    console().print(f"[bold]{store.run_id} / iter {iter_n}[/bold]")
    console().print(f"  parent: {parent_txt}")
    console().print(f"  strategy: {strategy}")

    # Full proposal (rationale + failure modes) if we have it
    prop_path = d / "proposal.json"
    if prop_path.exists():
        prop = json.loads(prop_path.read_text())
        if prop.get("rationale"):
            console().print("\n[bold magenta]rationale:[/bold magenta]")
            for line in prop["rationale"].splitlines():
                console().print(f"  {line}")
        if prop.get("hypothesis"):
            console().print("\n[bold]hypothesis:[/bold]")
            for line in prop["hypothesis"].splitlines():
                console().print(f"  {line}")
        if prop.get("expected_failure_modes"):
            console().print("\n[bold]expected failure modes:[/bold]")
            for m in prop["expected_failure_modes"]:
                console().print(f"  • {m}")
        console().print(
            f"\n[dim]predicted Δ: {prop.get('predicted_delta', '—')}  ·  "
            f"tokens: {prop.get('tokens_in', 0):,} in / {prop.get('tokens_out', 0):,} out[/dim]"
        )
    elif hyp:
        # Legacy runs — only hypothesis.md was written
        console().print("  hypothesis:")
        for line in hyp.splitlines():
            console().print(f"    {line}")

    # Diff vs parent
    try:
        parent_n = int(parent_txt)
    except ValueError:
        parent_n = None
    if parent_n is not None and store.iter_dir(parent_n).exists():
        parent_py = store.read_process(parent_n)
        this_py = store.read_process(iter_n)
        diff = diff_unified(parent_py, this_py, f"iter_{parent_n:03d}/process.py", f"iter_{iter_n:03d}/process.py")
        console().print("\n[bold]diff vs parent[/bold]")
        console().print(diff or "(no change)")

    # Eval breakdown
    eval_path = d / "eval.json"
    if eval_path.exists():
        ev = json.loads(eval_path.read_text())
        console().print(f"\n[bold]eval[/bold] — overall {ev['overall']:.3f} / target {ev['target']:.3f}")
        t = Table()
        t.add_column("doc")
        t.add_column("score", justify="right")
        t.add_column("error")
        for doc in ev["per_doc"]:
            t.add_row(doc["id"], f"{doc['score']:.3f}", doc.get("error") or "")
        console().print(t)
        if ev.get("failures"):
            console().print("\n[bold]worst failures[/bold]")
            for f in ev["failures"]:
                console().print(f"  - [cyan]{f['id']}[/cyan]: {f['summary']}")


@app.command()
def replay(
    run_id: str = typer.Argument(...),
    iter_n: int = typer.Argument(...),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Re-run a single iteration's process.py for debugging."""
    configure(log_level)
    store = RunStore(_runs_root(), run_id)
    if not store.iter_dir(iter_n).exists():
        raise typer.Exit(f"no such iter: {run_id}/{iter_n}")

    # Use the task config frozen inside the run
    import yaml

    run_yaml = yaml.safe_load((store.run_dir / "run.yaml").read_text())
    from .config import TaskConfig

    run_yaml["task_root"] = str(store.run_dir)
    task = TaskConfig.model_validate(run_yaml)
    worker_api_key = _preflight_worker(task)

    from .worker.sandbox import run_process_py
    import mimetypes

    inputs_root = Path(task.inputs_dir)
    inputs = []
    for p in sorted(inputs_root.iterdir()):
        if p.is_file() and not p.name.startswith("."):
            mime, _ = mimetypes.guess_type(p.name)
            inputs.append({"id": p.stem, "path": str(p), "mime": mime or "application/octet-stream"})

    out_dir = store.run_dir / f"replay_{iter_n:03d}_{int(time.time())}"
    ctx: dict[str, Any] = {
        "ollama_host": task.worker.ollama_host,
        "model_hint": task.worker.default_model,
        "backend": task.worker.backend,
    }
    if task.worker.backend == "openai_compat":
        ctx["base_url"] = task.worker.base_url
        ctx["api_key"] = worker_api_key or ""
    result = run_process_py(
        process_py_path=store.iter_dir(iter_n) / "process.py",
        inputs=inputs,
        ctx=ctx,
        out_dir=out_dir,
        timeout_s=task.budgets.per_iter_timeout_s,
        memory_mb=task.worker.memory_mb,
    )
    console().print(f"replay finished: {len(result.predictions)} predictions, elapsed {result.elapsed_s:.1f}s")
    if result.error:
        console().print(f"[red]error:[/red] {result.error}")
    console().print(f"artifacts: {out_dir}")


@app.command()
def tail(
    run_id: str = typer.Argument(...),
) -> None:
    """Live-follow history.jsonl with rich formatting."""
    configure("WARNING")
    store = RunStore(_runs_root(), run_id)
    hist_path = store.run_dir / "history.jsonl"
    if not hist_path.exists():
        raise typer.Exit(f"no such run: {run_id}")
    seen = 0
    try:
        while True:
            lines = hist_path.read_text().splitlines()
            for line in lines[seen:]:
                if not line.strip():
                    continue
                h = json.loads(line)
                score = f"{h['score']:.3f}" if h.get("score") is not None else "—"
                delta = f"{h['delta']:+.3f}" if h.get("delta") is not None else "—"
                tag = h.get("strategy_tag", "—")
                hyp = (h.get("hypothesis") or "").splitlines()[0][:80]
                console().print(
                    f"[dim]{h.get('ts','')}[/dim] iter {h['iter']:>3} "
                    f"[cyan]{tag:<22}[/cyan] score={score} Δ={delta} "
                    f"{'[green]*[/green]' if h.get('new_best') else ' '} {hyp}"
                )
            seen = len(lines)
            final_path = store.run_dir / "final.json"
            if final_path.exists():
                final = json.loads(final_path.read_text())
                console().print(f"\n[bold]final[/bold]: {final['halt_code']} — {final['halt_detail']}")
                return
            time.sleep(2.0)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    app()
