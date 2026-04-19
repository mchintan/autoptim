"""CLI entrypoint: run, inspect, resume, replay, tail."""
from __future__ import annotations

import json
import os
import platform
import shlex
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from .config import load_task
from .orchestrator import Orchestrator
from .storage.diff import unified as diff_unified
from .storage.run_store import RunStore, find_latest_for_task, list_runs, new_run_id
from .util.logging import configure, console

app = typer.Typer(add_completion=False, help="autoptim — agentic loop for verifiable processes.")

RUNS_ROOT_ENV = "AUTOPTIM_RUNS"
CRED_PATH = Path("~/.autoptim/credentials").expanduser()


def _runs_root() -> Path:
    return Path(os.environ.get(RUNS_ROOT_ENV, "runs")).expanduser().resolve()


_PROVIDER_ENV = {
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


def _spawn_dashboard_window(run_id: str) -> None:
    """Open a new terminal window running `autoptim tail <run_id>`.

    macOS uses AppleScript against Terminal.app. Linux tries the common
    terminal emulators. If nothing works, prints a manual instruction.
    """
    autoptim_bin = sys.argv[0]
    # Prefer the absolute path — if we were launched via PATH it'll still resolve
    autoptim_bin_abs = shutil.which(autoptim_bin) or autoptim_bin
    cwd = os.getcwd()
    shell_cmd = f"cd {shlex.quote(cwd)} && {shlex.quote(autoptim_bin_abs)} tail {shlex.quote(run_id)}"

    system = platform.system()
    try:
        if system == "Darwin":
            # AppleScript to open a new Terminal window running the command.
            escaped = shell_cmd.replace("\\", "\\\\").replace('"', '\\"')
            applescript = f'tell application "Terminal" to do script "{escaped}"'
            subprocess.Popen(
                ["osascript", "-e", applescript],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            console().print(f"[green]Dashboard window launched[/green] for run [cyan]{run_id}[/cyan]")
            return

        if system == "Linux":
            # Try common terminal emulators in order of ubiquity.
            candidates: list[list[str]] = [
                ["x-terminal-emulator", "-e", "bash", "-c", shell_cmd],
                ["gnome-terminal", "--", "bash", "-c", shell_cmd],
                ["konsole", "-e", "bash", "-c", shell_cmd],
                ["xfce4-terminal", "--command", f"bash -c '{shell_cmd}'"],
                ["alacritty", "-e", "bash", "-c", shell_cmd],
                ["kitty", "bash", "-c", shell_cmd],
                ["xterm", "-e", "bash", "-c", shell_cmd],
            ]
            for argv in candidates:
                if shutil.which(argv[0]):
                    subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    console().print(
                        f"[green]Dashboard window launched[/green] via [cyan]{argv[0]}[/cyan] for run [cyan]{run_id}[/cyan]"
                    )
                    return

        if system == "Windows":
            subprocess.Popen(
                ["cmd.exe", "/c", "start", "cmd.exe", "/k", f"{autoptim_bin_abs} tail {run_id}"]
            )
            console().print(f"[green]Dashboard window launched[/green] for run [cyan]{run_id}[/cyan]")
            return
    except OSError as e:
        console().print(f"[yellow]Dashboard window spawn failed ({e}).[/yellow]")

    console().print(
        f"[yellow]Could not auto-open a terminal window.[/yellow] "
        f"In another terminal, run: [bold]autoptim tail {run_id}[/bold]"
    )


def _preflight_worker(task: Any) -> str:
    """Preflight the OpenAI-compat worker endpoint. Returns the resolved API key
    (or a dummy for auth-less local servers like LM Studio / llama.cpp)."""
    import openai

    if task.worker.api_key_env:
        api_key = _resolve_worker_api_key(task.worker.api_key_env)
    else:
        # LM Studio / llama.cpp accept any non-empty string in the Authorization header.
        api_key = "not-needed"

    base_url = task.worker.base_url
    model = task.worker.default_model
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
            f"[green]Worker OK[/green] · [cyan]{model}[/cyan] @ {base_url} replied: {reply!r}"
        )
    except Exception as e:
        hint = (
            "For LM Studio: make sure the server is running (green 'Start Server' in the app) "
            "and the model in task.yaml matches what's currently loaded.\n"
            "For cloud providers: verify the API key and that the model id exists on your account."
        )
        console().print(
            f"[red]Worker rejected smoke test for [cyan]{model}[/cyan] @ {base_url}:[/red] "
            f"{type(e).__name__}: {e!s}\n\n{hint}"
        )
        raise typer.Exit(2)
    return api_key


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




@app.command()
def run(
    task_file: Path = typer.Argument(..., help="Path to task.yaml"),
    resume: bool = typer.Option(False, "--resume", help="Resume latest run for this task"),
    dashboard: bool = typer.Option(
        False,
        "--dashboard",
        help="Auto-open a second terminal window with the live Lightning-style dashboard",
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Start or resume an optimization run."""
    configure(log_level)
    task = load_task(task_file)
    worker_api_key = _preflight_worker(task)
    key = _resolve_api_key(task.meta.provider)
    if task.meta.provider == "gemini":
        _preflight_gemini(key, task.meta.model)

    # Pin a run_id upfront so we can tell any spawned dashboard where to look.
    if resume:
        run_id = find_latest_for_task(_runs_root(), task.name) or new_run_id(task.name)
    else:
        run_id = new_run_id(task.name)

    if dashboard:
        _spawn_dashboard_window(run_id)
    else:
        console().print(
            f"[dim]Tip: open another terminal and run "
            f"[bold]autoptim tail {run_id}[/bold] for a live dashboard view.[/dim]"
        )

    orch = Orchestrator(
        task,
        api_key=key,
        runs_root=_runs_root(),
        worker_api_key=worker_api_key,
        run_id=run_id,
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
        "model_hint": task.worker.default_model,
        "base_url": task.worker.base_url,
        "api_key": worker_api_key,
    }
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
    refresh: float = typer.Option(2.0, "--refresh", help="Seconds between dashboard repaints"),
) -> None:
    """Lightning-style live dashboard for a run: score sparkline, budget meters,
    recent iterations, and current/next status. Reads from disk — safe to run
    alongside `autoptim run` in another terminal, or on a finished run."""
    configure("WARNING")
    store = RunStore(_runs_root(), run_id)
    # If the dashboard was spawned a moment before `autoptim run` wrote the run dir,
    # poll briefly so we don't die on a race.
    waited = 0
    while not store.run_dir.exists() or not (store.run_dir / "run.yaml").exists():
        if waited == 0:
            console().print(f"[dim]Waiting for run {run_id} to start…[/dim]")
        time.sleep(0.5)
        waited += 1
        if waited > 120:  # 60s
            console().print(f"[red]Run {run_id} never appeared.[/red]")
            raise typer.Exit(2)

    try:
        with Live(_dashboard(store), refresh_per_second=4, console=console(), screen=False) as live:
            while True:
                time.sleep(refresh)
                live.update(_dashboard(store))
                if (store.run_dir / "final.json").exists():
                    live.update(_dashboard(store))
                    break
    except KeyboardInterrupt:
        pass


# ------------------------------ dashboard ------------------------------

_SPARK_BLOCKS = " ▁▂▃▄▅▆▇█"


def _spark(values: list[float], width: int = 60) -> Text:
    if not values:
        return Text("(no scored iterations yet)", style="dim")
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values
    out = Text()
    for v in sampled:
        # Scores live in [0, 1]; clamp and map to 0..8
        idx = min(len(_SPARK_BLOCKS) - 1, max(0, int(round(max(0.0, min(1.0, v)) * (len(_SPARK_BLOCKS) - 1)))))
        out.append(_SPARK_BLOCKS[idx], style="cyan")
    return out


def _bar(frac: float, width: int = 34, color: str = "cyan") -> Text:
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    t = Text()
    t.append("█" * filled, style=color)
    t.append("░" * (width - filled), style="dim")
    return t


def _dashboard(store: RunStore) -> Group:
    import yaml
    from .runspec import IterationRecord
    from .meta.strategy_scheduler import decide as decide_strategy

    hist = store.read_history()
    best = store.read_best() or {}
    final: dict[str, Any] = {}
    final_path = store.run_dir / "final.json"
    if final_path.exists():
        final = json.loads(final_path.read_text())
    run_yaml = yaml.safe_load((store.run_dir / "run.yaml").read_text()) if (store.run_dir / "run.yaml").exists() else {}
    target = run_yaml.get("metric", {}).get("target", 1.0)
    budgets = run_yaml.get("budgets", {})
    max_iters = budgets.get("max_iters", 0) or max(len(hist), 1)
    wall_cap_h = float(budgets.get("wall_clock_h", 1.0) or 1.0)
    usd_cap = float(budgets.get("frontier_usd", 1.0) or 1.0)

    # --- header ---
    if final:
        code = final.get("halt_code", "?")
        col = {"target": "green", "max_iters": "yellow", "wall_clock": "yellow", "usd_cap": "yellow"}.get(code, "red")
        status = Text.from_markup(f"[{col} bold]halted: {code}[/{col} bold] · {final.get('halt_detail','')}")
    else:
        status = Text.from_markup("[green bold]● running[/green bold]")
    header = Text.assemble(
        Text.from_markup(f"[bold cyan]{store.run_id}[/bold cyan]"),
        Text.from_markup(f"   target [bold]{target:.3f}[/bold]   "),
        status,
    )

    # --- trajectory panel ---
    scores = [h.get("score") for h in hist if h.get("score") is not None]
    spark = _spark(scores)
    target_overshoot = any(s >= target for s in scores)
    best_line = (
        f"best: iter [bold]{best.get('iter','—')}[/bold] = "
        f"[bold green]{best.get('score', 0.0):.3f}[/bold green]"
        if best else "best: —"
    )
    trajectory_lines = [
        Text.from_markup(f"scores over [cyan]{len(scores)}[/cyan] iterations  (▁=0.00 █=1.00, target {target:.3f}):"),
        spark,
        Text.from_markup(
            f"{best_line}   "
            + ("[green]✓ above target[/green]" if target_overshoot else "[yellow]below target[/yellow]")
        ),
    ]
    traj_panel = Panel(Group(*trajectory_lines), title="trajectory", border_style="cyan", padding=(0, 1))

    # --- budgets panel ---
    usd_spent = sum(h.get("frontier_usd", 0.0) or 0.0 for h in hist)
    wall_used_s = _wall_seconds(hist)
    wall_used_h = wall_used_s / 3600.0
    iters_used = len(hist)
    budgets_panel = Panel(
        Group(
            Text.assemble(
                Text.from_markup("[bold]iters[/bold]     "),
                _bar(iters_used / max(max_iters, 1), color="magenta"),
                Text.from_markup(f"  [dim]{iters_used}/{max_iters}[/dim]"),
            ),
            Text.assemble(
                Text.from_markup("[bold]wall[/bold]      "),
                _bar(wall_used_h / max(wall_cap_h, 1e-9), color="blue"),
                Text.from_markup(f"  [dim]{_fmt_dur(wall_used_s)} / {wall_cap_h:.1f}h[/dim]"),
            ),
            Text.assemble(
                Text.from_markup("[bold]frontier[/bold]  "),
                _bar(usd_spent / max(usd_cap, 1e-9), color="yellow"),
                Text.from_markup(f"  [dim]${usd_spent:.2f} / ${usd_cap:.2f}[/dim]"),
            ),
        ),
        title="budgets",
        border_style="dim",
        padding=(0, 1),
    )

    # --- recent iterations table ---
    tbl = Table(show_header=True, header_style="bold", show_lines=False, padding=(0, 1), expand=True)
    tbl.add_column("iter", justify="right", style="bold")
    tbl.add_column("parent", justify="right")
    tbl.add_column("strategy")
    tbl.add_column("score", justify="right")
    tbl.add_column("Δ", justify="right")
    tbl.add_column("kept", justify="center")
    tbl.add_column("hypothesis", overflow="ellipsis", no_wrap=True)
    for h in hist[-8:]:
        score_s = f"{h['score']:.3f}" if h.get("score") is not None else "—"
        delta_s = f"{h['delta']:+.3f}" if h.get("delta") is not None else "—"
        kept_marker = "[green]★[/green]" if h.get("new_best") else ("[green]✓[/green]" if h.get("kept") else "")
        strategy = h.get("strategy_tag", "—")
        if h.get("error"):
            strategy = f"[red]{strategy}[/red]"
        hyp = (h.get("hypothesis") or "").splitlines()[0]
        tbl.add_row(
            str(h.get("iter", "")),
            str(h.get("parent", "")),
            strategy,
            score_s,
            delta_s,
            kept_marker,
            hyp,
        )
    recent_panel = Panel(tbl if hist else Text("(no iterations yet)", style="dim"), title="recent iterations", border_style="dim")

    # --- current / next panel ---
    current = _current_phase(store, hist)
    # Predict next strategy via the scheduler
    next_hint: str
    if not final:
        recs = [_dict_to_rec(h) for h in hist]
        try:
            directive = decide_strategy(recs)
            if directive.forced_strategy:
                next_hint = f"scheduler will force [yellow]{directive.forced_strategy}[/yellow] on the next iter"
            else:
                next_hint = f"[dim]scheduler:[/dim] {directive.message}"
        except Exception:
            next_hint = ""
    else:
        next_hint = "[dim]run complete[/dim]"
    current_panel = Panel(
        Group(current, Text.from_markup(next_hint) if next_hint else Text("")),
        title="now",
        border_style="green" if not final else "dim",
        padding=(0, 1),
    )

    return Group(Panel(header, border_style="cyan"), traj_panel, budgets_panel, recent_panel, current_panel)


def _current_phase(store: RunStore, hist: list[dict]) -> Text:
    """Infer what the orchestrator is doing right now from the latest iter dir."""
    if not hist:
        return Text.from_markup("[yellow]waiting: seed iteration hasn't run yet[/yellow]")
    last_n = max((int(h.get("iter", -1)) for h in hist), default=-1)
    next_n = last_n + 1
    next_dir = store.iter_dir(next_n)
    if next_dir.exists():
        # meta-agent has proposed next iter; is the worker running?
        if (next_dir / "eval.json").exists():
            return Text.from_markup(f"[cyan]iter {next_n} evaluated, queuing iter {next_n + 1}[/cyan]")
        if (next_dir / "process.py").exists():
            prop_path = next_dir / "proposal.json"
            strategy = "?"
            rationale = ""
            if prop_path.exists():
                try:
                    p = json.loads(prop_path.read_text())
                    strategy = p.get("strategy_tag", "?")
                    rationale = (p.get("rationale") or "").replace("\n", " ")[:120]
                except json.JSONDecodeError:
                    pass
            return Text.from_markup(
                f"[green]● running worker[/green]  iter [bold]{next_n}[/bold]  "
                f"strategy=[cyan]{strategy}[/cyan]\n[dim]rationale: {rationale}[/dim]"
            )
    # Between iterations
    return Text.from_markup(
        f"[yellow]designing iter {next_n}[/yellow]  (frontier meta-agent thinking)"
    )


def _wall_seconds(hist: list[dict]) -> float:
    import datetime as dt

    if len(hist) < 2:
        return 0.0
    try:
        first = dt.datetime.fromisoformat(hist[0]["ts"])
        last = dt.datetime.fromisoformat(hist[-1]["ts"])
        return max(0.0, (last - first).total_seconds())
    except (ValueError, KeyError):
        return 0.0


def _fmt_dur(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    h, rem = divmod(seconds, 3600)
    return f"{h}h{rem // 60:02d}m"


def _dict_to_rec(h: dict):
    from .runspec import IterationRecord

    return IterationRecord(
        iter=h.get("iter", 0),
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


if __name__ == "__main__":
    app()
