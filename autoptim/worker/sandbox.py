"""Parent-side subprocess runner.

Launches `python -m autoptim.worker.runner` with resource limits, feeds it
inputs + ctx via a JSON file, enforces a wall-clock timeout, and reads back
predictions + timing + stdio.
"""
from __future__ import annotations

import json
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SandboxResult:
    predictions: list[dict[str, Any]]
    stdout: str
    stderr: str
    elapsed_s: float
    error: str | None
    returncode: int
    timed_out: bool


def _preexec_limits(memory_mb: int, cpu_s: int) -> "callable":
    """Closure that runs in the child before exec — apply RLIMITs and new process group."""

    def _apply() -> None:
        # New process group so we can kill the whole subtree reliably.
        os.setsid()
        mem_bytes = memory_mb * 1024 * 1024
        # Hard cap on virtual memory the worker can request.
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except (ValueError, OSError):
            pass
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        except (ValueError, OSError):
            pass
        # Prevent core-dump bombs.
        try:
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except (ValueError, OSError):
            pass

    return _apply


_ENV_ALLOWLIST = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "PYTHONPATH",
    "PYTHONUNBUFFERED",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "TZ",
)


def _scrub_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k in _ENV_ALLOWLIST}
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def run_process_py(
    process_py_path: str | Path,
    inputs: list[dict[str, Any]],
    ctx: dict[str, Any],
    out_dir: str | Path,
    *,
    timeout_s: int,
    memory_mb: int = 4096,
) -> SandboxResult:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = out_dir / "scratch"
    scratch_dir.mkdir(exist_ok=True)

    # ctx gets augmented with a scratch dir the worker can freely use
    ctx = dict(ctx)
    ctx["scratch_dir"] = str(scratch_dir)
    ctx["per_iter_timeout_s"] = timeout_s

    job = {
        "inputs": inputs,
        "ctx": ctx,
        "out_dir": str(out_dir),
    }
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", dir=str(out_dir), delete=False
    ) as f:
        json.dump(job, f, default=str)
        job_path = f.name

    cmd = [
        sys.executable,
        "-m",
        "autoptim.worker.runner",
        str(process_py_path),
        job_path,
    ]

    t0 = time.time()
    timed_out = False
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_scrub_env(),
        preexec_fn=_preexec_limits(memory_mb, max(timeout_s + 30, 60)),
        cwd=str(out_dir),
    )
    try:
        stdout_b, stderr_b = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            stdout_b, stderr_b = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            stdout_b, stderr_b = proc.communicate()
    elapsed = time.time() - t0

    stdout = stdout_b.decode("utf-8", errors="replace") if stdout_b else ""
    stderr = stderr_b.decode("utf-8", errors="replace") if stderr_b else ""

    # Load results if runner wrote them
    predictions: list[dict[str, Any]] = []
    runner_error: str | None = None
    rr_path = out_dir / "run_result.json"
    if rr_path.exists():
        try:
            rr = json.loads(rr_path.read_text())
            predictions = rr.get("predictions") or []
            runner_error = rr.get("error")
        except json.JSONDecodeError as e:
            runner_error = f"could not parse run_result.json: {e}"
    elif timed_out:
        runner_error = f"worker timed out after {timeout_s}s"
    elif proc.returncode != 0:
        runner_error = f"worker exited {proc.returncode} without results"

    # Cleanup job file
    try:
        os.unlink(job_path)
    except OSError:
        pass

    return SandboxResult(
        predictions=predictions,
        stdout=stdout,
        stderr=stderr,
        elapsed_s=elapsed,
        error=runner_error,
        returncode=proc.returncode or 0,
        timed_out=timed_out,
    )
