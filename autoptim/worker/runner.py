"""Child-side entrypoint.

Invoked as a subprocess by sandbox.py. Imports a candidate process.py,
calls its `run(inputs, ctx)`, writes predictions.jsonl and timing.json to the
scratch dir. Never imported by the parent process.

Usage:
    python -m autoptim.worker.runner <process_py_path> <job_json_path>

Where job_json_path points at a JSON file with:
    {"inputs": [...], "ctx": {...}, "out_dir": "..."}
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def _load_process(path: str):
    p = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("autoptim_candidate_process", p)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    if not hasattr(mod, "run"):
        raise AttributeError("process.py must define `def run(inputs, ctx)`")
    return mod.run


def _make_logger(out_dir: Path):
    log_path = out_dir / "worker_stdout.log"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        try:
            with open(log_path, "a") as f:
                f.write(line + "\n")
        except OSError:
            pass
        print(line, flush=True)

    return log


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(
            "usage: python -m autoptim.worker.runner <process.py> <job.json>",
            file=sys.stderr,
        )
        return 2
    process_path, job_path = argv[1], argv[2]
    job: dict[str, Any] = json.loads(Path(job_path).read_text())
    inputs: list[dict[str, Any]] = job["inputs"]
    ctx_raw: dict[str, Any] = job["ctx"]
    out_dir = Path(job["out_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = dict(ctx_raw)
    ctx["log"] = _make_logger(out_dir)

    t0 = time.time()
    result: dict[str, Any] = {
        "predictions": [],
        "elapsed_s": 0.0,
        "error": None,
    }
    try:
        run = _load_process(process_path)
        preds = run(inputs, ctx)
        if not isinstance(preds, list):
            raise TypeError(f"process.run must return list, got {type(preds).__name__}")
        norm: list[dict[str, Any]] = []
        for p in preds:
            if not isinstance(p, dict) or "id" not in p:
                raise TypeError("each prediction must be a dict with an 'id' key")
            norm.append({"id": p["id"], "prediction": p.get("prediction")})
        result["predictions"] = norm
    except Exception as e:
        tb = traceback.format_exc()
        result["error"] = f"{type(e).__name__}: {e}"
        Path(out_dir / "worker_stderr.log").write_text(tb)
    finally:
        result["elapsed_s"] = time.time() - t0
        (out_dir / "predictions.jsonl").write_text(
            "\n".join(json.dumps(p) for p in result["predictions"])
        )
        (out_dir / "run_result.json").write_text(json.dumps(result, default=str))

    return 0 if result["error"] is None else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
