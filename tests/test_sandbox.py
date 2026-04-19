import json
import textwrap
from pathlib import Path

from autoptim.worker.sandbox import run_process_py


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content))
    return path


def test_happy_path(tmp_path: Path):
    proc = _write(
        tmp_path / "process.py",
        '''
        def run(inputs, ctx):
            return [{"id": i["id"], "prediction": {"len": len(i["path"])}} for i in inputs]
        ''',
    )
    result = run_process_py(
        process_py_path=proc,
        inputs=[{"id": "a", "path": "/tmp/x", "mime": "text/plain"}],
        ctx={"base_url": "http://127.0.0.1:1234/v1", "model_hint": "local-model", "api_key": "x"},
        out_dir=tmp_path / "out",
        timeout_s=30,
        memory_mb=512,
    )
    assert result.error is None
    assert result.predictions == [{"id": "a", "prediction": {"len": 6}}]


def test_timeout_is_enforced(tmp_path: Path):
    proc = _write(
        tmp_path / "process.py",
        '''
        import time
        def run(inputs, ctx):
            time.sleep(60)
            return []
        ''',
    )
    result = run_process_py(
        process_py_path=proc,
        inputs=[{"id": "a", "path": "/tmp/x", "mime": "text/plain"}],
        ctx={"base_url": "http://127.0.0.1:1234/v1", "model_hint": "local-model", "api_key": "x"},
        out_dir=tmp_path / "out",
        timeout_s=2,
        memory_mb=256,
    )
    assert result.timed_out
    assert result.error


def test_raise_is_captured(tmp_path: Path):
    proc = _write(
        tmp_path / "process.py",
        '''
        def run(inputs, ctx):
            raise RuntimeError("boom")
        ''',
    )
    result = run_process_py(
        process_py_path=proc,
        inputs=[{"id": "a", "path": "/tmp/x", "mime": "text/plain"}],
        ctx={"base_url": "http://127.0.0.1:1234/v1", "model_hint": "local-model", "api_key": "x"},
        out_dir=tmp_path / "out",
        timeout_s=10,
        memory_mb=256,
    )
    assert result.error
    assert "RuntimeError" in result.error


def test_missing_run_function(tmp_path: Path):
    proc = _write(tmp_path / "process.py", "x = 1\n")
    result = run_process_py(
        process_py_path=proc,
        inputs=[],
        ctx={"base_url": "http://127.0.0.1:1234/v1", "model_hint": "local-model", "api_key": "x"},
        out_dir=tmp_path / "out",
        timeout_s=10,
        memory_mb=256,
    )
    assert result.error and "run" in result.error
