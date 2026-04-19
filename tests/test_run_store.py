from autoptim.runspec import IterationRecord
from autoptim.storage.run_store import RunStore, new_run_id, find_latest_for_task


def _rec(i=0, score=0.5):
    return IterationRecord(
        iter=i,
        ts="2024-01-01T00:00:00",
        parent="seed",
        strategy_tag="seed",
        hypothesis="baseline",
        score=score,
        delta=None,
        kept=True,
        new_best=True,
        frontier_tokens_in=0,
        frontier_tokens_out=0,
        frontier_usd=0.0,
        worker_seconds=1.0,
        worker_calls=0,
        predicted_delta=None,
    )


def test_create_and_write(tmp_path):
    rid = new_run_id("foo")
    store = RunStore.create(tmp_path, rid, {"name": "foo"})
    store.write_iter_process(0, "def run(inputs, ctx): return []\n")
    store.write_iter_hypothesis(0, "seed", "seed", "seed")
    store.write_iter_eval(0, {"overall": 0.5, "target": 0.9, "per_doc": [], "failures": [], "metric_name": "x", "extra": {}})
    store.append_history(_rec())
    store.update_best(0, 0.5)

    assert (tmp_path / rid / "iter_000" / "process.py").exists()
    assert store.read_best() == {"iter": 0, "score": 0.5, "path": str(store.iter_dir(0))}
    hist = store.read_history()
    assert len(hist) == 1 and hist[0]["iter"] == 0


def test_find_latest(tmp_path):
    rid1 = "foo-20240101-000000"
    rid2 = "foo-20240102-000000"
    (tmp_path / rid1).mkdir()
    (tmp_path / rid2).mkdir()
    (tmp_path / "bar-20240101-000000").mkdir()
    assert find_latest_for_task(tmp_path, "foo") == rid2


def test_atomic_write_replaces(tmp_path):
    rid = new_run_id("foo")
    store = RunStore.create(tmp_path, rid, {"name": "foo"})
    store.update_best(0, 0.5)
    store.update_best(1, 0.7)
    assert store.read_best()["iter"] == 1
