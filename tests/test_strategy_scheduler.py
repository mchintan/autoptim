from autoptim.meta.strategy_scheduler import (
    SAME_AXIS_LIMIT,
    STALL_DELTA_EPSILON,
    STALL_ITERS,
    decide,
)
from autoptim.runspec import IterationRecord


def _rec(i, tag, score, delta):
    return IterationRecord(
        iter=i,
        ts="",
        parent=0,
        strategy_tag=tag,
        hypothesis="",
        score=score,
        delta=delta,
        kept=False,
        new_best=False,
        frontier_tokens_in=0,
        frontier_tokens_out=0,
        frontier_usd=0.0,
        worker_seconds=0.0,
        worker_calls=0,
        predicted_delta=None,
    )


def test_empty_history_is_free():
    d = decide([])
    assert d.forced_strategy is None


def test_forces_off_same_axis():
    hist = [_rec(i, "prompt_mutation", 0.5 + i * 0.01, 0.01) for i in range(SAME_AXIS_LIMIT)]
    d = decide(hist)
    assert d.forced_strategy is not None
    assert d.forced_strategy != "prompt_mutation"


def test_forces_off_stall():
    hist = []
    for i in range(STALL_ITERS):
        hist.append(_rec(i, ["prompt_mutation", "pipeline_restructuring", "model_param_swap"][i], 0.5, STALL_DELTA_EPSILON / 2))
    d = decide(hist)
    assert d.forced_strategy is not None


def test_variety_does_not_force():
    hist = [
        _rec(0, "prompt_mutation", 0.5, 0.05),
        _rec(1, "model_param_swap", 0.55, 0.05),
    ]
    d = decide(hist)
    assert d.forced_strategy is None
