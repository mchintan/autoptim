"""Anti-local-minimum forcing function.

Tracks strategy_tag usage and score deltas. If the meta-agent is hill-climbing
one axis or stalled, the scheduler forces exploration of an unused axis.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from ..runspec import ALL_STRATEGIES, IterationRecord

STALL_DELTA_EPSILON = 0.005
SAME_AXIS_LIMIT = 3
STALL_ITERS = 3


@dataclass
class Directive:
    message: str
    forced_strategy: str | None  # None => agent picks freely


def decide(history: list[IterationRecord]) -> Directive:
    """Given the full history, return a directive for the next iteration."""
    if not history:
        return Directive(
            message="Seed baseline just established. Pick any strategy you think will help most; bias toward prompt_mutation or representation for early wins.",
            forced_strategy=None,
        )

    # Consider only kept or discarded-with-score iterations (not errors).
    scored = [h for h in history if h.score is not None]

    # (1) Same-axis-in-a-row check
    recent_tags = [h.strategy_tag for h in history[-SAME_AXIS_LIMIT:]]
    if len(recent_tags) >= SAME_AXIS_LIMIT and len(set(recent_tags)) == 1:
        stuck = recent_tags[0]
        unused = _next_unused_axis(history, avoid={stuck})
        return Directive(
            message=(
                f"You've used `{stuck}` for the last {SAME_AXIS_LIMIT} iterations. "
                f"Force exploration on a different axis — try `{unused}`."
            ),
            forced_strategy=unused,
        )

    # (2) Stall check: last STALL_ITERS deltas are all tiny
    last_deltas = [h.delta for h in scored[-STALL_ITERS:] if h.delta is not None]
    if len(last_deltas) == STALL_ITERS and all(abs(d) < STALL_DELTA_EPSILON for d in last_deltas):
        unused = _next_unused_axis(history)
        return Directive(
            message=(
                f"Progress has stalled (last {STALL_ITERS} deltas < {STALL_DELTA_EPSILON}). "
                f"Escape the local minimum with `{unused}` — something structurally different from "
                f"the recent changes."
            ),
            forced_strategy=unused,
        )

    # (3) Otherwise nudge toward under-sampled axes without forcing
    usage = Counter(h.strategy_tag for h in history)
    ranked = sorted(ALL_STRATEGIES, key=lambda s: usage.get(s, 0))
    under = [s for s in ranked if usage.get(s, 0) <= min(usage.values() or [0]) + 1]
    return Directive(
        message=(
            f"Strategy usage so far: "
            + ", ".join(f"{s}={usage.get(s, 0)}" for s in ALL_STRATEGIES)
            + f". Consider under-sampled axes: {under[:3]}. You may pick freely."
        ),
        forced_strategy=None,
    )


def _next_unused_axis(
    history: Iterable[IterationRecord], avoid: set[str] | None = None
) -> str:
    avoid = avoid or set()
    usage = Counter(h.strategy_tag for h in history)
    for s in ALL_STRATEGIES:
        if s in avoid:
            continue
        if usage.get(s, 0) == 0:
            return s
    # Fall back to least-used axis not in `avoid`
    candidates = [s for s in ALL_STRATEGIES if s not in avoid]
    return min(candidates, key=lambda s: usage.get(s, 0)) if candidates else ALL_STRATEGIES[0]
