"""Custom evaluator for quadratic_nn.

Compares predicted root pairs to ground-truth root pairs AFTER sorting both —
roots are a set, not an ordered tuple.

Scoring is *continuous* so the meta-agent gets smooth gradient signal even
when the seed is very wrong. For each item:

    mean_abs_err = (|p1 - e1| + |p2 - e2|) / 2
    score_per_item = max(0, 1 - mean_abs_err / ERR_SCALE)

A random/untrained predictor on roots in [-5, 5] typically has mean error ~2,
giving score ~0. A model within ~0.05 of both roots scores ~0.975. The field
is marked "matched" (for the failures panel) when mean error is under TOL_TIGHT.
"""
from __future__ import annotations

from typing import Any

from autoptim.runspec import EvalResult, Failure, FieldScore, PerDocScore

ERR_SCALE = 2.0      # the denominator in the continuous score — controls how
                     #  forgiving the curve is
TOL_TIGHT = 0.05     # "matched" flag cutoff used for the worst-failures panel


def _coerce_pair(v: Any) -> list[float] | None:
    """Accept `roots: [r1, r2]` or a dict with r1/r2 keys or x1/x2 keys."""
    if isinstance(v, list) and len(v) == 2:
        try:
            return sorted(float(x) for x in v)
        except (TypeError, ValueError):
            return None
    if isinstance(v, dict):
        for k1, k2 in (("r1", "r2"), ("x1", "x2"), ("root1", "root2")):
            if k1 in v and k2 in v:
                try:
                    return sorted([float(v[k1]), float(v[k2])])
                except (TypeError, ValueError):
                    return None
    return None


def score(
    predictions: list[dict[str, Any]],
    ground_truth: dict[str, dict[str, Any]],
) -> EvalResult:
    pred_by_id = {p["id"]: (p.get("prediction") or {}) for p in predictions if "id" in p}

    per_doc: list[PerDocScore] = []
    for qid, expected in ground_truth.items():
        exp_pair = _coerce_pair(expected.get("roots"))
        pred = pred_by_id.get(qid)
        pred_pair = _coerce_pair((pred or {}).get("roots")) if isinstance(pred, dict) else None

        if exp_pair is None:
            per_doc.append(
                PerDocScore(
                    id=qid,
                    score=0.0,
                    fields=[],
                    error="malformed ground truth (expected 'roots' to be a [r1, r2] list)",
                )
            )
            continue

        if pred_pair is None:
            per_doc.append(
                PerDocScore(
                    id=qid,
                    score=0.0,
                    fields=[
                        FieldScore(
                            name="roots",
                            expected=exp_pair,
                            predicted=(pred or {}).get("roots"),
                            match=False,
                            score=0.0,
                        )
                    ],
                    error="missing or malformed prediction",
                )
            )
            continue

        err1 = abs(pred_pair[0] - exp_pair[0])
        err2 = abs(pred_pair[1] - exp_pair[1])
        mean_err = (err1 + err2) / 2.0
        item_score = max(0.0, 1.0 - mean_err / ERR_SCALE)
        per_doc.append(
            PerDocScore(
                id=qid,
                score=item_score,
                fields=[
                    FieldScore(
                        name="roots",
                        expected=exp_pair,
                        predicted=pred_pair,
                        match=mean_err <= TOL_TIGHT,
                        score=item_score,
                    )
                ],
            )
        )

    overall = sum(d.score for d in per_doc) / len(per_doc) if per_doc else 0.0

    worst = sorted(per_doc, key=lambda d: d.score)[:3]
    failures: list[Failure] = []
    for d in worst:
        if d.score >= 0.999:
            continue
        summary = d.error or f"pred={d.fields[0].predicted if d.fields else None} exp={d.fields[0].expected if d.fields else None}"
        failures.append(
            Failure(
                id=d.id,
                summary=summary,
                expected={"roots": d.fields[0].expected if d.fields else None},
                predicted={"roots": d.fields[0].predicted if d.fields else None},
            )
        )

    return EvalResult(
        metric_name=f"quadratic_roots_continuous_err_scale_{ERR_SCALE}",
        overall=overall,
        target=0.90,
        per_doc=per_doc,
        failures=failures,
    )
