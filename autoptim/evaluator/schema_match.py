"""Declarative schema-based evaluator.

Supports per-field matchers (exact / fuzzy / numeric / date_iso / list_of_objects / contains),
weighted aggregation to an overall score, and collection of representative failures.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from rapidfuzz.fuzz import ratio as _fuzz_ratio

from ..config import FieldSpec, MetricSpec
from ..runspec import EvalResult, Failure, FieldScore, PerDocScore


def _to_text(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _match_exact(expected: Any, predicted: Any) -> float:
    return 1.0 if _to_text(expected).lower() == _to_text(predicted).lower() else 0.0


def _match_contains(expected: Any, predicted: Any) -> float:
    e = _to_text(expected).lower()
    p = _to_text(predicted).lower()
    if not e:
        return 0.0
    return 1.0 if e in p else 0.0


def _match_fuzzy(expected: Any, predicted: Any, threshold: float) -> float:
    e = _to_text(expected)
    p = _to_text(predicted)
    if not e and not p:
        return 1.0
    score = _fuzz_ratio(e.lower(), p.lower()) / 100.0
    return score if score >= threshold else 0.0


def _to_number(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = _to_text(v)
    s = re.sub(r"[^\d.\-]", "", s)
    if not s or s in {".", "-", "-."}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _match_numeric(expected: Any, predicted: Any, tol: float) -> float:
    e = _to_number(expected)
    p = _to_number(predicted)
    if e is None or p is None:
        return 0.0
    return 1.0 if abs(e - p) <= tol else 0.0


_DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%B %d, %Y",
    "%b %d, %Y",
]


def _parse_date(v: Any) -> date | None:
    from datetime import datetime

    if v is None:
        return None
    if isinstance(v, date):
        return v
    s = _to_text(v)
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _match_date(expected: Any, predicted: Any) -> float:
    e = _parse_date(expected)
    p = _parse_date(predicted)
    if e is None or p is None:
        return 0.0
    return 1.0 if e == p else 0.0


def _match_list_of_objects(
    expected: Any,
    predicted: Any,
    key: str,
    subfields: dict[str, FieldSpec],
) -> float:
    if not isinstance(expected, list) or not isinstance(predicted, list):
        return 0.0
    if not expected and not predicted:
        return 1.0
    if not expected:
        return 0.0
    pred_by_key = {}
    for item in predicted:
        if isinstance(item, dict) and key in item:
            pred_by_key[_to_text(item[key]).lower()] = item
    scores = []
    for item in expected:
        if not isinstance(item, dict):
            continue
        k = _to_text(item.get(key, "")).lower()
        p_item = pred_by_key.get(k)
        if p_item is None:
            scores.append(0.0)
            continue
        sub_scores: list[tuple[float, float]] = []
        for fname, fspec in subfields.items():
            s = _score_field(fspec, item.get(fname), p_item.get(fname))
            sub_scores.append((fspec.weight, s))
        if not sub_scores:
            scores.append(1.0)
        else:
            total_w = sum(w for w, _ in sub_scores) or 1.0
            scores.append(sum(w * s for w, s in sub_scores) / total_w)
    return sum(scores) / len(scores) if scores else 0.0


def _score_field(spec: FieldSpec, expected: Any, predicted: Any) -> float:
    if spec.match == "exact":
        return _match_exact(expected, predicted)
    if spec.match == "contains":
        return _match_contains(expected, predicted)
    if spec.match == "fuzzy":
        return _match_fuzzy(expected, predicted, spec.threshold or 0.85)
    if spec.match == "numeric":
        return _match_numeric(expected, predicted, spec.tol or 0.01)
    if spec.match == "date_iso":
        return _match_date(expected, predicted)
    if spec.match == "list_of_objects":
        return _match_list_of_objects(
            expected, predicted, spec.key or "key", spec.fields or {}
        )
    return 0.0


@dataclass
class SchemaMatchEvaluator:
    metric: MetricSpec

    def score(
        self,
        predictions: list[dict[str, Any]],
        ground_truth: dict[str, dict[str, Any]],
    ) -> EvalResult:
        assert self.metric.schema_ is not None, "schema_match requires metric.schema"
        schema = self.metric.schema_
        total_w = sum(spec.weight for spec in schema.values()) or 1.0

        pred_by_id: dict[str, Any] = {}
        for p in predictions:
            if not isinstance(p, dict) or "id" not in p:
                continue
            pred_by_id[p["id"]] = p.get("prediction") or {}

        per_doc: list[PerDocScore] = []
        failures: list[Failure] = []

        for doc_id, expected_fields in ground_truth.items():
            pred = pred_by_id.get(doc_id)
            if pred is None:
                per_doc.append(
                    PerDocScore(
                        id=doc_id,
                        score=0.0,
                        fields=[],
                        error="missing prediction",
                    )
                )
                failures.append(
                    Failure(
                        id=doc_id,
                        summary="no prediction returned",
                        expected=expected_fields,
                        predicted=None,
                    )
                )
                continue

            field_scores: list[FieldScore] = []
            weighted_sum = 0.0
            for fname, fspec in schema.items():
                exp_v = expected_fields.get(fname)
                pred_v = pred.get(fname) if isinstance(pred, dict) else None
                s = _score_field(fspec, exp_v, pred_v)
                field_scores.append(
                    FieldScore(
                        name=fname,
                        expected=exp_v,
                        predicted=pred_v,
                        match=s > 0.0,
                        score=s,
                    )
                )
                weighted_sum += fspec.weight * s
            doc_score = weighted_sum / total_w
            per_doc.append(PerDocScore(id=doc_id, score=doc_score, fields=field_scores))

        # Aggregate
        if self.metric.aggregate == "mean_score":
            overall = sum(d.score for d in per_doc) / len(per_doc) if per_doc else 0.0
        elif self.metric.aggregate == "min_score":
            overall = min((d.score for d in per_doc), default=0.0)
        else:  # weighted_f1 == mean of per-doc weighted scores for this implementation
            overall = sum(d.score for d in per_doc) / len(per_doc) if per_doc else 0.0

        # Build 3 worst failures (lowest score) for prompt feedback
        worst = sorted(per_doc, key=lambda d: d.score)[:3]
        for d in worst:
            if d.score >= 0.999:
                continue
            missed = [f for f in d.fields if not f.match]
            summary = "; ".join(
                f"{f.name}: expected={f.expected!r} got={f.predicted!r}" for f in missed[:4]
            ) or (d.error or "unknown failure")
            failures.append(
                Failure(
                    id=d.id,
                    summary=summary,
                    expected={f.name: f.expected for f in d.fields},
                    predicted={f.name: f.predicted for f in d.fields},
                )
            )

        return EvalResult(
            metric_name=f"schema_match/{self.metric.aggregate}",
            overall=overall,
            target=self.metric.target,
            per_doc=per_doc,
            failures=failures,
        )
