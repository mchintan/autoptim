from autoptim.config import FieldSpec, MetricSpec
from autoptim.evaluator.schema_match import SchemaMatchEvaluator


def _make_eval(**overrides):
    schema = {
        "invoice_no": FieldSpec(match="exact", weight=1),
        "date": FieldSpec(match="date_iso", weight=1),
        "vendor": FieldSpec(match="fuzzy", threshold=0.8, weight=1),
        "total": FieldSpec(match="numeric", tol=0.01, weight=2),
        "currency": FieldSpec(match="exact", weight=1),
    }
    metric = MetricSpec(type="schema_match", schema=schema, aggregate="mean_score", target=0.9, **overrides)
    return SchemaMatchEvaluator(metric)


def test_perfect_match():
    ev = _make_eval()
    gt = {
        "a": {"invoice_no": "X-1", "date": "2024-03-15", "vendor": "ACME Inc", "total": 100.00, "currency": "USD"},
    }
    preds = [{"id": "a", "prediction": dict(gt["a"])}]
    result = ev.score(preds, gt)
    assert abs(result.overall - 1.0) < 1e-6
    assert result.per_doc[0].score == 1.0


def test_numeric_tolerance():
    ev = _make_eval()
    gt = {"a": {"invoice_no": "X", "date": "2024-01-01", "vendor": "V", "total": 100.0, "currency": "USD"}}
    preds = [{"id": "a", "prediction": {"invoice_no": "X", "date": "2024-01-01", "vendor": "V", "total": 100.005, "currency": "USD"}}]
    assert ev.score(preds, gt).per_doc[0].score == 1.0


def test_numeric_fail_outside_tol():
    ev = _make_eval()
    gt = {"a": {"invoice_no": "X", "date": "2024-01-01", "vendor": "V", "total": 100.0, "currency": "USD"}}
    preds = [{"id": "a", "prediction": {"invoice_no": "X", "date": "2024-01-01", "vendor": "V", "total": 105.0, "currency": "USD"}}]
    result = ev.score(preds, gt)
    # Total weight=2, others 1+1+1+1=4; missing 2/6 of weighted score
    assert abs(result.per_doc[0].score - (4 / 6)) < 1e-6


def test_fuzzy_threshold():
    ev = _make_eval()
    gt = {"a": {"invoice_no": "X", "date": "2024-01-01", "vendor": "ACME Widgets Inc", "total": 1.0, "currency": "USD"}}
    preds = [{"id": "a", "prediction": {"invoice_no": "X", "date": "2024-01-01", "vendor": "ACME Widgets", "total": 1.0, "currency": "USD"}}]
    # Should pass fuzzy ratio above 0.80
    assert ev.score(preds, gt).per_doc[0].fields[2].match


def test_date_multiple_formats():
    ev = _make_eval()
    gt = {"a": {"invoice_no": "X", "date": "2024-03-15", "vendor": "V", "total": 1.0, "currency": "USD"}}
    preds = [{"id": "a", "prediction": {"invoice_no": "X", "date": "03/15/2024", "vendor": "V", "total": 1.0, "currency": "USD"}}]
    assert ev.score(preds, gt).per_doc[0].score == 1.0


def test_missing_prediction_counts_zero():
    ev = _make_eval()
    gt = {"a": {"invoice_no": "X", "date": "2024-01-01", "vendor": "V", "total": 1.0, "currency": "USD"}}
    result = ev.score([], gt)
    assert result.per_doc[0].score == 0.0
    assert result.per_doc[0].error == "missing prediction"
    assert result.failures


def test_worst_failures_captured():
    ev = _make_eval()
    gt = {
        "perfect": {"invoice_no": "X", "date": "2024-01-01", "vendor": "V", "total": 1.0, "currency": "USD"},
        "bad": {"invoice_no": "Y", "date": "2024-01-01", "vendor": "V", "total": 1.0, "currency": "USD"},
    }
    preds = [
        {"id": "perfect", "prediction": dict(gt["perfect"])},
        {"id": "bad", "prediction": {"invoice_no": "Z", "date": "2024-01-01", "vendor": "V", "total": 1.0, "currency": "USD"}},
    ]
    result = ev.score(preds, gt)
    assert any(f.id == "bad" for f in result.failures)
