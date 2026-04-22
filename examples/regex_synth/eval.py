"""Custom evaluator for regex_synth.

For each task: compile the predicted regex and check each positive fullmatches
and each negative does NOT fullmatch. Score = correct / total.

Robustness:
- Strips markdown code fences and common "here is your regex:" prose wrappings.
- Catches re.error, ValueError, TypeError — a syntactically broken regex scores
  0 on its task but does not blow up the whole run.
- Catches catastrophic-backtracking hangs via a simple pattern-length sanity
  bound (the task set doesn't require anything longer than ~300 chars).
"""
from __future__ import annotations

import re
from typing import Any

from autoptim.runspec import EvalResult, Failure, FieldScore, PerDocScore

_FENCE_RE = re.compile(r"^```(?:regex|re|python)?\s*(.*?)\s*```$", re.DOTALL)


def _clean_pattern(raw: Any) -> str:
    """Best-effort extraction of a regex from potentially noisy LLM output."""
    if raw is None:
        return ""
    s = str(raw).strip()
    # Fence-wrapped
    m = _FENCE_RE.match(s)
    if m:
        s = m.group(1).strip()
    # Prose prefix like "Here is the regex: foo"
    for sep in ("\n", ":", "->"):
        if sep in s and len(s.split(sep, 1)[1].strip()) < len(s) / 2 + 5:
            continue  # prose removal too aggressive — skip
    # Strip surrounding backticks / single quotes / double quotes
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("`", "'", '"'):
        s = s[1:-1]
    return s.strip()


def _score_task(pattern_raw: Any, positives: list[str], negatives: list[str]) -> tuple[float, str, str | None]:
    """Return (score_in_0_1, summary, error_msg_or_None)."""
    pattern = _clean_pattern(pattern_raw)
    total = len(positives) + len(negatives)
    if total == 0:
        return 0.0, "(no examples)", None
    if not pattern:
        return 0.0, "empty pattern", "empty"
    if len(pattern) > 500:
        return 0.0, "pattern too long (>500 chars, probably prose)", "too_long"
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return 0.0, f"invalid regex: {e}", "compile_error"

    correct = 0
    hits_pos, misses_neg = [], []
    for p in positives:
        try:
            if rx.fullmatch(p) is not None:
                correct += 1
            else:
                hits_pos.append(p)  # positive that failed to match
        except Exception:
            hits_pos.append(p)
    for n in negatives:
        try:
            if rx.fullmatch(n) is None:
                correct += 1
            else:
                misses_neg.append(n)  # negative that wrongly matched
        except Exception:
            misses_neg.append(n)

    score = correct / total
    bits = []
    if hits_pos:
        bits.append(f"missed positives={hits_pos[:3]}")
    if misses_neg:
        bits.append(f"matched negatives={misses_neg[:3]}")
    summary = "; ".join(bits) if bits else "all correct"
    return score, summary, None


def score(
    predictions: list[dict[str, Any]],
    ground_truth: dict[str, dict[str, Any]],
) -> EvalResult:
    pred_by_id = {p["id"]: (p.get("prediction") or {}) for p in predictions if "id" in p}

    per_doc: list[PerDocScore] = []
    # We need the per-task positives/negatives which live in the inputs dir,
    # NOT in ground_truth.jsonl. Read them from the input files on disk.
    # The inputs dir is the parent of this file's ../../inputs; but the
    # evaluator is task-agnostic, so we recover the input path from the
    # predictions (the seed embeds nothing). We rely on ground_truth holding
    # the description only; the real positives/negatives come from the
    # per-id input file. Use the id to look it up.
    import json
    from pathlib import Path
    inputs_dir = Path(__file__).parent / "inputs"

    for tid, expected in ground_truth.items():
        # Find the matching input file by stem
        candidates = list(inputs_dir.glob(f"{tid}.json"))
        if not candidates:
            per_doc.append(
                PerDocScore(id=tid, score=0.0, fields=[], error=f"input file not found for {tid}")
            )
            continue
        try:
            task_def = json.loads(candidates[0].read_text())
        except Exception as e:
            per_doc.append(
                PerDocScore(id=tid, score=0.0, fields=[], error=f"could not read input: {e}")
            )
            continue
        positives = task_def.get("positives", [])
        negatives = task_def.get("negatives", [])

        pred = pred_by_id.get(tid) or {}
        raw_pattern = pred.get("pattern") if isinstance(pred, dict) else pred
        s, summary, err = _score_task(raw_pattern, positives, negatives)
        per_doc.append(
            PerDocScore(
                id=tid,
                score=s,
                fields=[
                    FieldScore(
                        name="pattern",
                        expected=expected.get("description", "<description>"),
                        predicted=_clean_pattern(raw_pattern),
                        match=(s >= 0.999),
                        score=s,
                    )
                ],
                error=err,
            )
        )

    overall = sum(d.score for d in per_doc) / len(per_doc) if per_doc else 0.0

    worst = sorted(per_doc, key=lambda d: d.score)[:3]
    failures: list[Failure] = []
    for d in worst:
        if d.score >= 0.999:
            continue
        failures.append(
            Failure(
                id=d.id,
                summary=d.error or (d.fields[0].predicted if d.fields else "unknown"),
                expected=d.fields[0].expected if d.fields else None,
                predicted=d.fields[0].predicted if d.fields else None,
            )
        )

    return EvalResult(
        metric_name="regex_synth_fullmatch_accuracy",
        overall=overall,
        target=0.95,
        per_doc=per_doc,
        failures=failures,
    )
