"""Tests for the meta-agent tool-call schema: what gets accepted vs. rejected."""
import pytest
from pydantic import ValidationError

from autoptim.meta.agent import ProposedIteration


_VALID_PROCESS = '''
def run(inputs, ctx):
    return [{"id": i["id"], "prediction": {}} for i in inputs]
'''


_VALID_RATIONALE = (
    "The latest eval shows 3 docs failing on the `total` field, "
    "specifically when the invoice includes a tax line. "
    "Prior attempts tightened the schema; none addressed tax handling."
)


def test_valid_proposal_accepted():
    p = ProposedIteration(
        rationale=_VALID_RATIONALE,
        hypothesis="Tightening the JSON schema will reduce parse failures.",
        strategy_tag="prompt_mutation",
        process_py=_VALID_PROCESS,
        predicted_delta=0.05,
        expected_failure_modes=["model may still leak markdown"],
    )
    assert p.strategy_tag == "prompt_mutation"


def test_rejects_unknown_strategy():
    with pytest.raises(ValidationError):
        ProposedIteration(
            rationale=_VALID_RATIONALE,
            hypothesis="x",
            strategy_tag="refactor_everything",
            process_py=_VALID_PROCESS,
            predicted_delta=0.0,
            expected_failure_modes=[],
        )


def test_rejects_process_without_run():
    with pytest.raises(ValidationError):
        ProposedIteration(
            rationale=_VALID_RATIONALE,
            hypothesis="xxxxx",
            strategy_tag="prompt_mutation",
            process_py="def other(): pass",
            predicted_delta=0.0,
            expected_failure_modes=[],
        )


def test_rejects_short_hypothesis():
    with pytest.raises(ValidationError):
        ProposedIteration(
            rationale=_VALID_RATIONALE,
            hypothesis="no",
            strategy_tag="prompt_mutation",
            process_py=_VALID_PROCESS,
            predicted_delta=0.0,
            expected_failure_modes=[],
        )


def test_rejects_short_rationale():
    with pytest.raises(ValidationError):
        ProposedIteration(
            rationale="too short",
            hypothesis="Doing X improves Y because Z.",
            strategy_tag="prompt_mutation",
            process_py=_VALID_PROCESS,
            predicted_delta=0.0,
            expected_failure_modes=[],
        )
