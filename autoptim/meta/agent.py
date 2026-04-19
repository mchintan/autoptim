"""Meta-agent: renders prompts, calls the frontier LLM, validates the tool call."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel, Field, ValidationError, field_validator

from ..runspec import ALL_STRATEGIES, IterationRecord
from ..util.cost import CostTracker
from .providers import Provider, make_provider
from .strategy_scheduler import Directive, decide as decide_strategy


PROMPTS_DIR = Path(__file__).parent / "prompts"
CONTRACT_PATH = Path(__file__).resolve().parents[2] / "CONTRACT.md"


# --- Tool schema (also used for Pydantic validation) --------------------------

_TOOL_NAME = "propose_iteration"
_TOOL_DESCRIPTION = (
    "Propose the next `process.py` experiment. Return one full replacement "
    "file, a falsifiable hypothesis, a strategy tag, a predicted score delta, "
    "and expected failure modes."
)
_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rationale": {
            "type": "string",
            "description": (
                "Your reasoning for THIS specific experiment. Cite concrete evidence "
                "from the latest eval or history: which per-doc failures, which "
                "strategy gaps, which pattern in the prior scores. 3-6 sentences. "
                "This is how the operator tells whether the frontier budget is "
                "being spent on thinking or guessing."
            ),
        },
        "hypothesis": {
            "type": "string",
            "description": (
                "One to three sentences: the falsifiable claim this experiment tests. "
                "Form: 'Doing X will cause Y because Z.'"
            ),
        },
        "strategy_tag": {
            "type": "string",
            "enum": list(ALL_STRATEGIES),
            "description": "Which axis of experimentation you are exploring.",
        },
        "process_py": {
            "type": "string",
            "description": "Full contents of the new process.py file. No markdown fences.",
        },
        "predicted_delta": {
            "type": "number",
            "description": "Expected change in overall score vs. parent (can be negative).",
        },
        "expected_failure_modes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Short list of ways this hypothesis could turn out wrong.",
        },
    },
    "required": [
        "rationale",
        "hypothesis",
        "strategy_tag",
        "process_py",
        "predicted_delta",
        "expected_failure_modes",
    ],
    "additionalProperties": False,
}


class ProposedIteration(BaseModel):
    rationale: str = Field(min_length=20)
    hypothesis: str = Field(min_length=5)
    strategy_tag: str
    process_py: str = Field(min_length=10)
    predicted_delta: float
    expected_failure_modes: list[str] = Field(default_factory=list)

    @field_validator("strategy_tag")
    @classmethod
    def _valid_tag(cls, v: str) -> str:
        if v not in ALL_STRATEGIES:
            raise ValueError(f"strategy_tag must be one of {ALL_STRATEGIES}; got {v!r}")
        return v

    @field_validator("process_py")
    @classmethod
    def _has_run(cls, v: str) -> str:
        if "def run(" not in v:
            raise ValueError("process_py must define a `def run(inputs, ctx)` function")
        return v


# --- Agent --------------------------------------------------------------------


@dataclass
class MetaCallResult:
    proposal: ProposedIteration
    tokens_in: int
    tokens_out: int
    usd: float
    directive: Directive


class MetaAgent:
    def __init__(
        self,
        *,
        provider: str,
        api_key: str,
        model: str,
        task_name: str,
        metric_name: str,
        target: float,
        worker_models: list[str],
        max_history: int = 5,
        temperature: float | None = None,
    ):
        self._provider: Provider = make_provider(provider, api_key)
        self._provider_name = provider
        self.model = model
        self.task_name = task_name
        self.metric_name = metric_name
        self.target = target
        self.worker_models = worker_models
        self.max_history = max_history
        self.temperature = temperature
        self._env = Environment(
            loader=FileSystemLoader(str(PROMPTS_DIR)),
            autoescape=select_autoescape(disabled_extensions=("j2", "md")),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._contract_text = CONTRACT_PATH.read_text() if CONTRACT_PATH.exists() else ""
        self._strategies_text = (PROMPTS_DIR / "strategies.md").read_text()

    def _render_system(self) -> str:
        return self._env.get_template("system.j2").render(
            task_name=self.task_name,
            metric_name=self.metric_name,
            target=self.target,
            worker_models=", ".join(self.worker_models) or "(none — worker endpoint unreachable)",
            contract=self._contract_text,
            strategies=self._strategies_text,
        )

    def _render_user(
        self,
        *,
        iter_n: int,
        max_iters: int,
        wall_used_h: float,
        wall_cap_h: float,
        usd_spent: float,
        usd_cap: float,
        parent_process_py: str,
        parent_score: float | None,
        eval_json: dict[str, Any],
        history: list[IterationRecord],
        best: dict[str, Any] | None,
        directive: Directive,
        worker_log_tail: str = "",
    ) -> str:
        hist_view = [
            {
                "iter": h.iter,
                "parent": h.parent,
                "strategy_tag": h.strategy_tag,
                "hypothesis": (h.hypothesis or "").replace("\n", " ")[:120],
                "score": f"{h.score:.3f}" if h.score is not None else "—",
                "delta": f"{h.delta:+.3f}" if h.delta is not None else "—",
            }
            for h in history[-self.max_history :]
        ]
        # Trim eval for prompt budget
        eval_trimmed = {
            "overall": f"{eval_json.get('overall', 0):.3f}",
            "per_doc": [
                {"id": d["id"], "score": f"{d['score']:.3f}", "error": d.get("error")}
                for d in eval_json.get("per_doc", [])
            ],
            "failures": eval_json.get("failures", [])[:3],
        }
        return self._env.get_template("iteration.j2").render(
            iter=iter_n,
            max_iters=max_iters,
            wall_used_h=f"{wall_used_h:.2f}",
            wall_cap_h=f"{wall_cap_h:.2f}",
            usd_spent=f"{usd_spent:.2f}",
            usd_cap=f"{usd_cap:.2f}",
            usd_remaining=f"{max(0.0, usd_cap - usd_spent):.2f}",
            target=f"{self.target:.3f}",
            best=best,
            history=hist_view,
            strategy_directive=directive.message,
            forced_strategy=directive.forced_strategy,
            parent_process_py=parent_process_py,
            parent_score=f"{parent_score:.3f}" if parent_score is not None else "—",
            eval=eval_trimmed,
            worker_log_tail=worker_log_tail.strip(),
        )

    def propose(
        self,
        *,
        iter_n: int,
        max_iters: int,
        wall_used_h: float,
        wall_cap_h: float,
        cost_tracker: CostTracker,
        parent_process_py: str,
        parent_score: float | None,
        eval_json: dict[str, Any],
        history: list[IterationRecord],
        best: dict[str, Any] | None,
        worker_log_tail: str = "",
    ) -> MetaCallResult:
        directive = decide_strategy(history)
        system = self._render_system()
        user = self._render_user(
            iter_n=iter_n,
            max_iters=max_iters,
            wall_used_h=wall_used_h,
            wall_cap_h=wall_cap_h,
            usd_spent=cost_tracker.spent_usd,
            usd_cap=cost_tracker.cap_usd,
            parent_process_py=parent_process_py,
            parent_score=parent_score,
            eval_json=eval_json,
            history=history,
            best=best,
            directive=directive,
            worker_log_tail=worker_log_tail,
        )

        # Preflight cost check using a rough estimator
        est_in = _rough_tokens(system) + _rough_tokens(user)
        est_out = 4000  # process.py can be big
        ok, projected = cost_tracker.preflight(est_in, est_out)
        if not ok:
            raise BudgetExceeded(
                f"meta call would project to ${projected:.2f} — over cap ${cost_tracker.cap_usd:.2f}"
            )

        # First try
        attempt = 0
        last_err: str | None = None
        while attempt < 2:
            attempt += 1
            resp = self._provider.call_tool(
                system=system,
                user=user if attempt == 1 else user + _retry_suffix(last_err or ""),
                tool_name=_TOOL_NAME,
                tool_description=_TOOL_DESCRIPTION,
                tool_schema=_TOOL_SCHEMA,
                model=self.model,
                temperature=self.temperature,
                max_tokens=8000,
            )
            cost_tracker.record(resp.tokens_in, resp.tokens_out)
            try:
                proposal = ProposedIteration.model_validate(resp.tool_args)
                # Enforce forced strategy if any
                if directive.forced_strategy and proposal.strategy_tag != directive.forced_strategy:
                    raise ValidationError.from_exception_data(  # type: ignore[call-arg]
                        "ProposedIteration",
                        [
                            {
                                "type": "value_error",
                                "loc": ("strategy_tag",),
                                "msg": f"scheduler forced `{directive.forced_strategy}`",
                                "input": proposal.strategy_tag,
                                "ctx": {"error": ValueError("forced")},
                            }
                        ],
                    )
                return MetaCallResult(
                    proposal=proposal,
                    tokens_in=resp.tokens_in,
                    tokens_out=resp.tokens_out,
                    usd=cost_tracker.spent_usd,
                    directive=directive,
                )
            except ValidationError as e:
                last_err = str(e)
                continue
        raise MetaValidationError(
            f"meta-agent produced invalid tool call after 2 attempts: {last_err}"
        )


class BudgetExceeded(RuntimeError):
    pass


class MetaValidationError(RuntimeError):
    pass


def _rough_tokens(text: str) -> int:
    # ~4 chars per token is the standard-enough approximation for budgeting.
    return max(1, len(text) // 4)


def _retry_suffix(err: str) -> str:
    return (
        "\n\n---\n\n**RETRY**: your previous response was rejected with this "
        f"validation error:\n\n{err}\n\nFix the issue and respond again with a "
        "single valid `propose_iteration` tool call."
    )
