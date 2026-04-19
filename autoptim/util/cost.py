"""Token -> USD estimation for the meta-agent.

Prices are in USD per 1M tokens. Keep these conservative (high-side) so the
budget guard errs on the side of stopping early. Update as providers change.
"""
from __future__ import annotations

from dataclasses import dataclass


# Conservative public list prices, per 1M tokens.
PRICES: dict[tuple[str, str], tuple[float, float]] = {
    # (provider, model_prefix) -> (in_per_mtok, out_per_mtok)
    ("openai", "gpt-4o-mini"): (0.15, 0.60),
    ("openai", "gpt-4o"): (2.50, 10.0),
    ("openai", "o1"): (15.0, 60.0),
    ("openai", "gpt-4.1"): (2.0, 8.0),
    # openrouter is pass-through; we charge on the listed model prefix if known
    ("openrouter", "openai/gpt-4o"): (2.50, 10.0),
    # Gemini — conservative estimates (real prices vary by tier + context length)
    ("gemini", "gemini-3-pro"): (5.0, 20.0),
    ("gemini", "gemini-3"): (5.0, 20.0),
    ("gemini", "gemini-2.5-pro"): (1.25, 10.0),
    ("gemini", "gemini-2.5-flash"): (0.30, 2.50),
    ("gemini", "gemini-2.0-pro"): (1.25, 5.0),
    ("gemini", "gemini-2.0-flash"): (0.15, 0.60),
    ("gemini", "gemini-1.5-pro"): (1.25, 5.0),
    ("gemini", "gemini-1.5-flash"): (0.15, 0.60),
}

# Safe default if the model string doesn't match any prefix — priced on the high
# side so the budget guard errs toward stopping early.
DEFAULT_PRICE: tuple[float, float] = (5.0, 20.0)


def price_for(provider: str, model: str) -> tuple[float, float]:
    best: tuple[float, float] | None = None
    best_len = -1
    for (p, prefix), rate in PRICES.items():
        if p == provider and model.startswith(prefix) and len(prefix) > best_len:
            best = rate
            best_len = len(prefix)
    return best or DEFAULT_PRICE


def estimate_cost(provider: str, model: str, tokens_in: int, tokens_out: int) -> float:
    pin, pout = price_for(provider, model)
    return (tokens_in / 1_000_000.0) * pin + (tokens_out / 1_000_000.0) * pout


@dataclass
class CostTracker:
    cap_usd: float
    provider: str
    model: str
    spent_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0

    def preflight(self, est_tokens_in: int, est_tokens_out: int) -> tuple[bool, float]:
        """Return (ok_to_proceed, projected_total_after_call)."""
        projected = self.spent_usd + estimate_cost(
            self.provider, self.model, est_tokens_in, est_tokens_out
        )
        return projected <= self.cap_usd, projected

    def record(self, tokens_in: int, tokens_out: int) -> float:
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        delta = estimate_cost(self.provider, self.model, tokens_in, tokens_out)
        self.spent_usd += delta
        return delta

    def remaining(self) -> float:
        return max(0.0, self.cap_usd - self.spent_usd)
